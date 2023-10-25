import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
import pytorch_lightning as pl
import wandb
from baby_llama.utils import get_rotary_matrix, RMSnorm, SwiGLU


def causal_mask(size, device):
    x = torch.full(size, float("-inf"))
    return torch.triu(x, diagonal=1).to(device=device)
    
class LlamaSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        context_len: int,
        causal_attention: bool,
        n_heads: int,
        swiglu_d_moltiplier: float
        ):
        super().__init__()
        # self.embedding_size = torch.tensor(embedding_size)
        self.embedding_size = embedding_size
        self.causal_attention = causal_attention
        self.n_heads = n_heads
        assert self.embedding_size % self.n_heads == 0, f"Embedding size ({self.embedding_size}) must be divisable by the number of heads ({self.n_heads})"
        self.head_dim = self.embedding_size // self.n_heads
        self.R = get_rotary_matrix(context_len=context_len, embedding_dim=self.head_dim)   
        self.rms = RMSnorm(size=embedding_size)
        self.ff_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_v = nn.Linear(embedding_size, embedding_size, bias=False)
        
        # In Llama paper swiglu_d_moltiplier = 2/3 * 4 
        swiglu_size = int(swiglu_d_moltiplier * embedding_size)
        self.fc1 = nn.Linear(embedding_size, swiglu_size)
        self.activation = SwiGLU(size=swiglu_size)
        self.fc2 = nn.Linear(swiglu_size, embedding_size)
    
    def forward(self, x, is_inference=False):
        if is_inference:
            breakpoint()
        input_shape = x.shape
        resize = (x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        x_res = x
        x = self.rms(x) # pre-normalization
        query = self.ff_q(x).reshape(resize)
        key = self.ff_k(x).reshape(resize)
        value = self.ff_v(x).reshape(resize)
                
        # Apply rotation to query and key, separatly for each head  
        R_matrix = self.R[:resize[1], :, :].to(query.device) 
        query_rot = torch.einsum('bhld,ldd->bhld', query.permute(0,2,1,3), R_matrix)
        key_rot = torch.einsum('bhdl,ldd->bhdl', key.permute(0,2,3,1), R_matrix)
        
        score = query_rot @ key_rot
        if self.causal_attention:
            score += causal_mask(size=score.shape, device=score.device)
        score = score / torch.sqrt(torch.tensor(self.head_dim)) 
        attention = torch.softmax(score, dim=-1) 
        x = attention @ value.permute(0,2,1,3)
        x = x.permute(0, 2, 1, 3).reshape(input_shape)
        x += x_res
        
        x_res = x
        x = self.rms(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + x_res
        

class Llama(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        context_len: int,
        causal_attention: bool,
        n_heads: int,
        n_blocks: int,
        swiglu_d_moltiplier: float,
        ):
        super().__init__()
        self.context_len = context_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention_block = nn.ModuleList([LlamaSelfAttentionBlock(hidden_size, context_len, causal_attention, n_heads, swiglu_d_moltiplier) for _ in range(n_blocks)])
        self.unembedding = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, is_inference=False):
        x = self.embedding(x)
        for single_block in self.attention_block:
            x = single_block(x, is_inference)
        x = self.unembedding(x)
        return x


class SimpleModule(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        tokenizer: AutoTokenizer,
        ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.tokenizer = tokenizer
        
        self.logger_table_data = []   
        
    def forward(self, x, is_inference=False):
        return self.model(x, is_inference)
    
    def _single_generate(self, idx, context_len, temperature, top_k, top_p, greedy):
        logits = self(idx[:, -context_len:], is_inference=False)[:, -1, :]
        logits = logits / temperature
        
        if greedy:
            return torch.argmax(logits, dim=1).reshape(-1, 1)
        
        # Initialize mask with ones
        mask = torch.ones_like(logits).bool()
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1)
            sorted_mask = cumulative_probs > top_p
            # Ensure at least the most probable is included if sorted_mask contains all True 
            if sorted_mask.all():
                sorted_mask[..., :1] = 0
            to_scatter = sorted_mask.type_as(logits) * float('-inf')
            to_scatter[sorted_mask == 0] = logits.gather(1, sorted_indices)[sorted_mask == 0]
            logits.scatter_(1, sorted_indices, to_scatter)
        elif top_k > 0:
            top_k = min(top_k, logits.shape[1])            
            values, _ = torch.topk(logits, top_k)
            # smallest allowed value
            kth_values = values[..., -1]
            logits = torch.where(logits < kth_values.unsqueeze(-1), torch.tensor(float('-inf')).type_as(logits), logits)
    
                    
        probs = torch.softmax(logits, dim=1)
        m = Categorical(probs)
        idx_next_token = m.sample()
        return idx_next_token.reshape(-1, 1)

    def generate(self, context_len, max_output_token, temperature=1, top_k=0, top_p=0.9, greedy=False):
        idx = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0).to(self.device)
        for _ in range(max_output_token):
            next_token = self._single_generate(idx, context_len, temperature, top_k, top_p, greedy)
            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        decoded = self.tokenizer.decode(idx[0], skip_special_tokens=False)
        return idx, decoded
            
    def training_step(self, batch, batch_idx):
        _, loss = self._get_preds_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx, split_name="val"):
        _, loss = self._get_preds_loss(batch)
        self.log(f'{split_name}_loss', loss)        
        return loss
    
    def on_validation_end(self) -> None:
        _, output_decoded = self.generate(context_len=self.model.context_len, max_output_token=50)
        print(f"Full Text: \n{output_decoded}")
        current_epoch = len(self.logger_table_data) -1
        self.logger_table_data.append([current_epoch, output_decoded])
        self.logger.log_table(key="Example Text Generation", columns=["Epoch", "Text"], data=self.logger_table_data, )
        return super().on_validation_end()
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split_name="test")
    
    def _get_preds_loss(self, batch):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.view(-1, y_hat.shape[-1]), y.view(-1))
        return y_hat, loss
    
    def configure_optimizers(self):
        max_step = self.trainer.max_epochs * (len(self.trainer.datamodule.train_dataloader()))
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay = 0.1, betas=(0.9, 0.95))
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=3e-4,  # Maximum learning rate
                total_steps=max_step,  # Total number of training steps
                pct_start=0.03,  # Percentage of steps spent in the warmup phase
                anneal_strategy='cos',  # Cosine annealing
            ),
            'interval': 'step',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
