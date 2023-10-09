import torch
from torch import nn
from torch.distributions import Categorical
import pytorch_lightning as pl
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
        
        self.fc1 = nn.Linear(embedding_size, embedding_size*2)
        self.activation = SwiGLU(size=embedding_size*2)
        self.fc2 = nn.Linear(embedding_size*2, embedding_size)
    
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
        n_blocks: int
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention_block = nn.ModuleList([LlamaSelfAttentionBlock(hidden_size, context_len, causal_attention, n_heads) for _ in range(n_blocks)])
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
        ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x, is_inference=False):
        return self.model(x, is_inference)
    
    def predict(self, x):
        # Predict the last next word, from all the previous ones
        last_word = self(x, is_inference=False)[:,-1,:]
        return torch.softmax(last_word, dim=1)
    
    def _single_generate(self, idx, context_len):
        # Generate the next token
        probs = self.predict(idx[:,-context_len:])
        m = Categorical(probs)
        idx_next_token = m.sample()
        return idx_next_token.reshape(-1, 1)
    
    def generate(self, idx, context_len, max_output_token):
        for _ in range(max_output_token):
            next_token = self._single_generate(idx, context_len)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
            
    def training_step(self, batch, batch_idx):
        _, loss = self._get_preds_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx, split_name="val"):
        _, loss = self._get_preds_loss(batch)
        self.log(f'{split_name}_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split_name="test")
    
    def _get_preds_loss(self, batch):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.view(-1, y_hat.shape[-1]), y.view(-1))
        return y_hat, loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3) # weight_decay = , betas=()
    
