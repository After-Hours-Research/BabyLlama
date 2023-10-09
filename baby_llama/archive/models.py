import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        hidden_size, 
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.unembedding = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.unembedding(x)
        return x
    
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        causal_attention: bool,
        n_heads: int,
        activation: nn.Module
        ):
        super().__init__()
        self.embedding_size = torch.tensor(embedding_size)
        self.causal_attention = causal_attention
        self.n_heads = n_heads
        assert self.embedding_size % self.n_heads == 0, f"Embedding size ({self.embedding_size}) must be divisable by the number of heads ({self.n_heads})"
        self.head_dim = self.embedding_size // self.n_heads
        self.ff_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_v = nn.Linear(embedding_size, embedding_size, bias=False)
        
        self.fc1 = nn.Linear(embedding_size, embedding_size*2)
        self.activation = activation
        self.fc2 = nn.Linear(embedding_size*2, embedding_size)
    
    def forward(self, x):
        input_shape = x.shape
        resize = (x.shape[0], x.shape[1], self.n_heads, self.head_dim.item())
        key = self.ff_k(x).reshape(resize)
        query = self.ff_q(x).reshape(resize)
        value = self.ff_v(x).reshape(resize)
        
        score = query.permute(0,2,1,3) @ key.permute(0,2,3,1)
        if self.causal_attention:
            score += causal_mask(size=score.shape, device=score.device)
        score = score / torch.sqrt(self.head_dim)
        attention = torch.softmax(score, dim=-1) 
        x = attention @ value.permute(0,2,1,3)
        x = x.permute(0, 2, 1, 3).reshape(input_shape)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x