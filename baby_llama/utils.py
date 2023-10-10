from typing import List
import torch
from torch import nn
from tokenizers import pre_tokenizers

def words_level_tokeniser(data: List[str]):
    # Unique set of words
    words_list = list({w for sentence in data for w in sentence.split()})
    padding_token = 8  
    size = (len(data), max([len(s.split()) for s in data]))
    
    tokenized_data = torch.full(size, padding_token)
    for r, sentence in enumerate(data):
        for c, w in enumerate(sentence.split()):
            tokenized_data[r, c] = words_list.index(w)
            
    return tokenized_data
    
    
class SwiGLU(nn.Module):
    def __init__(
        self,
        size: int
    ):
        super().__init__()
        self.linearA = nn.Linear(size, size)
        self.linearB = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
    
    def forward(self, x):
        swish = self.linearB(x) * torch.sigmoid(self.beta * self.linearB(x))
        return swish * self.linearA(x) 
    
    
class RMSnorm(nn.Module):
    def __init__(
        self, 
        size: int,
        eps: float = 1e-5, 
    ):
        """
        Root-Mean-Square Layer Normalization.
        Assumes the shape of the input x is (batch, seq_len, d_model)

        Args:
            size: shape of the feature dimention (i.e. d_model)
            eps: For numerical stability. Defaults to 1e-5.
        """
        super(RMSnorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        
    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) # as an alternative can also use the frobenius norm to compute rms
        x_norm = x / rms
        
        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm
    
def get_rotary_matrix(context_len: int, embedding_dim: int) -> torch.Tensor:
    """
    Generate the Rotary Matrix for ROPE

    Args:
        context_len (int): context len
        embedding_dim (int): embedding dim

    Returns:
        torch.Tensor: the rotary matrix of dimension context_len x embedding_dim x embedding_dim
    """
    R = torch.zeros((context_len, embedding_dim, embedding_dim), requires_grad=False)
    
    positions = torch.arange(1, context_len+1).unsqueeze(1)
    # Create matrix theta (shape: context_window x embedding_dim // 2)
    slice_i = torch.arange(0, embedding_dim // 2)
    theta = 10000. ** (-2.0 * (slice_i.float()) / embedding_dim) 
    m_theta = positions * theta
    # Create sin and cos values
    cos_values = torch.cos(m_theta)
    sin_values = torch.sin(m_theta)
    # Populate the rotary matrix R using slicing
    R[:, 2*slice_i, 2*slice_i] = cos_values
    R[:, 2*slice_i, 2*slice_i+1] = -sin_values
    R[:, 2*slice_i+1, 2*slice_i] = sin_values
    R[:, 2*slice_i+1, 2*slice_i+1] = cos_values
    return R
