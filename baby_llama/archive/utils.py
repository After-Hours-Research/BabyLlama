import torch
from torch import nn

class Layernorm(nn.Module):
    def __init__(
        self, 
        size: int,
        eps: float = 1e-5, 
    ):
        """
        Layer Normalization.
        Assumes the shape of the input x is (batch, seq_len, d_model)

        Args:
            size: shape of the feature dimention (i.e. d_model)
            eps: For numerical stability. Defaults to 1e-5.
        """
        super(Layernorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(size), requires_grad=True)
        
    def forward(self, x):
        x_var, x_mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        x_std = torch.sqrt(x_var + self.eps)
        
        x_norm = (x - x_mean)/ x_std
        
        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm + self.beta.unsqueeze(0).unsqueeze(1)
    
    
class Batchnorm(nn.Module):
    def __init__(
        self, 
        size: int,
        eps: float = 1e-5, 
    ):
        """
        Batch Normalization.
        Assumes the shape of the input x is (batch, seq_len, d_model)

        Args:
            size: shape of the feature dimention (i.e. d_model)
            eps: For numerical stability. Defaults to 1e-5.
        """
        super(Layernorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(size), requires_grad=True)
        
    def forward(self, x):
        x_var, x_mean = torch.var_mean(x, dim=[0,1], keepdim=True, correction=0)
        x_std = torch.sqrt(x_var + self.eps)
        
        x_norm = (x - x_mean)/ x_std
        
        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm + self.beta.unsqueeze(0).unsqueeze(1)
    
    