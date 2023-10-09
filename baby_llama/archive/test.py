import torch
from torch import nn


def _test_norms():
    """
    """

    x = torch.randn((2, 5, 10))
    eps = 1e-5
    x_var, x_mean = torch.var_mean(x, dim=[0,1], keepdim=True, correction=0)
    x_std = torch.sqrt(x_var+eps)
    x_norm = (x - x_mean) / x_std
    
    m = nn.BatchNorm1d(5, affine=False, momentum=None, eps=eps)
    x_norm_1 = m(x)
    
    breakpoint()
    
if __name__ == '__main__':
    _test_norms()