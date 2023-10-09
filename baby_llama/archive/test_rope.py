"""
---
title: Rotary Positional Embeddings (RoPE)
summary: >
  Annotated implementation of RoPE from paper
  RoFormer: Enhanced Transformer with Rotary Position Embedding
---

# Rotary Positional Embeddings (RoPE)

This is an implementation of
[Rotary Positional Embeddings (RoPE)](https://papers.labml.ai/paper/2104.09864)
in [PyTorch](https://pytorch.org).

Rotary Positional Embeddings (RoPE) encode position information of tokens
with a rotation matrix that naturally incorporates explicit relative position
dependency.

Here's [the training code](experiment.html) for training a transformer model with RoPE
 on Tiny Shakespeare dataset.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        breakpoint()

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        #
        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    We override [multi-head attention from original transformer](../mha.html).
    """

    def __init__(self, heads: int, d_model: int, rope_percentage: float = 0.5, dropout_prob: float = 0.0):
        super().__init__(heads, d_model, dropout_prob)

        # Rotary positional embedding layers
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        """

        # Calculate dot-product with RoPE
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))

import time

def get_rotary_matrix(context_window, embedding_dim):
    start = time.time()
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    print(f"Time: {time.time() - start}")
    return R

import torch
import time
import numpy as np

def get_rotary_matrix_optimized_olde(context_window, embedding_dim):
    start = time.time()
    
    # Create theta values similar to original function
    theta = 10000 ** (-2.0 * (torch.arange(0, embedding_dim // 2).float() - 1) / embedding_dim)
    
    # Create position values
    positions = torch.arange(0, context_window).unsqueeze(1)
    
    # Create matrix theta (shape: context_window x embedding_dim // 2)
    m_theta = positions * theta
    
    # Create sin and cos values
    cos_values = torch.cos(m_theta)
    sin_values = torch.sin(m_theta)
    
    # Initialize rotary matrix R
    R = torch.zeros((context_window, embedding_dim, embedding_dim))
    
    # Populate the rotary matrix R using slicing
    R[:, ::2, ::2] = cos_values.unsqueeze(-1)
    R[:, ::2, 1::2] = -sin_values.unsqueeze(-1)
    R[:, 1::2, ::2] = sin_values.unsqueeze(-1)
    R[:, 1::2, 1::2] = cos_values.unsqueeze(-1)
    
    print(f"Time: {time.time() - start}")
    return R

def get_rotary_matrix_optimized_new(context_window, embedding_dim):
    start = time.time()
    R_torch = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    
    positions = torch.arange(0, context_window).unsqueeze(1)
    # Create matrix theta (shape: context_window x embedding_dim // 2)
    slice_i = torch.arange(0, embedding_dim // 2)
    theta = 10000. ** (-2.0 * (slice_i.float() - 1) / embedding_dim)
    print(theta.shape)
    m_theta = positions * theta
    print(m_theta.shape)
    # Create sin and cos values
    cos_values = torch.cos(m_theta)
    sin_values = torch.sin(m_theta)
    # Populate the rotary matrix R using slicing
    # Populate the rotary matrix R using slicing
    R_torch[:, 2*slice_i, 2*slice_i] = cos_values
    R_torch[:, 2*slice_i, 2*slice_i+1] = -sin_values
    R_torch[:, 2*slice_i+1, 2*slice_i] = sin_values
    R_torch[:, 2*slice_i+1, 2*slice_i+1] = cos_values
    print(f"Time: {time.time() - start}")
    return R_torch


def get_rotary_matrix_optimized(context_window, embedding_dim):
    start = time.time()
    R_torch = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        # Create theta values similar to original function
        slice_i = torch.arange(0, embedding_dim // 2)
        theta = 10000. ** (-2.0 * (slice_i.float() - 1) / embedding_dim)
        m_theta = position * theta
        # Create sin and cos values
        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)
        # Populate the rotary matrix R using slicing
        # Populate the rotary matrix R using slicing
        R_torch[position, 2*slice_i, 2*slice_i] = cos_values
        R_torch[position, 2*slice_i, 2*slice_i+1] = -sin_values
        R_torch[position, 2*slice_i+1, 2*slice_i] = sin_values
        R_torch[position, 2*slice_i+1, 2*slice_i+1] = cos_values
    print(f"Time: {time.time() - start}")
    return R_torch


# Original function for comparison
def get_rotary_matrix(context_window, embedding_dim):
    start = time.time()
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    R_torch = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)

    for position in range(context_window):
        cos_list = []
        sin_list = []
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
            cos_list.append(np.cos(m_theta))
            sin_list.append(np.sin(m_theta))
            
        # # Create theta values similar to original function
        # slice_i = torch.arange(0, embedding_dim // 2)
        # theta = 10000. ** (-2.0 * (slice_i.float() - 1) / embedding_dim)
        # m_theta = position * theta
        # # Create sin and cos values
        # cos_values = torch.cos(m_theta)
        # sin_values = torch.sin(m_theta)
        # # Populate the rotary matrix R using slicing
        # # Populate the rotary matrix R using slicing
        # R_torch[position, 2*slice_i, 2*slice_i] = cos_values
        # R_torch[position, 2*slice_i, 2*slice_i+1] = -sin_values
        # R_torch[position, 2*slice_i+1, 2*slice_i] = sin_values
        # R_torch[position, 2*slice_i+1, 2*slice_i+1] = cos_values
        # breakpoint()
    print(f"Time: {time.time() - start}")
    return R

  
def _test_rotary():
    """
    Testing RoPE with a simple example
    """
    config = {
        'd_model': 256,
        'context_window': 128,
        'batch_size': 16
    }

    R = get_rotary_matrix(config['context_window'], config['d_model'])
    R_torch = get_rotary_matrix_optimized(config['context_window'], config['d_model'])
    R_torch_new = get_rotary_matrix_optimized_new(config['context_window'], config['d_model'])
    assert torch.allclose(R, R_torch, atol=1e-5), "Matrices are not equivalent!"
    assert torch.allclose(R, R_torch_new, atol=1e-5), "Matrices are not equivalent!"
    assert torch.allclose(R_torch, R_torch_new, atol=1e-5), "Matrices are not equivalent!"
    print(R.shape)
    
    q = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
    print(q.shape)
    breakpoint()

if __name__ == '__main__':
    _test_rotary()