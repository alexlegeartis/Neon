import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from math import ceil


def u1s1v1t_torch(W, num_iter=20, eps=1e-8):
    """Power iteration using PyTorch operations"""
    # Ensure W is 2D
    if len(W.shape) > 2:
        W = W.reshape(W.size(0), -1)
    
    # Initialize v with correct shape
    v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
    v_norm = v.norm()
    if v_norm < eps:
        return torch.zeros_like(W)
    v /= v_norm
    
    for _ in range(num_iter):
        # Matrix-vector multiplication
        u = F.linear(v, W)
        u_norm = u.norm()
        if u_norm < eps:
            return torch.zeros_like(W)
        u /= u_norm
        
        # Transpose matrix-vector multiplication
        v = F.linear(u, W.T)
        v_norm = v.norm()
        if v_norm < eps:
            return torch.zeros_like(W)
        v /= v_norm
    
    # Compute first singular value - fix transpose warning
    sigma1 = (u.reshape(1, -1) @ F.linear(v, W)).squeeze()
    
    # Reshape u and v for outer product
    u = u.reshape(-1, 1)  # shape: (m, 1)
    v = v.reshape(-1, 1)  # shape: (n, 1)
    
    # Return scaled outer product
    return sigma1 * (u @ v.T)


import numpy as np
def u1s1v1t(W, num_iter=30):
    v = np.random.randn(W.shape[1])
    v /= np.linalg.norm(v)
    
    for _ in range(num_iter):
        u = W @ v
        u /= np.linalg.norm(u)
        v = W.T @ u
        v /= np.linalg.norm(v)
    sigma1 = (u.T @ W @ v)
    u = u.reshape(-1, 1)         # shape: (5, 1)
    v = v.reshape(-1, 1)         # shape: (3, 1)
    return sigma1 * u @ v.T


class Neon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                # Add numerical stability
                norm = p.data.norm()
                if norm < 1e-8:
                    continue
                p.data.mul_(len(p.data)**0.5 / norm)
                
                update = u1s1v1t_torch(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)