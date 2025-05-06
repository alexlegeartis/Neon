import math
import torch
import torch.nn.functional as F
from torch import nn
from matrix_functions import k_sv_svds_approximation_dlpack, one_sv_svds_approximation, svd_full_approximation

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Add numerical stability
    norm = X.norm() + eps
    if norm < eps:
        return torch.zeros_like(X)
    X /= norm
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

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

class Muon(torch.optim.Optimizer):
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
                
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                
                n, m = g.shape # d_out and d_in
                p.data.add_(update, alpha=-lr * math.sqrt(n / m))

class Neon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, k=1, tau = 0, nesterov=False,
                 neon_mode='fast', iter_num = 100):
        self.tau = tau
        self.k = k
        self.lanczos_iter_num = iter_num
        self.type = neon_mode
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
                
                # update = u1s1v1t_torch(g.reshape(len(g), -1)).view(g.shape)
                g_resh = g.reshape(len(g), -1)
                n, m = g_resh.shape # d_out and d_in
                if g_resh.shape[0] == 1 or g_resh.shape[1] == 1:
                    update = zeropower_via_newtonschulz5(g_resh).view(g.shape)
                    print("Using Muon is illegal!")
                else:
                    if self.type == 'fast':
                        update = one_sv_svds_approximation(g_resh, self.lanczos_iter_num)
                    elif self.type == 'accurate':
                        update, self.tau, self.k = k_sv_svds_approximation_dlpack(g_resh, self.k, self.tau, self.lanczos_iter_num)
                        # update, self.tau, self.k = svd_full_approximation(g_resh, self.tau) -- too slow
                p.data.add_(update.view(g.shape), alpha=-lr * math.sqrt(n / m)) 