import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from matrix_methods.matrix_functions import k_sv_svds_approximation_dlpack, one_sv_svds_approximation, svd_full_approximation, several_sv_svds_approximation


# this should be used for CIFAR, but not for NanoGPT: for NanoGPT, see logs
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-12):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Add numerical stability
    norm = X.norm() + eps
    if norm < eps:
        print("Norm lower than eps")
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


def newtonschulz5_with_nuclear(G, steps=3, eps=1e-12):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Add numerical stability
    norm = X.norm() + eps
    if norm < eps:
        print("Norm lower than eps")
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
    return X * torch.trace(G.bfloat16() @ X.T)


# archaic: it had been used in Neon before Lanczos
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
    return sigma1 * (u @ v.T) # now, in LMO paradigm, we do not need sigma


# archaic: for old F*-Neon. Now we use matrix_functions.py
def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


# the same as in airbench_muon.py
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
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                norm = p.data.norm()
                if norm < 1e-10:
                    norm = 1e-10
                p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step


# this function also supports F-Neon
class Neon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, k=1, tau = 0, nesterov=False,
                 neon_mode='fast', iter_num = 100, sgd_coeff=0):
        self.tau = tau
        self.k = k # target number of SVD componenets which we preserve
        self.lanczos_iter_num = iter_num
        self.type = neon_mode # fast (vanilla Neon), accurate (old F*-Neon), Ky-Fan (~Dion)
        self.sgd_coeff = sgd_coeff
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
                
                g_resh = g.reshape(len(g), -1)
                n, m = g_resh.shape # d_out and d_in
                if False: # g_resh.shape[0] == 1 or g_resh.shape[1] == 1:
                    # this is not run usually
                    update = zeropower_via_newtonschulz5(g_resh).view(g.shape)
                    print("Using Muon is illegal!")
                else:
                    if self.type == 'fast':
                        update, sigma1 = one_sv_svds_approximation(g_resh, self.lanczos_iter_num)
                    elif self.type == 'accurate':
                        # remember that it is not in LMO paradigm yet, so we must tune old F*-Neon very carefullys
                        update, self.tau, self.k = k_sv_svds_approximation_dlpack(g_resh, self.k, self.tau, self.lanczos_iter_num)

                    elif self.type == 'kyfan': # no EF here, so it is not Dion
                        u, s, vt = several_sv_svds_approximation(g_resh, self.k)
                        update = u @ vt
                        error = u @ torch.diag(s) @ vt
                update2 = (1-self.sgd_coeff) * update + self.sgd_coeff * g_resh / (g_resh.norm() + 1e-12)
                p.data.add_(update2.view(g.shape), alpha=-lr)


# the same as F-Muon, the copy is in airbench_muon.py
class NormalizedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                norm = p.data.norm()
                if norm < 1e-10:
                    norm = 1e-10
                p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                g_normalized = g / (g.norm() + eps)
                if self.sgd_coeff != 1:
                    update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                    update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                else:
                    update = self.sgd_coeff * g_normalized
                p.data.add_(update, alpha=-lr) # take a step


class MuonSGDStyle(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                norm = p.data.norm()
                if norm < 1e-10:
                    norm = 1e-10
                p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                g_normalized = g / (g.norm() + eps)
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                if self.sgd_coeff != 1:
                    update_part = newtonschulz5_with_nuclear(g.reshape(len(g), -1)).view(g.shape)
                    update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                else:
                    update = self.sgd_coeff * g_normalized
                p.data.add_(update, alpha=-lr) # take a step

# generated by Cursor, but I checked the logic. Unfortunately, I do not see much help, only increased variance
class Dion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, rank=1, momentum_decay=0.9, sgd_coeff=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        if momentum_decay < 0.0 or momentum_decay > 1.0:
            raise ValueError(f"Invalid momentum decay value: {momentum_decay}")
        if rank < 1:
            raise ValueError(f"Invalid rank value: {rank}")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.rank = rank
        self.momentum_decay = momentum_decay
        self.sgd_coeff = sgd_coeff
    '''
    def power_iter1(self, B, Q):
        """
        Single power iteration (from Q)
        B: gradient matrix (m x n)
        Q: right factor from previous iteration (n x r)
        Returns: P (m x r), R (n x r) where P is orthonormal
        """
        # P ← BQ
        P = B @ Q
        
        # P ← Orthogonalize(P) using QR decomposition
        P, _ = torch.linalg.qr(P)
        
        # R ← B^T P
        R = B.T @ P
        
        return P, R

    def column_normalize(self, R):
        """Normalize columns of R to unit norm"""
        norms = torch.norm(R, dim=0, keepdim=True)
        # Add small epsilon for numerical stability
        norms = torch.clamp(norms, min=1e-8)
        return R / norms
    '''
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                
                # Initialize state variables if not present
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                '''
                if 'Q' not in state:
                    # Initialize Q randomly with correct shape
                    n = p.numel()
                    m = g.numel()
                    if m > n:
                        state['Q'] = torch.randn(n, self.rank, device=p.device, dtype=p.dtype)
                    else:
                        state['Q'] = torch.randn(m, self.rank, device=p.device, dtype=p.dtype)
                
                Q = state['Q']
                '''
                buf = state['momentum_buffer']
                
                # Apply momentum
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                
                # Add numerical stability
                norm = p.data.norm()
                if norm < 1e-8:
                    continue
                p.data.mul_(len(p.data)**0.5 / norm)
                
                # Reshape gradient to matrix form
                g_reshaped = g.reshape(len(g), -1)
                m, n = g_reshaped.shape
                
                # Ensure Q has correct shape for current gradient
                '''
                if Q.shape[0] != n:
                    Q = torch.randn(n, self.rank, device=p.device, dtype=p.dtype)
                    state['Q'] = Q
                '''
                # Algorithm 1: Dion implementation
                # Step 3: Bt ← Mt−1 + Gt (in this case, just the gradient)
                Bt = g_reshaped
                
                # Step 4: Pt, Rt ← PowerIter1(Bt; Qt−1)
                # Pt, Rt = self.power_iter1(Bt, Q)
                u, s, vt = several_sv_svds_approximation(g_reshaped, self.rank)
                # Step 5: ∆t = Bt − PtR⊤t (approximation error)
                Delta_t = Bt - u @ torch.diag(s) @ vt
                
                # Step 6: Mt ← μBt + (1 − μ)∆t (error feedback)
                # This is equivalent to: Mt ← Bt − (1 − μ)PtR⊤t
                Mt = self.momentum_decay * Bt + (1 - self.momentum_decay) * Delta_t
                
                # Step 7: Qt ← ColumnNormalize(Rt)
                # Qt = self.column_normalize(Rt)
                
                # Step 8: Xt ← Xt−1 − η√(m/n) PtQ⊤t (scaled orthonormal update)
                update = u @ vt # Pt @ Qt.T
                # scaled_update = update * math.sqrt(m / n) # and you could also add a factor gamma
                update2 = (1-self.sgd_coeff) * update + self.sgd_coeff * g_reshaped / (g_reshaped.norm() + 1e-12)
                # Update the parameter
                p.data.add_(update2.view(g.shape), alpha=-lr)
                
                # Store Q for next iteration
                # state['Q'] = Qt


# it is not an LMO-based algorithm, only an archive
class SGDMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g
                p.data.add_(update, alpha=-lr) # take a step

            
# this one is in LMO, but it's worse than F-Muon
class SignSGDMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                update = (1 - self.sgd_coeff) * update_part + self.sgd_coeff * g.sign() * 0.01 # the last one is lr
                p.data.add_(update, alpha=-lr) # take a step


class RandomNormalizedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                norm = p.data.norm()
                if norm < 1e-10:
                    norm = 1e-10
                p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                if self.sgd_coeff < torch.rand(1).item():
                    update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                    update = update_part
                else:
                    g_normalized = g / (g.norm() + eps)
                    update = g_normalized
                p.data.add_(update, alpha=-lr) # take a step