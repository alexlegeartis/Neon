import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor, nuclear_norm
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

def zeropower_via_newtonschulz5_with_fro_norm(G, steps=3, eps=1e-12):
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
    return X, norm


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
    return X, torch.trace(G.bfloat16() @ X.T) # UV^t and nuclear norm


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
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

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

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step


class MuonSignedUpdate(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

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

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update

                p.data.add_(update.sign(), alpha=-lr) # take a step

class Muon2(torch.optim.Optimizer): # not ready yet!, we must have access to gradients by sample!
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    def step(self, grad_list):
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

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

class MuonCringeMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

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
                # buf.mul_(momentum).add_(g)
                g_old = g
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step
                buf.mul_(momentum).add_(g_old - lr * update)


# this function also supports F-Neon
class Neon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, k=1, tau = 0, nesterov=False,
                 neon_mode='fast', iter_num=1, sgd_coeff=0, norm_weight=True, sign_lr_mult=None):
        self.tau = tau
        self.k = k # target number of SVD componenets which we preserve
        self.lanczos_iter_num = iter_num
        self.type = neon_mode # fast (vanilla Neon), accurate (old F*-Neon), Ky-Fan (~Dion)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight
        self.sign_lr_mult = sign_lr_mult
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            pass
            # raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            pass
            # raise ValueError("Nesterov momentum requires a momentum")
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
                if self.norm_weight:
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
                        u, s, vt = several_sv_svds_approximation(g_resh, self.k, self.lanczos_iter_num)
                        update = u @ vt
                        error = u @ torch.diag(s) @ vt
                if self.sign_lr_mult is not None:
                    # Use sign-based update when sign_lr_mult is provided
                    update2 = (1-self.sgd_coeff) * update.view(g.shape) + self.sgd_coeff * g.sign() * self.sign_lr_mult
                else:
                    # Default: normalized gradient
                    update2 = (1-self.sgd_coeff) * update.view(g.shape) + self.sgd_coeff * g / (g.norm() + 1e-12)
                p.data.add_(update2, alpha=-lr)



# the same as F-Muon, the copy is in airbench_muon.py
class NormalizedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight

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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                if self.sgd_coeff != 1:
                    # zeropower expects 2D input: reshape and restore
                    update_part, fro_norm = zeropower_via_newtonschulz5_with_fro_norm(
                        g.reshape(g.size(0), -1)
                    )
                    update_part = update_part.view_as(g)

                    # normalize g by fro_norm in-place (safe after update_part computed)
                    # If you prefer not to mutate g, use g_div = g / fro_norm instead.
                    g.div_(fro_norm)

                    update = (1 - self.sgd_coeff) * update_part + self.sgd_coeff * g
                else:
                    # normalize g by its norm + eps (in-place), then scale by sgd_coeff
                    grad_norm = g.norm().add_(eps)
                    g.div_(grad_norm)
                    update = self.sgd_coeff * g

                # apply parameter update (no_grad context is active)
                p.data.add_(update, alpha=-lr)



class MuonSGDStyle(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight

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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                g_normalized = g / (g.norm() + eps)
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                if self.sgd_coeff != 1:
                    uvt, nuc_norm = newtonschulz5_with_nuclear(g.reshape(len(g), -1))
                    uvt = uvt.view(g.shape)
                    update_part = uvt * nuc_norm
                    update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                else:
                    update = self.sgd_coeff * g_normalized
                p.data.add_(update, alpha=-lr) # take a step


class NuclearNormalizedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight

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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                uvt, nuc_norm = newtonschulz5_with_nuclear(g.reshape(len(g), -1))
                uvt = uvt.view(g.shape)
                g_normalized = g / (nuc_norm + eps)
                if self.sgd_coeff != 1:
                    update_part = uvt
                    update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                else:
                    update = self.sgd_coeff * g_normalized
                p.data.add_(update, alpha=-lr) # take a step


# generated by Cursor, but I checked the logic. Unfortunately, I do not see much help, only increased variance
class Dion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, rank=1, momentum_decay=0.9, sgd_coeff=0, norm_weight=True):
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
        self.norm_weight = norm_weight

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]
                
                # Initialize state variables if not present
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                
                # Apply momentum
                # buf.mul_(momentum).add_(g)
                # g = g.add(buf, alpha=momentum) if group['nesterov'] else buf #
                # we need buf_new = buf + g
                g = g.add(buf) # now actually m in terms of Dion

                # Add numerical stability - do we need it here?
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-8:
                        continue
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                # Reshape gradient to matrix form
                g_reshaped = g.reshape(len(g), -1)
                m, n = g_reshaped.shape
                
                
                u, s, vt = several_sv_svds_approximation(g_reshaped, self.rank)
                Mt = g_reshaped - self.momentum_decay * (u @ torch.diag(s) @ vt)

                state["momentum_buffer"] = Mt.view(g.shape)
                
                update = u @ vt # Pt @ Qt.T
                # scaled_update = update * math.sqrt(m / n) # and you could also add a factor gamma
                update2 = (1-self.sgd_coeff) * update + self.sgd_coeff * g_reshaped / (g_reshaped.norm() + 1e-12)
                # Update the parameter
                p.data.add_(update2.view(g.shape), alpha=-lr)
                
                # Store Q for next iteration
                # state['Q'] = Qt


# it is not an LMO-based algorithm, only an archive
class SGDMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight

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

                if self.norm_weight:
                    p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g
                p.data.add_(update, alpha=-lr) # take a step

            
class SignSGDMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True, sign_lr_mult=1, sgd_coeff=0, signed=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight
        self.sign_lr_mult = sign_lr_mult
        self.signed = signed

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

                if self.norm_weight:
                    p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                if self.sgd_coeff != 1:
                    update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                    update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g.sign() * self.sign_lr_mult
                else:
                    update = g.sign() * self.sign_lr_mult
                if self.signed:
                    update = update.sign()
                p.data.add_(update, alpha=-lr) # take a step


class RandomNormalizedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight

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
                if self.norm_weight:
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


class SpectrallyNormalizedNeon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, k=1, tau = 0, nesterov=False,
                iter_num = 100, sgd_coeff=0, norm_weight=True):
        self.tau = tau
        self.k = k # target number of SVD componenets which we preserve
        self.lanczos_iter_num = iter_num
        self.type = 'kyfan'
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            pass
            # raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            pass
            # raise ValueError("Nesterov momentum requires a momentum")
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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-8:
                        continue
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                g_resh = g.reshape(len(g), -1)
                n, m = g_resh.shape # d_out and d_in
                u, s, vt = several_sv_svds_approximation(g_resh, self.k)
                update = u @ vt
                # error = u @ torch.diag(s) @ vt
                update2 = (1-self.sgd_coeff) * update.view(g.shape) + self.sgd_coeff * g / (s[0] + 1e-12)
                p.data.add_(update2, alpha=-lr)



def radius_for_volume(n, vol):
    """
    Return the radius r such that the n-dimensional ball of radius r
    has volume `vol`.
    """
    coef = math.gamma(n/2 + 1) / (math.pi ** (n/2))
    return (vol * coef) ** (1/n)

class MuonOrNSGD(torch.optim.Optimizer): # in strange 
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, nsgd_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.nsgd_coeff = nsgd_coeff # here it is for the norm with conv(ball_op, sgd_coeff * ball_Fro)
        self.norm_weight = norm_weight

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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                g_norm = g.norm()
                g_normalized = g / (g_norm + eps)
                # update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                uvt, nuc_norm = newtonschulz5_with_nuclear(g.reshape(len(g), -1))
                m, n = uvt.shape
                real_nsgd = radius_for_volume(min(m, n), (self.nsgd_coeff * 2)**min(m, n))
                # print(f"{m, n}: {real_nsgd}")
                uvt = uvt.view(g.shape)
                if g_norm * real_nsgd > nuc_norm:
                    update = real_nsgd * g_normalized
                    print("Normalized")
                else:
                    update = uvt
                    print("Muon")
                p.data.add_(update, alpha=-lr) # take a step


class MuonOrSign(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sign_coeff=0, norm_weight=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sign_coeff = sign_coeff # here it is for the norm with conv(ball_op, sign_coeff * ball_sign)
        self.norm_weight = norm_weight

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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight
                uvt, nuc_norm = newtonschulz5_with_nuclear(g.reshape(len(g), -1))
                m, n = uvt.shape
                # print(f"{m, n}: {real_nsgd}")
                uvt = uvt.view(g.shape)
                sign_drop = g.abs().mean() * self.sign_coeff * m * n
                # print(f"Muon vs SignSGD {m, n}: {nuc_norm} vs {sign_drop}")
                if (sign_drop + nuc_norm) / 2 > 0.24 / 0.42 * nuc_norm:
                    update = self.sign_coeff * g.sign() * (1 - 0.5) + uvt * 0.5
                    # print("Sign")
                else:
                    update = uvt * 0.24 / 0.42
                    # print("Muon")
                p.data.add_(update, alpha=-lr) # take a step


class RealFanion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, k_share=1, nesterov=False,
                 iter_num=1, sgd_coeff=0, norm_weight=True):
        self.k_share = k_share # target number of SVD componenets which we preserve
        self.lanczos_iter_num = iter_num
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            pass
            # raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            pass
            # raise ValueError("Nesterov momentum requires a momentum")
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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-8:
                        continue
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                g_resh = g.reshape(len(g), -1)
                n, m = g_resh.shape # d_out and d_in
                
                    
                u, s, vt = several_sv_svds_approximation(g_resh, 1, self.lanczos_iter_num)
                update1 = u @ vt

                foo, nuc_norm = newtonschulz5_with_nuclear(g_resh)
                update2 = foo.view(g.shape)
                
                k = math.ceil(min(m, n) * self.k_share)
                if s[0] > nuc_norm / k:
                    update = update1
                    # print(f"Neon, {k}")
                else:
                    update = update2 / k
                    # print(f"Muon, {k}")
                        
                update_full = (1-self.sgd_coeff) * update.view(g.shape) + self.sgd_coeff * g / (g.norm() + 1e-12)
                p.data.add_(update_full, alpha=-lr)

class NeonMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, neon_share=1, nesterov=False,
                 iter_num=1, sgd_coeff=0, norm_weight=True):
        self.neon_share = neon_share # target number of SVD componenets which we preserve
        self.lanczos_iter_num = iter_num
        self.sgd_coeff = sgd_coeff
        self.norm_weight = norm_weight
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            pass
            # raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            pass
            # raise ValueError("Nesterov momentum requires a momentum")
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
                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-8:
                        continue
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                g_resh = g.reshape(len(g), -1)
                n, m = g_resh.shape # d_out and d_in
                
                    
                u, s, vt = several_sv_svds_approximation(g_resh, 1, self.lanczos_iter_num)
                update1 = u @ vt

                update2, nuc_norm = newtonschulz5_with_nuclear(g_resh)
                
                update = update1 * self.neon_share + update2 * (1 - self.neon_share)
                        
                update_full = (1-self.sgd_coeff) * update.view(g.shape) + self.sgd_coeff * g / (g.norm() + 1e-12)
                p.data.add_(update_full, alpha=-lr)



def zeropower_via_newtonschulz5NorMuon(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5NorMuon(update, steps=ns_steps).float()
    if original_shape is not None:
        update = update.reshape(original_shape)
    ################ NorMuon added ###################
    vnorm = update.norm(dim=(-2,-1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
    ##################################################
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95, norm_weight=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[...,0:1], dtype=torch.float32) # added dtype=torch.float32
                if self.norm_weight:
                    eps = 1e-12
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight - added for CIFAR
                
                update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                if group["weight_decay"] and had_grad:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss