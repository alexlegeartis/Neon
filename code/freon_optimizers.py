import math
import torch
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

# ==============================================================================
# Helper Functions & Constants
# ==============================================================================

# Generated for p_values=[0.0, 0.25, 0.5, 0.75, 1.0, 2], c_values=[0.5, 0.6666666666666666, 0.6, 0.8, 0.75]
REMEZ_COEFFS = {
    # Denominator b=2
    2: [
        (54288352.331898, 736806272583903.250000, 736806326872254.625000),
        (378.653183, 35655.481641, 36033.134824),
        (7.513645, 10.606894, 17.120540),
        (3.083575, 1.085322, 3.168897),
        (3.000004, 1.000004, 3.000008),
    ],
    # Denominator b=4
    4: [
        (3.168523, 9.045696, 9.271302),
        (5.969065, 95.335812, 102.645546),
        (9.223686, 31.968272, 40.960691),
        (12.054078, 32.342112, 43.415815),
        (12.046803, 32.472646, 43.519467),
    ],
    # Denominator b=6
    6: [
        (2.121532, 7.052988, 7.186960),
        (2.019028, 6.674413, 7.874411),
        (2.466513, 6.437068, 7.911456),
        (2.488754, 6.456088, 7.944849),
        (2.488053, 6.463830, 7.951883),
    ],
    # Denominator b=8
    8: [
        (1.584199, -0.298385, 0.793962),
        (1.515856, 3.072078, 3.631530),
        (1.578244, 3.054035, 3.632701),
        (1.581282, 3.061915, 3.643199),
        (1.581588, 3.061624, 3.643212),
    ],
    # Denominator b=10
    10: [
        (1.349635, -0.601891, -0.078390),
        (1.171437, -0.246297, -0.103622),
        (1.104854, -0.208210, -0.110551),
        (1.101728, -0.213110, -0.111831),
        (1.101487, -0.214376, -0.112886),
    ],
}

def _matrix_power(M, power):
    """Helper to compute integer matrix powers correctly handling power == 0"""
    if power == 0:
        return torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    if power == 1: 
        return M
        
    res = M
    for _ in range(power - 1):
        res = res @ M
    return res


# ==============================================================================
# Functional Updates
# ==============================================================================

def kaon_update(G, steps=5, eps=1e-12):
    """Kaon's chaotic spectral iteration"""
    assert len(G.shape) == 2
    X = G.bfloat16()
    
    # [SAFETY FIX]: Explicit 0-norm handler
    norm = X.norm()
    if norm < eps:
        return torch.zeros_like(X)
    norm += eps
    
    X = X / norm
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    I = torch.eye(X.size(0), device=X.device, dtype=X.dtype)
    
    for _ in range(steps):
        A = X @ X.T
        B = I - A
        B = B @ B  # (I - A)^2
        X = 4.1 * (B @ X)
        
    X = X / 1.175 
    
    if transposed:
        X = X.T
        
    return X


def freon_update(G, a, b, eps=1e-12):
    """Coupled QDWH Cholesky Iteration for (GG^T)^{-a/b} G"""
    assert len(G.shape) == 2
    
    if b % 2 != 0:
        a, b = 2 * a, 2 * b
    r = b // 2
    
    if b not in REMEZ_COEFFS:
        raise ValueError(f"b={b} is not supported. Use denominators equivalent to b=2, 4, 6, 8, 10.")

    X = G.float() 
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    k = X.size(0)
    
    # [SAFETY FIX]: Explicit 0-norm handler
    nu = X.norm()
    if nu < eps:
        return torch.zeros_like(G)
    nu += eps
    
    X = X / nu
    
    I_k = torch.eye(k, device=X.device, dtype=X.dtype)
    K0 = torch.cat([X.T, math.sqrt(eps) * I_k], dim=0)
    
    Q0, R0 = torch.linalg.qr(K0) 
    L = R0.T
    L0 = L.clone()  # Saved for log-determinant computation
    C = I_k.clone()
    
    for (alpha_t, beta_t, gamma_t) in REMEZ_COEFFS[b]:
        # [MATH DOMAIN FIX]: Handle negative Remez denominator coefficients gracefully
        if gamma_t > 0:
            rho_t = beta_t / gamma_t
            
            # Block Matrix K optimal construction avoiding numeric blowup
            if gamma_t <= 1.0:
                K = torch.cat([math.sqrt(gamma_t) * L.T, I_k], dim=0)
            else:
                K = torch.cat([L.T, (1.0 / math.sqrt(gamma_t)) * I_k], dim=0)
                
            Q_K, R_K = torch.linalg.qr(K)
            
            Q2 = Q_K[-k:, :]
            V = Q2 @ Q2.T
            
            W = rho_t * I_k + (alpha_t - rho_t) * V
        else:
            # If gamma_t <= 0, the block QR identity fails (imaginary sqrt).
            # However, for negative gamma, (I + gamma_t * LL^T) is highly well-conditioned,
            # making explicit linear solve perfectly safe and stable.
            A_mat = L @ L.T
            B_mat = I_k + gamma_t * A_mat
            W = torch.linalg.solve(B_mat, alpha_t * I_k + beta_t * A_mat)
            
        W = 0.5 * (W + W.T)
        
        L = _matrix_power(W, r) @ L
        C = W @ C
        
    out = (nu ** (1.0 - 2.0 * a / b)) * (_matrix_power(C, a) @ X)
    
    # [MU SCALING FIX]: Restores scale invariance for non-Muon norms
    c = a / b
    q = 2.0 * (1.0 - c)
    if abs(q) < 1e-6:
        # q=0 limit computes det(GG^T)^{1/2k} efficiently using Cholesky factor L0
        log_det = 2.0 * torch.sum(torch.log(torch.abs(torch.diag(L0)))) + 2.0 * k * math.log(nu)
        mu = torch.exp(log_det / (2.0 * k))
    else:
        # mean q-norm trace scaling
        inner = torch.sum(out * X) * (nu / k)
        inner = torch.clamp(inner, min=eps)
        mu = inner ** ((1.0 - q) / q)
    out = mu * out
    
    if transposed:
        out = out.T
        
    return out.to(G.dtype)


def kaon_update_with_fro_norm(G, steps=5, eps=1e-12):
    """Kaon's chaotic spectral iteration returning both update and Fro-norm."""
    assert len(G.shape) == 2
    X = G.bfloat16()
    
    # [SAFETY FIX]: Explicit 0-norm handler
    fro_norm = X.norm()
    if fro_norm < eps:
        return torch.zeros_like(G), fro_norm
    fro_norm += eps
    
    X = X / fro_norm
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    I = torch.eye(X.size(0), device=X.device, dtype=X.dtype)
    
    for _ in range(steps):
        A = X @ X.T
        B = I - A
        B = B @ B 
        X = 4.1 * (B @ X)
        
    X = X / 1.175 
    
    if transposed:
        X = X.T
        
    return X.to(G.dtype), fro_norm


def freon_qdwh_update_with_fro_norm(G, a, b, eps=1e-12):
    if b % 2 != 0:
        a, b = 2 * a, 2 * b
    r = b // 2
    
    X = G.float() 
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T; transposed = True
        
    k = X.size(0)
    
    # [SAFETY FIX]: Explicit 0-norm handler
    fro_norm = X.norm()
    if fro_norm < eps:
        return torch.zeros_like(G), fro_norm
    fro_norm += eps
    
    X = X / fro_norm
    
    I_k = torch.eye(k, device=X.device, dtype=X.dtype)
    K0 = torch.cat([X.T, math.sqrt(eps) * I_k], dim=0)
    
    Q0, R0 = torch.linalg.qr(K0) 
    L = R0.T
    L0 = L.clone()  # Saved for log-determinant computation
    C = I_k.clone()
    
    for (alpha_t, beta_t, gamma_t) in REMEZ_COEFFS[b]:
        # [MATH DOMAIN FIX]: Handle negative Remez denominator coefficients gracefully
        if gamma_t > 0:
            rho_t = beta_t / gamma_t
            if gamma_t <= 1.0:
                K = torch.cat([math.sqrt(gamma_t) * L.T, I_k], dim=0)
            else:
                K = torch.cat([L.T, (1.0 / math.sqrt(gamma_t)) * I_k], dim=0)
                
            Q_K, R_K = torch.linalg.qr(K)
            Q2 = Q_K[-k:, :]
            V = Q2 @ Q2.T
            W = rho_t * I_k + (alpha_t - rho_t) * V
        else:
            # Explicit linear solve for negative/zero gamma
            A_mat = L @ L.T
            B_mat = I_k + gamma_t * A_mat
            W = torch.linalg.solve(B_mat, alpha_t * I_k + beta_t * A_mat)
            
        W = 0.5 * (W + W.T)
        
        L = _matrix_power(W, r) @ L
        C = W @ C
        
    out = (fro_norm ** (1.0 - 2.0 * a / b)) * (_matrix_power(C, a) @ X)
    
    # [MU SCALING FIX]
    c = a / b
    q = 2.0 * (1.0 - c)
    if abs(q) < 1e-6:
        log_det = 2.0 * torch.sum(torch.log(torch.abs(torch.diag(L0)))) + 2.0 * k * math.log(fro_norm)
        mu = torch.exp(log_det / (2.0 * k))
    else:
        inner = torch.sum(out * X) * (fro_norm / k)
        inner = torch.clamp(inner, min=eps)
        mu = inner ** ((1.0 - q) / q)
    out = mu * out
    
    if transposed: out = out.T
    return out.to(G.dtype), fro_norm


def freon_svd_update_with_fro_norm(G, c, eps=1e-12):
    X = G.float()
    
    # [SAFETY FIX]: Explicit 0-norm handler
    fro_norm = X.norm()
    if fro_norm < eps:
        return torch.zeros_like(G), fro_norm
    fro_norm += eps
    
    X = X / fro_norm
    
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    
    # [SAFETY FIX]: Clamp S to prevent division by zero / NaNs if c > 0.5
    S_clamped = torch.clamp(S, min=eps)
    S_scaled = S_clamped ** (1.0 - 2.0 * c)
    out = (U * S_scaled.unsqueeze(0)) @ Vh
    
    out = (fro_norm ** (1.0 - 2.0 * c)) * out
    
    # [MU SCALING FIX]
    k = min(G.size(0), G.size(1))
    q = 2.0 * (1.0 - c)
    if abs(q) < 1e-6:
        S_orig = S_clamped * fro_norm
        mu = torch.exp(torch.sum(torch.log(S_orig)) / k)
    else:
        inner = torch.sum(out * X) * (fro_norm / k)
        inner = torch.clamp(inner, min=eps)
        mu = inner ** ((1.0 - q) / q)
    out = mu * out
    
    return out.to(G.dtype), fro_norm


def dyn_spectral_update_with_fro_norm(G, p_t, steps=5, eps=1e-12):
    """
    Implements Algorithm 1 & 2 from the DynMuon paper.
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    
    # [SAFETY FIX]: Explicit 0-norm handler
    fro_norm = X.norm()
    if fro_norm < eps:
        return torch.zeros_like(G), fro_norm
    fro_norm += eps
        
    if p_t >= 0.25:
        return X.to(G.dtype), fro_norm
        
    X_n = X / fro_norm
    
    transposed = False
    if X_n.size(0) > X_n.size(1):
        X_n = X_n.T
        transposed = True
        
    Y = X_n
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = Y @ Y.T
        B = b * A + c * (A @ A)
        Y = a * Y + B @ Y
        
    if p_t >= 0.0:
        if transposed:
            Y = Y.T
        return Y.to(G.dtype), fro_norm
        
    A_mat = X_n @ X_n.T
    I = torch.eye(A_mat.size(0), device=A_mat.device, dtype=A_mat.dtype)
    E = A_mat - I
    delta = p_t / 2.0
    
    E2 = E @ E
    C_mat = I + delta * E + 0.5 * delta * (delta - 1.0) * E2
    
    X_tilde = C_mat @ Y
    
    if transposed:
        X_tilde = X_tilde.T
        
    X_tilde = (fro_norm ** p_t) * X_tilde
    
    return X_tilde.to(G.dtype), fro_norm


# ==============================================================================
# Optimizers
# ==============================================================================

class Kaon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True, steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            steps = group["steps"]
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
                    p.data.mul_(len(p.data)**0.5 / norm) 
                
                update = kaon_update(g.reshape(len(g), -1), steps=steps).view(g.shape)
                p.data.add_(update, alpha=-lr)

class KaonSignedUpdate(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, norm_weight=True, steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            steps = group["steps"]
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
                    p.data.mul_(len(p.data)**0.5 / norm) 
                
                update = kaon_update(g.reshape(len(g), -1), steps=steps).view(g.shape)
                p.data.add_(update.sign(), alpha=-lr)


class Freon(torch.optim.Optimizer):
    def __init__(self, params, a=2, b=3, lr=1e-3, momentum=0.0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        test_b = b * 2 if b % 2 != 0 else b
        if test_b not in REMEZ_COEFFS:
            raise ValueError(f"Fraction {a}/{b} forces denominator to {test_b}, which lacks precomputed Remez coefficients." 
                             f" Use equivalents for 1/4, 1/3, 1/2, 2/3, 3/4, or 1/1.")
                             
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, a=a, b=b)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            a = group["a"]
            b = group["b"]
            
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                update = freon_update(g.reshape(len(g), -1), a, b).view(g.shape)
                p.data.add_(update, alpha=-lr)

class FreonSignedUpdate(torch.optim.Optimizer):
    def __init__(self, params, a=2, b=3, lr=1e-3, momentum=0.0, nesterov=False, norm_weight=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        test_b = b * 2 if b % 2 != 0 else b
        if test_b not in REMEZ_COEFFS:
            raise ValueError(f"Fraction {a}/{b} forces denominator to {test_b}, which lacks precomputed Remez coefficients." 
                             f" Use equivalents for 1/4, 1/3, 1/2, 2/3, 3/4, or 1/1.")
                             
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, a=a, b=b)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            a = group["a"]
            b = group["b"]
            
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                if self.norm_weight:
                    norm = p.data.norm()
                    if norm < 1e-10:
                        norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / norm)
                
                update = freon_update(g.reshape(len(g), -1), a, b).view(g.shape)
                p.data.add_(update.sign(), alpha=-lr)


class FKaon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, sgd_coeff=0.0, norm_weight=True, steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, sgd_coeff=sgd_coeff, steps=steps)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            sgd_coeff = group["sgd_coeff"]
            steps = group["steps"]
            
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                if self.norm_weight:
                    w_norm = p.data.norm()
                    if w_norm < 1e-10:
                        w_norm = 1e-10
                    p.data.mul_(len(p.data)**0.5 / w_norm) 
                    
                if sgd_coeff != 1.0:
                    update_part, fro_norm = kaon_update_with_fro_norm(g.reshape(len(g), -1), steps=steps)
                    update_part = update_part.view_as(g)
                    
                    g_div = g / fro_norm
                    update = (1 - sgd_coeff) * update_part + sgd_coeff * g_div
                else:
                    grad_norm = g.norm().add(eps)
                    g_div = g / grad_norm
                    update = sgd_coeff * g_div

                p.data.add_(update, alpha=-lr)


class FFreon(torch.optim.Optimizer):
    def __init__(self, params, c=None, p=None, a=None, b=None, lr=1e-3, momentum=0.0, nesterov=False, sgd_coeff=0.0, norm_weight=True):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
            
        if p is not None:
            c_val = (1.0 - p) / 2.0
        elif c is not None:
            c_val = float(c)
        elif a is not None and b is not None:
            c_val = a / b
        else:
            c_val = 0.5 
            
        best_a, best_b = None, None
        if a is None or b is None:
            frac = Fraction(c_val).limit_denominator(10)
            test_b = frac.denominator * 2 if frac.denominator % 2 != 0 else frac.denominator
            if test_b in REMEZ_COEFFS:
                best_a, best_b = frac.numerator, frac.denominator
        else:
            best_a, best_b = a, b
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, sgd_coeff=sgd_coeff, c=c_val, a=best_a, b=best_b)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            sgd_coeff = group["sgd_coeff"]
            c, a, b = group["c"], group["a"], group["b"]
            
            can_use_qdwh = False
            if a is not None and b is not None:
                test_b = b * 2 if b % 2 != 0 else b
                if test_b in REMEZ_COEFFS:
                    can_use_qdwh = True
            
            for p_param in group["params"]:
                g = p_param.grad
                if g is None: continue
                    
                state = self.state[p_param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                if self.norm_weight:
                    w_norm = p_param.data.norm()
                    if w_norm < 1e-10:
                        w_norm = 1e-10
                    p_param.data.mul_(len(p_param.data)**0.5 / w_norm)
                    
                if sgd_coeff != 1.0:
                    G_2d = g.reshape(len(g), -1)
                    if can_use_qdwh:
                        update_part, fro_norm = freon_qdwh_update_with_fro_norm(G_2d, a, b)
                    else:
                        update_part, fro_norm = freon_svd_update_with_fro_norm(G_2d, c)
                        
                    update_part = update_part.view_as(g)
                    g_div = g / fro_norm
                    update = (1 - sgd_coeff) * update_part + sgd_coeff * g_div
                else:
                    grad_norm = g.norm().add(eps)
                    g_div = g / grad_norm
                    update = sgd_coeff * g_div
                    
                p_param.data.add_(update, alpha=-lr)


class FDynFreon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, sgd_coeff=0.0, norm_weight=True, 
                 total_steps=1000, p_max=1.0, p_min=-0.5, tau=0.5, w=0.1, steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, sgd_coeff=sgd_coeff,
                        total_steps=total_steps, p_max=p_max, p_min=p_min, tau=tau, w=w, steps=steps)
        super().__init__(params, defaults)
        self.norm_weight = norm_weight

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            sgd_coeff = group["sgd_coeff"]
            steps = group["steps"]
            
            total_steps = group["total_steps"]
            p_max, p_min = group["p_max"], group["p_min"]
            tau, w = group["tau"], group["w"]
            
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            t = group["step"]
            
            u = (t / total_steps - tau) / w
            u = max(min(u, 50.0), -50.0) 
            a = 1.0 / (1.0 + math.exp(u))
            p_t = p_min + a * (p_max - p_min)
            
            for p_param in group["params"]:
                g = p_param.grad
                if g is None:
                    continue
                    
                state = self.state[p_param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                eps = 1e-12
                if self.norm_weight:
                    w_norm = p_param.data.norm()
                    if w_norm < 1e-10:
                        w_norm = 1e-10
                    p_param.data.mul_(len(p_param.data)**0.5 / w_norm) 
                    
                if sgd_coeff != 1.0:
                    update_part, fro_norm = dyn_spectral_update_with_fro_norm(g.reshape(len(g), -1), p_t, steps=steps)
                    update_part = update_part.view_as(g)
                    
                    g_div = g / fro_norm
                    update = (1 - sgd_coeff) * update_part + sgd_coeff * g_div
                else:
                    grad_norm = g.norm().add(eps)
                    g_div = g / grad_norm
                    update = sgd_coeff * g_div

                p_param.data.add_(update, alpha=-lr)


# ==============================================================================
# REMEZ Coefficient Generator (for reference/generation usage)
# ==============================================================================

def get_exact_qdwh_b2(iters=5):
    """Exact analytical formula for b=2 (c=1/2, p=0, Muon case)"""
    l_t = 1e-11
    coeffs = []
    for i in range(iters):
        d1 = (4 * (1 - l_t**2) / (l_t**4))**(1/3)
        f = 8 * (2 - l_t**2) / (l_t**2 * (1 + d1)**(1/2))
        alpha = (1 + d1)**(1/2) + 0.5 * (8 - 4*d1 + f)**(1/2)
        beta = (alpha - 1)**2 / 4
        gamma = alpha + beta - 1
        coeffs.append((alpha, beta, gamma))
        
        W_l = (alpha + beta * l_t) / (1 + gamma * l_t)
        l_t = l_t * (W_l**2)
    return coeffs

def get_dynamic_remez_fixed(b, iters=5):
    """Dynamic Remez minimax optimizer for arbitrary even denominator b"""
    if b == 2:
        return get_exact_qdwh_b2(iters)
        
    l_t = (10**-11)**(2/b)
    u_t = 1.0
    coeffs = []
    
    for t in range(iters):
        xs = np.logspace(np.log10(l_t), np.log10(u_t), 1000)
        
        def loss(params):
            alpha, beta, gamma = params
            W = (alpha + beta * xs) / (1 + gamma * xs)
            f_x = xs * (np.abs(W)**b)
            return np.max(np.abs(1 - f_x))
            
        if t == 0:
            alpha0, gamma0 = (b+1)/(b-1), (b+1)/(b-1)
            beta0 = -4*b / ((b-1)**2)
        else:
            alpha0, beta0, gamma0 = coeffs[-1]
            
        res = minimize(loss, [alpha0, beta0, gamma0], method='Nelder-Mead', options={'maxiter': 10000})
        alpha, beta, gamma = res.x
        coeffs.append((alpha, beta, gamma))
        
        W = (alpha + beta * xs) / (1 + gamma * xs)
        f_x = xs * (np.abs(W)**b)
        l_t = np.min(f_x)
        u_t = np.max(f_x)
        
    return coeffs

def generate_remez_dict(p_values=[], c_values=[], iters=5):
    required_b = set()
    
    c_targets = list(c_values)
    for p in p_values:
        c_targets.append((1.0 - p) / 2.0)
        
    for c in c_targets:
        frac = Fraction(c).limit_denominator(100)
        b = frac.denominator
        if b % 2 != 0:
            b *= 2
        required_b.add(b)
        
    required_b = sorted(list(required_b))
    print(f"# Generated for p_values={p_values}, c_values={c_values}")
    print("REMEZ_COEFFS = {")
    for b in required_b:
        print(f"    # Denominator b={b}")
        coeffs = get_dynamic_remez_fixed(b, iters)
        print(f"    {b}: [")
        for c in coeffs:
            print(f"        ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}),")
        print("    ],")
    print("}")


if __name__ == "__main__":
    p_targets = [0.0, 0.25, 0.5, 0.75, 1.0, 2]
    c_targets = [1/2, 2/3, 3/5, 4/5, 3/4]
    
    print("Precompiling Remez coefficients... This may take a few seconds.")
    generate_remez_dict(p_values=p_targets, c_values=c_targets)