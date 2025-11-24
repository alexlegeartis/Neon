# maxiumum 93% instead of Muon's 94%
import torch
from optimizers import zeropower_via_newtonschulz5
from torch.optim import Optimizer
from matrix_methods.matrix_functions import several_sv_svds_approximation

class WrongMLion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                
                state = self.state[p]

                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                
                update = zeropower_via_newtonschulz5(exp_avg.reshape(len(g), -1)).view(exp_avg.shape) # whiten the update
                # p.add_(torch.sign(exp_avg), alpha=-lr)
                p.add_(update, alpha=-lr)#  * max(1, p.size(-2) / p.size(-1)) ** 0.5)

        return loss

class NLion(Optimizer): # does not work for airbench
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                
                u, s, vt = several_sv_svds_approximation(exp_avg.reshape(len(g), -1), 1, 10)
                update = (u @ vt).view(exp_avg.shape)
                # p.add_(torch.sign(exp_avg), alpha=-lr)
                p.add_(update, alpha=-lr)#  * max(1, p.size(-2) / p.size(-1)) ** 0.5)

        return loss



class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if 'mom' not in state:
                    state['mom'] = torch.zeros_like(p)
                mom = state['mom']

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Compute the interpolated value c_t = β1 * m + (1 − β1) * g
                c = mom.mul(beta1).add(g, alpha=(1 - beta1))

                # Compute the signed update
                update = torch.sign(c)

                # Parameter update
                p.add_(update, alpha=-lr)

                # Momentum update: m_t = β2 * m + (1 − β2) * g
                mom.mul_(beta2).add_(g, alpha=(1 - beta2))

        return loss


class MLion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if 'mom' not in state:
                    state['mom'] = torch.zeros_like(p)
                mom = state['mom']

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Compute the interpolated value c_t = β1 * m + (1 − β1) * g
                c = mom.mul(beta1).add(g, alpha=(1 - beta1))

                # Compute the signed update
                # update = torch.sign(c)
                
                update = zeropower_via_newtonschulz5(c.reshape(len(g), -1)).view(c.shape)

                # Parameter update
                p.add_(update, alpha=-lr)

                # Momentum update: m_t = β2 * m + (1 − β2) * g
                mom.mul_(beta2).add_(g, alpha=(1 - beta2))

        return loss


