# maxiumum 93% instead of Muon's 94%
import torch
from optimizers import zeropower_via_newtonschulz5
from torch.optim import Optimizer

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
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                
                update = zeropower_via_newtonschulz5(exp_avg.reshape(len(g), -1)).view(exp_avg.shape) # whiten the update
                # p.add_(torch.sign(exp_avg), alpha=-lr)
                p.add_(update, alpha=-lr * max(1, p.size(-2) / p.size(-1)) ** 0.5)

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
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                
                # update = zeropower_via_newtonschulz5(exp_avg.reshape(len(g), -1)).view(exp_avg.shape) # whiten the update
                p.add_(torch.sign(exp_avg), alpha=-lr)
                # p.add_(update, alpha=-lr * max(1, p.size(-2) / p.size(-1)) ** 0.5)

        return loss

