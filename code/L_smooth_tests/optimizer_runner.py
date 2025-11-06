import time
from typing import Any, Callable, Dict, Optional
from math import sqrt

import torch


class MatrixProblem:
    """
    Interface for matrix optimization problems.

    A problem must define:
    - objective(X): returns scalar loss tensor
    - gradient(X): returns tensor with the same shape as X (d objective / dX)
    """

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def run_optimizer_on_problem(
    name: str,
    optimizer_class: Callable[..., torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    problem: MatrixProblem,
    X_init: torch.Tensor,
    num_iterations: int = 500,
    record_interval: int = 1,
    verbose: bool = False,
    lr_scheduler: Optional[Callable[[int, float], float]] = None,
) -> Dict[str, Any]:
    """
    Run an optimizer on a matrix problem and record metrics.

    Returns a dictionary with:
      - iterations: list[int]
      - cumulative_time: list[float]
      - losses: list[float]
      - grad_frobenius_norms: list[float]  (||∇f(X)||_F)
      - grad_spectral_norms: list[float]   (||∇f(X)||_2)
      - grad_nuclear_norms: list[float]    (||∇f(X)||_* )
    """
    print(f"Testing {name}")
    X = X_init.clone()

    optimizer = optimizer_class([X], **optimizer_kwargs)

    # Capture base learning rates per param group to support scheduling
    base_lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]

    iterations: list[int] = []
    cumulative_time: list[float] = []
    losses: list[float] = []
    grad_frobenius_norms: list[float] = []
    grad_spectral_norms: list[float] = []
    grad_nuclear_norms: list[float] = []

    start_time = time.time()
    overhead_time = 0.0  # time spent on costly gradient norm computations (excluded from timing)
    
    for t in range(num_iterations):
        # Compute gradient and set it manually
        grad = problem.gradient(X)
        X.grad = grad
        # Record metrics on schedule
        if t % record_interval == 0:
            with torch.no_grad():
                cur_time = time.time() - start_time - overhead_time
                iterations.append(t)
                cumulative_time.append(cur_time)
                losses.append(problem.objective(X).item())
                grad_now = problem.gradient(X)
                # Frobenius, spectral, and nuclear norms
                _oh_start = time.time()
                grad_frobenius_norms.append(torch.linalg.norm(grad_now, ord='fro').item())
                grad_spectral_norms.append(torch.linalg.norm(grad_now, ord=2).item())
                grad_nuclear_norms.append(torch.linalg.norm(grad_now, ord='nuc').item())
                overhead_time += time.time() - _oh_start

        # Optional progress
        if verbose and (t % max(1, (num_iterations // 10)) == 0):
            with torch.no_grad():
                loss_val = problem.objective(X).item()
                _oh_start_verbose = time.time()
                grad_fro_val = torch.linalg.norm(grad, ord='fro').item()
                grad_spec_val = torch.linalg.norm(grad, ord=2).item()
                grad_nuc_val = torch.linalg.norm(grad, ord='nuc').item()
                overhead_time += time.time() - _oh_start_verbose
                print(
                    f"Iter {t}: loss={loss_val:.6f}, "
                    f"||∇f(X)||_F={grad_fro_val:.6f}, "
                    f"||∇f(X)||_2={grad_spec_val:.6f}, "
                    f"||∇f(X)||_*={grad_nuc_val:.6f}"
                )

        # Apply learning rate scheduler, if any (T = t+1 to avoid division by zero)
        if lr_scheduler is not None:
            T = t + 1
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = lr_scheduler(T, base_lr)

        # Optimizer step
        optimizer.step()

        
        # Reset explicit grad storage
        X.grad = None

    print(f"Gradient norm computation overhead (excluded from timing): {overhead_time:.6f}s")

    return {
        "iterations": iterations,
        "cumulative_time": cumulative_time,
        "losses": losses,
        "grad_frobenius_norms": grad_frobenius_norms,
        "grad_spectral_norms": grad_spectral_norms,
        "grad_nuclear_norms": grad_nuclear_norms,
        "final_X": X,
    }


