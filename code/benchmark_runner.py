import time
from typing import Any, Callable, Dict, Optional

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
    optimizer_class: Callable[..., torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    problem: MatrixProblem,
    X_init: torch.Tensor,
    num_iterations: int = 500,
    record_interval: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run an optimizer on a matrix problem and record metrics.

    Returns a dictionary with:
      - iterations: list[int]
      - cumulative_time: list[float]
      - losses: list[float]
      - grad_frobenius_norms: list[float]  (||∇f(X)||_F)
    """

    X = X_init.clone()

    optimizer = optimizer_class([X], **optimizer_kwargs)

    iterations: list[int] = []
    cumulative_time: list[float] = []
    losses: list[float] = []
    grad_fro_norms: list[float] = []

    start_time = time.time()
    
    for t in range(num_iterations):
        # Compute gradient and set it manually
        grad = problem.gradient(X)
        X.grad = grad
        # Record metrics on schedule
        if t % record_interval == 0:
            with torch.no_grad():
                cur_time = time.time() - start_time
                iterations.append(t)
                cumulative_time.append(cur_time)
                losses.append(problem.objective(X).item())
                grad_now = problem.gradient(X)
                grad_fro_norms.append(torch.linalg.norm(grad_now, ord="fro").item())

        # Optional progress
        if verbose and (t % max(1, (num_iterations // 10)) == 0):
            with torch.no_grad():
                loss_val = problem.objective(X).item()
                grad_norm_val = torch.linalg.norm(grad, ord="fro").item()
                print(f"Iter {t}: loss={loss_val:.6f}, ||∇f(X)||_F={grad_norm_val:.6f}")

        # Optimizer step
        optimizer.step()

        
        # Reset explicit grad storage
        X.grad = None

    return {
        "iterations": iterations,
        "cumulative_time": cumulative_time,
        "losses": losses,
        "grad_frobenius_norms": grad_fro_norms,
        "final_X": X,
    }


