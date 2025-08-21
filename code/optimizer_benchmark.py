import time
import math
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

# Local optimizers
from optimizers import Muon, Neon


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


class SinFrobeniusNormProblem(MatrixProblem):
    """
    Problem: f(X) = sin(||X||_F)

    Gradient:
      Let r = ||X||_F. Then ∂f/∂X = cos(r) * X / r for r > 0, and 0 when r = 0.
    """

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(X, ord="fro")
        return torch.sin(r)

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(X, ord="fro")
        if r.item() == 0.0:
            return torch.zeros_like(X)
        return torch.cos(r) * X / r


def benchmark_optimizer_on_problem(
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

    # Clone the initial point so repeated benchmarks are independent
    X = X_init.clone()

    # Our custom optimizers expect manual gradients set on parameters
    # We do not use autograd here
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

        # Optionally log progress
        if verbose and (t % max(1, (num_iterations // 10)) == 0):
            with torch.no_grad():
                loss_val = problem.objective(X).item()
                print(f"Iter {t}: loss={loss_val:.6f}, ||X||_F={torch.linalg.norm(X, ord='fro').item():.6f}")

        # Step
        optimizer.step()

        # Record metrics on schedule
        if t % record_interval == 0:
            with torch.no_grad():
                cur_time = time.time() - start_time
                iterations.append(t)
                cumulative_time.append(cur_time)
                losses.append(problem.objective(X).item())
                # Recompute gradient at the updated X to measure ||∇f(X)||_F
                grad_now = problem.gradient(X)
                grad_fro_norms.append(torch.linalg.norm(grad_now, ord="fro").item())

        # Prevent growing autograd graphs (we don't use autograd but be safe)
        X.grad = None

    return {
        "iterations": iterations,
        "cumulative_time": cumulative_time,
        "losses": losses,
        "grad_frobenius_norms": grad_fro_norms,
        "final_X": X,
    }


def plot_from_descriptions(
    panel_descriptions: List[Dict[str, Any]],
    ncols: int = 2,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Generic plotting utility.

    panel_descriptions: list of panels; each panel is a dict with keys:
      - title: str
      - x_label: str
      - y_label: str
      - series: list of dict, each with keys:
          - name: str
          - x: list/array-like
          - y: list/array-like
          - color: Optional[str]
          - linestyle: Optional[str]
          - linewidth: Optional[float]
      - yscale: Optional[str] (e.g., 'log')
    """

    num_panels = len(panel_descriptions)
    nrows = math.ceil(num_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, panel in enumerate(panel_descriptions):
        ax = axes_list[idx]
        for s in panel.get("series", []):
            ax.plot(
                s["x"],
                s["y"],
                label=s.get("name", "series"),
                color=s.get("color", None),
                linestyle=s.get("linestyle", "-"),
                linewidth=s.get("linewidth", 2),
            )
        ax.set_xlabel(panel.get("x_label", ""))
        ax.set_ylabel(panel.get("y_label", ""))
        base_title = panel.get("title", "")
        ax.set_title(base_title + (f" ({title_suffix})" if title_suffix else ""))
        if panel.get("yscale"):
            ax.set_yscale(panel["yscale"]) 
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_panels, len(axes_list)):
        fig.delaxes(axes_list[j])

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def run_sin_fro_benchmark(
    m: int = 1000,
    n: int = 100,
    num_iterations: int = 300,
    record_interval: int = 1,
    lr: float = 0.01,
    momentum: float = 0.005,
    neon_mode: str = "fast",
    neon_iter_num: int = 50,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Benchmark Muon and Neon on f(X) = sin(||X||_F) with matrix variable X in R^{m x n}.
    Returns (muon_results, neon_results). neon_results can be None if CUDA is unavailable.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    problem = SinFrobeniusNormProblem()

    # Initialize X with a larger norm to make the sine landscape non-trivial
    X0 = (100.0 * torch.randn(m, n, dtype=torch.float32, device=device))

    # Muon benchmark
    muon_kwargs = dict(lr=lr, momentum=momentum, nesterov=True)
    muon_results = benchmark_optimizer_on_problem(
        optimizer_class=Muon,
        optimizer_kwargs=muon_kwargs,
        problem=problem,
        X_init=X0,
        num_iterations=num_iterations,
        record_interval=record_interval,
        verbose=verbose,
    )

    # Neon benchmark (requires CUDA for best performance; skip on CPU if unavailable)
    neon_results: Optional[Dict[str, Any]] = None
    if device.type == "cuda":
        neon_kwargs = dict(lr=lr, nesterov=True, momentum=momentum, neon_mode=neon_mode, iter_num=neon_iter_num)
        neon_results = benchmark_optimizer_on_problem(
            optimizer_class=Neon,
            optimizer_kwargs=neon_kwargs,
            problem=problem,
            X_init=X0,
            num_iterations=num_iterations,
            record_interval=record_interval,
            verbose=verbose,
        )
    else:
        print("WARNING: CUDA not available. Skipping Neon benchmark.")

    # Assemble plot descriptions
    panels: List[Dict[str, Any]] = []

    # Loss vs Iteration
    panels.append({
        "title": "Loss vs Iteration",
        "x_label": "Iteration",
        "y_label": "Loss",
        "series": [
            {"name": "Muon", "x": muon_results["iterations"], "y": muon_results["losses"], "color": "b"},
        ] + (
            [{"name": "Neon", "x": neon_results["iterations"], "y": neon_results["losses"], "color": "r"}] if neon_results is not None else []
        )
    })

    # Loss vs Time
    panels.append({
        "title": "Loss vs Time",
        "x_label": "Time (s)",
        "y_label": "Loss",
        "series": [
            {"name": "Muon", "x": muon_results["cumulative_time"], "y": muon_results["losses"], "color": "b"},
        ] + (
            [{"name": "Neon", "x": neon_results["cumulative_time"], "y": neon_results["losses"], "color": "r"}] if neon_results is not None else []
        )
    })

    # Grad Fro norm vs Iteration
    panels.append({
        "title": "Gradient Frobenius Norm vs Iteration",
        "x_label": "Iteration",
        "y_label": "||∇f(X)||_F",
        "series": [
            {"name": "Muon", "x": muon_results["iterations"], "y": muon_results["grad_frobenius_norms"], "color": "b"},
        ] + (
            [{"name": "Neon", "x": neon_results["iterations"], "y": neon_results["grad_frobenius_norms"], "color": "r"}] if neon_results is not None else []
        )
    })

    # Grad Fro norm vs Time
    panels.append({
        "title": "Gradient Frobenius Norm vs Time",
        "x_label": "Time (s)",
        "y_label": "||∇f(X)||_F",
        "series": [
            {"name": "Muon", "x": muon_results["cumulative_time"], "y": muon_results["grad_frobenius_norms"], "color": "b"},
        ] + (
            [{"name": "Neon", "x": neon_results["cumulative_time"], "y": neon_results["grad_frobenius_norms"], "color": "r"}] if neon_results is not None else []
        )
    })

    # Plot
    save_path = "sin_fro_benchmark.png"
    plot_from_descriptions(panels, ncols=2, title_suffix=f"{m}x{n}", save_path=save_path)

    return muon_results, neon_results


def main() -> None:
    """
    Execute the default benchmark of Muon and Neon on f(X) = sin(||X||_F).
    """
    run_sin_fro_benchmark(
        m=1000,
        n=100,
        num_iterations=200,
        record_interval=1,
        lr=0.01,
        momentum=0.005,
        neon_mode="fast",
        neon_iter_num=50,
        seed=42,
        verbose=True,
    )


if __name__ == "__main__":
    main()


