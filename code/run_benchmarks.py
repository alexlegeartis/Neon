from typing import Any, Dict, Optional, Tuple

import torch

from optimizers import Muon, Neon, NormalizedMuon
from benchmark_runner import MatrixProblem, run_optimizer_on_problem
from benchmark_plotter import build_default_panels, plot_from_descriptions
from problems import RandomQuadraticPSDProblem as SimpleQuadratic

class SinFrobeniusNormProblem(MatrixProblem):
    """
    Problem: f(X) = sin(||X||_F)
    ∂f/∂X = cos(r) * X / r, r = ||X||_F, and 0 if r = 0.
    """

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(X, ord="fro")
        return torch.sin(r)

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(X, ord="fro")
        if r.item() == 0.0:
            return torch.zeros_like(X)
        return torch.cos(r) * X / r


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define problem
    
    # Initial point
    m, n = 10, 100
    problem = SimpleQuadratic(m, n, device=device, seed=42)  # or SinFrobeniusNormProblem()
    grad_eps = 1e-1
    lipsch_const = problem.lipschitz_constant()
    common_lr = grad_eps / lipsch_const
    iter_num = int(lipsch_const * 10 /  grad_eps / grad_eps)
    print(iter_num, common_lr)
    X0 = 0.1 * torch.randn(m, n, dtype=torch.float32, device=device)
    # Define optimizers and settings
    experiments: Dict[str, Dict[str, Any]] = {}
    nsgd_name = "NSGD"
    nsgd_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=common_lr, momentum=0.95, nesterov=True, sgd_coeff=1),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num,
        record_interval=10,
        verbose=True,
    )
    experiments[nsgd_name] = nsgd_results

    muon_name = "Muon"
    muon_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=common_lr, momentum=0.95, nesterov=True),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num,
        record_interval=10,
        verbose=True,
    )
    experiments[muon_name] = muon_results

    nsgd_muon_name = "NSGD Muon"
    nsgd_muon_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=common_lr, momentum=0.95, nesterov=True, sgd_coeff=0.5),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num,
        record_interval=10,
        verbose=True,
    )
    experiments[nsgd_muon_name] = nsgd_muon_results

    if device.type == "cuda":
        neon_name = "Neon"
        neon_results = run_optimizer_on_problem(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=common_lr, nesterov=True, momentum=0.95, neon_mode="fast", iter_num=50),
            problem=problem,
            X_init=X0,
            num_iterations=iter_num,
            record_interval=10,
            verbose=True,
        )
        experiments[neon_name] = neon_results
    else:
        print("CUDA not available: skipping Neon run")

    if device.type == "cuda":
        nsgd_neon_name = "NSGD Neon"
        nsgd_neon_results = run_optimizer_on_problem(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=common_lr, nesterov=True, momentum=0.95, neon_mode="fast", iter_num=50, sgd_coeff=0.5),
            problem=problem,
            X_init=X0,
            num_iterations=iter_num,
            record_interval=10,
            verbose=True,
        )
        experiments[nsgd_neon_name] = nsgd_neon_results
    else:
        print("CUDA not available: skipping Neon run")

    sgd_name = "SGD"
    sgd_results = run_optimizer_on_problem(
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs=dict(lr=0.04, momentum=0.95, nesterov=True),
        problem=problem,
        X_init=X0,
        num_iterations=200,
        record_interval=1,
        verbose=True,
    )
    # experiments[sgd_name] = sgd_results
    # Build panels and plot
    panels = build_default_panels(experiments)
    plot_from_descriptions(panels, ncols=2, title_suffix=f"{m}x{n}", save_path="sin_fro_benchmark.png")


if __name__ == "__main__":
    main()


