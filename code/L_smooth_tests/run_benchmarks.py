from typing import Any, Dict, Optional, Tuple
from math import sqrt

import torch

from optimizers import Muon, Neon, NormalizedMuon # you must add optimizers to the same folder!
from optimizer_runner import MatrixProblem, run_optimizer_on_problem
from benchmark_plotter import build_default_panels, plot_from_descriptions, plot_and_save_default_panels
from problems import RandomQuadraticPSDProblem as SimpleQuadratic, LogisticRegressionProblem

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
    m, n = 50, 50
    # problem = LogisticRegressionProblem(m, n) 
    problem = SimpleQuadratic(m, n, device=device, seed=42, eig_range_M=(0,1), eig_range_N=(0,1))
    delta = 250 # let it be
    grad_op_eps = 1
    lipsch_const_op = problem.lipschitz_constant(norm=2)
    lipsch_const_nuc = problem.lipschitz_constant(norm='nuc')
    lipsch_const_fro = problem.lipschitz_constant(norm='fro')
    print(lipsch_const_op, lipsch_const_nuc, lipsch_const_fro)

    beta_op = n / 2
    beta_nuc = 1
    beta_fro = sqrt(n)/2

    lr_op = grad_op_eps / beta_op / lipsch_const_op
    lr_nuc = grad_op_eps / beta_nuc / lipsch_const_nuc
    lr_fro = grad_op_eps / beta_fro / lipsch_const_fro
    print(f"Muon lr: {lr_op: .2E}, Neon iter: {lr_nuc: .2e}, Frobenius iter: {lr_fro:.2e}")

    iter_num_op = int(lipsch_const_op * delta / grad_op_eps / grad_op_eps / beta_op / beta_op)
    iter_num_nuc = int(lipsch_const_nuc * delta / grad_op_eps / grad_op_eps / beta_nuc / beta_nuc)
    iter_num_fro = int(lipsch_const_fro * delta / grad_op_eps / grad_op_eps / beta_fro / beta_fro)

    print(lr_op, lr_nuc, lr_fro)
    print(f"Muon iter: {iter_num_op}, Neon iter: {iter_num_nuc}, Frobenius iter: {iter_num_fro}")
    iter_num_nuc = iter_num_op = iter_num_fro = 10000
    # iter_num = 100000
    X0 = 0.1 * torch.randn(m, n, dtype=torch.float32, device=device)
    # Define optimizers and settings
    experiments: Dict[str, Dict[str, Any]] = {}
    nsgd_name = "NSGD"
    nsgd_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=lr_fro, momentum=0.95, nesterov=True, sgd_coeff=1),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num_fro,
        record_interval=10,
        verbose=True,
        name=nsgd_name
    )
    experiments[nsgd_name] = nsgd_results

    muon_name = "Muon"
    muon_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=lr_op, momentum=0.95, nesterov=True),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num_op,
        record_interval=100,
        verbose=True,
        name=muon_name
    )
    experiments[muon_name] = muon_results

    nsgd_muon_name = "NSGD Muon"
    nsgd_muon_results = run_optimizer_on_problem(
        optimizer_class=NormalizedMuon,
        optimizer_kwargs=dict(lr=lr_fro, momentum=0.95, nesterov=True, sgd_coeff=0.5),
        problem=problem,
        X_init=X0,
        num_iterations=iter_num_fro,
        record_interval=100,
        verbose=True,
        name=nsgd_muon_name
    )
    experiments[nsgd_muon_name] = nsgd_muon_results

    if device.type == "cuda":
        neon_name = "Neon"
        neon_results = run_optimizer_on_problem(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=lr_nuc, nesterov=True, momentum=0.95, neon_mode="fast", iter_num=50),
            problem=problem,
            X_init=X0,
            num_iterations=iter_num_nuc,
            record_interval=100,
            verbose=True,
            name=neon_name
        )
        experiments[neon_name] = neon_results
    else:
        print("CUDA not available: skipping Neon run")

    if device.type == "cuda":
        nsgd_neon_name = "NSGD Neon"
        nsgd_neon_results = run_optimizer_on_problem(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=lr_nuc, nesterov=True, momentum=0.95, neon_mode="fast", iter_num=50, sgd_coeff=0.5),
            problem=problem,
            X_init=X0,
            num_iterations=iter_num_nuc,
            record_interval=100,
            verbose=True,
            name=nsgd_neon_name
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
        num_iterations=1000,
        record_interval=1,
        verbose=True,
        name=sgd_name
    )
    experiments[sgd_name] = sgd_results
    # experiments[sgd_name] = sgd_results
    # Build panels and plot
    panels = build_default_panels(experiments)
    
    # Save individual plots in simple_lls folder
    plot_and_save_default_panels(experiments, title_suffix=f"{m}x{n}")
    
    # Also save the combined plot (optional)
    plot_from_descriptions(panels, ncols=2, title_suffix=f"{m}x{n}", save_path="sin_fro_benchmark.png")


if __name__ == "__main__":
    main()


