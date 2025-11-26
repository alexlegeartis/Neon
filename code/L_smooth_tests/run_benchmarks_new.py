from typing import Any, Dict, Optional, Tuple
from math import sqrt
import matplotlib.cm as cm
import numpy as np

import torch

from optimizers import Muon, Neon, NormalizedMuon, RandomNormalizedMuon, NeonMuon, SignSGDMuon
from mlion import MLion, Lion
from L_smooth_tests.optimizer_runner import MatrixProblem, run_optimizer_on_problem
from L_smooth_tests.benchmark_plotter import build_default_panels, plot_from_descriptions, plot_and_save_default_panels, save_experiments_to_csv
from L_smooth_tests.problems import RandomQuadraticPSDProblem as SimpleQuadratic, LogisticRegressionProblem

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


def run_experiments_from_specs(
    specs: Dict[str, Dict[str, Any]],
    problem: MatrixProblem,
    X_init: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple experiments defined in a dictionary of specifications.

    Each spec value can contain the following keys:
    - optimizer_class: optimizer class (required)
    - optimizer_kwargs: dict of kwargs for the optimizer (required)
    - num_iterations: int number of iterations (required)
    - record_interval: int interval to record metrics (required)
    - verbose: bool verbosity flag (optional, default True)
    - lr_scheduler: callable lr scheduler (optional)
    - name: override name for the experiment (optional; defaults to dict key)
    - requires_cuda: bool to skip when CUDA unavailable (optional)
    """
    results: Dict[str, Dict[str, Any]] = {}
    device = X_init.device
    for key, spec in specs.items():
        name = spec.get("name", key)
        requires_cuda = bool(spec.get("requires_cuda", False))
        if requires_cuda and device.type != "cuda":
            print(f"CUDA not available: skipping {name} run")
            continue

        optimizer_class = spec["optimizer_class"]
        optimizer_kwargs = spec.get("optimizer_kwargs", {})
        optimizer_kwargs["norm_weight"] = False # very important
        num_iterations = spec["num_iterations"]
        record_interval = spec["record_interval"]
        verbose = spec.get("verbose", True)
        lr_scheduler = spec.get("lr_scheduler")
        color = spec.get("color")
        linestyle = spec.get("linestyle")

        run_result = run_optimizer_on_problem(
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            problem=problem,
            X_init=X_init,
            num_iterations=num_iterations,
            record_interval=record_interval,
            verbose=verbose,
            name=name,
            lr_scheduler=lr_scheduler,
        )
        # Attach optional styling info for downstream plotting
        if color is not None:
            run_result["color"] = color
        if linestyle is not None:
            run_result["linestyle"] = linestyle
        results[name] = run_result
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run Neon benchmarks and optionally save results to CSV.")
    parser.add_argument("--save-csv", type=str, default=None, help="Path to CSV to save experiments (e.g., L_smooth_tests/results/exp.csv)")
    parser.add_argument("--no-plot", action="store_true", help="If set, skip plotting after running experiments")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional title suffix for saved plots (defaults to size m x n)")
    parser.add_argument("--combined-plot-path", type=str, default=None, help="Optional path to save combined panel figure (PDF)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initial point
    m, n = 500, 500
    # problem = LogisticRegressionProblem(m, n) 
    problem = SimpleQuadratic(m, n, device=device, seed=42, eig_range_M=(0,1), eig_range_N=(0,1), shift_scale=0)
    iter_num_nuc = iter_num_op = iter_num_fro = 1200
    record_period=25
    # iter_num = 100000
    X0 = 0.1 * torch.randn(m, n, dtype=torch.float32, device=device)
    # Define optimizers and settings
    experiments: Dict[str, Dict[str, Any]] = {}
    # Inverse-square-root scheduler: lr(T) = lr0 / sqrt(T)
    def inv_sqrt_lr(T: int, base_lr: float) -> float:
        return base_lr / T**(1/2)
    def const_lr(T: int, base_lr: float) -> float:
        return base_lr


    muon_lr = 0.007
    fmuon_lr = 0.015
    smuon_lr = 0.011
    
    muon_mom = 0.5
    fmuon_mom = 0.7
    smuon_mom = 0.9

    # Define experiment specifications
    colors = cm.tab20b(np.linspace(0, 1, 5))[::-1]
    experiment_specs: Dict[str, Dict[str, Any]] = {
        "NSGD": dict(
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict(lr=0.25, momentum=0.9, nesterov=True, sgd_coeff=1),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            color="gray"
        ),
        #  "Random F-Muon": dict(
        #     optimizer_class=RandomNormalizedMuon,
        #     optimizer_kwargs=dict(lr=1, momentum=0.95, nesterov=True, sgd_coeff=0.5),
        #     num_iterations=iter_num_fro,
        #     record_interval=100,
        #     verbose=True,
        #     lr_scheduler=inv_sqrt_lr,
        # ),
        # "NeonMuon": dict(
        #     optimizer_class=NeonMuon,
        #     optimizer_kwargs=dict(lr=0.021, nesterov=True, momentum=0.6, iter_num=5, neon_share=0.5),
        #     num_iterations=iter_num_nuc,
        #     record_interval=100,
        #     verbose=True,
        #     lr_scheduler=const_lr,
        #     requires_cuda=True,
        #     color=colors[0]
        # ),
        # "F-NeonMuon": dict(
        #     optimizer_class=NeonMuon,
        #     optimizer_kwargs=dict(lr=0.055, nesterov=True, momentum=0.9, iter_num=5, neon_share=0.5, sgd_coeff=0.33),
        #     num_iterations=iter_num_fro,
        #     record_interval=50,
        #     verbose=True,
        #     lr_scheduler=const_lr,
        #     requires_cuda=True,
        #     color=colors[0]
        # ),
        "Neon": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=muon_lr, nesterov=True, momentum=muon_mom, neon_mode="kyfan", iter_num=5),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[0]
        ),
        "F-Neon": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=fmuon_lr, nesterov=True, momentum=fmuon_mom, neon_mode="kyfan", iter_num=5, sgd_coeff=0.5),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[0]
        ),
        "Fanion-2": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=muon_lr, nesterov=True, momentum=muon_mom, neon_mode="kyfan", iter_num=5, k=2),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[1]
        ),
        "F-Fanion-2": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=fmuon_lr, nesterov=True, momentum=fmuon_mom, neon_mode="kyfan", iter_num=5, sgd_coeff=0.5, k = 2),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[1]
        ),
        "Fanion-10": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=muon_lr, nesterov=True, momentum=muon_mom, neon_mode="kyfan", iter_num=5, k=10),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[2]
        ),
        "F-Fanion-10": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=fmuon_lr, nesterov=True, momentum=fmuon_mom, neon_mode="kyfan", iter_num=5, sgd_coeff=0.5, k = 10),
            num_iterations=iter_num_nuc,
            record_interval=100,
            verbose=True,
            lr_scheduler=inv_sqrt_lr,
            requires_cuda=True,
            color=colors[2]
        ),
        "Fanion-100": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=muon_lr, nesterov=True, momentum=muon_mom, neon_mode="kyfan", iter_num=5, k=100),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[3]
        ),
        "F-Fanion-100": dict(
            optimizer_class=Neon,
            optimizer_kwargs=dict(lr=fmuon_lr, nesterov=True, momentum=fmuon_mom, neon_mode="kyfan", iter_num=5, sgd_coeff=0.5, k = 100),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            requires_cuda=True,
            color=colors[3]
        ),

        # "MLion": dict(
        #     optimizer_class=MLion,
        #     optimizer_kwargs=dict(lr=0.1),
        #     num_iterations=iter_num_op,
        #     record_interval=100,
        #     verbose=True,
        #     lr_scheduler=inv_sqrt_lr,
        #     color=colors[3]
        # ),
        # "Lion": dict(
        #     optimizer_class=Lion,
        #     optimizer_kwargs=dict(lr=0.1),
        #     num_iterations=iter_num_op,
        #     record_interval=100,
        #     verbose=True,
        #     lr_scheduler=inv_sqrt_lr,
        #     color=colors[3]
        # ),
        "S-Muon": dict(
            optimizer_class=SignSGDMuon,
            optimizer_kwargs=dict(lr=smuon_lr, momentum=smuon_mom, nesterov=True, sign_lr_mult=0.01, sgd_coeff=0.5), # lr=0.035, momentum=0.8 for 0.01 loss
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            color=colors[3]
        ),
        "SignSGD": dict(
            optimizer_class=SignSGDMuon,
            optimizer_kwargs=dict(lr=0.055, momentum=0.95, nesterov=True, sign_lr_mult=0.01, sgd_coeff=1),
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            color=colors[3]
        ),
        "Muon": dict(
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict(lr=muon_lr, momentum=muon_mom, nesterov=True), # 0.025, 0.5 for 0.01 loss
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            color=colors[4]
        ),
        "F-Muon": dict(
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict[str, float](lr=fmuon_lr, momentum=fmuon_mom, nesterov=True, sgd_coeff=0.5), # lr=0.035, momentum=0.5 for 0.01 loss
            num_iterations=iter_num_fro,
            record_interval=record_period,
            verbose=True,
            lr_scheduler=const_lr,
            color=colors[4]
        ),
        # "SGD": dict(
        #     optimizer_class=torch.optim.SGD,
        #     optimizer_kwargs=dict(lr=0.04, momentum=0.95, nesterov=True),
        #     num_iterations=iter_num_nuc,
        #     record_interval=1,
        #     verbose=True,
        #     lr_scheduler=inv_sqrt_lr,
        # ),
    }
    linestyles = ["--", "-",]
    for idx, key in enumerate(experiment_specs.keys()):
        spec = experiment_specs[key]
        # spec["color"] = tab20_hex[idx % len(tab20_hex)]
        spec["linestyle"] = linestyles[idx % len(linestyles)]
        #if idx == 0:
        #    spec["linestyle"] = "-"

    # Run all experiments via the unified helper
    experiments = run_experiments_from_specs(experiment_specs, problem=problem, X_init=X0)
    # experiments[sgd_name] = sgd_results
    # Build panels and plot
    # Optionally save to CSV
    if args.save_csv is not None:
        save_experiments_to_csv(experiments, args.save_csv)

    if not args.no_plot:
        panels = build_default_panels(experiments)
        title_suffix = args.title_suffix if args.title_suffix is not None else f"{m}x{n}"
        # Save individual plots
        plot_and_save_default_panels(experiments, title_suffix=title_suffix)
        # Also save the combined plot (optional)
        if args.combined_plot_path is not None:
            plot_from_descriptions(panels, ncols=2, title_suffix=title_suffix, save_path=args.combined_plot_path)


if __name__ == "__main__":
    main()


