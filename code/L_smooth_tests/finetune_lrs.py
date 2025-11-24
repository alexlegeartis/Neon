from typing import Any, Dict, List, Optional, Tuple
from math import sqrt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

import torch

from optimizers import Muon, Neon, NormalizedMuon, RandomNormalizedMuon, NeonMuon
from mlion import MLion, Lion
from L_smooth_tests.optimizer_runner import MatrixProblem, run_optimizer_on_problem
from L_smooth_tests.benchmark_plotter import build_default_panels, plot_from_descriptions, plot_and_save_default_panels, save_experiments_to_csv
from L_smooth_tests.problems import RandomQuadraticPSDProblem as SimpleQuadratic, LogisticRegressionProblem


def grid_search_learning_rates(
    algorithm_specs: List[Dict[str, Any]],
    learning_rates: np.ndarray,
    momentums: Optional[np.ndarray],
    problem: MatrixProblem,
    X_init: torch.Tensor,
    csv_path: str,
    loss_threshold: float = 0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform grid search over learning rates and momentums for multiple algorithms.
    Writes results to CSV incrementally as they are computed.
    
    Args:
        algorithm_specs: List of algorithm specifications. Each spec should contain:
            - name: str algorithm name
            - optimizer_class: optimizer class
            - optimizer_kwargs: dict of kwargs (without 'lr' and 'momentum', which will be set dynamically)
            - num_iterations: int number of iterations
            - record_interval: int interval to record metrics
            - requires_cuda: bool (optional)
            - use_momentum: bool (optional) - if True, momentum will be grid searched; if False/None, momentum from kwargs is used
        learning_rates: Array of learning rates to try
        momentums: Array of momentums to try (None if momentum should not be grid searched)
        problem: MatrixProblem instance
        X_init: Initial point tensor
        csv_path: Path to CSV file for incremental writing
        loss_threshold: Loss threshold for below_thresh_iter
        verbose: Whether to print progress
        
    Returns:
        List of dicts with keys: algorithm_name, learning_rate, momentum, below_thresh_iter
    """
    results: List[Dict[str, Any]] = []
    device = X_init.device
    
    # Calculate total runs
    if momentums is not None:
        total_runs = len(algorithm_specs) * len(learning_rates) * len(momentums)
    else:
        total_runs = len(algorithm_specs) * len(learning_rates)
    run_count = 0
    
    # Create parent directory if needed
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    
    # Initialize CSV file with headers (overwrite if exists to start fresh)
    if momentums is not None:
        header = "algorithm_name,learning_rate,momentum,below_thresh_iter\n"
    else:
        header = "algorithm_name,learning_rate,below_thresh_iter\n"
    with open(csv_path, 'w', newline='') as f:
        f.write(header)
    
    for spec in algorithm_specs:
        name = spec["name"]
        requires_cuda = bool(spec.get("requires_cuda", False))
        if requires_cuda and device.type != "cuda":
            print(f"CUDA not available: skipping {name} runs")
            continue
        
        optimizer_class = spec["optimizer_class"]
        base_optimizer_kwargs = spec["optimizer_kwargs"].copy()
        # Check if this algorithm should use momentum grid search
        use_momentum = spec.get("use_momentum", momentums is not None)
        
        # Remove 'lr' and 'momentum' from base kwargs if present (will be set dynamically)
        base_optimizer_kwargs.pop("lr", None)
        if use_momentum and momentums is not None:
            base_optimizer_kwargs.pop("momentum", None)
        
        num_iterations = spec["num_iterations"]
        record_interval = spec["record_interval"]
        
        # Determine momentum values to iterate over
        if use_momentum and momentums is not None:
            momentum_values = momentums
        else:
            # Use momentum from kwargs if specified, otherwise single None value
            momentum_values = [base_optimizer_kwargs.get("momentum", None)]
        
        for lr in learning_rates:
            for momentum in momentum_values:
                run_count += 1
                if verbose:
                    print()
                    if momentum is not None:
                        print(f"\n[{run_count}/{total_runs}] Testing {name} with lr={lr:.6f}, momentum={momentum:.6f}")
                    else:
                        print(f"\n[{run_count}/{total_runs}] Testing {name} with lr={lr:.6f}")
                
                # Create optimizer kwargs with current learning rate and momentum
                optimizer_kwargs = {**base_optimizer_kwargs, "lr": float(lr)}
                if momentum is not None:
                    optimizer_kwargs["momentum"] = float(momentum)
                    optimizer_kwargs["nesterov"] = True
                
                # Clone initial point for fair comparison
                X_init_clone = X_init.clone()
                
                # Run optimizer (no lr_scheduler for grid search - fixed learning rates)
                if momentum is not None:
                    name_suffix = f"_lr_{lr:.6f}_mom_{momentum:.6f}"
                else:
                    name_suffix = f"_lr_{lr:.6f}"
                run_result = run_optimizer_on_problem(
                    f"{name}{name_suffix}",
                    optimizer_class=optimizer_class,
                    optimizer_kwargs=optimizer_kwargs,
                    problem=problem,
                    X_init=X_init_clone,
                    num_iterations=num_iterations,
                    record_interval=record_interval,
                    verbose=False,  # Keep individual runs quiet
                    lr_scheduler=None,  # Fixed learning rates for grid search
                    loss_threshold=loss_threshold,
                )
                
                below_thresh_iter = run_result["below_thresh_iter"]
                # Convert -10 (never reached) to -1 for cleaner logging
                if below_thresh_iter == -10:
                    below_thresh_iter = -1
                
                result = {
                    "algorithm_name": name,
                    "learning_rate": lr,
                    "below_thresh_iter": below_thresh_iter,
                }
                if momentums is not None:
                    result["momentum"] = momentum if momentum is not None else ""
                
                results.append(result)
                
                # Write result immediately to CSV
                with open(csv_path, 'a', newline='') as f:
                    if momentums is not None:
                        momentum_str = f"{momentum}" if momentum is not None else ""
                        f.write(f"{name},{lr},{momentum_str},{below_thresh_iter}\n")
                    else:
                        f.write(f"{name},{lr},{below_thresh_iter}\n")
    
    return results


def save_grid_search_results_to_csv(
    results: List[Dict[str, Any]],
    csv_path: str,
) -> None:
    """
    Save grid search results to CSV file.
    
    Args:
        results: List of dicts with keys: algorithm_name, learning_rate, below_thresh_iter
        csv_path: Path to save CSV file
    """
    df = pd.DataFrame(results)
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Grid search results saved to {csv_path}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print summary table showing best learning rate and momentum per algorithm.
    
    Args:
        results: List of dicts with keys: algorithm_name, learning_rate, momentum (optional), below_thresh_iter
    """
    df = pd.DataFrame(results)
    has_momentum = "momentum" in df.columns
    
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    
    # Find best LR (and momentum if applicable) for each algorithm (lowest below_thresh_iter, ignoring -1)
    best_configs = {}
    for algo in df["algorithm_name"].unique():
        algo_df = df[df["algorithm_name"] == algo].copy()
        # Filter out runs that never reached threshold
        reached = algo_df[algo_df["below_thresh_iter"] != -1]
        
        if len(reached) > 0:
            best = reached.loc[reached["below_thresh_iter"].idxmin()]
            best_configs[algo] = {
                "lr": best["learning_rate"],
                "below_thresh_iter": int(best["below_thresh_iter"]),
            }
            if has_momentum and "momentum" in best:
                momentum_val = best["momentum"]
                # Handle momentum value (could be float, string, or empty)
                if pd.notna(momentum_val) and momentum_val != "":
                    try:
                        best_configs[algo]["momentum"] = float(momentum_val)
                    except (ValueError, TypeError):
                        best_configs[algo]["momentum"] = None
                else:
                    best_configs[algo]["momentum"] = None
        else:
            best_configs[algo] = {
                "lr": None,
                "below_thresh_iter": "N/A (never reached threshold)",
            }
            if has_momentum:
                best_configs[algo]["momentum"] = None
    
    # Print summary table
    if has_momentum:
        print(f"\n{'Algorithm':<20} {'Best LR':<15} {'Best Momentum':<15} {'Below Thresh Iter':<20}")
        print("-" * 80)
        for algo, info in sorted(best_configs.items()):
            lr_str = f"{info['lr']:.6f}" if info['lr'] is not None else "N/A"
            mom_str = f"{info.get('momentum', ''):.6f}" if info.get('momentum') is not None and info.get('momentum') != "" else "N/A"
            iter_str = str(info['below_thresh_iter'])
            print(f"{algo:<20} {lr_str:<15} {mom_str:<15} {iter_str:<20}")
    else:
        print(f"\n{'Algorithm':<20} {'Best LR':<15} {'Below Thresh Iter':<20}")
        print("-" * 80)
        for algo, info in sorted(best_configs.items()):
            lr_str = f"{info['lr']:.6f}" if info['lr'] is not None else "N/A"
            iter_str = str(info['below_thresh_iter'])
            print(f"{algo:<20} {lr_str:<15} {iter_str:<20}")
    
    print("\n" + "="*80)
    print(f"Total runs: {len(results)}")
    print("="*80 + "\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Grid search learning rates and momentums for optimizers and log results to CSV.")
    parser.add_argument("--lr-start", type=float, default=1e-3, help="Start of LR range (log10, default: 1e-3)")
    parser.add_argument("--lr-end", type=float, default=1e0, help="End of LR range (log10, default: 1e0)")
    parser.add_argument("--lr-num", type=int, default=10, help="Number of LR points (default: 10)")
    parser.add_argument("--momentum-start", type=float, default=None, help="Start of momentum range (default: None, no momentum grid search)")
    parser.add_argument("--momentum-end", type=float, default=None, help="End of momentum range (default: None)")
    parser.add_argument("--momentum-num", type=int, default=5, help="Number of momentum points (default: 5)")
    parser.add_argument("--loss-threshold", type=float, default=0.01, help="Loss threshold for below_thresh_iter (default: 0.01)")
    parser.add_argument("--log-csv", type=str, default="L_smooth_tests/results/lr_grid_search.csv", help="Path to save grid search results CSV")
    parser.add_argument("--no-plot", action="store_true", help="If set, skip plotting after running experiments")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional title suffix for saved plots (defaults to size m x n)")
    parser.add_argument("--combined-plot-path", type=str, default=None, help="Optional path to save combined panel figure (PDF)")
    parser.add_argument("--verbose", action="store_true", help="Print progress for each run")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initial point
    m, n = 500, 500
    problem = SimpleQuadratic(m, n, device=device, seed=42, eig_range_M=(0,1), eig_range_N=(0,1), shift_scale=0)
    iter_num_nuc = iter_num_op = iter_num_fro = 12000
    X0 = 0.1 * torch.randn(m, n, dtype=torch.float32, device=device)
    
    # Generate learning rate grid
    # lr_start_log = np.log10(args.lr_start)
    # lr_end_log = np.log10(args.lr_end)
    # learning_rates = np.logspace(lr_start_log, lr_end_log, args.lr_num)
    learning_rates = np.concatenate((
        np.round(np.linspace(0.01, 0.1, 19), 3),
        np.round(np.linspace(0.15, 1, 18), 3)
    ))
    print(f"Learning rate grid: {learning_rates}")
    
    # Generate momentum grid if specified
    momentums = None
    if True:# args.momentum_start is not None and args.momentum_end is not None:
        momentums = [0, 0.5, 0.9, 0.95, 0.99]# np.linspace(args.momentum_start, args.momentum_end, args.momentum_num)
        momentums = np.round(momentums, 3)
        print(f"Momentum grid: {momentums}")
        print(f"Loss threshold: {args.loss_threshold}")
        print(f"Total runs: {len(learning_rates)} learning rates × {len(momentums)} momentums × number of algorithms\n")
    else:
        print(f"Loss threshold: {args.loss_threshold}")
        print(f"Total runs: {len(learning_rates)} learning rates × number of algorithms\n")
    
    # Define algorithm specifications (without fixed lr and momentum if grid searching)
    algorithm_specs: List[Dict[str, Any]] = [
        dict(
            name="NSGD",
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict(nesterov=True, sgd_coeff=1),
            num_iterations=5000,
            record_interval=50,
            use_momentum=momentums is not None,  # Use momentum grid search if momentums are provided
        ),
        # dict(
        #     name="MLion",
        #     optimizer_class=MLion,
        #     optimizer_kwargs=dict(),
        #     num_iterations=iter_num_op,
        #     record_interval=100,
        # ),
        # dict(
        #     name="Lion",
        #     optimizer_class=Lion,
        #     optimizer_kwargs=dict(),
        #     num_iterations=iter_num_op,
        #     record_interval=100,
        # ),
        dict(
            name="Muon",
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict(nesterov=True),
            num_iterations=3500,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-Muon",
            optimizer_class=NormalizedMuon,
            optimizer_kwargs=dict(nesterov=True, sgd_coeff=0.5),
            num_iterations=3000,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-Neon",
            optimizer_class=Neon,
            optimizer_kwargs=dict(nesterov=True, neon_mode="kyfan", iter_num=1, sgd_coeff=0.5),
            num_iterations=8500,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-NeonMuon",
            optimizer_class=NeonMuon,
            optimizer_kwargs=dict(nesterov=True, neon_share=0.5, iter_num=10, sgd_coeff=0.33),
            num_iterations=3000,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-Fanion-2",
            optimizer_class=Neon,
            optimizer_kwargs=dict(nesterov=True, neon_mode="kyfan", iter_num=10, sgd_coeff=0.5, k=2),
            num_iterations=7500,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-Fanion-5",
            optimizer_class=Neon,
            optimizer_kwargs=dict(nesterov=True, neon_mode="kyfan", iter_num=10, sgd_coeff=0.5, k=5),
            num_iterations=7500,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        dict(
            name="F-Fanion-100",
            optimizer_class=Neon,
            optimizer_kwargs=dict(nesterov=True, neon_mode="kyfan", iter_num=10, sgd_coeff=0.5, k=100),
            num_iterations=5000,
            record_interval=50,
            use_momentum=momentums is not None,
        ),
        # dict(
        #     name="Neon",
        #     optimizer_class=Neon,
        #     optimizer_kwargs=dict(nesterov=True, momentum=0.6, neon_mode="kyfan", iter_num=10, sgd_coeff=0),
        #     num_iterations=1000,
        #     record_interval=100,
        # ),
        # dict(
        #     name="NeonMuon",
        #     optimizer_class=NeonMuon,
        #     optimizer_kwargs=dict(nesterov=True, momentum=0.6, neon_share=0.5, iter_num=10, sgd_coeff=0),
        #     num_iterations=4500,
        #     record_interval=100,
        # ),
        
    ]
    
    # Run grid search (writes to CSV incrementally)
    print(f"Writing results incrementally to {args.log_csv}")
    grid_search_results = grid_search_learning_rates(
        algorithm_specs=algorithm_specs,
        learning_rates=learning_rates,
        momentums=momentums,
        problem=problem,
        X_init=X0,
        csv_path=args.log_csv,
        loss_threshold=args.loss_threshold,
        verbose=args.verbose,
    )
    
    print(f"Grid search complete. Results saved to {args.log_csv}")
    
    # Print summary
    print_summary(grid_search_results)
    
    # Optional plotting (if requested and not disabled)
    if not args.no_plot:
        # For plotting, we could optionally run the best LR for each algorithm
        # But for now, we'll skip plotting in grid search mode
        print("Plotting skipped in grid search mode. Use individual runs for plotting.")


if __name__ == "__main__":
    main()
