"""All-rank follow-up: reproducible bounds and audits, with no training runs.

Run: python theory_attempts/fanion_analysis.py
All outputs stay in theory_attempts/results/fanions. Original files are read only.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import LAYERS, polar_k, write_csv


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "results" / "fanions"
RANKS = (1, 2, 10, 100, 500)


def source_record(path):
    return {"path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def capped_allocation(s, p, cap, budget):
    """Optimal singular corrections: rank <= p, op <= cap, nuclear <= budget.

    Input singular values must be nonnegative and nonincreasing. This is a
    relaxation of an update trajectory, not a simulation of Fanion iterates.
    """
    s = np.asarray(s, dtype=float)
    p = min(max(int(p), 0), len(s))
    correction = np.zeros_like(s)
    if p == 0 or cap <= 0 or budget <= 0:
        return correction
    target = min(float(budget), float(np.minimum(s[:p], cap).sum()))
    if target >= float(np.minimum(s[:p], cap).sum()):
        correction[:p] = np.minimum(s[:p], cap)
        return correction
    lo, hi = 0.0, float(s[0])
    for _ in range(70):
        mid = (lo + hi) / 2
        if np.clip(s[:p] - mid, 0, cap).sum() > target:
            lo = mid
        else:
            hi = mid
    correction[:p] = np.clip(s[:p] - (lo + hi) / 2, 0, cap)
    return correction


def spectral_barrier(s, p, cap, budget):
    correction = capped_allocation(s, p, cap, budget)
    return float(np.sum((s - correction) ** 2) / 2)


def family_barrier(s, k, steps, gamma_sum, sqrt_l, family, a=0.5, b=0.5,
                   eta=0.01, entries=None):
    """Combine total singular budgets and low-rank-part + dense-part budgets."""
    r = len(s)
    if entries is None:
        entries = r * r
    if family == "plain":
        return spectral_barrier(s, k * steps, sqrt_l * gamma_sum,
                                sqrt_l * k * gamma_sum)
    dense_radius = 1.0 if family == "F" else eta * np.sqrt(entries)
    nuclear_radius = a * k + b * np.sqrt(r) * dense_radius
    op_radius = a + b * dense_radius
    total = spectral_barrier(s, r, sqrt_l * gamma_sum * op_radius,
                             sqrt_l * gamma_sum * nuclear_radius)
    low = spectral_barrier(s, k * steps, sqrt_l * a * gamma_sum,
                           sqrt_l * a * k * gamma_sum)
    split = max(np.sqrt(2 * low) - sqrt_l * b * dense_radius * gamma_sum, 0) ** 2 / 2
    return float(max(total, split))


def kyfan_interval(fro, op, nuc, r, k):
    """Proved moment bounds for S_k from (Frobenius, operator, nuclear) norms.

    These are safe analytic intervals, not a claim that the bounds are sharp.
    """
    if k == 1:
        return float(op), float(op)
    if k == r:
        return float(nuc), float(nuc)
    n, p, q = r - 1, k - 1, r - k
    rem_sum = max(nuc - op, 0.0)
    rem_square = max(fro * fro - op * op, 0.0)
    spread = max(rem_square - rem_sum * rem_sum / n, 0.0)
    upper = min(p * op, rem_sum,
                p * rem_sum / n + np.sqrt(p * q / n * spread))
    root_sum = p * op + rem_sum
    disc = max(root_sum * root_sum - 4 * p * rem_square, 0.0)
    root_lower = (2 * p * rem_square / (root_sum + np.sqrt(disc))
                  if root_sum else 0.0)
    lower = max(p * rem_sum / n, root_lower, rem_sum - q * op, 0.0)
    if lower > upper + 1e-6 * max(nuc, 1.0):
        raise ValueError("Input norms are inconsistent with the supplied rank")
    return float(op + lower), float(op + upper)


def q_f(sk, fro, k, a=0.5, b=0.5):
    x = sk / fro
    return (a * x + b) ** 2 / (a * a * k + b * b + 2 * a * b * x)


def current_specs():
    """Inspect the source AST; do not import or execute benchmark/training code."""
    path = REPO / "code" / "L_smooth_tests" / "run_benchmarks_new.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            try:
                val = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = val
    specs = next(node.value for node in ast.walk(tree)
                 if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name)
                 and node.target.id == "experiment_specs")

    def scalar(node):
        return constants[node.id] if isinstance(node, ast.Name) else ast.literal_eval(node)

    result = {}
    for key, val in zip(specs.keys, specs.values):
        name = ast.literal_eval(key)
        fields = {x.arg: x.value for x in val.keywords}
        kw = {x.arg: scalar(x.value) for x in fields["optimizer_kwargs"].keywords}
        family = "F" if name.startswith("F-") else "S" if name.startswith("S-") else "plain"
        rank = 500 if "Muon" in name else kw.get("k", 1)
        if name in ("NSGD", "SignSGD"):
            family, rank = name, None
        result[name] = {"family": family, "k": rank,
                        "lr": kw["lr"], "momentum": kw["momentum"],
                        "scheduler": fields["lr_scheduler"].id,
                        "dense_coefficient": kw.get("sgd_coeff", 0),
                        "eta": kw.get("sign_lr_mult", None)}
    return result, source_record(path)


def audit_quadratic():
    path = REPO / "code" / "long_run"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    data = {}
    for row in raw:
        if row["x_key"] == "iterations":
            data.setdefault(row["experiment"], {}).setdefault(row["y_key"], []).append(
                (int(float(row["x"])), float(row["y"])))
    specs, script_source = current_specs()
    rows, curve_rows = [], []
    for name, metrics in data.items():
        points = np.array(sorted(metrics["losses"]))
        times, losses = points.T
        assert len(times) == 500 and np.all(np.diff(times) == 10)
        assert np.all(np.isfinite(losses)) and np.all(losses >= 0)
        row = {"method": name, **specs[name], "observations": len(times),
               "initial_loss": losses[0], "last_iteration": int(times[-1]),
               "last_loss": losses[-1], "minimum_saved_loss": losses.min()}
        for threshold in (100, 10, 1, 0.01, 0.001):
            hits = np.flatnonzero(losses <= threshold)
            row["first_saved_time_loss_" + str(threshold)] = int(times[hits[0]]) if hits.size else None
        rows.append(row)
        curve_rows += [{"method": name, "iteration": int(t), "loss": float(loss)}
                       for t, loss in points]
    write_csv(OUT / "saved_quadratic_summary.csv", rows)
    write_csv(OUT / "saved_quadratic_curves.csv", curve_rows)

    reference = np.array(sorted(data["Neon"]["losses"]))
    initial_loss = reference[0, 1]
    collapse = []
    for k in (2, 10, 100):
        name = "Fanion-" + str(k)
        arr = np.array(sorted(data[name]["losses"]))
        keep = (arr[:, 0] > 0) & (arr[:, 0] * k <= reference[-1, 0])
        arr = arr[keep]
        ref_loss = np.interp(arr[:, 0] * k, reference[:, 0], reference[:, 1])
        err = (arr[:, 1] - ref_loss) / initial_loss
        collapse.append({"method": name, "k": k, "comparison_snapshots": len(arr),
                         "last_compared_iteration": int(arr[-1, 0]),
                         "max_abs_error_divided_by_initial_loss": float(np.abs(err).max()),
                         "rms_error_divided_by_initial_loss": float(np.sqrt(np.mean(err ** 2)))})
    write_csv(OUT / "rank_time_collapse.csv", collapse)

    # Endpoint predictions use only the initial saved norms. The sign term
    # uses a stated Gaussian-entry model, not a measured entrywise L1 norm.
    norms = {key: sorted(value)[0][1] for key, value in data["Neon"].items() if key != "losses"}
    f0, op0, nuc0 = (norms[x] for x in
                    ("grad_frobenius_norms", "grad_spectral_norms", "grad_nuclear_norms"))
    initial = []
    for name in ("Neon", "F-Neon", "S-Neon"):
        spec = specs[name]
        a, b = 1 - spec["dense_coefficient"], spec["dense_coefficient"]
        support = (op0 if name == "Neon" else a * op0 + b * f0 if name == "F-Neon"
                   else a * op0 + b * spec["eta"] * np.sqrt(2 / np.pi) * 500 * f0)
        observed = initial_loss - sorted(data[name]["losses"])[1][1]
        predicted = 10 * spec["lr"] * support
        initial.append({"method": name, "observed_first_10_step_loss_drop": observed,
                        "linear_model_prediction": predicted,
                        "relative_error": predicted / observed - 1,
                        "uses_Gaussian_sign_model": name == "S-Neon"})
    write_csv(OUT / "initial_progress_models.csv", initial)

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(RANKS)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for ax, prefix in zip(axes, ("", "F-", "S-")):
        for k, color in zip(RANKS, colors):
            suffix = "Neon" if k == 1 else "Muon" if k == 500 else f"Fanion-{k}"
            name = prefix + suffix
            points = np.array(sorted(data[name]["losses"]))
            scheduled = specs[name]["scheduler"] != "const_lr"
            label = f"k={k}" + (r", $1/\sqrt{t}$ lr" if scheduled else "")
            ax.semilogy(points[:, 0], points[:, 1], label=label, color=color,
                        linestyle="--" if scheduled else "-", linewidth=1.5)
        ax.set(title=(prefix or "Plain ") + "Fanions", xlabel="Iterations", ylim=(0.0003, 400))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Saved quadratic loss")
    fig.savefig(OUT / "saved_fanion_families.png", dpi=180)
    fig.savefig(OUT / "saved_fanion_families.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for k, color in zip((1, 2, 10, 100), colors):
        name = "Neon" if k == 1 else f"Fanion-{k}"
        arr = np.array(sorted(data[name]["losses"]))
        keep = arr[:, 0] * k <= 5000
        ax.plot(k * 0.007 * arr[keep, 0], arr[keep, 1] / initial_loss,
                label=f"k={k}", linewidth=1.6, color=color,
                marker="o" if k == 100 else None, markersize=3)
    ax.set(xlabel=r"Cumulative nuclear budget $k\gamma T$", ylabel="Loss / initial loss",
           title="Saved low-rank Fanions nearly share a time scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "rank_time_collapse.png", dpi=180)
    fig.savefig(OUT / "rank_time_collapse.pdf")
    plt.close(fig)
    return {"sources": [source_record(path), script_source], "summary": rows,
            "collapse": collapse, "initial_progress": initial,
            "initial_norms": {"frobenius": f0, "spectral": op0, "nuclear": nuc0},
            "provenance_limit": "CSV does not serialize optimizer settings; assignments use current source."}


def first_possible_time(bound, target):
    hi = 1
    while bound(hi) > target:
        hi *= 2
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bound(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def fresh_budget_bounds():
    r, target = 500, 0.001
    rng = np.random.default_rng(142)
    left, right = rng.uniform(0, 1, r), rng.uniform(0, 1, r)
    x = rng.normal(0, 0.1, (r, r))
    s = np.linalg.svd(np.sqrt(left)[:, None] * x * np.sqrt(right)[None, :], compute_uv=False)
    sqrt_l = float(np.sqrt(left.max() * right.max()))
    rows = []
    for family, lr in (("plain", 0.007), ("F", 0.015), ("S", 0.011)):
        for k in RANKS:
            capacity = k if family == "plain" else 0.5 * k + 0.5 * np.sqrt(r) * (1 if family == "F" else 0.01 * r)
            bound = lambda t: family_barrier(s, k, t, lr * t, sqrt_l, family)
            earliest = first_possible_time(bound, target)
            old = math.ceil(max(s.sum() - np.sqrt(2 * r * target), 0) / (sqrt_l * lr * capacity))
            rows.append({"family": family, "k": k, "lr": lr, "schedule": "constant",
                         "nuclear_capacity": capacity, "old_necessary_iterations": old,
                         "stronger_necessary_iterations": earliest})
            assert earliest >= old
    # An upper bound on cumulative learning rate gives a safe obstruction for
    # gamma_t = lr/sqrt(t+1), without constructing a many-million-element array.
    k, lr = 10, 0.015
    gamma_upper = lambda t: lr * (2 * np.sqrt(t) - 1) if t else 0
    bound = lambda t: family_barrier(s, k, t, gamma_upper(t), sqrt_l, "F")
    rows.append({"family": "F", "k": k, "lr": lr, "schedule": "inverse_sqrt",
                 "nuclear_capacity": 0.5 * k + 0.5 * np.sqrt(r),
                 "old_necessary_iterations": None,
                 "stronger_necessary_iterations": first_possible_time(bound, target)})
    write_csv(OUT / "stronger_quadratic_barriers.csv", rows)

    fractions = np.linspace(0, 1, 301)
    new = [spectral_barrier(s, r, s[0], f * s.sum()) for f in fractions]
    old = [max(s.sum() * (1 - f), 0) ** 2 / (2 * r) for f in fractions]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.plot(fractions, new, label="Full-spectrum lower bound", linewidth=2)
    ax.plot(fractions, old, label="Previous nuclear-norm lower bound", linestyle="--")
    ax.set(xlabel="Allowed nuclear correction / initial nuclear norm", ylabel="Loss lower bound",
           title="Sharper finite-budget obstruction (fresh quadratic draw)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "stronger_barrier.png", dpi=180)
    fig.savefig(OUT / "stronger_barrier.pdf")
    plt.close(fig)
    return {"seed": 142, "dimension": r, "initial_loss": float(np.dot(s, s) / 2),
            "initial_nuclear_norm": float(s.sum()), "sqrt_L": sqrt_l,
            "target_loss": target, "necessary_times": rows,
            "not_a_replication": True}


def cifar_rank_intervals():
    path = REPO / "code" / "cifar_norms" / "grad_norms_Muon_run0_20251208_162732.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    points, summary = [], []
    for layer, shape in LAYERS.items():
        r = min(shape)
        for k in sorted({1, 2, 5, 10, min(100, r), r}):
            group = []
            for row in raw:
                f, a, s = (float(row[layer + "_" + n]) for n in ("frobenius", "spectral", "nuclear"))
                lower, upper = kyfan_interval(f, a, s, r, k)
                q_lo, q_hi = lower ** 2 / (k * f * f), upper ** 2 / (k * f * f)
                qm, qn = s * s / (r * f * f), a * a / (f * f)
                fm = q_f(s, f, r)
                item = {"layer": layer, "r": r, "k": k, "step": int(row["step"]),
                        "S_k_lower": lower, "S_k_upper": upper,
                        "q_k_lower": q_lo, "q_k_upper": q_hi,
                        "q_Muon": qm, "q_Neon": qn,
                        "Muon_certified_above_Fanion_k": qm > q_hi * (1 + 1e-9),
                        "Fanion_k_certified_above_Muon": q_lo > qm * (1 + 1e-9),
                        "Fanion_k_certified_above_Neon": q_lo > qn * (1 + 1e-9),
                        "F_Muon_certified_above_F_Fanion_k": fm > q_f(upper, f, k) * (1 + 1e-9)}
                points.append(item)
                group.append(item)
            summary.append({"layer": layer, "r": r, "k": k, "snapshots": len(group),
                            "median_interval_width_q": float(np.median([p["q_k_upper"] - p["q_k_lower"] for p in group])),
                            **{key + "_fraction": float(np.mean([p[key] for p in group])) for key in (
                                "Muon_certified_above_Fanion_k", "Fanion_k_certified_above_Muon",
                                "Fanion_k_certified_above_Neon", "F_Muon_certified_above_F_Fanion_k")}})
    write_csv(OUT / "cifar_KyFan_intervals.csv", points)
    write_csv(OUT / "cifar_KyFan_summary.csv", summary)
    return {"source": source_record(path), "summary": summary,
            "architecture_source": source_record(REPO / "code" / "cifar_bench" / "norm_logger_airbench.py"),
            "architecture_assumption": "Layer ranks inferred from current CifarNet: first two 64, last four 256.",
            "finite_loss_rows": int(sum(np.isfinite(float(row["loss"])) for row in raw)),
            "snapshot_rows": len(raw),
            "interpretation": "Intervals for common-gradient Frobenius surrogates only, not measured convergence."}


def same_moments_example():
    r, total, square = 64, 16.0, 6.5
    rows, spectra = [], []
    n, mean = r - 1, (total - 1) / (r - 1)
    variance = (square - 1) / n - mean * mean
    for count in (4, 20):
        high = mean + np.sqrt(variance * (n - count) / count)
        low = mean - np.sqrt(variance * count / (n - count))
        s = np.r_[1.0, np.full(count, high), np.full(n - count, low)]
        assert 1 >= high >= low >= 0
        np.testing.assert_allclose([s[0], s.sum(), np.dot(s, s)], [1, total, square])
        sk = s[:5].sum()
        qm, q5 = total ** 2 / (r * square), sk ** 2 / (5 * square)
        rows.append({"high_count_below_top": count, "op": s[0], "nuclear": s.sum(),
                     "frobenius_squared": np.dot(s, s), "S_5": sk,
                     "q_Muon": qm, "q_Fanion_5": q5,
                     "q_F_Muon": q_f(total, np.sqrt(square), r),
                     "q_F_Fanion_5": q_f(sk, np.sqrt(square), 5)})
        spectra.append(s)
    assert rows[0]["q_Fanion_5"] > rows[0]["q_Muon"]
    assert rows[1]["q_Fanion_5"] < rows[1]["q_Muon"]
    assert rows[0]["q_F_Fanion_5"] > rows[0]["q_F_Muon"]
    assert rows[1]["q_F_Fanion_5"] < rows[1]["q_F_Muon"]
    write_csv(OUT / "same_moments_counterexample.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for s, row in zip(spectra, rows):
        k = np.arange(1, r + 1)
        q = np.cumsum(s) ** 2 / (k * square)
        label = str(row["high_count_below_top"]) + " elevated modes below the top"
        axes[0].step(k, s, where="mid", label=label)
        axes[1].plot(k, q, label=label)
        axes[1].scatter([5], [row["q_Fanion_5"]], s=25)
    axes[0].set(xlabel="Singular-value index", ylabel="Singular value",
                title="Identical three logged norms")
    axes[1].set(xlabel="Fanion rank k", ylabel=r"$q_k=S_k^2/(k\|G\|_F^2)$",
                title="Opposite Fanion-5 / Muon comparisons")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "same_moments.png", dpi=180)
    fig.savefig(OUT / "same_moments.pdf")
    plt.close(fig)
    return rows


def flat_rank_profiles():
    """Exact common-gradient scores; not fixed-learning-rate trajectories."""
    r, eta = 256, 0.01
    ranks = np.arange(1, r + 1)
    x = ranks / np.sqrt(r)

    def score(t):
        # t is the dense normalized-gradient coefficient divided by the
        # partial-polar coefficient. For a=b, t=1 in the F family.
        return (x + t) ** 2 / (ranks + t * t + 2 * t * x)

    profiles = {"Plain": ranks / r, "F": score(1.0),
                "S_dense_Hadamard": score(eta * r),
                "S_diagonal": score(eta * np.sqrt(r))}
    rows = [{"r": r, "eta": eta, "k": int(k),
             **{name + "_q": float(values[k - 1]) for name, values in profiles.items()}}
            for k in ranks]
    write_csv(OUT / "flat_hybrid_rank_profiles.csv", rows)
    fig, ax = plt.subplots(figsize=(6.7, 3.7), constrained_layout=True)
    labels = {"Plain": "Plain Fanion", "F": "F-Fanion",
              "S_dense_Hadamard": "S-Fanion, dense Hadamard gradient",
              "S_diagonal": "S-Fanion, diagonal gradient"}
    for name, values in profiles.items():
        ax.semilogx(ranks, values, label=labels[name], linewidth=1.7)
    ax.set(xlabel="Rank k (log scale)", ylabel="Exact common-gradient cosine squared",
           title=r"Flat singular spectrum, r=256, $a=b=1/2$, $\eta=.01$", ylim=(0, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(OUT / "flat_hybrid_rank_profiles.png", dpi=180)
    fig.savefig(OUT / "flat_hybrid_rank_profiles.pdf")
    plt.close(fig)
    return {"r": r, "eta": eta, "analytic_F_minimizer_continuous_k": r / (np.sqrt(r) + 2),
            "interpretation": "Exact flat-spectrum Frobenius scores at a common gradient; scalar step optimized separately."}


def verify_results():
    rng = np.random.default_rng(9821)
    counts = {"KyFan_moment_intervals": 0, "budget_dominates_previous_bounds": 0,
              "actual_quadratic_trajectory_barriers": 0, "flat_fixed_step_identity": 0,
              "convexified_flow_support_identity": 0, "all_family_oracle_bounds": 0,
              "code_Nesterov_unrolling_and_tracking": 0}
    for trial in range(160):
        r = int(rng.integers(4, 80))
        s = np.sort(rng.uniform(size=r) ** rng.uniform(0.1, 4))[::-1]
        k = int(rng.integers(1, r + 1))
        lo, hi = kyfan_interval(np.linalg.norm(s), s[0], s.sum(), r, k)
        assert lo <= s[:k].sum() + 1e-9 <= hi + 2e-9
        counts["KyFan_moment_intervals"] += 1
        p = int(rng.integers(0, r + 1))
        cap, budget = rng.uniform(0, 2), rng.uniform(0, s.sum())
        lower = spectral_barrier(s, p, cap, budget)
        old = max(s.sum() - budget, 0) ** 2 / (2 * r)
        tail = float(np.dot(s[p:], s[p:]) / 2)
        assert lower + 1e-10 >= max(old, tail)
        counts["budget_dominates_previous_bounds"] += 1

        # Validate the explicit continuous-time formula away from breakpoints.
        t = rng.uniform(0.05, 0.95) * s.sum() / k
        eps = 1e-7
        e = s - capped_allocation(s, r, t, k * t)
        next_e = s - capped_allocation(s, r, t + eps, k * (t + eps))
        d = (e - next_e) / eps
        assert d.min() >= -1e-6 and d.max() <= 1 + 1e-6 and d.sum() <= k + 1e-5
        np.testing.assert_allclose(np.dot(e, d), np.sort(e)[::-1][:k].sum(), atol=5e-6, rtol=5e-6)
        counts["convexified_flow_support_identity"] += 1

    for r in (8, 17, 64):
        for k in (1, 2, min(5, r), r):
            residual = np.ones(r)
            gamma = 1 / 16
            for t in range(math.ceil(r / (k * gamma)) + 1):
                q, p = divmod(min(k * t, r * 16), r)
                formula = ((r - p) * (1 - q * gamma) ** 2
                           + p * (1 - (q + 1) * gamma) ** 2) / 2
                np.testing.assert_allclose(np.dot(residual, residual) / 2, formula, atol=1e-12)
                indices = np.argsort(-residual, kind="stable")[:k]
                residual[indices] = np.maximum(residual[indices] - gamma, 0)
                counts["flat_fixed_step_identity"] += 1

    for trial in range(40):
        m, n, steps = 9, 7, 4
        r, k = min(m, n), int(rng.integers(1, min(m, n) + 1))
        left, right = rng.uniform(0.1, 1, m), rng.uniform(0.1, 1, n)
        initial = rng.normal(size=(m, n))
        sqrt_l = float(np.sqrt(left.max() * right.max()))
        s = np.linalg.svd(np.sqrt(left)[:, None] * initial * np.sqrt(right)[None, :], compute_uv=False)
        a, b, eta = 0.5, 0.5, 0.07
        for family in ("plain", "F", "S"):
            x, gamma_sum = initial.copy(), 0.0
            for t in range(steps):
                g = left[:, None] * x * right[None, :]
                h = g + 0.2 * rng.normal(size=(m, n))
                d = polar_k(h, k)
                support = lambda z: np.linalg.svd(z, compute_uv=False)[:k].sum()
                if family == "F":
                    d = a * d + b * h / np.linalg.norm(h)
                    phi = lambda z: a * support(z) + b * np.linalg.norm(z)
                    radius = a * np.sqrt(k) + b
                elif family == "S":
                    d = a * d + b * eta * np.sign(h)
                    phi = lambda z: a * support(z) + b * eta * np.abs(z).sum()
                    radius = a * np.sqrt(k) + b * eta * np.sqrt(m * n)
                else:
                    phi, radius = support, np.sqrt(k)
                assert np.sum(g * d) + 1e-8 >= phi(g) - 2 * phi(h - g)
                assert np.linalg.norm(d) <= radius + 1e-10
                counts["all_family_oracle_bounds"] += 1
                gamma = rng.uniform(0.05, 0.5)
                x -= gamma * d
                gamma_sum += gamma
                actual_loss = float(np.sum(left[:, None] * x * x * right[None, :]) / 2)
                lower = family_barrier(s, k, t + 1, gamma_sum, sqrt_l, family,
                                       a=a, b=b, eta=eta, entries=m * n)
                assert actual_loss + 1e-8 >= lower
                counts["actual_quadratic_trajectory_barriers"] += 1

    # Check the actual code's Nesterov convention, including its initial mass
    # deficit. This is distinct from replacing it by a normalized EMA.
    for beta in (0.0, 0.5, 0.7, 0.9, 0.99):
        alpha, gamma, k = 1 - beta, 0.03, 3
        x = rng.normal(size=(8, 6))
        g_initial, buf = x.copy(), np.zeros_like(x)
        noisy_gradients, noises = [], []
        v = (1 - beta * beta) ** 2 + alpha * beta ** 4 / (1 + beta)
        for t in range(80):
            true_g = x.copy()  # f(X)=||X||_F^2/2, L=1
            noise = 0.2 * rng.normal(size=x.shape)
            g = true_g + noise
            noisy_gradients.append(g)
            noises.append(noise)
            buf = beta * buf + g
            h_scaled = alpha * (g + beta * buf)
            weights = np.r_[1 - beta * beta, alpha * beta ** (np.arange(1, t + 1) + 1)]
            unrolled = sum(w * past for w, past in zip(weights, reversed(noisy_gradients)))
            weighted_noise = sum(w * past for w, past in zip(weights, reversed(noises)))
            np.testing.assert_allclose(h_scaled, unrolled, atol=1e-10)
            np.testing.assert_allclose(weights.sum(), 1 - beta ** (t + 2), atol=1e-12)
            assert np.dot(weights, weights) <= v + 1e-12
            assert v <= 2 * alpha + 1e-12
            tracking = np.linalg.norm(h_scaled - true_g - weighted_noise)
            bound = beta ** (t + 2) * np.linalg.norm(g_initial) + gamma * np.sqrt(k) * beta ** 2 / alpha
            assert tracking <= bound + 1e-9
            x -= gamma * polar_k(g + beta * buf, k)
            counts["code_Nesterov_unrolling_and_tracking"] += 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = verify_results()
    result = {"checks": checks, "quadratic_audit": audit_quadratic(),
              "fresh_budget_bounds": fresh_budget_bounds(),
              "same_moments_example": same_moments_example(),
              "flat_rank_profiles": flat_rank_profiles(),
              "cifar_intervals": cifar_rank_intervals()}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Passed", sum(checks.values()), "numerical checks:", json.dumps(checks))
    print("Rank-time collapse:", json.dumps(result["quadratic_audit"]["collapse"]))
    print("Wrote", (OUT / "summary.json").relative_to(HERE).as_posix())


if __name__ == "__main__":
    main()
