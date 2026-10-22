"""Reproduce the calculations in bounds.tex; write only inside theory_attempts.

Run from the repository root:
    python theory_attempts/analyze.py

Optional saved matrix diagnostic:
    python theory_attempts/analyze.py --snapshot theory_attempts/example.npz
The NPZ must contain `gradient` and may contain `oracle_input`, `left`, `right`.
Here `gradient` is the true/evaluation gradient; `left`, `right` specify the
quadratic Hessian D -> left @ D @ right. See README.txt for interpretation.
"""

import os

# Small dense problems are faster and more reproducible without BLAS oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "results"
LAYERS = {
    "layers.1.conv1.weight": (64, 24 * 9),
    "layers.1.conv2.weight": (64, 64 * 9),
    "layers.2.conv1.weight": (256, 64 * 9),
    "layers.2.conv2.weight": (256, 256 * 9),
    "layers.3.conv1.weight": (256, 256 * 9),
    "layers.3.conv2.weight": (256, 256 * 9),
}


def fro2(a):
    return float(np.sum(a * a))


def kyfan(a, k):
    return float(np.linalg.svd(a, compute_uv=False)[:k].sum())


def polar_k(a, k):
    """Minimal rank maximizer: do not fill zero singular spaces arbitrarily."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if not s.size or s[0] == 0:
        return np.zeros_like(a)
    active = int(np.count_nonzero(s > 1e-12 * s[0]))
    take = min(k, active)
    return u[:, :take] @ vh[:take]


def make_directions(h, k, alpha=0.5, sign_scale=0.003):
    d = polar_k(h, k)
    hnorm = np.linalg.norm(h)
    normalized = h / hnorm if hnorm else np.zeros_like(h)
    return {
        "Fanion": d,
        "F-Fanion": alpha * d + (1 - alpha) * normalized,
        "S-Fanion": alpha * d + (1 - alpha) * sign_scale * np.sign(h),
        "NSGD": normalized,
        "SignSGD": np.sign(h),
    }


def metrics(g, d, left=None, right=None):
    """A^2/B is a smoothness surrogate, not an observed loss reduction."""
    a = float(np.sum(g * d))
    b = fro2(d)
    g2 = fro2(g)
    positive = max(a, 0.0)
    c = None if left is None else float(np.sum(d * (left @ d @ right)))
    return {
        "inner_product": a,
        "frobenius_squared": b,
        "cosine_squared_if_descent": positive**2 / (g2 * b) if g2 * b else 0.0,
        "frobenius_score_L1": positive**2 / b if b else 0.0,
        "quadratic_curvature": c,
        "quadratic_score": positive**2 / c if c is not None and c > 0 else None,
        "quadratic_optimal_step": positive / c if c is not None and c > 0 else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_formulas(seed=142, trials=160):
    """Independent numerical checks of the identities and inequalities in the note."""
    rng = np.random.default_rng(seed)
    counts = {
        "Fanion_angle_identity": 0,
        "F_mixture_identity_and_improvement": 0,
        "S_mixture_retention": 0,
        "inexact_oracle_inequality": 0,
        "quadratic_exact_decrease": 0,
        "quadratic_rank_lower_bound": 0,
        "quadratic_nuclear_lower_bound": 0,
    }
    max_relative_identity_error = 0.0
    for trial in range(trials):
        m, n = (11, 8) if trial % 2 else (7, 10)
        r = min(m, n)
        g = rng.normal(size=(m, n))
        u, s, vh = np.linalg.svd(g, full_matrices=False)
        k = int(rng.integers(1, r + 1))
        d = u[:, :k] @ vh[:k]
        a, b = rng.uniform(0.05, 0.95, size=2)
        gf = np.linalg.norm(g)
        sk = s[:k].sum()
        qk = sk**2 / (k * gf**2)
        actual = metrics(g, d)["cosine_squared_if_descent"]
        np.testing.assert_allclose(actual, qk, rtol=1e-11, atol=1e-12)
        counts["Fanion_angle_identity"] += 1

        x = sk / gf
        df = a * d + b * g / gf
        qf = (a * x + b) ** 2 / (a * a * k + b * b + 2 * a * b * x)
        np.testing.assert_allclose(metrics(g, df)["cosine_squared_if_descent"], qf,
                                   rtol=1e-11, atol=1e-12)
        assert qf + 1e-12 >= qk
        counts["F_mixture_identity_and_improvement"] += 1

        eta = float(rng.uniform(0.001, 0.2))
        ds = a * d + b * eta * np.sign(g)
        rho = b * eta * np.sqrt(np.count_nonzero(g) / k) / a
        assert metrics(g, ds)["cosine_squared_if_descent"] + 1e-12 >= qk / (1 + rho)**2
        counts["S_mixture_retention"] += 1

        h = g + rng.normal(size=(m, n)) * 0.4
        # A feasible, deliberately imperfect LMO, with its exact support deficit.
        dh = polar_k(h, k) * 0.83
        deficit = kyfan(h, k) - float(np.sum(h * dh))
        rhs = kyfan(g, k) - 2 * kyfan(h - g, k) - deficit
        assert float(np.sum(g * dh)) + 1e-10 >= rhs
        counts["inexact_oracle_inequality"] += 1

        z = rng.normal(size=(m, n))
        left0 = rng.normal(size=(m, m))
        right0 = rng.normal(size=(n, n))
        left = left0.T @ left0 + 0.1 * np.eye(m)
        right = right0.T @ right0 + 0.1 * np.eye(n)
        gradient = left @ z @ right
        loss0 = 0.5 * float(np.sum(z * gradient))
        for direction in make_directions(gradient, k, a, eta).values():
            mm = metrics(gradient, direction, left, right)
            step = mm["quadratic_optimal_step"]
            new = z - step * direction
            loss1 = 0.5 * float(np.sum(new * (left @ new @ right)))
            expected = 0.5 * mm["quadratic_score"]
            err = abs((loss0 - loss1) - expected) / max(1.0, expected)
            max_relative_identity_error = max(max_relative_identity_error, err)
            np.testing.assert_allclose(loss0 - loss1, expected, rtol=1e-9, atol=1e-8)
            counts["quadratic_exact_decrease"] += 1

        ev_m, q_m = np.linalg.eigh(left)
        ev_n, q_n = np.linalg.eigh(right)
        sqrt_m = (q_m * np.sqrt(ev_m)) @ q_m.T
        sqrt_n = (q_n * np.sqrt(ev_n)) @ q_n.T
        initial_s = np.linalg.svd(sqrt_m @ z @ sqrt_n, compute_uv=False)
        cap = float(np.sqrt(ev_m[-1] * ev_n[-1]))
        total_step = 0.0
        current = z.copy()
        for t in range(1, 5):
            random_direction = polar_k(rng.normal(size=(m, n)), k)
            step = float(rng.uniform(0.001, 0.05))
            current -= step * random_direction
            total_step += step
            loss = 0.5 * float(np.sum(current * (left @ current @ right)))
            tail_bound = 0.5 * float(np.sum(initial_s[min(k * t, r):]**2))
            budget_bound = max(initial_s.sum() - cap * k * total_step, 0.0)**2 / (2 * r)
            assert loss + 1e-8 >= tail_bound
            assert loss + 1e-8 >= budget_bound
            counts["quadratic_rank_lower_bound"] += 1
            counts["quadratic_nuclear_lower_bound"] += 1

    return {"seed": seed, "trials": trials, "checks": counts,
            "max_relative_quadratic_identity_error": max_relative_identity_error}


def flat_quadratic():
    """Exact line search on 0.5 ||X-I||_F^2, using diagonal iterates."""
    rows = []
    trajectories = {}
    for r in (32, 128, 500):
        for k in sorted({1, 2, 5, 10, min(100, r), r}):
            residual = np.ones(r)
            losses = [float(np.sum(residual**2) / r)]
            for t in range(math.ceil(r / k)):
                active = np.flatnonzero(np.abs(residual) > 1e-12)
                selected = active[np.argsort(-np.abs(residual[active]), kind="stable")[:k]]
                direction = np.zeros(r)
                direction[selected] = np.sign(residual[selected])
                step = float(np.dot(residual, direction) / np.dot(direction, direction))
                residual -= step * direction
                losses.append(float(np.sum(residual**2) / r))
            np.testing.assert_allclose(losses[-1], 0.0, atol=1e-20)
            for t, loss in enumerate(losses):
                np.testing.assert_allclose(loss, max(1 - k * t / r, 0), atol=1e-12)
            rows.append({"r": r, "k": k, "exact_iterations": len(losses) - 1,
                         "rank_lower_bound_iterations": math.ceil(r / k)})
            if r == 128 and k in (1, 2, 10, 100, 128):
                trajectories[k] = losses
    write_csv(OUT / "flat_quadratic.csv", rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for k, losses in trajectories.items():
        label = "Muon (k=128)" if k == 128 else f"Fanion-{k}"
        ax.plot(range(len(losses)), losses, label=label, linewidth=1.7)
    ax.set(xlabel="Iterations (exact line search)", ylabel="Loss / initial loss",
           title="A tight rank separation: r=128")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "rank_barrier.pdf")
    fig.savefig(OUT / "rank_barrier.png", dpi=180)
    plt.close(fig)
    return rows


def paper_distribution_quadratic(size=500, seed=142):
    """A new draw from the paper's distribution, not its original experiment seed.

    Work in the eigenbases of M and N. This preserves the joint distribution
    of the Gaussian initialization and all quantities used in these bounds.
    No claim of rotation invariance is made for the entrywise sign algorithm.
    N(0, 0.01) is interpreted as variance 0.01 (standard deviation 0.1).
    """
    rng = np.random.default_rng(seed)
    m_eigs = rng.uniform(0, 1, size=size)
    n_eigs = rng.uniform(0, 1, size=size)
    x = rng.normal(scale=0.1, size=(size, size))
    e = np.sqrt(m_eigs)[:, None] * x * np.sqrt(n_eigs)[None, :]
    s = np.linalg.svd(e, compute_uv=False)
    threshold = 0.001
    tail = np.r_[np.cumsum((s**2)[::-1])[::-1], 0.0] / 2
    needed_rank = int(np.flatnonzero(tail <= threshold)[0])
    required_nuclear = max(float(s.sum()) - np.sqrt(2 * size * threshold), 0.0)
    sqrt_l = float(np.sqrt(m_eigs.max() * n_eigs.max()))
    rows = []
    for k in sorted({1, 2, 10, min(100, size), size}):
        for name, lr, capacity in [
            ("Fanion", 0.007, float(k)),
            ("F-Fanion", 0.015, 0.5 * k + 0.5 * np.sqrt(size)),
            ("S-Fanion", 0.011, 0.5 * k + 0.5 * 0.01 * size * np.sqrt(size)),
        ]:
            rank_floor = math.ceil(needed_rank / k) if name == "Fanion" else None
            budget_floor = math.ceil(required_nuclear / (sqrt_l * lr * capacity))
            rows.append({"family": name, "k": k, "constant_step": lr,
                         "nuclear_capacity_upper_bound": capacity,
                         "necessary_iterations_rank": rank_floor,
                         "necessary_iterations_nuclear_budget": budget_floor})
    write_csv(OUT / "quadratic_lower_bounds.csv", rows)
    return {"size": size, "seed": seed, "initial_loss": float(fro2(e) / 2),
            "initial_nuclear_norm": float(s.sum()), "sqrt_L": sqrt_l,
            "threshold": threshold, "necessary_approximation_rank": needed_rank,
            "bounds": rows}


def mixture_q(x, k, alpha=0.5):
    b = 1 - alpha
    return (alpha * x + b)**2 / (alpha**2 * k + b**2 + 2 * alpha * b * x)


def audit_logs():
    """Use a predeclared cohort (all nine Dec 8 unprefixed runs), all six convs.

    These files record raw minibatch gradients, not momentum matrices or full
    gradients. All outputs are explicitly surrogates, not measured speedups.
    """
    source_dir = REPO / "code" / "cifar_norms"
    paths = sorted(source_dir.glob("grad_norms_*_run0_20251208_*.csv"))
    point_rows, summary_rows, sources = [], [], []
    ref_ratios, ref_rows = {}, []
    reference_name = "grad_norms_Muon_run0_20251208_162732.csv"
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        losses = np.array([float(row["loss"]) for row in rows])
        source = {
            "path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "last_step": int(rows[-1]["step"]),
            "last_epoch": int(rows[-1]["epoch"]),
            "finite_loss_rows": int(np.isfinite(losses).sum()),
        }
        sources.append(source)
        for layer, (m, n) in LAYERS.items():
            r = min(m, n)
            arr = np.array([[float(row[layer + "_" + norm])
                             for norm in ("frobenius", "spectral", "nuclear")]
                            for row in rows])
            assert np.all(np.isfinite(arr)) and np.all(arr > 0), (path, layer)
            f, spec, nuc = arr.T
            assert np.all(spec <= f * 1.005), (path, layer)
            assert np.all(f <= nuc * 1.005), (path, layer)
            assert np.all(nuc <= np.sqrt(r) * f * 1.005), (path, layer)
            q1 = spec**2 / f**2
            qr = nuc**2 / (r * f**2)
            ratio = nuc**2 / (r * spec**2)
            qfm = mixture_q(nuc / f, r)
            qfn = mixture_q(spec / f, 1)
            for i, row in enumerate(rows):
                point_rows.append({
                    "file": path.name, "step": int(row["step"]), "layer": layer,
                    "r": r, "neon_q_raw_gradient": float(q1[i]),
                    "muon_q_raw_gradient": float(qr[i]),
                    "muon_neon_surrogate_ratio": float(ratio[i]),
                    "fmuon_muon_surrogate_ratio": float(qfm[i] / qr[i]),
                    "fneon_muon_surrogate_ratio": float(qfn[i] / qr[i]),
                })
            summary = {
                "file": path.name, "layer": layer, "m": m, "n": n,
                "count": len(rows), "median_neon_q": float(np.median(q1)),
                "median_muon_q": float(np.median(qr)),
                "median_muon_neon_surrogate_ratio": float(np.median(ratio)),
                "fraction_muon_neon_ratio_above_one": float(np.mean(ratio > 1)),
                "median_fmuon_muon_surrogate_ratio": float(np.median(qfm / qr)),
                "median_fneon_muon_surrogate_ratio": float(np.median(qfn / qr)),
                "smuon_guaranteed_q_retention_at_eta_0003":
                    float((1 + 0.003 * np.sqrt(m * n / r))**-2),
            }
            summary_rows.append(summary)
            if path.name == reference_name:
                ref_ratios[layer] = ratio
                ref_rows.append(summary)
    write_csv(OUT / "cifar_gradient_surrogates.csv", point_rows)
    write_csv(OUT / "cifar_summary.csv", summary_rows)
    if ref_rows:
        tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
               r"Layer & $r$ & median $q_r/q_1$ & $\Pr(q_r>q_1)$ & median $q_{F,r}/q_r$ & median $q_{F,1}/q_r$ \\",
               r"\midrule"]
        for row in ref_rows:
            label = row["layer"].replace("layers.", "").replace(".weight", "")
            tex.append("{} & {} & {:.2f} & {:.1f}\\% & {:.2f} & {:.2f} \\\\".format(
                label, row["m"], row["median_muon_neon_surrogate_ratio"],
                100 * row["fraction_muon_neon_ratio_above_one"],
                row["median_fmuon_muon_surrogate_ratio"],
                row["median_fneon_muon_surrogate_ratio"]))
        tex.extend([r"\bottomrule", r"\end{tabular}"])
        (OUT / "reference_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
        ax.boxplot(list(ref_ratios.values()), showfliers=False,
                   tick_labels=[x.replace("layers.", "").replace(".weight", "")
                                for x in ref_ratios])
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(ylabel="q(Muon) / q(Neon), raw-gradient surrogate",
               title="Saved CIFAR Muon trajectory: all logged steps, six conv layers")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(OUT / "cifar_diagnostics.pdf")
        fig.savefig(OUT / "cifar_diagnostics.png", dpi=180)
        plt.close(fig)
    return {"cohort": "grad_norms_*_run0_20251208_*.csv", "sources": sources,
            "reference": reference_name, "reference_summary": ref_rows,
            "total_layer_snapshots": len(point_rows)}


def reverse_example():
    r, delta = 64, 0.001
    g = np.diag(np.r_[1.0, np.full(r - 1, delta)])
    result = {}
    for k in (1, r):
        d = polar_k(g, k)
        mm = metrics(g, d, np.eye(r), np.eye(r))
        result[str(k)] = mm
    ratio = result[str(r)]["quadratic_score"] / result["1"]["quadratic_score"]
    np.testing.assert_allclose(ratio, (1 + (r - 1) * delta)**2 / r)
    return {"r": r, "small_singular_value": delta,
            "Muon_to_Neon_exact_one_step_decrease_ratio": ratio,
            "Neon_to_Muon_exact_one_step_decrease_ratio": 1 / ratio}


def curvature_example():
    """A member of the paper's quadratic class where curvature changes ranking.

    f(X)=0.5 <X, Lambda X>, X0=I. Muon is the Newton direction here.
    This is an explanatory example, not the paper's random initialization.
    """
    r = 128
    lam = np.geomspace(1.0, 0.001, r)
    g = np.diag(lam)
    left, right = np.diag(lam), np.eye(r)
    loss0 = float(lam.sum() / 2)
    rows = []
    for k in (1, 5, 32, r):
        for name, d in make_directions(g, k, 0.5, 0.003).items():
            mm = metrics(g, d, left, right)
            rows.append({"family": name, "k": k, **mm,
                         "loss_fraction_after_exact_line_search":
                             max(0.0, 1 - mm["quadratic_score"] / (2 * loss0))})
    write_csv(OUT / "curvature_example.csv", rows)
    return {"r": r, "smallest_eigenvalue": float(lam[-1]),
            "initial_loss": loss0, "directions": rows}


def small_step_counterexample():
    """Noiseless scalar Muon cannot travel distance two with total step one."""
    rows = []
    for horizon in (10, 100, 1000, 10000):
        step = 1.0 / horizon
        # Before update t, x_t = 2 - t / T and all signs remain positive.
        smallest_recorded_gradient = 2 - (horizon - 1) * step
        assert smallest_recorded_gradient >= 1.0
        rows.append({"T": horizon, "learning_rate": step,
                     "minimum_gradient_over_t_less_than_T": smallest_recorded_gradient,
                     "last_gradient_after_T_updates": 1.0})
    return rows


def snapshot_diagnostics(path):
    with np.load(path, allow_pickle=False) as data:
        g = np.asarray(data["gradient"], dtype=float)
        h = np.asarray(data["oracle_input"], dtype=float) if "oracle_input" in data else g
        left = np.asarray(data["left"], dtype=float) if "left" in data else None
        right = np.asarray(data["right"], dtype=float) if "right" in data else None
    if g.ndim != 2 or h.shape != g.shape or not np.all(np.isfinite(g + h)):
        raise ValueError("gradient and oracle_input must be finite matrices of the same shape")
    if (left is None) != (right is None):
        raise ValueError("Provide both left and right or neither")
    if left is not None:
        for a, size in ((left, g.shape[0]), (right, g.shape[1])):
            if a.shape != (size, size) or not np.all(np.isfinite(a)):
                raise ValueError("Incorrect quadratic factor shape or nonfinite factor")
            if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() < -1e-10:
                raise ValueError("Quadratic factors must be symmetric positive semidefinite")
    rows = []
    r = min(g.shape)
    for k in sorted({1, min(5, r), r}):
        err = h - g
        err_k = kyfan(err, k)
        err_f = float(np.linalg.norm(err))
        err_1 = float(np.sum(np.abs(err)))
        error_bounds = {
            "Fanion": 2 * err_k,
            "F-Fanion": err_k + err_f,
            "S-Fanion": err_k + 0.003 * err_1,
            "NSGD": 2 * err_f,
            "SignSGD": 2 * err_1,
        }
        for name, d in make_directions(h, k).items():
            rows.append({"algorithm": name, "k": k, **metrics(g, d, left, right),
                         "support_error_bound": error_bounds[name]})
    write_csv(OUT / "snapshot_diagnostics.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--quadratic-size", type=int, default=500)
    args = parser.parse_args()
    if args.quadratic_size < 1:
        parser.error("--quadratic-size must be positive")
    OUT.mkdir(exist_ok=True)
    summary = {
        "verification": verify_formulas(),
        "flat_quadratic": flat_quadratic(),
        "paper_distribution_quadratic": paper_distribution_quadratic(args.quadratic_size),
        "reverse_example": reverse_example(),
        "curvature_example": curvature_example(),
        "small_step_schedule_counterexample": small_step_counterexample(),
        "cifar_audit": audit_logs(),
    }
    if args.snapshot:
        snapshot_diagnostics(args.snapshot)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Algebra checks:", summary["verification"])
    quad = summary["paper_distribution_quadratic"]
    print("New quadratic draw: initial loss={:.6f}; nuclear norm={:.6f}; necessary rank={}"
          .format(quad["initial_loss"], quad["initial_nuclear_norm"],
                  quad["necessary_approximation_rank"]))
    for row in quad["bounds"]:
        if row["family"] == "Fanion":
            print("k={k}: constant-step iteration lower bound {necessary_iterations_nuclear_budget}"
                  .format(**row))
    print("Reverse example: Neon one-step improvement / Muon = {:.3f}"
          .format(summary["reverse_example"]["Neon_to_Muon_exact_one_step_decrease_ratio"]))
    print("CIFAR layer snapshots:", summary["cifar_audit"]["total_layer_snapshots"])
    for row in summary["cifar_audit"]["reference_summary"]:
        print("{layer}: median qMu/qNe={median_muon_neon_surrogate_ratio:.3f}; "
              "median qFNe/qMu={median_fneon_muon_surrogate_ratio:.3f}".format(**row))
    print("Results written to", OUT)


if __name__ == "__main__":
    main()
