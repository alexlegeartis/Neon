"""CPU checks and illustrative models for the CIFAR/LLM theoretical description.

These are mathematical examples, not CIFAR or language-model training runs.
All generated files stay under theory_attempts/results/deep_learning.
"""

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
ROOT = HERE.parent
OUT = HERE / "results" / "deep_learning"
COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MEDIUM_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def ns_filter(x, steps, coefficients=None):
    y = np.asarray(x, dtype=float).copy()
    sequence = coefficients if coefficients is not None else [COEFFICIENTS]*steps
    for a, b, c in sequence:
        y = a * y + b * y**3 + c * y**5
    return y


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    # Exact reindexing of normalization-before-update into a tangent retraction.
    retraction_error = 0.0
    for _ in range(150):
        y, direction = rng.normal(size=(2, 37))
        radius = math.sqrt(7)
        y *= radius / np.linalg.norm(y)
        gamma = 0.2 * radius / np.linalg.norm(direction)
        radial = y @ direction / radius**2
        tangent = direction - radial * y
        effective_gamma = gamma / (1 - gamma * radial)
        direct = y - gamma * direction
        direct *= radius / np.linalg.norm(direct)
        retracted = y - effective_gamma * tangent
        retracted *= radius / np.linalg.norm(retracted)
        error = np.linalg.norm(direct - retracted)
        retraction_error = max(retraction_error, float(error))
        check(error < 2e-14, "normalize-before-update identity")
        check(abs(y @ tangent) < 2e-13, "tangent projection")

    # Constant-beta conventions coincide after a positive rescaling. With
    # variable beta the explicit weights below describe the normalized buffer.
    beta = 0.95
    raw, ema = np.zeros(8), np.zeros(8)
    for _ in range(80):
        grad = rng.normal(size=8)
        raw = beta * raw + grad
        ema = beta * ema + (1 - beta) * grad
        check(np.allclose((1-beta) * (grad + beta*raw),
                          (1-beta)*grad + beta*ema, atol=1e-14),
              "constant-beta buffer equivalence")
    betas = np.linspace(0.85, 0.95, 70)
    grads = rng.normal(size=(len(betas), 8))
    ema = np.zeros(8)
    for t, beta in enumerate(betas):
        ema = beta*ema + (1-beta)*grads[t]
        h = (1-beta)*grads[t] + beta*ema
        weights = np.zeros(t+1)
        weights[t] = 1-beta**2
        for s in range(t):
            weights[s] = beta * (1-betas[s]) * np.prod(betas[s+1:t+1])
        check(np.allclose(h, weights @ grads[:t+1], atol=1e-14),
              "variable-beta momentum weights")
        check(abs(weights.sum() - (1-beta*np.prod(betas[:t+1]))) < 2e-15,
              "variable-beta mass")

    # Newton alignment gives the exact optimal direction in a modal model,
    # even when the modal curvature has cross terms.
    for _ in range(40):
        n = 12
        m = rng.uniform(size=(n, n))
        curvature = m.T @ m + np.eye(n)
        ones = np.ones(n)
        gradient = curvature @ ones
        q_muon = gradient.sum()**2 / (ones @ curvature @ ones)
        optimum = gradient @ np.linalg.solve(curvature, gradient)
        check(abs(q_muon-optimum) < 1e-10, "full-modal Newton optimality")
        for k in (1, 2, 5, 12):
            d = np.zeros(n)
            d[np.argsort(gradient)[-k:]] = 1
            for w in (d, 0.5*d + 0.5*gradient/np.linalg.norm(gradient)):
                score = (gradient @ w)**2 / (w @ curvature @ w)
                check(score <= q_muon*(1+1e-12), "modal efficiency at most one")

    # Comparable Newton errors yield a curvature-mass bound for all ranks.
    for _ in range(40):
        lambdas = np.exp(rng.uniform(-4, 1, 24))
        errors = rng.uniform(0.8, 1.2, 24)
        gradients = lambdas * errors
        q_muon = (lambdas @ errors)**2 / lambdas.sum()
        for k in (1, 2, 5, 12, 24):
            indices = np.argsort(gradients)[-k:]
            q_k = gradients[indices].sum()**2 / lambdas[indices].sum()
            bound = (errors.min()/errors.max())**2 * (
                lambdas.sum()/np.sort(lambdas)[-k:].sum())
            check(q_muon/q_k >= bound*(1-1e-12), "curvature-mass comparison")

    # An exactly defined Kronecker quadratic. In the U,V coordinates,
    # H[D]_ij = sqrt(lambda_i lambda_j) D_ij, and X0 = U V^T.
    r = 128
    lambdas = np.arange(1, r+1, dtype=float)**(-0.9)
    curvature_weights = np.sqrt(lambdas[:, None]*lambdas[None, :])
    u, _ = np.linalg.qr(rng.normal(size=(r, r)))
    v, _ = np.linalg.qr(rng.normal(size=(r, r)))
    grad_rotated = (u*lambdas) @ v.T
    grad_norm = np.linalg.norm(lambdas)
    total_curvature = lambdas.sum()
    rows = []
    ranks = (1, 2, 5, 10, 32, 64, 128)
    for basis in ("diagonal", "rotated"):
        sign_modal = (np.eye(r) if basis == "diagonal"
                      else u.T @ np.sign(grad_rotated) @ v)
        for k in ranks:
            plain = np.zeros(r)
            plain[:k] = 1
            directions = {
                "plain": np.diag(plain),
                "F": np.diag(0.5*plain + 0.5*lambdas/grad_norm),
                "S": 0.5*np.diag(plain) + 0.5*0.003*sign_modal,
            }
            for family, direction in directions.items():
                alignment = lambdas @ np.diag(direction)
                curvature = np.sum(curvature_weights * direction**2)
                score = alignment**2 / curvature
                fro_score = alignment**2 / np.sum(direction**2)
                gamma = alignment / curvature
                remaining = np.eye(r) - gamma*direction
                next_loss = 0.5*np.sum(curvature_weights * remaining**2)
                check(abs(next_loss - 0.5*(total_curvature-score)) < 2e-13,
                      "exact quadratic one-step decrease")
                check(score <= total_curvature*(1+1e-12), "quadratic optimality")
                off = direction.copy()
                np.fill_diagonal(off, 0)
                rows.append(dict(basis=basis, family=family, rank=k,
                                 alignment=float(alignment), curvature=float(curvature),
                                 score=float(score), fraction_loss_removed=float(score/total_curvature),
                                 frobenius_score=float(fro_score), optimal_gamma=float(gamma),
                                 off_modal_curvature_fraction=float(np.sum(curvature_weights*off**2)/curvature)))
        for name, steps, coefficients in (("NS3",3,None), ("NS5",5,None),
                                          ("NSM",5,MEDIUM_COEFFICIENTS)):
            weights = ns_filter(lambdas/grad_norm, steps, coefficients)
            alignment = lambdas @ weights
            curvature = lambdas @ (weights**2)
            score = alignment**2 / curvature
            rows.append(dict(basis=basis, family=name, rank=r,
                             alignment=float(alignment), curvature=float(curvature),
                             score=float(score), fraction_loss_removed=float(score/total_curvature),
                             frobenius_score=float(alignment**2/(weights@weights)),
                             optimal_gamma=float(alignment/curvature),
                             off_modal_curvature_fraction=0.0))
            check(score <= total_curvature*(1+1e-12), "finite NS efficiency")
    write_csv("curvature_modes.csv", rows)
    muon = next(x for x in rows if x["basis"] == "rotated" and x["family"] == "plain" and x["rank"] == r)
    neon = next(x for x in rows if x["basis"] == "rotated" and x["family"] == "plain" and x["rank"] == 1)
    check(muon["score"] > neon["score"], "true curvature favors Muon")
    check(muon["frobenius_score"] < neon["frobenius_score"], "Frobenius surrogate reverses ranking")

    # Exact finite-filter representation: matrix iteration and scalar iteration.
    matrix = rng.normal(size=(8, 13))
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    for steps, coefficients in ((3,None), (5,None), (5,MEDIUM_COEFFICIENTS)):
        iterated = matrix/np.linalg.norm(matrix)
        sequence = coefficients if coefficients is not None else [COEFFICIENTS]*steps
        for a, b, c in sequence:
            gram = iterated @ iterated.T
            iterated = a*iterated + (b*gram + c*(gram@gram))@iterated
        filtered = (left*ns_filter(singular/np.linalg.norm(singular), steps, coefficients)) @ right
        check(np.linalg.norm(iterated-filtered) < 1e-11, "matrix/scalar NS equivalence")
    grid = np.geomspace(1e-5, 1, 450)
    write_csv("ns_filter.csv", [dict(normalized_singular_value=float(x),
              ns3=float(ns_filter(x, 3)), ns5=float(ns_filter(x, 5)),
              medium_ns5=float(ns_filter(x, 5, MEDIUM_COEFFICIENTS))) for x in grid])

    # Potential Frobenius step sizes at the inherited CIFAR plain-family rate.
    cifar_scales = []
    for m, n in ((64,216), (64,576), (256,576), (256,2304)):
        for k in (1, 5, min(m,n)):
            cifar_scales.append(dict(out_channels=m, flattened_input=n, rank=k,
                                     weight_radius=math.sqrt(m), direction_frobenius=math.sqrt(k),
                                     nominal_lr=0.24,
                                     relative_step_scale=0.24*math.sqrt(k/m),
                                     frobenius_matched_lr=0.24*math.sqrt(min(m,n)/k)))
    write_csv("cifar_step_scales.csv", cifar_scales)

    # Classification error is controlled by the smoothed cross-entropy gap,
    # with an exact constant obtained by pooling the correct and wrong class.
    smoothing, classes = 0.2, 10
    q = np.full(classes, smoothing/classes)
    q[0] += 1-smoothing
    entropy = -q @ np.log(q)
    boundary = q.copy()
    boundary[:2] = (q[0]+q[1])/2
    mistake_gap = q @ np.log(q/boundary)
    check(abs(mistake_gap - (q[0]*math.log(2*q[0]/(q[0]+q[1]))
                            + q[1]*math.log(2*q[1]/(q[0]+q[1])))) < 1e-14,
          "sharp classification-gap constant")
    for _ in range(2200):
        p = rng.dirichlet(np.full(classes, 0.5))
        kl = q @ np.log(q/p)
        check(int(np.argmax(p) != 0) <= kl/mistake_gap + 1e-12,
              "classification-gap bound")
    check(np.argmax(boundary) in (0,1), "misclassification boundary")

    # Variable-objective telescoping retains context-schedule changes.
    x = rng.normal(size=5)
    curvatures, drift, alignments = [], [], []
    lambdas_time = 1 + np.arange(22)/50
    gamma = 0.04
    initial_loss = 0.5*lambdas_time[0]*(x@x)
    for t in range(21):
        lam = lambdas_time[t]
        d = x + rng.normal(scale=0.1, size=5)
        alignments.append(lam * (x@d))
        curvatures.append(lam * (d@d))
        x = x-gamma*d
        drift.append(0.5*(lambdas_time[t+1]-lam)*(x@x))
    final_loss = 0.5*lambdas_time[-1]*(x@x)
    rhs = initial_loss-final_loss + 0.5*gamma**2*sum(curvatures) + sum(drift)
    check(abs(gamma*sum(alignments)-rhs) < 1e-13, "context-drift telescoping")

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "savefig.bbox":"tight", "pdf.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for family, color in (("plain", "#2867a0"), ("F", "#d16b28"), ("S", "#33865b")):
        selected = [x for x in rows if x["basis"] == "rotated" and x["family"] == family]
        axes[0].plot([x["rank"] for x in selected],
                     [x["fraction_loss_removed"] for x in selected],
                     "o-", label=family+"-Fanion" if family != "plain" else "Fanion", color=color)
    axes[0].set(xscale="log", xlabel="Rank k", ylabel="Fraction of initial loss removed",
                title="Curvature-aware, best scalar step", ylim=(0,1.04))
    axes[0].legend(frameon=False)
    fro_plain = [np.sum(lambdas[:k])**2/k for k in ranks]
    true_plain = [np.sum(lambdas[:k]) for k in ranks]
    axes[1].plot(ranks, np.array(true_plain)/true_plain[0], "o-", label="True curvature", color="#2867a0")
    axes[1].plot(ranks, np.array(fro_plain)/fro_plain[0], "s--", label="Frobenius surrogate", color="#aa4568")
    axes[1].set(xscale="log", xlabel="Rank k", ylabel="Score / Neon score",
                title="Same gradient, different rank predictions")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(r"Illustrative model: $\lambda_i=i^{-0.9}$, equal modal errors, $r=128$", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("curvature_rank."+suffix), dpi=190)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4,3.4))
    ax.semilogx(grid, ns_filter(grid,3), label="CIFAR: 3 steps", color="#2867a0")
    ax.semilogx(grid, ns_filter(grid,5), label="NanoGPT: 5 steps", color="#d16b28")
    ax.semilogx(grid, ns_filter(grid,5,MEDIUM_COEFFICIENTS),
                label="Medium: 5 varying polynomials", color="#33865b", lw=1.3)
    ax.axhline(1, color="black", lw=0.9, ls="--", label="Exact polar (positive modes)")
    ax.set(xlabel=r"Input singular value / Frobenius norm", ylabel="Output signed gain",
           title="The implemented Muon retains a finite spectral filter")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("ns_filter."+suffix), dpi=190)
    plt.close(fig)

    inputs = ["jota_submission/main.tex", "code/optimizers.py",
              "code/cifar_bench/airbench_muon.py", "code/cifar_bench/airbench_my_muon.py",
              "code/cifar_bench/norm_logger_airbench.py", "code/gpt_pretrain/fmuon_pretrain.py",
              "code/gpt_pretrain/smuon_pretrain.py", "code/gpt_pretrain/medium_f_s_muon.py"]
    summary = dict(checks=checks, seed=20261021, max_retraction_error=retraction_error,
                   classification=dict(classes=classes, smoothing=smoothing,
                                       entropy=entropy, mistake_gap=float(mistake_gap),
                                       optimal_logit_margin=math.log(q[0]/q[1])),
                   curvature_example=dict(rank=r, exponent=0.9,
                                          true_muon_neon_ratio=muon["score"]/neon["score"],
                                          fro_muon_neon_ratio=muon["frobenius_score"]/neon["frobenius_score"],
                                          family_rows=[x for x in rows if x["basis"] == "rotated" and x["rank"] in (1,5,128)]),
                   input_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
                   limitations=["No neural-network training, checkpoint extraction, or fitting to benchmark losses.",
                                "The curvature model is an illustrative exact quadratic, not a measured neural Hessian.",
                                "NS curves use exact arithmetic; BF16/FP8 rounding and the normalization epsilon are omitted.",
                                "Current scripts contain edited alternatives, not immutable manifests for published runs.",
                                "CIFAR relative step scales precede tangent projection; they are not measured angular steps."])
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"checks":checks, "max_retraction_error":retraction_error,
                      "true_muon_neon_ratio":summary["curvature_example"]["true_muon_neon_ratio"],
                      "fro_muon_neon_ratio":summary["curvature_example"]["fro_muon_neon_ratio"],
                      "classification":summary["classification"]}, indent=2))
    for x in summary["curvature_example"]["family_rows"]:
        print(x["family"], x["rank"], "efficiency", round(x["fraction_loss_removed"], 8))


if __name__ == "__main__":
    main()
