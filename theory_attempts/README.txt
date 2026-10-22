Theory attempts for the JOTA submission -- 22 October 2026
=======================================================

Start with SUMMARY.txt for the consolidated findings, evidence, limitations,
and the distinction between norm-agnostic rates and an optimizer ranking.
The quadratic mechanisms are proved; the neural explanation remains conditional.

Use deep_learning_model.pdf for the CIFAR/LLM theoretical description,
or deep_learning_model.html / deep_learning_model.tex. The all-rank attempt
is fanion_followup.pdf; the original attempt is bounds.pdf. All have
standalone offline HTML and LaTeX sources.

CIFAR / LLM description
-----------------------
The third note proposes a curvature-adjusted explanation: many modes may
need comparable Newton corrections even when gradient singular values are
very unequal. In a precisely defined local model this makes full Muon
optimal; every Fanion rank loses a quantifiable fraction of useful curvature
mass, and hybrid directions retain a fraction given by their angle in the
curvature metric. A matrix quadratic example favors Muon over Neon by
6.82127 using true curvature, while the Frobenius surrogate gives 0.363514
and favors Neon. This is an exact illustrative example, not a neural fit.

The actual training settings require separate models:

  * CIFAR: a batch-dependent smoothed cross-entropy objective, normalized
    convolution weights, tangent updates, and auxiliary SGD. The source
    normalizes BEFORE subtracting the update; an exact reindexing gives a
    sphere retraction with a radial-dependent step. At the inherited rate,
    ideal Neon has an 8-16 times smaller nominal relative Frobenius step
    than ideal Muon, before tangent projection. The current script also
    contains a commented Neon normalization flag inconsistent with the
    manuscript's table description; original run settings are needed.
  * LLM: additive shape-scaled hidden-weight updates, auxiliary Adam, actual
    hybrid component scaling, tensor normalization groups, and a growing
    attention context. The finite-horizon bound retains objective drift.
    Momentum, dependent token noise, and nonlinear oracle bias are explicit.
  * Implemented Muon is a finite spectral filter: three repeated polynomials
    for CIFAR, five for NanoGPT, and five different polynomials for Medium.
    Hard-rank Fanions, these filters, and F/S mixtures fit the same modal
    direction/curvature framework.

For the actual CIFAR smoothing, target entropy is 0.86689391 and any wrong
prediction costs at least 0.48773025 above it. Hence classification error
is bounded by (mean smoothed loss - entropy)/0.48773025 for the same
prediction mechanism and distribution. Validation accuracy additionally
requires generalization and evaluation-state control. The saved gradient
logs without weight normalization do not verify the normalized model.

The note proves these conditional results and supplies 3,378 CPU numerical
checks. Whether the broad Newton-correction hypothesis holds for the actual
networks remains unresolved: independent probe alignment, directional
curvature, and per-rank tuning are the discriminating measurements. The
current LLM results do not include an all-rank Fanion/Neon comparison.

New files:
  deep_learning_model.tex/pdf/html    Full theoretical description and proofs.
  paper_insert_deep_learning.tex     Candidate manuscript discussion/proposition.
  deep_learning_analysis.py          Exact examples and numerical checks.
  results/deep_learning/summary.json Results, input hashes, limitations.
  results/deep_learning/*.csv        Rank curves, filters, nominal step scales.
  results/deep_learning/*.pdf/*.png  Standalone research figures.

Reproduce from the repository root:

  python theory_attempts/deep_learning_analysis.py
  python theory_attempts/build_report.py --stem deep_learning_model

All-rank continuation
---------------------
The second note covers every k in plain, F-, and S-Fanions, with proofs for:

  * A common-gradient-norm convergence bound retaining each family's distinct
    support function, curvature radius, cumulative step sizes, and input error.
  * Tracking for the actual code's Nesterov buffer convention. Under a broad
    spectrum condition, the plain-family deterministic bound has an r/k
    factor; its stochastic leading iteration term has an (r/k)^2 factor.
  * A whole-spectrum quadratic loss obstruction with rank, operator, and
    nuclear budgets, plus a separate bound on hybrid dense corrections.
  * Exact flat finite-step Fanion curves, and a convexified continuous model
    with extinction time max(initial operator norm, initial nuclear norm/k).
    Hybrid models have different rank crossover scales; sign updates also
    depend on entry density, which singular values alone do not determine.
  * Nonmonotone hybrid rank curves: even at a flat spectrum, F-Neon's
    optimized common-point decrease can exceed F-Fanion-5's while F-Muon
    is best. For S-Fanions the corresponding curve depends on the entry basis.

The saved quadratic file code/long_run contains all 17 methods and matches
the manuscript's Muon/F-Muon/S-Muon threshold times. Without fitting a scale,
Fanion-2 and Fanion-10 match Neon after time rescaling t -> k*t, with maximum
loss discrepancies of 0.000252% and 0.01884% of the initial loss, respectively,
over their shared recorded horizons (249 and 49 noninitial points).

An audit finding matters: the current benchmark specification gives
F-Fanion-10 an inverse-square-root learning-rate schedule, while the other
Fanions use constant rates. At 5000 updates it accumulates 2.09952 learning
rate units instead of 75. The CSV has no serialized configuration; settings
are assigned from the current source, with this provenance limit recorded.
Neither source nor saved experiments were changed.

The CIFAR norm logs do not determine the intermediate ranks. An explicit
pair of gradients has identical operator/Frobenius/nuclear norms but opposite
Fanion-5 versus Muon comparisons, for both plain and F families. Safe moment
intervals certify Muon's common-gradient Frobenius certificate above
Fanion-5 in 21.2% of the 1500 reference snapshots; most remain unresolved.
This is a limitation of these logged quantities, not evidence of a runtime
or validation-accuracy ordering. Full spectra, actual momentum directions,
and directional curvature remain the useful next measurements.

New files:
  fanion_followup.tex/pdf/html       Full continuation with proofs and figures.
  paper_insert_fanions.tex          Compact candidate article section.
  fanion_analysis.py                CPU calculations, source audit, checks.
  results/fanions/summary.json      Complete results, input hashes, limitations.
  results/fanions/*.csv             Rank-time, family, moment, and budget tables.
  results/fanions/*.pdf and *.png   Standalone figures.

Reproduce the continuation from the repository root:

  python theory_attempts/fanion_analysis.py
  python theory_attempts/build_report.py --stem fanion_followup

The analysis passes 4,322 numerical checks; proofs and assumptions are in the
report. No new neural-network training or original-seed replication is claimed.

Main result
-----------
For the article's quadratic, put E0 = sqrt(M) (X0-Z) sqrt(N),
r = min(m,n), and L = ||M||op ||N||op. Every additive rank-k Fanion
trajectory, with any momentum and step sizes gamma_t >= 0, satisfies

  f(X_T) >= 1/2 sum_{j>kT} sigma_j(E0)^2,
  f(X_T) >= [||E0||_* - sqrt(L) k sum_t gamma_t]_+^2 / (2r).

The second inequality is substantially stronger at the small learning rates
used in the experiments. On a fresh draw from the paper's 500 x 500 random
problem distribution, Neon needs at least 65,172 steps at lr=0.007 to attain
loss 0.001. This is a necessary time, independent of momentum, not a training
result or a replication of the original PyTorch seed.

There is also an exact example where optimally stepped Fanion-k needs
ceil(r/k) iterations and Muon needs one, plus explicit upper bounds using
the gradient spectrum, directional curvature, noise, and hybrid coefficients.

What remains unresolved
-----------------------
These results do not prove the entire CIFAR/LLM ranking. A common Frobenius
curvature surrogate favors Muon over Neon on most saved CIFAR gradients,
but incorrectly favors F-Neon over Muon. The report preserves that negative
finding and identifies the actual momentum/curvature measurements needed.
The saved reference CSV has nonfinite losses at 242 of 250 rows; all norm
data used here are finite. No loss-rate fitting uses those loss entries.

Relevant theory has advanced: in particular, Kovalev--Borodich v2 (June 2026)
has improved guarantees under structural assumptions. The report gives
primary sources, qualifications, and a counterexample to a schedule claim
in another recent preprint. No novelty claim is made for standard descent
or rank-approximation arguments.

Files
-----
bounds.tex / bounds.pdf: full derivations, assumptions, proofs, literature,
                        data audit, limitations, and next measurements.
bounds.html:            standalone MathML report with embedded figures.
paper_insert.tex:        compact self-contained section for possible inclusion
                        in the article; does not edit the submission.
analyze.py:              reproducible CPU calculations and matrix diagnostics.
build_report.py:         offline HTML/PDF exporter using Pandoc and Edge/Chrome.
results/summary.json:   numerical checks, input-file hashes, data quality counts.
results/quadratic_lower_bounds.csv: rank and nuclear step-budget obstructions.
results/flat_quadratic.csv: exact r/k example for r=32, 128, and 500.
results/curvature_example.csv: exact one-step comparisons with known curvature.
results/cifar_summary.csv: all nine trajectories and six convolution layers.
results/cifar_gradient_surrogates.csv: all 13,500 individual layer snapshots.
results/rank_barrier.pdf and cifar_diagnostics.pdf: standalone figures.

Reproduce
---------
From the repository root:

  python theory_attempts/analyze.py

Python 3.9+, NumPy, and Matplotlib 3.9+ are sufficient. No GPU, PyTorch,
training dataset, package installation, or network is needed. The checked
environment used NumPy 2.0.2 and Matplotlib 3.9.2. Outputs always stay in
theory_attempts/results. The random draw uses NumPy seed 142; it is not
identical to the article's PyTorch seed 42. The original code disables weight
normalization for its synthetic experiment, as required by the additive
iterate result. A different quadratic size can be requested with
--quadratic-size, but the report's numerical tables then need updating.

To reproduce the supplied standalone HTML/PDF, from the repository root:

  python theory_attempts/build_report.py

This requires Pandoc and Edge/Chrome (already available here), runs the browser
headless with a temporary profile under .build, and writes only in this folder.
--html-only skips the PDF step. The supplied PDF uses this route because the
local MiKTeX launcher cannot initialize in the restricted execution account.
No TeX configuration or installed packages were changed.

For a standard LaTeX build instead, from theory_attempts:

  pdflatex -interaction=nonstopmode -halt-on-error bounds.tex
  pdflatex -interaction=nonstopmode -halt-on-error bounds.tex

Optional matrix diagnostic
--------------------------
Save an NPZ with:

  gradient:     true or independent evaluation gradient, a 2-D matrix;
  oracle_input: optional actual momentum/gradient matrix used by the LMO;
  left, right:  optional symmetric PSD quadratic factors M,N, both required.

Then run:

  python theory_attempts/analyze.py --snapshot theory_attempts/example.npz

This writes results/snapshot_diagnostics.csv with actual gradient projections,
Frobenius scores, and exact quadratic curvature scores when M,N are supplied.
Directions are ideal exact-SVD Fanions and mixtures at alpha=0.5, eta=0.003;
they are not a simulation of the repository's Newton--Schulz approximation.
If gradient is only a minibatch gradient, its scores remain stochastic
surrogates, not population-gradient convergence guarantees.

All work is confined to theory_attempts. The submission and existing code
were read only. The pre-existing modification to the submission bibliography
was not touched.
