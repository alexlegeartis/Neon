cupy_svds_profiler.py and cupy_svds_profiler_variance: we see that almost all time is spent on Lanczos, not copying.

playground: with SVDS, power iterations etc.

svds_vs_ns.py: on 10K x 10K SVDS even with k=10 is 4x faster than NS. In case of 1K x 1K NS is 10x faster. Very strange.

matrix_functions.py: the best code.