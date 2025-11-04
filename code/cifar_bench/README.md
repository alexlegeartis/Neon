airbench_muon.py: best results on NormalizedMuon (aka F-Muon)

compare_sgd_coeffs.py: checks the dependency of F-Muon on alpha

plot_sgd_coeffs_results.py: creates plots for the article

create_fmuon_testing.py: a helper function for compare_sgd_coeffs.py. It also uses airbench_muon.py.

base_airbench.py: playground for different LMO optimizers from optimizers.py

muon_vs_neon.py: comparison of Muon, Neon, SGD with plots

profile_script.sh: runs a profiler on muon_vs_neon.py --> profiler.svg

matrix_functions.py and optimizer.py: have been moved to the root.

norm_logger_airbench.py: a draft of the approach to track the Frobenius, spectral, and nuclear norms of gradients of the layers.
There we discovered that those norms don't actually decrease significantly, and for Neon they even increase.