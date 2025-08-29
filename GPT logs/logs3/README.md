Only NormMuon (Normalized Muon, aka F-Muon) works. NormNeon could not be tested, as it uses cupy. FastNormMuon does not work correctly.


Notable logs:

664f1c87-01af-4a16-b8cd-e7415caadede_baseline.txt - Muon baseline.

1756057473914_007lr_095mom.txt and 1756058551408_007lr_095mom.txt: 3.2809 and 3.2811 loss respectively, though probably not always reproducible.

1756055424649_007lr_096mom_NSGD.txt: 3.2846 loss, pure NSGD.

1756060088739_ns_4not5.txt: 4 iterations of NS instead of 5. 3.2876, does not seem to be profitable (time decreased, but not significantly - by about 2s when compared to F-Muon with 5 NS, and it's still much slower than with Muon)

Bottom line: F-Muon here is not better.