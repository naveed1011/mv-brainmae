Evidence from the CPU smoke run v0.2 (synthetic brains, planted atrophy signal):
  summary.md           -> H0 gate PASSED; H3/H4 SUPPORTED (expected on synthetic)
  downstream_probe.csv -> label-efficiency tables, four encoder arms
  anomaly_metrics.json -> AUC 1.0, spearman(nWBV) = -0.93 (planted signal recovered)
  fig_*.png            -> the six auto-generated figures of that run
  recon_grid_*.png     -> epoch-2 reconstruction grids (noise is NORMAL at 2 epochs)
Regenerate: bash scripts/smoke_test.sh   (~10 min on CPU which does not needs data)
