# Verification Guide — how to know this project does what it claims

Verification is layered: **(A)** code correctness on synthetic data with a
planted signal (runs anywhere, ~10 min, CPU), **(B)** stage-by-stage checks on
real data (inside the Kaggle notebooks), **(C)** result-level checks before
believing/publishing anything.

## A. Synthetic end-to-end gate (run locally, before Kaggle)

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh        # synthetic brains -> all 6 stages -> PASS lines
```

The synthetic brains (`scripts/make_synthetic_data.py`) have **ventricle size
growing and cortex thinning with CDR class**, plus correlated `nWBV`/`age`
fields. Therefore a *correct* pipeline MUST show:

| Check | Expected on synthetic |
|---|---|
| probe TEST balanced accuracy | ≫ 0.25 (H0 gate: > 0.40) |
| label-efficiency curve | monotone-ish improvement 25 % → 100 % |
| anomaly AUC, Spearman(score, nWBV) | AUC high, ρ strongly **negative** |
| age probe Pearson r | high (> 0.6) |

If any of these is at chance on synthetic data, the code is broken — **do not
spend GPU hours**, fix first. (`summary.md` prints these verdicts as H0/H3/H4.)

## B. Stage checks (each Kaggle notebook ends with a VERIFICATION cell)

**Stage 1 (factory).**
- `factory_log.json`: `n_skipped` should be 0 or explainable (corrupt volumes).
- Slice counts: OASIS-1 ≈ 416 subjects × ~60–90 kept axial + coronal slices →
  tens of thousands of rows; QC histogram of `brain_frac` unimodal, no mass at 0.
- `splits.csv`: subject uniqueness asserted; class counts per split non-degenerate.
- QC figure: axial/coronal slices look like brains in radiological orientation.

**Stage 2 (pretrain).**
- `pretrain_*.jsonl`: training loss decreasing; val reconstruction MSE
  decreasing; **no NaNs**.
- Recon grid PNGs: by ~epoch 20–30 the reconstructions must be recognizable
  brain slices (not noise). Noise at epoch 60 = broken data pipeline.
- Resume test (optional, local): kill and rerun — epoch counter continues.

**Stage 3 (downstream).**
- Probe table: `scratch` should be worst or near-worst; curves ordered
  scratch ≤ imagenet ≤ mae arms *on average* (not guaranteed — that's H1/H2).
- CV-mean vs TEST columns should agree within ~±0.08 (small test split!) —
  large disagreement means overfitting to the test split or leakage.
- Leakage audit (in notebook): every `preds_*` file contains TEST subjects only.
- Paired check: the same subject subsets per fraction across arms — verified by
  construction (`_stratified_subjects`, seeded).

**Stage 4 (anomaly).**
- Heatmaps concentrate on ventricles/cortex boundaries, not skull/background
  (brain mask applied).
- Score distributions shift upward with CDR class in the scatter figure.
- `spearman_nwbv_p` < 0.05 for a meaningful effect; if ρ ≈ 0 while AUC ≈ 0.5
  the encoder is not anatomy-aware — inspect recon grids from Stage 2 first.

**Stage 5 (analysis).**
- `results/summary.md` verdicts are computed from CSVs with thresholds from
  `configs/analysis.yaml`; spot-check two numbers by hand from the CSVs.

## C. Result-level checks before claiming anything

1. **Direction vs magnitude.** Report balanced accuracy and macro-F1, not raw
   accuracy (4-class OASIS is imbalanced). Never headline slice-level accuracy.
2. **Uncertainty.** Probe numbers = mean ± std over 5 GroupKFold folds; a
   cross−single difference smaller than the fold std is INCONCLUSIVE, full stop.
3. **TEST discipline.** The test split is touched once per arm (final refit
   row). If you ever iterate on test numbers, bump the seed and re-plan.
4. **External sanity.** Published OASIS-1 4-class subject-level accuracies for
   strong supervised models sit ~0.7–0.85; a probe on *frozen* features much
   above that smells of leakage; much below 0.5 smells of broken features.
5. **Anomaly plausibility.** Literature AUCs for unsupervised AD-vs-CN from
   structural MRI alone are ~0.7–0.9; >0.95 on 7 test subjects = tiny-sample
   luck, say so explicitly.

## Artifact checklist (what to keep from Kaggle)

- `results/*.csv`, `results/*.json`, `results/summary.md`
- `figures/*.png` (+ `recon_samples_*.png` from both pretrain runs)
- `outputs/pretrain_{single,cross}/encoder.pt` (your foundation weights)
- `data/factory/metadata.csv`, `splits.csv`, `folds.csv`, `stats.json`
  (the exact data fingerprint of the run)
- Notebook versions (Kaggle keeps them) — they are the run log.

## Known non-determinism

- No `cudnn.deterministic` (speed); run-to-run jitter on GPU is a few 0.001s in
  loss and up to ±0.02 in subject metrics — hence fold means±stds everywhere.
- PNG16 quantization is lossless w.r.t. the float slices we store (round-trip
  tested), so the factory output itself is bit-reproducible for a fixed seed.
