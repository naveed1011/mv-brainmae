# MV-BrainMAE — Experiment Plan

**the summary.** We build our own masked-autoencoder foundation encoder for
brain MRI, and ask whether a *multi-view* pretraining objective — reconstruct an
axial slice **and** a coronal slice of the same subject from mutually masked
tokens — learns representations that beat standard single-view MAE and
ImageNet transfer on Alzheimer's (CDR) staging and unsupervised atrophy
detection.

**Author:** Naveed Ahmad · **Status:** pre-registered design, code + smoke tests complete
**Data:** OASIS-1 cross-sectional (free registration, oasisbrains.org) — see `KAGGLE_SETUP.md`

---

## 1. The Motivation

Self-supervised foundation encoders are the current workhorse of medical
imaging (Swin UNETR, MAE-style recipes, brain-specific SSL such as BrainIAC /
BrainSegFounder). Two observations that drove me to this project:

1. **Brain MRI is multi-planar by acquisition.** A radiologist never reads one
   plane alone; axial, coronal and sagittal views are complementary. Yet MAE
   pretraining for medical images is almost always *single-image*: each slice
   is masked and reconstructed from itself only.
2. **Cross-plane consistency is free supervision.** Axial and coronal slices of
   the same subject share the same 3-D anatomy. Forcing the encoder to
   reconstruct one plane while only seeing masked fragments of *another* plane
   injects a 3-D anatomical-consistency prior into a 2-D encoder — no extra
   labels, no 3-D convolutions, no extra data.

Prior art we build on (and cite): MAE (He et al., 2021), ViT (Dosovitskiy et
al., 2020), multi-view fusion in AD classification (our own AD-TriFuseViT line
of work fuses views at *classification* time — here views shape *pretraining*),
brain SSL encoders (BrainIAC, BrainSegFounder — supervised/contrastive or
segmentation-oriented; different objective and different evaluation). The
**cross-plane masked-reconstruction objective + this exact evaluation protocol
is the original contribution of this repository.**

## 2. Research questions & pre-registered hypotheses

| ID | Hypothesis | Metric | Pre-registered threshold |
|----|-----------|--------|--------------------------|
| **H0** | *Pipeline validity gate (synthetic data only):* a dataset with a planted atrophy signal must be learnable end-to-end | TEST subject-balanced accuracy (probes) | > chance(0.25) + 0.15 |
| **H1** | Domain-specific brain SSL beats ImageNet transfer for CDR staging | TEST balanced acc, probe arm, all label fracs | `mae_single − imagenet > +0.02` |
| **H2** | The multi-view twist pays off | TEST balanced acc, probe + finetune | `mae_cross − mae_single ≥ +0.005` |
| **H3** | Reconstruction error is an unsupervised atrophy biomarker | AUC(CDR≥0.5 vs CDR0) on TEST; Spearman(score, nWBV) | AUC ≥ 0.60 **and** ρ < 0 |
| **H4** | Frozen encoder features encode subject age (sanity) | Pearson r(age) on TEST | r ≥ 0.40 |

Verdicts are **computed, not eyeballed** — `mvbrainmae.analysis` reads the
thresholds from `configs/analysis.yaml` and writes them into
`results/summary.md` as SUPPORTED / REFUTED / INCONCLUSIVE.

## 3. Design

```
Stage 1  factory        volumes -> axial+coronal 16-bit PNG slices, QC,
                        subject-level 70/10/20 split + 5 GroupKFold folds
Stage 2  pretraining    2A single-view MAE   |   2B cross-plane MAE (the twist)
                        identical everything-else (seeds, arch, schedule, data)
Stage 3  downstream     frozen [CLS||mean] features -> linear probes
                        (subject GroupKFold; label fracs 10/25/50/100%)
                        + finetune (attention-pool head) + age probe
                        arms: mae_single, mae_cross, imagenet(ViT-S), scratch(ViT-S)
Stage 4  anomaly        brain-masked |x - x̂| per slice -> subject score
                        -> AUC vs CDR, Spearman vs nWBV/age, heatmaps
Stage 5  analysis       figures + auto hypothesis verdicts (summary.md)
```

**Cross-view objective (2B).** For subject *s*, draw an axial slice *a* and a
coronal slice *c*. Mask each at 75 %. The encoder sees
`[CLS | visible(a) + pos + view0 | visible(c) + pos + view1]`; the decoder
reconstructs the masked patches of **both** planes. Learned view embeddings
(let zero-initialized) keep the planes distinguishable; shared positional
embeddings keep parameter count equal to 2A. The `adjacent` mode (2.5-D: axial
+ neighbouring axial) exists as an ablation for "any extra context helps" vs
the cross-plane mechanism specifically.

**Leakage firewall.** OASIS-1 subjects contribute multiple slices *and*
sometimes multiple visits. Every split/fold in this repo is at **subject**
level; classification metrics average slice probabilities per subject before
scoring; the TEST split is scored exactly once per arm; verification cells in
every notebook assert split disjointness and TEST-only prediction files.

**Paired comparisons.** At each label fraction the *same* stratified subject
subset feeds every encoder arm (seeded), so arm differences cannot be
explained by lucky subsamples.

## 4. Compute budget (Kaggle free tier)

| Stage | Hardware | Est. wall time | Session-safe? |
|---|---|---|---|
| 1 factory | CPU | 15–30 min | yes |
| 2A pretrain single (60 ep, ViT-S/16, bs128, AMP) | T4×2 | 3–5 h | `max_minutes=600` guard + resume |
| 2B pretrain cross | T4×2 | 5–8 h | same |
| 3 features+probes | GPU | 30–60 min | yes |
| 3 finetune (2 arms × 3 folds × 8 ep) | GPU | 1–3 h | yes |
| 4 anomaly | GPU | 20–60 min | yes |
| 5 analysis | CPU | <5 min | yes |

Total ≈ 12–20 GPU-hours → fits the 30 h/week free quota across 2–3 sessions.
P100 users: roughly same wall time (16 GB VRAM is enough for bs 128 @ 224).

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| OASIS-1 is small (~416 subjects) → noisy verdicts | 5-fold GroupKFold means±stds; verdict thresholds include INCONCLUSIVE band; synthetic H0 gate proves the chain itself |
| Cross-view may not help (H2 refuted) | A refuted H2 **is a result**: report it with the adjacent-mode ablation to explain *why* (context-plane specificity) |
| Class imbalance (few CDR2) | balanced class weights, balanced accuracy as primary metric, confusion matrices reported |
| Session kills on Kaggle | atomic resumable checkpoints + `max_minutes` guard |
| timm download fails offline | arm auto-skips with a log line; MAE arms unaffected |

## 6. Reproducibility contract

- Seeds: 42 everywhere (factory splits/folds, subsamples, training);
  cudnn determinism intentionally OFF (documented in `utils.set_seed`).
- Every stage writes JSONL/CSV logs; `results/` + `figures/` + `encoder.pt`
  weights are the published artifacts.
- `bash scripts/smoke_test.sh` re-verifies the *code* end-to-end on synthetic
  data in ~10 min on CPU — run it after any code change.
