# Kaggle Setup — data, GPU, run order

## 1. Get the data (OASIS-1 cross-sectional)

Kaggle mirrors of OASIS mostly contain **pre-sliced 2D images**, which cannot be
used here (the cross-view objective needs raw 3-D volumes to cut its own
planes). Get the official volumes — free, requires a quick registration and
data-use agreement:

1. Register at <https://sites.wustl.edu/oasisbrains/> and accept the OASIS DUA.
2. Download the **OASIS-1 cross-sectional** image disk (MPRAGE volumes,
   incl. the Talairach-registered `*_t88_masked_gfc` files we prefer) and
   `oasis_cross-sectional.csv` (labels: CDR, age, sex, MMSE, nWBV, eTIV…).
3. Upload to Kaggle as a **private dataset**:
   ```bash
   pip install kaggle
   mkdir -p ~/oasis1 && cp -r <downloaded disk contents> ~/oasis1/
   cd ~ && kaggle datasets init -p oasis1   # edit the JSON (title: oasis1)
   kaggle datasets create -p oasis1
   ```
   Expected layout inside the dataset (the factory auto-discovers it):
   ```
   OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_..._t88_masked_gfc.img|.hdr
   ...
   oasis_cross-sectional.csv
   ```
4. In notebook 00, set `cfg["root"] = "/kaggle/input/oasis1"` (or your dataset id).

**Optional extra healthy brains:** the IXI dataset (~600 healthy T1s, free
direct download) can be dropped into the same dataset folder to enlarge the
pretraining pool. IXI has no CDR labels — the factory handles missing labels
(pretrain-only subjects).

**Licensing note:** OASIS-1 is distributed for research use under its DUA; keep
your Kaggle dataset private and cite the OASIS-1 paper (Marcus et al., 2007) in
anything you publish, including the GitHub README.

## 2. Kaggle account settings

- Verify phone number → unlocks GPU (T4×2 or P100) and longer sessions.
- Quotas (free tier): ~30 GPU h/week, 12 h per session, 16 GB VRAM per GPU,
  ~13 GB RAM (more with GPU attached), ~20 GB dataset / 100 GB output space.
- Settings → **Internet: ON** for notebook 03 (timm downloads ViT-S ImageNet
  weights once, ~90 MB) — optional for other notebooks.

## 3. Run order & time budget

| # | Notebook | GPU? | Est. time | Output |
|---|----------|------|-----------|--------|
| 00 | data factory | CPU | 15–30 min | `data/factory/*` |
| 01 | pretrain single | GPU | 3–5 h (resumable) | `outputs/pretrain_single/*` |
| 02 | pretrain cross | GPU | 5–8 h (resumable) | `outputs/pretrain_cross/*` |
| 03 | downstream | GPU | 1.5–4 h | `results/*` |
| 04 | anomaly | GPU | 20–60 min | `results/*`, heatmaps |
| 05 | analysis | CPU | <5 min | `figures/*`, `summary.md` |

Each notebook first runs a setup cell that clones **your** GitHub copy of this
repo into `/kaggle/working` (set your username there once). Commit notebook
versions after each stage: `/kaggle/working` persists per version, so stage *n*
reads stage *n−1* outputs.

## 4. Session dies mid-pretraining? (normal on the free tier)

The loop stops itself at `max_minutes=600` and writes an atomic `ckpt.pt`.
To continue: open a new notebook version of the same notebook, re-run setup +
run cell — it resumes from `ckpt.pt` in `/kaggle/working/outputs/...`
(epoch counter continues; loss curve stays contiguous in the JSONL).
If `/kaggle/working` was wiped, attach the previous version's output as an
input dataset and set `cfg["resume_ckpt"] = "/kaggle/input/<...>/ckpt.pt"`.

## 5. Collecting artifacts

Notebook 05 zips `results/ + figures/ + encoder.pt + training logs` into
`/kaggle/working/mv_brainmae_artifacts.zip` → download it, unzip into the repo
root (respecting `.gitignore`-tracked vs published files), and rerun
`python -m mvbrainmae.analysis` locally to confirm the numbers survived the trip.

## 6b. Laptop WITHOUT a GPU: the CPU pilot on REAL data

You cannot train the full experiment on a laptop CPU (weeks of wall time), but
you CAN run a shrunken **pilot** on real OASIS-1 volumes — ~120 subjects,
128 px slices, tiny ViT, 15 epochs — in roughly **1.5–3 h** on a 4-core laptop.
It exercises the entire real-data chain (factory → both pretrain arms →
probes → anomaly → verdicts), so when you later switch to Kaggle GPU you know
every path works with real volumes:

```bash
# put OASIS-1 volumes under data/oasis1 (or edit configs/pilot/factory.yaml root)
bash scripts/pilot_cpu.sh
```

Pilot outputs live in `outputs/pilot/*`, `results_pilot/`, `figures_pilot/`
and are clearly labelled — they are a *pipeline proof*, not the science run.
Treat the Kaggle GPU run as the experiment whose numbers go on GitHub.

## 6. Local development loop (this sandbox / your laptop)

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py --out data/synthetic --n 32
bash scripts/smoke_test.sh          # ~10 min CPU: proves the code, not the science
pytest -q                            # unit tests (masking, split integrity, png16)
```
