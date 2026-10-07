"""Generate the six Kaggle-ready notebooks (thin, documented wrappers over src/).

Run from repo root:  python scripts/make_notebooks.py
Every notebook: setup cell (clone repo / fix sys.path) -> GPU check -> stage run
-> VERIFICATION cell (asserts + inline figures). Nothing is duplicated between
notebooks and src/ — the notebooks are orchestration + verification only.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks")
OUT.mkdir(exist_ok=True)

SETUP = '''import os, sys, subprocess
from pathlib import Path

IN_KAGGLE = Path("/kaggle").is_dir()
if IN_KAGGLE:
    repo = Path("mv-brainmae")
    if not repo.is_dir():
        # set your repo URL after pushing to GitHub
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/YOUR_GITHUB_USERNAME/mv-brainmae.git"],
                       check=True)
else:
    repo = Path("..")  # local: notebooks/ inside a clone
sys.path.insert(0, str(repo.resolve()))
os.chdir(repo)
print("repo:", repo.resolve())

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'''

COMMON_VER = '''# ---- VERIFICATION (do not skip) ----
import json
from pathlib import Path
import pandas as pd
md = pd.read_csv("data/factory/metadata.csv")
sp = pd.read_csv("data/factory/splits.csv")
assert sp.subject_id.is_unique, "LEAK: subject appears in two splits"
assert not set(sp[sp.split=="train"].subject_id) & set(sp[sp.split=="test"].subject_id)
print("slices:", len(md), "| subjects:", md.subject_id.nunique(),
      "| views:", md.view.value_counts().to_dict())
print("subject counts per split:", sp.split.value_counts().to_dict())
print("class counts (subjects):",
      md.drop_duplicates("subject_id").cdr_class.value_counts().sort_index().to_dict())
log = json.load(open("data/factory/factory_log.json"))
print("skipped visits:", log["n_skipped"], "| runtime:", log["runtime_sec"], "s")'''


def nb(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    (OUT / name).write_text(nbf.writes(nb))
    print("wrote", OUT / name)


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ---------------------------------------------------------------- 00 factory
nb("00_data_factory.ipynb", [
    md("""# Stage 0 — Data factory: OASIS-1 volumes → QC'd slices + splits

Converts raw T1 volumes into axial + coronal 16-bit PNG slices, joins the
cross-sectional clinical CSV (CDR, age, sex, nWBV…), and writes the
**subject-level** train/val/test splits + GroupKFold fold ids that every later
stage must use. **~15–30 min on Kaggle CPU.**

Checks performed here: slice QC counts, split disjointness, class balance.
If anything looks off, STOP — everything downstream inherits these files."""),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.factory import run_factory

cfg = load_config("configs/data_factory.yaml")
if IN_KAGGLE:
    cfg["root"] = "/kaggle/input/oasis1"   # <-- your attached dataset id here
summary = run_factory(cfg)"""),
    code(COMMON_VER),
    code("""# eyeball a few slices straight from the factory output
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for v, row in zip(("axial", "coronal"), axes):
    allf = sorted(Path("data/factory/slices", v).glob("*.png"))
    files = allf[:: max(1, len(allf)//4)][:4]
    for ax, f in zip(row, files):
        ax.imshow(np.asarray(Image.open(f)) / 65535.0, cmap="gray"); ax.axis("off")
        ax.set_title(f.stem.split("_")[-2] + " " + f.stem.split("_")[-1], fontsize=8)
fig.suptitle("factory QC samples — top: axial, bottom: coronal (radiological)")
plt.tight_layout(); plt.show()"""),
    md("""### Before closing this notebook
Commit the notebook version. `data/factory/` lives in `/kaggle/working` and is
versioned with the notebook — later notebooks can read it from there, or you
can publish it as a private Kaggle dataset for faster re-attach."""),
])

# ---------------------------------------------------------------- 01/02 pretrain
PRETRAIN_TMPL = """# Stage 2{tag} — {title}

{blurb}

**Kaggle reality notes**
- Enable GPU + Internet (timm not needed here; Internet optional).
- `max_minutes=600` stops cleanly before the 12 h session limit; rerun the SAME
  notebook version (or a new version) and it **resumes** from `ckpt.pt`.
- Estimated wall time on T4 x2: single ≈ 3–5 h, cross ≈ 5–8 h for 60 epochs.
"""

nb("01_pretrain_single.ipynb", [
    md(PRETRAIN_TMPL.format(tag="A", title="single-view MAE (control arm)",
        blurb="Standard masked autoencoder pretraining on brain-MRI slices "
               "(75% masking, normalized-pixel targets). This is the CONTROL "
               "against which the multi-view twist is judged.")),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.pretrain import run_pretrain

cfg = load_config("configs/pretrain_single.yaml")
summary = run_pretrain(cfg)   # resumable: rerun after a session kill = continues"""),
    code("""# ---- VERIFICATION ----
import json
from pathlib import Path
import pandas as pd
recs = [json.loads(l) for l in open("outputs/pretrain_single/pretrain_single.jsonl")]
ep = pd.DataFrame([r for r in recs if r["event"] == "epoch"])
print(ep[["epoch", "loss"]].tail(8).to_string(index=False))
rc = pd.DataFrame([r for r in recs if r["event"] == "recon"])
print("val recon MSE trajectory:"); print(rc[["epoch", "val_recon_mse"]].to_string(index=False))
assert ep.loss.iloc[-1] < ep.loss.iloc[0], "loss did not decrease — check config"
assert Path("outputs/pretrain_single/encoder.pt").exists(), "encoder not saved"
print("VERIFIED: loss decreased and encoder checkpoint exists")"""),
    code("""from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 7))
latest = sorted(Path("outputs/pretrain_single").glob("recon_samples_e*.png"))[-1]
ax.imshow(Image.open(latest)); ax.axis("off")
ax.set_title(f"latest reconstruction grid — {latest.name}")
plt.tight_layout(); plt.show()"""),
])

nb("02_pretrain_cross.ipynb", [
    md(PRETRAIN_TMPL.format(tag="B", title="MULTI-VIEW cross-plane MAE (the twist)",
        blurb="Both an axial AND a coronal slice of the same subject are 75%-masked; "
               "each view's visible tokens are the context for reconstructing the "
               "other. The encoder must internalize cross-plane 3-D anatomical "
               "consistency. Learned view embeddings keep the planes distinguishable. "
               "**Everything else is identical to Stage 2A** — same data, same seeds, "
               "same architecture, same schedule (paired comparison by design).")),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.pretrain import run_pretrain

cfg = load_config("configs/pretrain_cross.yaml")
summary = run_pretrain(cfg)"""),
    code("""# ---- VERIFICATION ----
import json, pandas as pd
a = pd.DataFrame([json.loads(l) for l in open("outputs/pretrain_single/pretrain_single.jsonl")
                  if json.loads(l)["event"] == "epoch"])
b = pd.DataFrame([json.loads(l) for l in open("outputs/pretrain_cross/pretrain_cross.jsonl")
                  if json.loads(l)["event"] == "epoch"])
cmp = pd.DataFrame({"single": a.loss.values[:len(b)], "cross": b.loss.values})
print(cmp.tail(10).to_string(index=False))
print("final losses ->", cmp.iloc[-1].to_dict())"""),
    code("""from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 7))
latest = sorted(Path("outputs/pretrain_cross").glob("recon_samples_e*.png"))[-1]
ax.imshow(Image.open(latest)); ax.axis("off")
ax.set_title(f"cross-arm reconstruction grid — {latest.name}")
plt.tight_layout(); plt.show()"""),
])

# ---------------------------------------------------------------- 03 downstream
nb("03_downstream.ipynb", [
    md("""# Stage 3 — Downstream: does pretraining pay off?

Runs, in one notebook:
1. **Feature caching** for all four arms (our MAE-single, our MAE-cross,
   ImageNet-pretrained ViT-S, scratch ViT-S) — needs Internet ON once for timm.
2. **Linear probes** with subject-level GroupKFold + label-efficiency fractions
   (10/25/50/100 % of training *subjects*, same subset for every arm).
3. **Fine-tuning** (attention-pool head) for the two MAE arms.
4. **Age probe** sanity check on frozen features.

All results land in `results/` as CSVs — every table in the README comes from
them. Budget: feature cache ~20–40 min; probes minutes; finetune ~1–3 h."""),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.downstream import run_downstream

cfg = load_config("configs/downstream.yaml")
summary = run_downstream(cfg)"""),
    code("""# ---- VERIFICATION ----
import pandas as pd
probe = pd.read_csv("results/downstream_probe.csv")
print(probe.pivot_table(index=["encoder", "label_frac"], columns="scope",
                        values="balanced_accuracy").round(3))
ft = pd.read_csv("results/downstream_finetune.csv")
print(ft.groupby("encoder")["balanced_accuracy"].agg(["mean", "std"]).round(3))
age = pd.read_csv("results/downstream_age.csv")
print(age[["encoder", "pearson_r", "mae_years"]].round(3))
# leakage audit: test subjects must never appear in any training predictions file
sp = pd.read_csv("data/factory/splits.csv").set_index("subject_id").split
for f in sorted(Path("results").glob("preds_probe_*frac1.0.csv")):
    d = pd.read_csv(f)
    assert set(sp[d.subject].unique()) == {"test"}, f"LEAK in {f}"
print("leakage audit OK: prediction files contain TEST subjects only")"""),
])

# ---------------------------------------------------------------- 04 anomaly
nb("04_anomaly.ipynb", [
    md("""# Stage 4 — Unsupervised anomaly maps from reconstruction error

No labels, no fine-tuning: the pretrained MAE reconstructs held-out slices;
brain-masked |x − x̂| is the anomaly score. Subject score = top-25 % slice mean.

Checks: AUC (demented vs CDR0) on TEST, Spearman(score, nWBV) expected
**negative**, Spearman(score, age) positive-but-weaker, plus example heatmaps.
Budget: ~20–60 min GPU."""),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.anomaly import run_anomaly

cfg = load_config("configs/anomaly.yaml")
metrics = run_anomaly(cfg)
print(metrics)"""),
    code("""# ---- VERIFICATION ----
import json
from PIL import Image
import matplotlib.pyplot as plt
m = json.load(open("results/anomaly_metrics.json"))
for arm, v in m.items():
    print(f"{arm}: AUC_topk={v['auc_topk']:.3f}  spearman(nWBV)={v['spearman_nwbv']:+.3f}"
          f" (p={v['spearman_nwbv_p']:.3g})")
for f in sorted(Path("figures").glob("anomaly_examples_*.png")):
    fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(Image.open(f)); ax.axis("off")
    plt.tight_layout(); plt.show()"""),
])

# ---------------------------------------------------------------- 05 analysis
nb("05_analysis.ipynb", [
    md("""# Stage 5 — Analysis: figures + auto hypothesis verdicts

Aggregates every CSV/JSONL into six figures and `results/summary.md`, where
each pre-registered hypothesis (H0–H4) gets a machine-computed verdict against
the thresholds in `configs/analysis.yaml`. **Read summary.md first, then look
at the figures — verdicts must not depend on which figure looks nicer.**"""),
    code(SETUP),
    code("""from mvbrainmae.utils import load_config
from mvbrainmae.analysis import run_analysis

cfg = load_config("configs/analysis.yaml")
print(run_analysis(cfg))"""),
    code("""# ---- VERIFICATION: print the whole report ----
print(Path("results/summary.md").read_text())"""),
    code("""from PIL import Image
import matplotlib.pyplot as plt
for f in sorted(Path("figures").glob("fig_*.png")):
    fig, ax = plt.subplots(figsize=(11, 6)); ax.imshow(Image.open(f)); ax.axis("off")
    ax.set_title(f.name); plt.tight_layout(); plt.show()"""),
    code("""# bundle everything you want to keep (Kaggle output = downloadable/versioned)
import os, zipfile
from pathlib import Path
out_zip = "/kaggle/working/mv_brainmae_artifacts.zip" if IN_KAGGLE \\
          else "mv_brainmae_artifacts.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("results", "figures"):
        for root, _, files in os.walk(d):
            for fn in files:
                z.write(os.path.join(root, fn))
    for p in Path("outputs").glob("*/encoder.pt"):
        z.write(p)
    for p in Path("outputs").glob("*/pretrain_*.jsonl"):
        z.write(p)
print("artifacts zipped ->", out_zip)"""),
])

print("done")
