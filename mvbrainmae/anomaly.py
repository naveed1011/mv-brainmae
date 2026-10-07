"""Stage 4 — Reconstruction-error anomaly maps (unsupervised, no fine-tuning).

Premise: an MAE pretrained to reconstruct brain anatomy should reconstruct
HEALTHY-looking tissue well and fail on atrophy/pathology. Per-slice anomaly
score = brain-masked mean |x - x_hat|. Subject score = mean and top-25% mean
of slice scores.

Verification targets:
  * AUC (CDR>=0.5 vs CDR0) on the TEST split, per encoder
  * Spearman(subject score, nWBV) — expected NEGATIVE (more atrophy -> lower
    normalized brain volume -> higher reconstruction error)
  * Spearman(subject score, age) — expected positive but weaker
  * Qualitative: example heatmaps for low-CDR and high-CDR subjects

Run: python -m mvbrainmae.anomaly --config configs/anomaly.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import FactoryData, load_slice
from .mae import build_from_checkpoint
from .metrics import binary_auc, correlations
from .utils import RunLogger, ensure_dir, get_device, set_seed


# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_slices(model, fd: FactoryData, rows: pd.DataFrame, device,
                 bg_thresh: float, batch_size: int = 64) -> pd.DataFrame:
    """Per-slice brain-masked reconstruction error."""
    model.eval()
    mean, std = fd.mean, fd.std
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        x01 = torch.stack([load_slice(fd.dir, f, 0.0, 1.0) for f in chunk["file"]])
        x_norm = (x01 - mean) / std
        x_hat = model.reconstruct(x_norm.to(device), view_id=0).cpu()
        x_hat01 = (x_hat * std + mean).clamp(0, 1)
        err = (x01 - x_hat01).abs().squeeze(1)                 # (B,H,W) in [0,1] space
        mask = (x01.squeeze(1) > bg_thresh).float()
        score = (err * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)
        for i, (_, r) in enumerate(chunk.iterrows()):
            out.append({"file": r["file"], "subject_id": r["subject_id"],
                        "slice_idx": int(r["slice_idx"]),
                        "cdr_class": r.get("cdr_class"), "split": r["split"],
                        "score": float(score[i]),
                        "err_map_path": None})
    return pd.DataFrame(out)


def subject_scores(slice_df: pd.DataFrame, topk: float = 0.25) -> pd.DataFrame:
    g = slice_df.groupby("subject_id", dropna=False)
    out = g["score"].mean().rename("score_mean").to_frame()
    out["score_topk"] = g["score"].apply(
        lambda s: float(s.nlargest(max(1, int(len(s) * topk))).mean()))
    out["n_slices"] = g["score"].size()
    return out.reset_index()


def make_heatmap_figure(model, fd: FactoryData, subj_rows: pd.DataFrame,
                        device, path: Path, title: str) -> None:
    """One row per subject: original | reconstruction | error (brain-masked)."""
    mean, std = fd.mean, fd.std
    n = len(subj_rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r_i, (_, r) in enumerate(subj_rows.iterrows()):
        x01 = load_slice(fd.dir, r["file"], 0.0, 1.0).unsqueeze(0)
        xh = model.reconstruct(((x01 - mean) / std).to(device), 0).cpu()
        xh01 = (xh * std + mean).clamp(0, 1)
        err = (x01 - xh01).abs().squeeze()
        m = (x01.squeeze() > fd.stats["bg_thresh"]).float()
        axes[r_i, 0].imshow(x01.squeeze(), cmap="gray"); axes[r_i, 0].set_title("original")
        axes[r_i, 1].imshow(xh01.squeeze(), cmap="gray"); axes[r_i, 1].set_title("recon")
        im = axes[r_i, 2].imshow(err * m, cmap="inferno")
        axes[r_i, 2].set_title(f"error | score={r.get('score', float('nan')):.4f}")
        plt.colorbar(im, ax=axes[r_i, 2], fraction=0.046)
        for ax in axes[r_i]:
            ax.axis("off")
        axes[r_i, 0].set_ylabel(str(r["subject_id"]), rotation=0, labelpad=60,
                                fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run_anomaly(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    fd = FactoryData(cfg["factory_dir"])
    results_dir = ensure_dir(cfg["results_dir"])
    figures_dir = ensure_dir(cfg.get("figures_dir", "figures"))
    log = RunLogger(results_dir, "anomaly")
    bg = float(cfg.get("bg_thresh", fd.stats.get("bg_thresh", 0.08)))
    topk = float(cfg.get("topk", 0.25))

    # subjects to score: test + val splits (all classes) + train CDR0 (healthy ref)
    rows = fd.rows(labeled_only=True, view="axial")
    healthy_train = rows[(rows.split == "train") & (rows.cdr_class == 0)]
    eval_rows = rows[rows.split.isin(["test", "val"])]
    target_rows = pd.concat([eval_rows, healthy_train], ignore_index=True)

    meta = fd.metadata.drop_duplicates("subject_id").set_index("subject_id")
    results = {}
    for name in cfg["encoders"]:
        mode = name.split("mae_", 1)[-1]
        ckpt = Path(cfg["outputs_root"]) / f"pretrain_{mode}" / "encoder.pt"
        if not ckpt.exists():
            log.log(f"[{name}] SKIP (missing {ckpt})")
            continue
        model = build_from_checkpoint(str(ckpt)).to(device)
        slice_df = score_slices(model, fd, target_rows, device, bg,
                                int(cfg.get("batch_size", 64)))
        subj = subject_scores(slice_df, topk)
        subj = subj.join(meta[["cdr", "cdr_class", "age", "nwbv", "split"]],
                         on="subject_id")
        subj.to_csv(results_dir / f"anomaly_subjects_{name}.csv", index=False)
        slice_df.to_csv(results_dir / f"anomaly_slices_{name}.csv", index=False)

        te = subj[(subj.split == "test") & subj.cdr_class.notna()].copy()
        te["demented"] = (te["cdr_class"] >= 1).astype(int)
        auc_top = binary_auc(te["score_topk"].values, te["demented"].values)
        auc_mean = binary_auc(te["score_mean"].values, te["demented"].values)
        c_nwbv = correlations(te["score_topk"].values,
                              pd.to_numeric(te["nwbv"], errors="coerce").values)
        c_age = correlations(te["score_topk"].values, te["age"].values)
        results[name] = {
            "auc_topk": auc_top, "auc_mean": auc_mean,
            "n_test_subjects": int(len(te)),
            "spearman_nwbv": c_nwbv["spearman_rho"],
            "spearman_nwbv_p": c_nwbv["spearman_p"],
            "spearman_age": c_age["spearman_rho"],
            "healthy_ref_mean": float(subj[(subj.split == "train")
                                           & (subj.cdr_class == 0)]
                                      ["score_topk"].mean()),
        }
        log.log(f"[{name}] AUC(topk)={auc_top:.3f} AUC(mean)={auc_mean:.3f} "
                f"spearman(nwbv)={c_nwbv['spearman_rho']:.3f} "
                f"(p={c_nwbv['spearman_p']:.3g})")

        # example heatmaps: healthiest-looking CDR0 vs worst CDR2+ in TEST
        ex = []
        cdr0 = te[te.cdr_class == 0].nsmallest(1, "score_topk")
        cdr2 = te[te.cdr_class >= 2].nlargest(1, "score_topk")
        for _, s in pd.concat([cdr0, cdr2]).iterrows():
            s_rows = slice_df[slice_df.subject_id == s["subject_id"]]
            mid = s_rows.iloc[(s_rows["score"] - s_rows["score"].median())
                              .abs().argsort()[:1]]
            mid = mid.assign(score=s["score_topk"])
            ex.append(mid)
        if ex:
            make_heatmap_figure(model, fd, pd.concat(ex), device,
                                figures_dir / f"anomaly_examples_{name}.png",
                                f"{name}: reconstruction-error anomaly maps "
                                f"(top: lowest-score CDR0, bottom: highest CDR2+)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(results_dir / "anomaly_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log.log(f"Anomaly done for {list(results)}")
    return results


def main() -> None:
    import argparse
    from .utils import apply_overrides, load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    r = run_anomaly(apply_overrides(load_config(args.config), args.overrides))
    print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
