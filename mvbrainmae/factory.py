"""Stage 1 — Data factory: raw MRI volumes -> QC'd 16-bit PNG slices.

Outputs (under cfg['out_dir']):
    slices/axial/<visit>_axial_<idx>.png      16-bit grayscale, radiological convention
    slices/coronal/<visit>_coronal_<idx>.png
    metadata.csv    one row per slice (+ joined clinical labels when available)
    splits.csv      subject_id -> train/val/test   (SUBJECT level: no leakage)
    folds.csv       train subjects -> fold 0..K-1  (GroupKFold assignments)
    stats.json      intensity stats + config snapshot (for reproducibility)
    factory_log.json

Run:  python -m mvbrainmae.factory --config configs/data_factory.yaml
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import oasis
from .utils import RunLogger, ensure_dir, load_config, set_seed

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# Slice geometry (canonical RAS+ arrays)
# --------------------------------------------------------------------------- #
def to_axial(arr: np.ndarray, z: int) -> np.ndarray:
    """Axial slice, radiological convention: anterior up, patient-right on image-left."""
    return np.flipud(arr[:, :, z].T).copy()


def to_coronal(arr: np.ndarray, y: int) -> np.ndarray:
    """Coronal slice, radiological convention: superior up, patient-right on image-left."""
    return np.flipud(arr[:, y, :].T).copy()


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def resize_float(img: np.ndarray, size: int) -> np.ndarray:
    """float32-preserving bilinear resize (PIL mode 'F')."""
    im = Image.fromarray(img.astype(np.float32), mode="F")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def normalize_volume(arr: np.ndarray) -> np.ndarray:
    """Per-volume robust intensity normalization to [0, 1].

    Clip at percentiles so rare hyperintensities don't wash out contrast,
    then min-max. Deterministic per volume (no dataset-level stats needed).
    """
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return out


def save_png16(img01: np.ndarray, path: Path) -> None:
    q = np.clip(img01 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(q).save(path)  # uint16 -> 16-bit PNG (lossless, verified in tests)


# --------------------------------------------------------------------------- #
# Main factory
# --------------------------------------------------------------------------- #
def run_factory(cfg: dict) -> dict:
    t0 = time.time()
    root = Path(cfg["root"])
    out = ensure_dir(cfg["out_dir"])
    img_size = int(cfg.get("img_size", 224))
    min_brain_frac = float(cfg.get("min_brain_frac", 0.05))
    bg_thresh = float(cfg.get("bg_thresh", 0.08))
    views = cfg.get("views", ["axial", "coronal"])
    seed = int(cfg.get("seed", 42))
    split_ratios = cfg.get("split_ratios", {"train": 0.7, "val": 0.1, "test": 0.2})
    n_folds = int(cfg.get("n_folds", 5))
    source_tag = str(cfg.get("source", "oasis1"))

    set_seed(seed)
    log = RunLogger(out, "factory")
    (out / "slices" / "axial").mkdir(parents=True, exist_ok=True)
    (out / "slices" / "coronal").mkdir(parents=True, exist_ok=True)

    visits = oasis.discover_visits(root)
    meta = oasis.load_metadata(root)
    log.log(f"Discovered {len(visits)} visits under {root} "
            f"(metadata rows: {len(meta)})")
    max_subjects = cfg.get("max_subjects")
    if max_subjects:
        keep = set(sorted({v.subject_id for v in visits})[:int(max_subjects)])
        visits = [v for v in visits if v.subject_id in keep]
        log.log(f"max_subjects={max_subjects} cap applied -> {len(visits)} visits")

    rows, skipped = [], []
    for vi, v in enumerate(visits):
        try:
            arr = normalize_volume(oasis.load_volume_ras(v.volume_path))
        except Exception as e:  # corrupt/partial downloads happen; log & continue
            skipped.append({"visit_id": v.visit_id, "reason": f"load_error: {e}"})
            continue

        nx, ny, nz = arr.shape
        mrow = meta.loc[v.visit_id] if v.visit_id in meta.index else None
        n_kept = 0
        for view in views:
            n = nz if view == "axial" else (ny if view == "coronal" else nx)
            fn = {"axial": to_axial, "coronal": to_coronal}.get(view)
            if fn is None:  # sagittal kept simple: no display flip conventions needed
                fn = lambda a, i: a[i, :, :].T[::-1].copy()
            for idx in range(n):
                sl = fn(arr, idx)
                sl_r = resize_float(pad_to_square(sl), img_size)
                brain_frac = float((sl_r > bg_thresh).mean())
                if brain_frac < min_brain_frac:
                    continue
                fname = f"{v.visit_id}_{view}_{idx:04d}.png"
                save_png16(sl_r, out / "slices" / view / fname)
                rows.append({
                    "file": f"slices/{view}/{fname}", "view": view,
                    "visit_id": v.visit_id, "subject_id": v.subject_id,
                    "slice_idx": idx, "n_slices": n,
                    "rel_pos": idx / max(1, n - 1),
                    "brain_frac": brain_frac, "volume_kind": v.kind,
                    "source": source_tag,
                })
                n_kept += 1
        if n_kept == 0:
            skipped.append({"visit_id": v.visit_id, "reason": "no_slices_passed_qc"})
        if (vi + 1) % 25 == 0:
            log.log(f"  processed {vi + 1}/{len(visits)} visits "
                    f"({len(rows)} slices kept)")

    if not rows:
        raise RuntimeError("No slices passed QC — check data root/layout.")

    md = pd.DataFrame(rows)
    # Join clinical metadata when available (OASIS-1 CSV / synthetic CSV)
    if len(meta):
        join_cols = [c for c in ("cdr", "cdr_class", "age", "sex", "mmse",
                                 "nwbv", "etiv", "educ", "ses") if c in meta.columns]
        md = md.merge(meta[join_cols].reset_index().rename(
            columns={"index": "visit_id"}), on="visit_id", how="left")
    md = md.sort_values(["visit_id", "view", "slice_idx"]).reset_index(drop=True)
    md.to_csv(out / "metadata.csv", index=False)

    # ---- Subject-level splits (the leakage firewall of this project) --------
    subjects = np.array(sorted(md["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subjects))
    n = len(subjects)
    n_tr = int(round(n * split_ratios["train"]))
    n_va = int(round(n * split_ratios["val"]))
    split_of = {}
    for rank, i in enumerate(perm):
        split_of[subjects[i]] = ("train" if rank < n_tr else
                                 "val" if rank < n_tr + n_va else "test")
    pd.DataFrame({"subject_id": subjects,
                  "split": [split_of[s] for s in subjects]}).to_csv(
        out / "splits.csv", index=False)

    # ---- GroupKFold fold ids over TRAIN subjects ----------------------------
    train_subs = [s for s in subjects if split_of[s] == "train"]
    rng2 = np.random.default_rng(seed + 1)
    order = rng2.permutation(len(train_subs))
    folds = [{"subject_id": train_subs[i], "fold": k % n_folds}
             for k, i in enumerate(order)]
    pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)

    # ---- Intensity stats over a sample of slices (for z-scoring at train) ---
    sample = md.sample(min(2000, len(md)), random_state=seed)
    pix_sum, pix_sq, pix_n = 0.0, 0.0, 0
    for f in sample["file"]:
        a = np.asarray(Image.open(out / f), dtype=np.float32) / 65535.0
        pix_sum += float(a.sum()); pix_sq += float((a ** 2).sum()); pix_n += a.size
    mean = pix_sum / pix_n
    std = float(np.sqrt(max(pix_sq / pix_n - mean ** 2, 1e-8)))
    stats = {"img_size": img_size, "pixel_mean": mean, "pixel_std": std,
             "bg_thresh": bg_thresh, "quantization": "uint16 / 65535"}
    with open(out / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    summary = {
        "n_visits_found": len(visits), "n_visits_used": md["visit_id"].nunique(),
        "n_subjects": int(md["subject_id"].nunique()),
        "n_slices": int(len(md)),
        "slices_per_view": md.groupby("view").size().to_dict(),
        "split_subject_counts": pd.Series([split_of[s] for s in subjects]
                                          ).value_counts().to_dict(),
        "class_counts_subjects": ({str(k): int(v) for k, v in
                                   md.drop_duplicates("subject_id")
                                   ["cdr_class"].value_counts(dropna=False)
                                   .sort_index().items()}
                                  if "cdr_class" in md else {}),
        "skipped": skipped[:50], "n_skipped": len(skipped),
        "runtime_sec": round(time.time() - t0, 1),
        "config": {k: str(v) for k, v in cfg.items()},
    }
    with open(out / "factory_log.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.log(f"Factory done in {summary['runtime_sec']}s — "
            f"{summary['n_slices']} slices / {summary['n_subjects']} subjects")
    return summary


def main() -> None:
    import argparse
    from .utils import apply_overrides
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    summary = run_factory(cfg)
    print(json.dumps({k: summary[k] for k in
                      ("n_visits_used", "n_subjects", "n_slices",
                       "split_subject_counts", "runtime_sec")}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
