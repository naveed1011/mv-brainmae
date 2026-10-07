"""Generate a small SYNTHETIC OASIS-style dataset for local smoke testing.

Why this exists: before spending any Kaggle GPU hours, the ENTIRE pipeline
(factory -> pretrain -> downstream -> anomaly -> analysis) must be verifiable
on a laptop/CPU. The synthetic brains carry a PLANTED signal — ventricle size
grows and cortex thins with CDR class, nWBV decreases with class — so a
working pipeline MUST detect it. If downstream AUC/accuracy on synthetic data
is at chance, the code is broken, not the science.

Usage: python scripts/make_synthetic_data.py --out data/synthetic --n 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

CLASSES = [0.0, 0.5, 1.0, 2.0]


def make_brain(rng: np.random.Generator, cls_idx: int, shape=(64, 72, 64)) -> np.ndarray:
    """Synthetic T1-like volume in [0,1]; atrophy signal correlates with class."""
    nx, ny, nz = shape
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    r_head = np.sqrt((x / 0.85) ** 2 + (y / 0.95) ** 2 + (z / 0.85) ** 2)

    vol = np.zeros(shape, dtype=np.float32)
    head = r_head < 1.0
    # white matter core + cortical rim (cortex thins with class)
    cortex_inner = 0.88 - 0.035 * cls_idx          # smaller -> thinner cortex ring
    vol[head] = 0.62                                # white matter base
    rim = head & (r_head > cortex_inner)
    vol[rim] = 0.85                                 # gray matter brighter
    # lateral ventricles: two dark blobs, radius grows with class (THE signal)
    vr = 0.10 + 0.045 * cls_idx + rng.normal(0, 0.008)
    for sx in (-0.22, 0.22):
        d = np.sqrt(((x - sx) / (vr * 1.6)) ** 2 + (y / (vr * 2.2)) ** 2
                    + ((z - 0.05) / vr) ** 2)
        vol[d < 1.0] = 0.10
    # smooth bias field + noise + texture
    bias = 1.0 + 0.06 * np.sin(2.1 * x + 0.7) * np.cos(1.7 * z)
    vol = vol * bias + rng.normal(0, 0.015, shape).astype(np.float32)
    vol[~head] = 0.0
    return np.clip(vol, 0, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=32, help="number of subjects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        rng = np.random.default_rng(args.seed + i)
        cls_idx = i % len(CLASSES)
        cdr = CLASSES[cls_idx]
        vol = make_brain(rng, cls_idx)
        sid = f"SYN{i + 1:04d}"
        visit = f"{sid}_MR1"
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # RAS+ 2mm voxels
        nib.save(nib.Nifti1Image(vol, affine), out / f"{visit}.nii.gz")
        rows.append({
            "ID": visit, "M/F": "F" if i % 2 else "M", "Hand": "R",
            "Age": round(float(62 + 6 * cls_idx + rng.normal(0, 3)), 1),
            "Educ": int(rng.integers(8, 21)), "SES": int(rng.integers(1, 6)),
            "MMSE": round(float(np.clip(30 - 4 * cls_idx + rng.normal(0, 1.5),
                                        4, 30)), 1),
            "CDR": cdr,
            "eTIV": round(float(1450 + rng.normal(0, 90)), 1),
            "nWBV": round(float(0.80 - 0.055 * cls_idx + rng.normal(0, 0.012)), 4),
            "ASF": round(float(1600 / (1450 + rng.normal(0, 90))), 3),
        })
    pd.DataFrame(rows).to_csv(out / "synthetic_cross-sectional.csv", index=False)
    dist = pd.DataFrame(rows)["CDR"].value_counts().sort_index().to_dict()
    print(f"Wrote {args.n} synthetic subjects to {out} | CDR distribution: {dist}")


if __name__ == "__main__":
    main()
