"""Readers for OASIS-1-style MRI collections (also handles synthetic/IXI layouts).

OASIS-1 cross-sectional disk layout (official, from oasisbrains.org):

    <root>/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/
        OAS1_0001_MR1_mpr_n3_anon_111_t88_masked_gfc.img (+ .hdr)   <- preferred
    <root>/oasis_cross-sectional.csv                                  <- labels

Volume preference order:
  1. *_t88_masked_gfc*  : Talairach-registered + skull-stripped  (consistent
                          frame across subjects; ideal for cross-view pairing)
  2. *mpr_n3_anon*      : N3 bias-corrected native space (we reorient to RAS)
  3. any *.nii.gz / *.nii / *.img : generic fallback (IXI, synthetic data)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

VISIT_RE = re.compile(r"(OAS\d_\d{4}_MR\d)")
GENERIC_SUBJECT_RE = re.compile(r"^([A-Za-z]+[\-_]?\d+)")

# OASIS-1 cross-sectional CSV uses 'M/F' and 'ID' columns; we normalize.
CDR_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 3}
CLASS_NAMES = ["CDR0 (non-demented)", "CDR0.5 (very mild)",
               "CDR1 (mild)", "CDR2-3 (moderate+)"]


@dataclass
class Visit:
    visit_id: str          # e.g. OAS1_0001_MR1 or SYN0001_MR1
    subject_id: str        # e.g. OAS1_0001  (grouping key -> no leakage)
    volume_path: Path
    kind: str              # t88_masked | n3 | generic


def discover_visits(root: str | Path) -> list[Visit]:
    """Find one volume per visit directory under `root`, by preference order."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    candidates: dict[str, tuple[int, Path]] = {}  # visit_id -> (rank, path)

    def offer(rank: int, path: Path, visit_id: str | None = None,
              subject_id: str | None = None) -> None:
        if visit_id is None:
            m = VISIT_RE.search(str(path))
            visit_id = m.group(1) if m else path.stem.split(".")[0]
        cur = candidates.get(visit_id)
        if cur is None or rank < cur[0]:
            candidates[visit_id] = (rank, path)

    # Rank 1: skull-stripped Talairach (OASIS-1 processed)
    for p in sorted(root.rglob("*masked_gfc.img")):
        offer(1, p)
    for p in sorted(root.rglob("*masked_gfc.nii*")):
        offer(1, p)
    # Rank 2: N3-corrected native
    for p in sorted(root.rglob("*mpr_n3_anon*.img")):
        if "t88" not in p.name:
            offer(2, p)
    for p in sorted(root.rglob("*mpr_n3_anon*.nii*")):
        if "t88" not in p.name:
            offer(2, p)
    # Rank 3: generic volumes (IXI / synthetic)
    for pat in ("*.nii.gz", "*.nii", "*.img"):
        for p in sorted(root.rglob(pat)):
            if "masked_gfc" in p.name or "mpr_n3_anon" in p.name:
                continue
            offer(3, p)

    visits = []
    for visit_id, (rank, path) in sorted(candidates.items()):
        m = VISIT_RE.search(visit_id)
        subject_id = m.group(1).rsplit("_MR", 1)[0] if m else _generic_subject(path)
        kind = {1: "t88_masked", 2: "n3", 3: "generic"}[rank]
        visits.append(Visit(visit_id, subject_id, path, kind))
    return visits


def _generic_subject(path: Path) -> str:
    stem = path.name.split(".")[0]
    m = VISIT_RE.search(stem)
    if m:
        return m.group(1).rsplit("_MR", 1)[0]
    m = GENERIC_SUBJECT_RE.match(stem)
    return m.group(1) if m else stem


def load_metadata(root: str | Path) -> pd.DataFrame:
    """Load the cross-sectional metadata CSV (OASIS-1 format) if present.

    Returns df indexed by visit_id with columns: subject_id, age, sex, mmse,
    cdr, cdr_class, nwbv, etiv, educ, ses. Missing file -> empty frame.
    """
    root = Path(root)
    csv = None
    for name in ("oasis_cross-sectional.csv", "oasis_longitudinal.csv",
                 "metadata.csv", "synthetic_cross-sectional.csv"):
        cand = list(root.rglob(name))
        if cand:
            csv = sorted(cand)[0]
            break
    if csv is None:
        return pd.DataFrame()

    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    df = df.rename(columns={id_col: "visit_id", "M/F": "sex", "nWBV": "nwbv",
                            "eTIV": "etiv", "Educ": "educ", "SES": "ses",
                            "MMSE": "mmse", "CDR": "cdr", "Age": "age"})
    keep = [c for c in ("visit_id", "age", "sex", "mmse", "cdr", "nwbv",
                        "etiv", "educ", "ses") if c in df.columns]
    df = df[keep].copy()
    df["subject_id"] = df["visit_id"].astype(str).str.extract(
        r"(OAS\d_\d{4}|[A-Za-z]+[\-_]?\d+)", expand=False)
    if "cdr" in df.columns:
        df["cdr"] = pd.to_numeric(df["cdr"], errors="coerce")
        df["cdr_class"] = df["cdr"].map(CDR_TO_CLASS).astype("Int64")
    return df.set_index("visit_id")


def load_volume_ras(path: str | Path) -> np.ndarray:
    """Load any supported volume -> float32 array in canonical RAS+ orientation.

    RAS+: axis0 = Left->Right, axis1 = Posterior->Anterior, axis2 = Inferior->Superior.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape} for {path}")
    return arr
