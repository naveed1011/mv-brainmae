"""PyTorch datasets over factory outputs.

Three pretraining modes (the experimental axis of this project):
  single    : standard MAE on one randomly chosen slice (axial or coronal)
  cross     : MULTI-VIEW twist — an axial and a coronal slice of the SAME
              subject are both 75%-masked; each view's visible tokens serve as
              context for reconstructing the other view. The encoder must learn
              cross-plane 3D anatomical consistency, not just in-plane texture.
  adjacent  : 2.5-D ablation — target axial slice + a NEIGHBOURING axial slice
              as context (controls for 'any extra context helps' vs the
              cross-plane mechanism specifically).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Factory output container
# --------------------------------------------------------------------------- #
class FactoryData:
    def __init__(self, factory_dir: str | Path):
        self.dir = Path(factory_dir)
        self.metadata = pd.read_csv(self.dir / "metadata.csv")
        with open(self.dir / "stats.json") as f:
            self.stats = json.load(f)
        sp = pd.read_csv(self.dir / "splits.csv")
        self.splits = dict(zip(sp["subject_id"], sp["split"]))
        self.metadata["split"] = self.metadata["subject_id"].map(self.splits)
        folds_path = self.dir / "folds.csv"
        self.folds = pd.read_csv(folds_path) if folds_path.exists() else None
        self.img_size = int(self.stats["img_size"])
        self.mean = float(self.stats["pixel_mean"])
        self.std = float(self.stats["pixel_std"])

    def rows(self, split: str | None = None, view: str | None = None,
             labeled_only: bool = False) -> pd.DataFrame:
        df = self.metadata
        if split is not None:
            df = df[df["split"] == split]
        if view is not None:
            df = df[df["view"] == view]
        if labeled_only:
            df = df[df["cdr_class"].notna()]
        return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_slice(factory_dir: Path, rel_path: str, mean: float, std: float) -> torch.Tensor:
    """PNG16 -> float32 tensor (1, H, W), z-scored with factory stats."""
    a = np.asarray(Image.open(Path(factory_dir) / rel_path), dtype=np.float32)
    a /= 65535.0
    a = (a - mean) / std
    return torch.from_numpy(a).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Pretraining datasets
# --------------------------------------------------------------------------- #
class PretrainDataset(Dataset):
    """mode: 'single' | 'cross' | 'adjacent' (see module docstring)."""

    def __init__(self, fd: FactoryData, mode: str = "cross",
                 split: str = "train", adjacent_delta: int = 2,
                 seed: int = 42):
        assert mode in ("single", "cross", "adjacent")
        self.fd, self.mode = fd, mode
        self.rng = np.random.default_rng(seed)
        rows = fd.rows(split=split)

        if mode == "single":
            self.items = rows[["file"]].to_dict("records")
        elif mode == "cross":
            self.by_visit_ax = {v: g["file"].tolist() for v, g in
                                rows[rows["view"] == "axial"].groupby("visit_id")}
            self.by_visit_co = {v: g["file"].tolist() for v, g in
                                rows[rows["view"] == "coronal"].groupby("visit_id")}
            self.visits = sorted(set(self.by_visit_ax) & set(self.by_visit_co))
            if not self.visits:
                raise RuntimeError("cross mode needs both axial and coronal slices")
            self.n_per_visit_ax = {v: len(self.by_visit_ax[v]) for v in self.visits}
            self.items = [None] * int(sum(self.n_per_visit_ax.values()))
        else:  # adjacent
            ax = rows[rows["view"] == "axial"].sort_values(
                ["visit_id", "slice_idx"]).reset_index(drop=True)
            self.axial = ax[["visit_id", "slice_idx", "file"]].to_dict("records")
            self.delta = adjacent_delta
            self.items = [None] * len(self.axial)

    def __len__(self) -> int:
        return len(self.items)

    def _img(self, rel: str) -> torch.Tensor:
        return load_slice(self.fd.dir, rel, self.fd.mean, self.fd.std)

    def __getitem__(self, i: int):
        if self.mode == "single":
            return self._img(self.items[i]["file"]), torch.tensor(0)

        if self.mode == "cross":
            v = self.visits[i % len(self.visits)]
            fa = self.rng.choice(self.by_visit_ax[v])
            fc = self.rng.choice(self.by_visit_co[v])
            return self._img(fa), self._img(fc)  # view ids 0, 1 assigned in loop

        # adjacent: same-plane neighbour as context
        rec = self.axial[i]
        d = int(self.rng.integers(1, self.delta + 1)) * (
            1 if self.rng.random() < 0.5 else -1)
        j = int(np.clip(i + d, 0, len(self.axial) - 1))
        while self.axial[j]["visit_id"] != rec["visit_id"] and j != i:
            j = i  # fall back to self-pair if at visit boundary
        return self._img(rec["file"]), self._img(self.axial[j]["file"])


def pretrain_collate(batch, mode: str):
    """Packs samples into the multi-view list format the MAE expects."""
    if mode == "single":
        imgs = torch.stack([b[0] for b in batch])
        return [imgs], [0]
    a = torch.stack([b[0] for b in batch])
    c = torch.stack([b[1] for b in batch])
    return [a, c], [0, 1]


def imagenet_normalize(x01: torch.Tensor) -> torch.Tensor:
    """For the timm ImageNet-pretrained baseline arm only (3-channel input)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((x01 - mean) / std)
