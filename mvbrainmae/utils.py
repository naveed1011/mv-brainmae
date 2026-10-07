"""Shared utilities: config loading, seeding, logging, checkpointing, schedules."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `--set key=value` / `--set a.b=value` CLI overrides (YAML-parsed values)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        val = yaml.safe_load(raw)  # parses ints/floats/bools/lists; falls back to str
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: we intentionally do NOT enable cudnn.deterministic — it slows T4
    # training a lot. Seeds + fixed data order give run-to-run comparability;
    # exact bitwise determinism is not required for this study's claims.


# --------------------------------------------------------------------------- #
# Device / timing
# --------------------------------------------------------------------------- #
def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeGuard:
    """Stops training before Kaggle's 12h session limit; checkpoint stays resumable."""

    def __init__(self, max_minutes: float | None):
        self.t0 = time.time()
        self.max_sec = (max_minutes or 1e9) * 60

    def expired(self) -> bool:
        return (time.time() - self.t0) > self.max_sec

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60


# --------------------------------------------------------------------------- #
# Logging (plain-text + JSONL so results are machine-verifiable)
# --------------------------------------------------------------------------- #
class RunLogger:
    def __init__(self, out_dir: str | Path, name: str = "run"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / f"{name}.jsonl"
        self.txt_path = self.dir / f"{name}.log"

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.txt_path, "a") as f:
            f.write(line + "\n")

    def log_json(self, record: dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0,
                    scaler=None, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        obj["scaler"] = scaler.state_dict()
    if extra:
        obj.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic: never leave a half-written checkpoint


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None,
                    map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def save_encoder(path: str | Path, model) -> None:
    """Save ONLY the encoder (+ patch embed / pos embed / view embed / cls token)
    so downstream stages never depend on the decoder."""
    enc_keys = {}
    for k, v in model.state_dict().items():
        if k.startswith("decoder.") or k.startswith("decoder_"):
            continue
        enc_keys[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": enc_keys, "config": getattr(model, "cfg_dict", {})}, path)


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def lr_at(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.05,
          min_lr_frac: float = 0.01) -> float:
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * step / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
