"""Stage 2 — Masked pretraining (single-view MAE vs multi-view cross-plane MAE).

Designed for Kaggle free-tier GPU reality:
  * AMP autocast on CUDA
  * resumable checkpoints (atomic writes) — a session can die at hour 11 safely
  * --set max_minutes=600 style time guard: stops cleanly, ckpt stays resumable
  * max_steps_per_epoch for smoke tests
  * JSONL logs of every step/epoch -> machine-verifiable training record

Run:  python -m mvbrainmae.pretrain --config configs/pretrain_cross.yaml
"""
from __future__ import annotations

import itertools
import json
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FactoryData, PretrainDataset, load_slice, pretrain_collate
from .mae import MaskedAutoencoder
from .utils import (RunLogger, TimeGuard, count_params, ensure_dir, get_device,
                    load_checkpoint, lr_at, save_checkpoint, save_encoder,
                    set_seed)


def build_loader(fd: FactoryData, cfg: dict, mode: str, split: str = "train"):
    ds = PretrainDataset(fd, mode=mode, split=split,
                         adjacent_delta=int(cfg.get("adjacent_delta", 2)),
                         seed=int(cfg.get("seed", 42)))
    return DataLoader(
        ds, batch_size=int(cfg["batch_size"]), shuffle=(mode == "single"),
        num_workers=int(cfg.get("num_workers", 2)), drop_last=True,
        collate_fn=partial(pretrain_collate, mode=("single" if mode == "single"
                                                   else "cross")),
        persistent_workers=int(cfg.get("num_workers", 2)) > 0)


def param_groups(model: torch.nn.Module, lr: float, wd: float):
    """MAE convention: no weight decay on biases, norms, embeddings, pos/cls/mask tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim <= 1 or "pos_embed" in name or "view_embed" in name
                or "cls_token" in name or "mask_token" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0}]


@torch.no_grad()
def eval_reconstruction(model, val_imgs: torch.Tensor, device) -> float:
    model.eval()
    x = val_imgs.to(device)
    x_hat = model.reconstruct(x, view_id=0)
    mse = float(torch.mean((x - x_hat) ** 2))
    model.train()
    return mse


def save_recon_samples(model, val_imgs: torch.Tensor, device, path: Path,
                       mean: float, std: float, epoch: int) -> None:
    """Grid: rows = [original | reconstruction | |error|] for 4 val slices."""
    model.eval()
    x = val_imgs[:4].to(device)
    x_hat = model.reconstruct(x, view_id=0).cpu()
    x = x.cpu()
    to01 = lambda t: (t * std + mean).clip(0, 1).squeeze(1)  # noqa: E731
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.6))
    err = (to01(x) - to01(x_hat)).abs()
    for c in range(4):
        axes[0, c].imshow(to01(x)[c], cmap="gray"); axes[0, c].set_title("original")
        axes[1, c].imshow(to01(x_hat)[c], cmap="gray"); axes[1, c].set_title("recon")
        im = axes[2, c].imshow(err[c], cmap="inferno")
        axes[2, c].set_title("|error|"); plt.colorbar(im, ax=axes[2, c], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Reconstruction samples — epoch {epoch}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    model.train()


def run_pretrain(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    mode = str(cfg["mode"])                       # single | cross | adjacent
    out = ensure_dir(cfg["out_dir"])
    log = RunLogger(out, f"pretrain_{mode}")
    device = get_device(cfg.get("device", "auto"))
    guard = TimeGuard(cfg.get("max_minutes"))

    fd = FactoryData(cfg["factory_dir"])
    loader = build_loader(fd, cfg, mode, split=cfg.get("train_split", "train"))
    max_steps = cfg.get("max_steps_per_epoch")

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("img_size", fd.img_size)
    model = MaskedAutoencoder(**model_cfg).to(device)
    n_params = count_params(model)
    log.log(f"mode={mode} device={device} params={n_params / 1e6:.2f}M "
            f"slices(train)={len(fd.rows(split='train'))}")

    opt = torch.optim.AdamW(param_groups(model, float(cfg["lr"]),
                                         float(cfg.get("weight_decay", 0.05))),
                            betas=(0.9, 0.95))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    ckpt_path = Path(cfg.get("resume_ckpt", out / "ckpt.pt"))
    if cfg.get("resume", True) and Path(ckpt_path).exists():
        ck = load_checkpoint(ckpt_path, model, opt, scaler)
        start_epoch = int(ck.get("epoch", 0)) + 1
        log.log(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    # fixed val batch for reconstruction monitoring + sample grids
    val_rows = fd.rows(split="val", view="axial")
    if len(val_rows) == 0:
        val_rows = fd.rows(split="train", view="axial").head(8)
    vr = val_rows.sample(min(8, len(val_rows)), random_state=0)
    val_imgs = torch.stack([load_slice(fd.dir, f, fd.mean, fd.std)
                            for f in vr["file"]])

    epochs = int(cfg["epochs"])
    steps_per_epoch = len(loader) if not max_steps else min(int(max_steps), len(loader))
    total_steps = epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    model.train()
    stop_reason = "completed"

    for epoch in range(start_epoch, epochs):
        t_ep, run_loss, n_seen = time.time(), 0.0, 0
        batch_iter = itertools.islice(loader, steps_per_epoch) if max_steps else loader
        for views, view_ids in batch_iter:
            for lr_g in opt.param_groups:      # cosine schedule w/ warmup
                lr_g["lr"] = lr_at(step, total_steps, float(cfg["lr"]),
                                   warmup_frac=float(cfg.get("warmup_frac", 0.05)))
            views = [v.to(device, non_blocking=True) for v in views]
            with torch.autocast(device.type, enabled=use_amp):
                loss, _ = model(views, view_ids)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = views[0].shape[0] * len(views)
            run_loss += float(loss) * bs; n_seen += bs; step += 1
            if step % int(cfg.get("log_every", 50)) == 0:
                log.log_json({"event": "step", "mode": mode, "step": step,
                              "epoch": epoch, "loss": float(loss),
                              "lr": opt.param_groups[0]["lr"],
                              "imgs_per_s": n_seen / max(time.time() - t_ep, 1e-9)})
            if guard.expired():
                stop_reason = f"time_guard_{cfg.get('max_minutes')}min"
                break
        ep_loss = run_loss / max(n_seen, 1)
        log.log(f"epoch {epoch + 1}/{epochs} loss={ep_loss:.5f} "
                f"({time.time() - t_ep:.0f}s, elapsed {guard.elapsed_min():.0f}min)")
        log.log_json({"event": "epoch", "mode": mode, "epoch": epoch,
                      "loss": ep_loss, "elapsed_min": guard.elapsed_min()})

        if (epoch + 1) % int(cfg.get("eval_every", 2)) == 0 or epoch == epochs - 1:
            mse = eval_reconstruction(model, val_imgs, device)
            log.log_json({"event": "recon", "mode": mode, "epoch": epoch,
                          "val_recon_mse": mse})
            log.log(f"  val recon MSE (normalized space): {mse:.5f}")
            save_recon_samples(model, val_imgs, device,
                               out / f"recon_samples_e{epoch + 1:03d}.png",
                               fd.mean, fd.std, epoch + 1)

        if (epoch + 1) % int(cfg.get("save_every", 5)) == 0 or epoch == epochs - 1:
            save_checkpoint(out / "ckpt.pt", model, opt, epoch, scaler,
                            extra={"mode": mode})
        if guard.expired():
            break

    save_checkpoint(out / "ckpt.pt", model, opt, epochs - 1, scaler,
                    extra={"mode": mode})
    save_encoder(out / "encoder.pt", model)
    summary = {"mode": mode, "epochs_run": f"{start_epoch}->{epochs}",
               "stop_reason": stop_reason, "params_M": round(n_params / 1e6, 3),
               "device": str(device), "torch": torch.__version__,
               "seed": cfg.get("seed", 42), "minutes": round(guard.elapsed_min(), 1),
               "config": cfg}
    with open(out / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.log(f"Done. stop_reason={stop_reason} encoder -> {out / 'encoder.pt'}")
    return summary


def main() -> None:
    import argparse
    from .utils import apply_overrides, load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    s = run_pretrain(cfg)
    print(json.dumps({k: s[k] for k in ("mode", "epochs_run", "stop_reason",
                                        "params_M", "minutes")}, indent=2))


if __name__ == "__main__":
    main()
