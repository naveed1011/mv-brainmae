"""Stage 3 — Downstream evaluation: does the pretraining actually pay off?

Protocol (leakage firewall everywhere):
  * ALL folds are subject-level (GroupKFold via factory folds.csv)
  * Slice probabilities are averaged per subject before scoring
  * The same subject subsets are used across encoders at every label fraction
    (paired comparison — fair by construction)
  * The test split is touched exactly once per arm (final refit rows)

Encoder arms:
  mae_single : our MAE, standard single-view pretraining
  mae_cross  : our MAE, multi-view cross-plane pretraining  (the twist)
  imagenet   : timm ViT-S/16 ImageNet-pretrained (transfer-learning baseline)
  scratch    : same ViT-S/16, random init (lower bound)

Run: python -m mvbrainmae.downstream --config configs/downstream.yaml
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .datasets import FactoryData, imagenet_normalize, load_slice
from .heads import TimmEncoderAdapter
from .mae import build_from_checkpoint
from .metrics import (classification_report, subject_level_predictions,
                      correlations)
from .utils import (RunLogger, ensure_dir, get_device, lr_at, set_seed)

CLASS_NAMES = ["CDR0", "CDR0.5", "CDR1", "CDR2-3"]


# --------------------------------------------------------------------------- #
# Encoder wrappers: unify feature/token APIs + per-arm input normalization
# --------------------------------------------------------------------------- #
class MAEArm(nn.Module):
    kind = "mae"

    def __init__(self, ckpt_path: str, device):
        super().__init__()
        self.model = build_from_checkpoint(ckpt_path).to(device)
        self.dim = self.model.cfg_dict["embed_dim"] * 2  # [CLS || mean]
        # downstream only ever uses the encoder path -> freeze decoder params
        for name, p in self.model.named_parameters():
            if name.startswith(("dec_", "mask_token", "decoder")):
                p.requires_grad = False

    def features(self, x01: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        x = (x01 - mean) / std
        return self.model.extract_features(x, view_id=0)

    def forward_tokens(self, x01: torch.Tensor, mean: float, std: float):
        return self.model.forward_tokens((x01 - mean) / std, view_id=0)


class TimmArm(nn.Module):
    kind = "timm"

    def __init__(self, name: str, pretrained: bool, device, cfg_name: str,
                 img_size: int | None = None):
        super().__init__()
        import timm
        kw = {}
        if img_size is not None and img_size != 224:
            # build the baseline ViT at the native slice size (timm interpolates
            # the pretrained pos-embed) — keeps memory small on non-224 data
            kw["img_size"] = img_size
        m = timm.create_model(cfg_name, pretrained=pretrained, num_classes=0, **kw)
        self.model = TimmEncoderAdapter(m).to(device)
        self.dim = self.model.dim * 2
        self.name = name
        try:
            self.expected_size = int(m.patch_embed.img_size[0])
        except Exception:
            self.expected_size = 224

    def _prep(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-1] != self.expected_size:  # keep baseline arms size-agnostic
            x01 = torch.nn.functional.interpolate(
                x01, size=(self.expected_size, self.expected_size),
                mode="bilinear", align_corners=False)
        x = x01.repeat(1, 3, 1, 1)
        return imagenet_normalize(x)

    def features(self, x01, mean, std):
        t = self.model.forward_tokens(self._prep(x01))
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=-1)

    def forward_tokens(self, x01, mean, std):
        return self.model.forward_tokens(self._prep(x01))


def build_arm(name: str, cfg: dict, device, img_size: int | None = None) -> nn.Module:
    if name.startswith("mae_"):
        mode = name.split("mae_", 1)[1]
        path = Path(cfg["outputs_root"]) / f"pretrain_{mode}" / "encoder.pt"
        if not path.exists():
            raise FileNotFoundError(f"{name}: run pretraining first (missing {path})")
        return MAEArm(str(path), device)
    timm_name = cfg.get("timm_model", "vit_small_patch16_224")
    if name == "imagenet":
        return TimmArm(name, True, device, timm_name, img_size)
    if name == "scratch":
        return TimmArm(name, False, device, timm_name, img_size)
    raise ValueError(f"Unknown encoder arm: {name}")


# --------------------------------------------------------------------------- #
# Slice rows -> raw [0,1] images (normalization applied per-arm)
# --------------------------------------------------------------------------- #
class SliceRows(Dataset):
    def __init__(self, fd: FactoryData, rows: pd.DataFrame):
        self.fd, self.rows = fd, rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = load_slice(self.fd.dir, r["file"], 0.0, 1.0)  # raw [0,1]
        lbl = int(r["cdr_class"]) if pd.notna(r.get("cdr_class")) else -1
        return img, lbl, i


@torch.no_grad()
def extract_features(arm, fd: FactoryData, rows: pd.DataFrame, device,
                     batch_size: int = 256) -> np.ndarray:
    arm.eval()
    ds = SliceRows(fd, rows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=getattr(fd, "num_workers", 2))
    feats = np.zeros((len(rows), arm.dim), dtype=np.float32)
    for imgs, _, idx in dl:
        f = arm.features(imgs.to(device), fd.mean, fd.std)
        feats[idx.numpy()] = f.float().cpu().numpy()
    return feats


def cache_features(cfg: dict, fd: FactoryData, rows: pd.DataFrame, log: RunLogger,
                   device) -> dict[str, Path]:
    """Extract (or reuse cached) features for every arm. Paired rows -> fair."""
    cache_root = ensure_dir(Path(cfg["outputs_root"]) / "features")
    paths = {}
    for name in cfg["encoders"]:
        d = ensure_dir(cache_root / name)
        feat_p, meta_p = d / "feat.npy", d / "meta.csv"
        rows[["file", "subject_id", "cdr_class", "split", "view"]].to_csv(meta_p, index=False)
        if feat_p.exists() and not cfg.get("recompute_features", False):
            log.log(f"[{name}] reusing cached features {feat_p}")
        else:
            t0 = time.time()
            try:
                arm = build_arm(name, cfg, device, fd.img_size)
            except Exception as e:
                log.log(f"[{name}] SKIP (could not build: {e})")
                continue
            feats = extract_features(arm, fd, rows, device,
                                     int(cfg.get("feat_batch_size", 256)))
            np.save(feat_p, feats)
            log.log(f"[{name}] cached {feats.shape} features in "
                    f"{time.time() - t0:.0f}s")
            del arm
            if device.type == "cuda":
                torch.cuda.empty_cache()
        paths[name] = feat_p
    return paths


# --------------------------------------------------------------------------- #
# Linear probes (sklearn) with GroupKFold + label efficiency
# --------------------------------------------------------------------------- #
def _make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0,
                                               class_weight="balanced"))])


def _subject_metrics(probs, labels, subjects) -> dict:
    sdf = subject_level_predictions(probs, labels, subjects)
    rep = classification_report(sdf)
    return rep, sdf


def _full_probs(model_pipe, X) -> np.ndarray:
    """predict_proba mapped back into the FULL label space (a subsample may
    miss a class entirely; columns of predict_proba then shift)."""
    n_classes = len(CLASS_NAMES)
    probs = model_pipe.predict_proba(X)
    classes = model_pipe.named_steps["lr"].classes_
    full = np.zeros((len(probs), n_classes), dtype=probs.dtype)
    full[:, classes] = probs
    return full


def _stratified_subjects(rows: pd.DataFrame, train_subs, frac: float,
                         seed: int) -> set:
    """Pick a subject subsample with every CDR class represented (paired across
    encoder arms — the same subjects for all arms at a given fraction)."""
    rng = np.random.default_rng(seed + int(frac * 1000))
    one = rows[rows.subject_id.isin(train_subs)].drop_duplicates("subject_id")
    chosen = []
    for _, g in one.groupby("cdr_class"):
        k = max(1, int(round(len(g) * frac)))
        chosen.extend(rng.choice(g["subject_id"].values,
                                 size=min(k, len(g)), replace=False))
    return set(chosen)


def run_probes(cfg: dict, fd: FactoryData, rows: pd.DataFrame,
               feat_paths: dict, results_dir: Path, log: RunLogger) -> pd.DataFrame:
    n_classes = int(rows["cdr_class"].max()) + 1
    folds_df = fd.folds
    fold_of = dict(zip(folds_df["subject_id"], folds_df["fold"]))
    train_subs = np.array(sorted(rows.loc[rows.split == "train", "subject_id"].unique()))
    test_mask = rows["split"] == "test"
    fracs = cfg.get("label_fracs", [0.1, 0.25, 0.5, 1.0])
    seed = int(cfg.get("seed", 42))
    n_folds = int(folds_df["fold"].max()) + 1
    idx = {f: i for i, f in enumerate(rows["file"])}

    out_rows = []
    for frac in fracs:
        subs = _stratified_subjects(rows, train_subs, frac, seed)
        sub_rows = rows[rows.subject_id.isin(subs)]

        for name, fp in feat_paths.items():
            X_all = np.load(fp)
            sel = np.array([idx[f] for f in sub_rows["file"]])
            X, y = X_all[sel], sub_rows["cdr_class"].values
            subj = sub_rows["subject_id"].values
            fold = np.array([fold_of.get(s, -1) for s in subj])

            fold_metrics = []
            for kfold in range(n_folds):
                tr, va = fold != kfold, fold == kfold
                if len(np.unique(y[tr])) < 2 or va.sum() == 0:
                    continue
                m = _make_model().fit(X[tr], y[tr])
                probs = _full_probs(m, X[va])
                rep, _ = _subject_metrics(probs, y[va], subj[va])
                fold_metrics.append(rep)
            agg = {k: float(np.mean([fm[k] for fm in fold_metrics]))
                   for k in fold_metrics[0]} if fold_metrics else {}
            agg_std = {k: float(np.std([fm[k] for fm in fold_metrics]))
                       for k in fold_metrics[0]} if fold_metrics else {}

            # final: refit on ALL subsample train subjects, evaluate on TEST once
            m = _make_model().fit(X, y)
            X_te = X_all[np.array([idx[f] for f in rows.loc[test_mask, "file"]])]
            te = rows[test_mask]
            probs_te = _full_probs(m, X_te)
            rep_te, sdf_te = _subject_metrics(probs_te, te["cdr_class"].values,
                                              te["subject_id"].values)
            sdf_te.to_csv(results_dir / f"preds_probe_{name}_frac{frac}.csv",
                          index=False)
            out_rows.append({
                "protocol": "probe", "encoder": name, "label_frac": frac,
                "scope": "cv_mean", **agg,
                **{f"{k}_std": v for k, v in agg_std.items()}})
            out_rows.append({"protocol": "probe", "encoder": name,
                             "label_frac": frac, "scope": "test", **rep_te})
            log.log(f"[probe] {name} frac={frac}: CV bal_acc="
                    f"{agg.get('balanced_accuracy', float('nan')):.3f} "
                    f"test bal_acc={rep_te['balanced_accuracy']:.3f}")
    df = pd.DataFrame(out_rows)
    df.to_csv(results_dir / "downstream_probe.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Fine-tuning arms (encoder + attention head), subject-level GroupKFold
# --------------------------------------------------------------------------- #
def run_finetune(cfg: dict, fd: FactoryData, rows: pd.DataFrame,
                 results_dir: Path, log: RunLogger, device) -> pd.DataFrame:
    ft = cfg.get("finetune", {})
    arms = ft.get("encoders", [])
    if not arms:
        return pd.DataFrame()
    n_classes = int(rows["cdr_class"].max()) + 1
    epochs = int(ft.get("epochs", 8))
    bs = int(ft.get("batch_size", 64))
    enc_lr = float(ft.get("encoder_lr", 3e-4))
    head_mult = float(ft.get("head_mult", 5.0))
    folds_df = fd.folds
    fold_of = dict(zip(folds_df["subject_id"], folds_df["fold"]))
    use_folds = list(range(int(ft.get("folds", 3))))
    view = cfg.get("view", "axial")
    rows_v = rows[rows["view"] == view].reset_index(drop=True)

    out_rows = []
    for name in arms:
        for kfold in use_folds:
            tr_subs = {s for s, f in fold_of.items() if f != kfold}
            va_subs = {s for s, f in fold_of.items() if f == kfold}
            tr_rows = rows_v[rows_v.subject_id.isin(tr_subs)]
            va_rows = rows_v[rows_v.subject_id.isin(va_subs)]
            if len(tr_rows) == 0 or len(va_rows) == 0:
                continue
            set_seed(cfg.get("seed", 42) + kfold)
            try:
                arm = build_arm(name, cfg, device, fd.img_size)
            except Exception as e:
                log.log(f"[finetune:{name}] SKIP ({e})")
                break

            # class weights from SUBJECT counts (imbalance is per-patient)
            sub_cls = tr_rows.drop_duplicates("subject_id")["cdr_class"].values
            counts = np.bincount(sub_cls, minlength=n_classes).astype(np.float32)
            w = torch.tensor((counts.sum() / np.maximum(counts, 1)) /
                             (counts.sum() / np.maximum(counts, 1)).mean(),
                             dtype=torch.float32, device=device)

            ft_model = FineTuneModelEncoder(arm, n_classes).to(device)
            opt = torch.optim.AdamW(
                [{"params": list(arm.parameters()), "lr": enc_lr,
                  "weight_decay": 0.05},
                 {"params": list(ft_model.head.parameters()),
                  "lr": enc_lr * head_mult, "weight_decay": 0.0}], betas=(0.9, 0.95))
            ce = nn.CrossEntropyLoss(weight=w)
            use_amp = device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            tr_dl = DataLoader(SliceRows(fd, tr_rows), batch_size=bs, shuffle=True,
                               num_workers=2, drop_last=True)
            steps_per_epoch = max(1, len(tr_dl))
            total = epochs * steps_per_epoch
            step = 0
            best = {"balanced_accuracy": -1}
            for ep in range(epochs):
                ft_model.train()
                for imgs, lbl, _ in tr_dl:
                    for gi, g in enumerate(opt.param_groups):
                        base_lr = enc_lr if gi == 0 else enc_lr * head_mult
                        g["lr"] = lr_at(step, total, base_lr, warmup_frac=0.1)
                    imgs = imgs.to(device)
                    with torch.autocast(device.type, enabled=use_amp):
                        logits = ft_model(imgs, fd.mean, fd.std)
                        loss = ce(logits, lbl.to(device))
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                    step += 1
                rep, sdf = eval_subject_level(ft_model, fd, va_rows, device,
                                              fd.mean, fd.std, n_classes)
                if rep["balanced_accuracy"] > best["balanced_accuracy"]:
                    best = {**rep, "epoch": ep + 1}
            sdf.to_csv(results_dir /
                       f"preds_ft_{name}_fold{kfold}.csv", index=False)
            out_rows.append({"protocol": "finetune", "encoder": name,
                             "fold": kfold, **best})
            log.log(f"[finetune] {name} fold{kfold}: best bal_acc="
                    f"{best['balanced_accuracy']:.3f} (epoch {best.get('epoch')})")
            del ft_model, arm, opt, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(out_rows)
    if len(df):
        df.to_csv(results_dir / "downstream_finetune.csv", index=False)
    return df


class FineTuneModelEncoder(nn.Module):
    """Attention-pool classifier over an arm's full-image tokens."""

    def __init__(self, arm, n_classes: int):
        super().__init__()
        self.arm = arm
        from .heads import AttentionPoolHead
        self.head = AttentionPoolHead(arm.dim // 2, n_classes)

    def forward(self, x01, mean, std):
        tokens = self.arm.forward_tokens(x01, mean, std)
        return self.head(tokens[:, 1:])


@torch.no_grad()
def eval_subject_level(ft_model, fd, rows, device, mean, std, n_classes):
    ft_model.eval()
    dl = DataLoader(SliceRows(fd, rows), batch_size=128, shuffle=False,
                    num_workers=0)
    probs, ys = [], []
    for imgs, lbl, _ in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = ft_model(imgs.to(device), mean, std)
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        ys.append(lbl.numpy())
    subs = rows["subject_id"].values  # shuffle=False -> loader preserves row order
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    sdf = subject_level_predictions(probs, ys, subs)
    return classification_report(sdf), sdf


# --------------------------------------------------------------------------- #
# Sanity probe: can frozen features predict AGE? (encoder quality control)
# --------------------------------------------------------------------------- #
def run_age_probe(cfg: dict, fd: FactoryData, rows: pd.DataFrame,
                  feat_paths: dict, results_dir: Path, log: RunLogger) -> pd.DataFrame:
    if "age" not in rows.columns or rows["age"].isna().all():
        return pd.DataFrame()
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    idx = {f: i for i, f in enumerate(rows["file"])}
    tr = rows[rows.split == "train"]; te = rows[rows.split == "test"]
    # one slice per subject (middle axial) to keep subjects independent
    def per_subject(df):
        d = df[df.view == "axial"].reset_index(drop=True)
        pos = (d["rel_pos"] - 0.5).abs().groupby(d["subject_id"]).idxmin()
        return d.iloc[pos.values]
    tr_s, te_s = per_subject(tr), per_subject(te)
    out = []
    for name, fp in feat_paths.items():
        X = np.load(fp)
        m = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])
        m.fit(X[[idx[f] for f in tr_s["file"]]], tr_s["age"].values)
        pred = m.predict(X[[idx[f] for f in te_s["file"]]])
        c = correlations(pred, te_s["age"].values)
        out.append({"encoder": name, **c,
                    "mae_years": float(np.mean(np.abs(pred - te_s["age"].values)))})
        pd.DataFrame({"encoder": name, "subject": te_s["subject_id"].values,
                      "age_true": te_s["age"].values, "age_pred": pred}).to_csv(
            results_dir / f"age_probe_{name}.csv", index=False)
        log.log(f"[age-probe] {name}: pearson r={c['pearson_r']:.3f} "
                f"MAE={out[-1]['mae_years']:.1f}y")
    df = pd.DataFrame(out)
    if len(df):
        df.to_csv(results_dir / "downstream_age.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
def run_downstream(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    results_dir = ensure_dir(cfg["results_dir"])
    log = RunLogger(results_dir, "downstream")

    fd = FactoryData(cfg["factory_dir"])
    fd.num_workers = int(cfg.get("num_workers", 2))
    rows = fd.rows(labeled_only=True, view=cfg.get("view", "axial"))
    log.log(f"Downstream rows: {len(rows)} slices / "
            f"{rows.subject_id.nunique()} subjects / device={device}")
    if len(rows) == 0:
        raise RuntimeError("No labeled slices found — check factory metadata join.")

    feat_paths = cache_features(cfg, fd, rows, log, device)
    probe_df = run_probes(cfg, fd, rows, feat_paths, results_dir, log)
    ft_df = run_finetune(cfg, fd, rows, results_dir, log, device)
    age_df = run_age_probe(cfg, fd, rows, feat_paths, results_dir, log)

    summary = {"n_rows": len(rows), "encoders": list(feat_paths),
               "probe_rows": len(probe_df), "finetune_rows": len(ft_df),
               "age_rows": len(age_df)}
    with open(results_dir / "downstream_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.log(f"Downstream done: {summary}")
    return summary


def main() -> None:
    import argparse
    from .utils import apply_overrides, load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    run_downstream(apply_overrides(load_config(args.config), args.overrides))


if __name__ == "__main__":
    main()
