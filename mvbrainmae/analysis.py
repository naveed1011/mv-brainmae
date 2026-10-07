"""Stage 5 — Analysis: aggregate every result CSV/JSONL into figures + verdicts.

Produces (under cfg['figures_dir'] / cfg['results_dir']):
  fig_pretrain_loss.png      single vs cross reconstruction-loss curves
  fig_label_efficiency.png   subject-level balanced accuracy vs label fraction
  fig_main_bars.png          CV + TEST balanced accuracy per encoder (probe & FT)
  fig_confusion_test.png     confusion matrix of best probe arm on TEST
  fig_anomaly.png            AUC bars + score distributions + nWBV correlation
  fig_age_probe.png          age prediction scatter (encoder sanity check)
  summary.md                 all tables + hypothesis verdicts (H1..H4 + H0 gate)

Run: python -m mvbrainmae.analysis --config configs/analysis.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import RunLogger, ensure_dir

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

ARM_ORDER = ["scratch", "imagenet", "mae_single", "mae_cross"]
ARM_COLORS = {"scratch": "#9e9e9e", "imagenet": "#4c72b0",
              "mae_single": "#dd8452", "mae_cross": "#c44e52"}
CLASS_NAMES = ["CDR0", "CDR0.5", "CDR1", "CDR2-3"]


def _arm_sort(names):
    return sorted(names, key=lambda n: ARM_ORDER.index(n) if n in ARM_ORDER else 99)


# --------------------------------------------------------------------------- #
def fig_pretrain(cfg: dict, figures: Path, log: RunLogger) -> None:
    curves = {}
    for mode in cfg.get("pretrain_modes", ["single", "cross"]):
        d = Path(cfg["outputs_root"]) / f"pretrain_{mode}"
        jp = d / f"pretrain_{mode}.jsonl"
        if not jp.exists():
            continue
        recs = [json.loads(l) for l in open(jp)]
        ep = [r for r in recs if r.get("event") == "epoch"]
        if ep:
            curves[mode] = ([r["epoch"] + 1 for r in ep], [r["loss"] for r in ep])
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", ms=3, label=f"MAE-{mode}")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked reconstruction loss")
    ax.set_title("Pretraining curves (train loss)")
    ax.legend(); fig.tight_layout()
    fig.savefig(figures / "fig_pretrain_loss.png"); plt.close(fig)
    log.log("wrote fig_pretrain_loss.png")


def _probe_df(results: Path) -> pd.DataFrame:
    p = results / "downstream_probe.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fig_label_efficiency(results: Path, figures: Path, log: RunLogger) -> None:
    df = _probe_df(results)
    if df.empty:
        return
    cv = df[(df["protocol"] == "probe") & (df["scope"] == "cv_mean")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in _arm_sort(cv["encoder"].unique()):
        g = cv[cv.encoder == name].sort_values("label_frac")
        ax.errorbar(g["label_frac"], g["balanced_accuracy"],
                    yerr=g.get("balanced_accuracy_std", 0), marker="o",
                    color=ARM_COLORS.get(name), label=name)
    ax.set_xscale("log"); ax.set_xticks(sorted(cv["label_frac"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("fraction of training SUBJECTS labeled")
    ax.set_ylabel("subject-level balanced acc (CV mean)")
    ax.set_title("Label efficiency — frozen features + linear probe")
    ax.legend(); fig.tight_layout()
    fig.savefig(figures / "fig_label_efficiency.png"); plt.close(fig)
    log.log("wrote fig_label_efficiency.png")


def fig_main_bars(results: Path, figures: Path, log: RunLogger) -> None:
    df = _probe_df(results)
    ftp = results / "downstream_finetune.csv"
    if df.empty:
        return
    test = df[(df["protocol"] == "probe") & (df["scope"] == "test")
              & (df["label_frac"] == df["label_frac"].max())]
    arms = _arm_sort(set(test["encoder"]) |
                     (set(pd.read_csv(ftp)["encoder"]) if ftp.exists() else set()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(arms)); w = 0.35
    axes[0].bar(x - w / 2, test.set_index("encoder")
                .reindex(arms)["balanced_accuracy"], w,
                color=[ARM_COLORS.get(a) for a in arms], label="probe (TEST)")
    cv = df[(df["protocol"] == "probe") & (df["scope"] == "cv_mean")
            & (df["label_frac"] == df["label_frac"].max())]
    axes[0].bar(x + w / 2, cv.set_index("encoder")
                .reindex(arms)["balanced_accuracy"], w, alpha=0.55,
                yerr=cv.set_index("encoder").reindex(arms)
                .get("balanced_accuracy_std"),
                color=[ARM_COLORS.get(a) for a in arms], label="probe (CV mean)")
    axes[0].set_xticks(x, arms, rotation=15); axes[0].legend()
    axes[0].set_ylabel("subject-level balanced acc")
    axes[0].set_title("Frozen-feature probes")
    axes[0].axhline(0.25, ls="--", c="k", lw=1)
    axes[0].text(0, 0.26, "4-class chance", fontsize=8)
    if ftp.exists():
        ftd = pd.read_csv(ftp)
        agg = ftd.groupby("encoder")["balanced_accuracy"].agg(["mean", "std"])
        arms2 = _arm_sort(agg.index)
        agg = agg.reindex(arms2)
        axes[1].bar(np.arange(len(arms2)), agg["mean"],
                    yerr=agg["std"], color=[ARM_COLORS.get(a) for a in arms2])
        axes[1].set_xticks(np.arange(len(arms2)), arms2, rotation=15)
    axes[1].set_title("Fine-tuned (GroupKFold val)")
    axes[1].set_ylabel("subject-level balanced acc")
    axes[1].axhline(0.25, ls="--", c="k", lw=1)
    fig.tight_layout(); fig.savefig(figures / "fig_main_bars.png"); plt.close(fig)
    log.log("wrote fig_main_bars.png")


def fig_confusion(results: Path, figures: Path, log: RunLogger) -> None:
    cands = sorted(results.glob("preds_probe_*frac1.0.csv"))
    if not cands:
        return
    best, best_acc = None, -1
    for c in cands:
        sdf = pd.read_csv(c)
        acc = (sdf.true_label == sdf.pred_label).mean()
        if acc > best_acc:
            best, best_acc, = c, acc
    sdf = pd.read_csv(best)
    labels = sorted(set(sdf.true_label) | set(sdf.pred_label))
    cm = pd.crosstab(sdf.true_label, sdf.pred_label)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), [CLASS_NAMES[i] if i < 4 else i for i in labels],
                  rotation=30)
    ax.set_yticks(range(len(labels)), [CLASS_NAMES[i] if i < 4 else i for i in labels])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "k")
    ax.set_title(f"TEST confusion — {best.stem.replace('preds_probe_', '')}")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(figures / "fig_confusion_test.png"); plt.close(fig)
    log.log("wrote fig_confusion_test.png")


def fig_anomaly(results: Path, figures: Path, log: RunLogger) -> None:
    mp = results / "anomaly_metrics.json"
    if not mp.exists():
        return
    metrics = json.loads(mp.read_text())
    if not metrics:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = _arm_sort(metrics.keys())
    axes[0].bar(names, [metrics[n]["auc_topk"] for n in names],
                color=[ARM_COLORS.get(n) for n in names])
    axes[0].axhline(0.5, ls="--", c="k", lw=1)
    axes[0].set_ylabel("AUC (demented vs CDR0, TEST)")
    axes[0].set_title("Unsupervised anomaly detection")
    axes[0].set_ylim(0, 1)
    # score distributions by class (first arm with a subjects csv)
    for n in names:
        sp = results / f"anomaly_subjects_{n}.csv"
        if sp.exists():
            subj = pd.read_csv(sp)
            subj = subj[(subj.split == "test") & subj.cdr_class.notna()]
            for cls in sorted(subj.cdr_class.unique()):
                vals = subj[subj.cdr_class == cls]["score_topk"]
                axes[1].scatter(np.full(len(vals), cls) + np.random.default_rng(0)
                                .normal(0, 0.05, len(vals)), vals, s=18, alpha=0.75,
                                label=CLASS_NAMES[int(cls)] if int(cls) < 4 else cls,
                                color=ARM_COLORS.get(n))
            break
    axes[1].set_xlabel("CDR class"); axes[1].set_ylabel("subject anomaly score (top-25%)")
    axes[1].set_title("Anomaly score by CDR class (TEST)")
    axes[1].legend(fontsize=8)
    # nWBV correlation scatter
    for n in names:
        sp = results / f"anomaly_subjects_{n}.csv"
        if sp.exists():
            subj = pd.read_csv(sp)
            subj = subj[(subj.split == "test")]
            s = subj.dropna(subset=["nwbv", "score_topk"])
            if len(s) > 3:
                axes[2].scatter(pd.to_numeric(s["nwbv"]), s["score_topk"], s=22,
                                alpha=0.8, color=ARM_COLORS.get(n), label=n)
    axes[2].set_xlabel("nWBV (normalized whole-brain volume)")
    axes[2].set_ylabel("anomaly score")
    axes[2].set_title("Score vs clinical atrophy marker")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "fig_anomaly.png"); plt.close(fig)
    log.log("wrote fig_anomaly.png")


def fig_age(results: Path, figures: Path, log: RunLogger) -> None:
    preds = sorted(results.glob("age_probe_*.csv"))
    if not preds:
        return
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.8),
                             squeeze=False)
    for ax, p in zip(axes[0], preds):
        d = pd.read_csv(p)
        ax.scatter(d.age_true, d.age_pred, s=22, alpha=0.8)
        lo, hi = min(d.age_true.min(), d.age_pred.min()), max(d.age_true.max(),
                                                              d.age_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = np.corrcoef(d.age_true, d.age_pred)[0, 1]
        ax.set_title(f"{p.stem.replace('age_probe_', '')}\nPearson r={r:.2f}")
        ax.set_xlabel("true age"); ax.set_ylabel("predicted age")
    fig.tight_layout(); fig.savefig(figures / "fig_age_probe.png"); plt.close(fig)
    log.log("wrote fig_age_probe.png")


# --------------------------------------------------------------------------- #
def verdicts(cfg: dict, results: Path, log: RunLogger) -> str:
    """Auto-verdicts for the pre-registered hypotheses (docs/EXPERIMENT_PLAN.md).

    Thresholds live in cfg so verdicts are computed, not eyeballed.
    """
    th = cfg.get("thresholds", {})
    h1_margin = float(th.get("h1_margin", 0.02))
    h2_margin = float(th.get("h2_margin", 0.005))
    h3_auc = float(th.get("h3_auc", 0.60))
    h4_r = float(th.get("h4_r", 0.40))
    h0_margin = float(th.get("h0_margin", 0.15))
    synthetic = bool(cfg.get("synthetic_run", False))

    lines = []
    df = _probe_df(results)

    def probe_test_bal(name):
        if df.empty:
            return float("nan")
        r = df[(df.protocol == "probe") & (df.scope == "test")
               & (df.encoder == name) & (df.label_frac == df.label_frac.max())]
        return float(r["balanced_accuracy"].iloc[0]) if len(r) else float("nan")

    # H0 — pipeline validity gate (decisive on synthetic data)
    if synthetic:
        accs = {n: probe_test_bal(n) for n in df.encoder.unique()} if not df.empty else {}
        best = max(accs.values()) if accs else float("nan")
        ok = best > 0.25 + h0_margin
        lines.append(f"- **H0 (pipeline validity, synthetic):** best test bal-acc "
                     f"{best:.3f} vs chance 0.25 — "
                     f"**{'PASS' if ok else 'FAIL'}** "
                     f"({'planted signal detected' if ok else 'CHECK CODE — signal not detected'})")

    # H1 — domain SSL beats ImageNet transfer
    ms, im = probe_test_bal("mae_single"), probe_test_bal("imagenet")
    if np.isfinite(ms) and np.isfinite(im):
        v = "SUPPORTED" if ms > im + h1_margin else (
            "REFUTED" if ms < im - h1_margin else "INCONCLUSIVE")
        lines.append(f"- **H1 (brain-SSL > ImageNet transfer):** mae_single {ms:.3f} "
                     f"vs imagenet {im:.3f} (margin {h1_margin}) — **{v}**")

    # H2 — cross-view pretraining >= single-view
    mc = probe_test_bal("mae_cross")
    if np.isfinite(mc) and np.isfinite(ms):
        d = mc - ms
        v = "SUPPORTED" if d >= h2_margin else (
            "REFUTED" if d <= -h1_margin else "INCONCLUSIVE")
        lines.append(f"- **H2 (cross-view twist pays off):** mae_cross {mc:.3f} vs "
                     f"mae_single {ms:.3f} (Δ={d:+.3f}) — **{v}**")

    # H3 — reconstruction error tracks dementia
    amp = results / "anomaly_metrics.json"
    if amp.exists():
        am = json.loads(amp.read_text())
        for n, m in am.items():
            auc = m.get("auc_topk", float("nan"))
            rho = m.get("spearman_nwbv", float("nan"))
            v = "SUPPORTED" if (np.isfinite(auc) and auc >= h3_auc and
                                np.isfinite(rho) and rho < 0) else \
                ("REFUTED" if np.isfinite(auc) and auc < 0.55 else "INCONCLUSIVE")
            lines.append(f"- **H3 (recon-error anomaly signal, {n}):** AUC {auc:.3f} "
                         f"(≥{h3_auc}), Spearman(score,nWBV) {rho:+.3f} (expect <0) "
                         f"— **{v}**")

    # H4 — age probe sanity
    agp = results / "downstream_age.csv"
    if agp.exists():
        ag = pd.read_csv(agp)
        for _, r in ag.iterrows():
            ok = r["pearson_r"] >= h4_r
            lines.append(f"- **H4 (age-probe sanity, {r.encoder}):** Pearson r="
                         f"{r['pearson_r']:.3f}, MAE {r['mae_years']:.1f}y — "
                         f"**{'PASS' if ok else 'WEAK (features may be undertrained)'}**")
    return "\n".join(lines) if lines else "_(no results found — run stages 2-4 first)_"


def summary_tables(results: Path) -> str:
    out = []
    df = _probe_df(results)
    if not df.empty:
        piv = df[df.protocol == "probe"].pivot_table(
            index=["encoder", "label_frac"], columns="scope",
            values="balanced_accuracy")
        out.append("## Probe results — subject-level balanced accuracy\n\n"
                   + piv.round(3).to_markdown() + "\n")
    ftp = results / "downstream_finetune.csv"
    if ftp.exists():
        ftd = pd.read_csv(ftp).groupby("encoder")["balanced_accuracy"].agg(
            ["mean", "std", "count"])
        out.append("## Fine-tune results (GroupKFold val)\n\n"
                   + ftd.round(3).to_markdown() + "\n")
    amp = results / "anomaly_metrics.json"
    if amp.exists():
        out.append("## Anomaly detection (reconstruction error, TEST)\n\n```\n"
                   + json.dumps(json.loads(amp.read_text()), indent=2) + "\n```\n")
    return "\n".join(out) if out else ""


def run_analysis(cfg: dict) -> dict:
    results = Path(cfg["results_dir"])
    figures = ensure_dir(cfg.get("figures_dir", "figures"))
    log = RunLogger(results, "analysis")
    fig_pretrain(cfg, figures, log)
    fig_label_efficiency(results, figures, log)
    fig_main_bars(results, figures, log)
    fig_confusion(results, figures, log)
    fig_anomaly(results, figures, log)
    fig_age(results, figures, log)

    md = ["# MV-BrainMAE — Results Summary\n",
          f"_Auto-generated by `mvbrainmae.analysis` "
          f"({'SYNTHETIC smoke run' if cfg.get('synthetic_run') else 'REAL data run'})._\n",
          "## Hypothesis verdicts\n", verdicts(cfg, results, log), "\n",
          summary_tables(results)]
    (results / "summary.md").write_text("\n".join(md))
    log.log("wrote summary.md")
    return {"summary": str(results / "summary.md"), "figures": str(figures)}


def main() -> None:
    import argparse
    from .utils import apply_overrides, load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    r = run_analysis(apply_overrides(load_config(args.config), args.overrides))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
