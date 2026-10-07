"""Evaluation metrics with SUBJECT-LEVEL aggregation.

Rationale (core rigor point of this repo): a single patient contributes ~100
slices. Plain slice-level accuracy lets a model 'recognize' patients it has
already seen through other slices -> leakage. Every classification metric here
first averages slice probabilities per subject, then scores subject predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)


def subject_level_predictions(probs: np.ndarray, labels: np.ndarray,
                              subjects: np.ndarray) -> pd.DataFrame:
    """Average slice softmax probs per subject -> subject-level pred.

    Returns DataFrame: subject, true_label, pred_label, per-class mean probs (c0..cK).
    """
    n_cls = probs.shape[1]
    prob_cols = [f"c{i}" for i in range(n_cls)]
    df = pd.DataFrame(np.asarray(probs), columns=prob_cols)
    df["_label"] = np.asarray(labels)
    df["_subject"] = np.asarray(subjects)
    grp = df.groupby("_subject")
    out = grp[prob_cols].mean()
    out["true_label"] = grp["_label"].first()
    out["pred_label"] = out[prob_cols].values.argmax(axis=1)
    return out.reset_index().rename(columns={"_subject": "subject"})


def classification_report(subject_df: pd.DataFrame) -> dict:
    y_true = subject_df["true_label"].values
    y_pred = subject_df["pred_label"].values
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n_subjects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
    }


def binary_auc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """AUC with a degenerate-input guard (single-class -> nan)."""
    if len(np.unique(binary_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(binary_labels, scores))


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 5:
        return {"n": int(len(x)), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan")}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp)}
