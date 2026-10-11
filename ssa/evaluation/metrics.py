"""Metric helpers used across model, retrieval and prioritisation evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y_true, prob, bins: int = 10) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(prob, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_table(y_true, prob, bins: int = 10) -> list[dict]:
    y = np.asarray(y_true, float)
    p = np.asarray(prob, float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean())})
    return rows


def at_capacity(y_true, score, fraction: float) -> dict:
    y = np.asarray(y_true, int)
    s = np.asarray(score, float)
    k = max(1, int(round(len(s) * fraction)))
    top = np.argsort(-s, kind="stable")[:k]
    tp = int(y[top].sum())
    pos = int(y.sum())
    base = y.mean() if len(y) else 0.0
    prec = tp / k
    return {"fraction": fraction, "k": k, "precision": prec, "recall": tp / pos if pos else 0.0,
            "lift": prec / base if base > 0 else None, "max_possible_recall": min(1.0, k / pos) if pos else None}


def bootstrap_auc_ci(y_true, score, n_boot: int = 300, seed: int = 0) -> dict:
    y = np.asarray(y_true, int)
    s = np.asarray(score, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return {"lower": float(np.quantile(vals, 0.025)), "upper": float(np.quantile(vals, 0.975)), "n_boot": len(vals)}


def binary_summary(y_true, prob) -> dict:
    y = np.asarray(y_true, int)
    p = np.asarray(prob, float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None}
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "ece_10": ece(y, p),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))
