"""Quality metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, brier_score_loss

__all__ = ["adaptive_ece", "score"]


def adaptive_ece(y_true, p, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    order = np.argsort(p)
    p, y_true = p[order], np.asarray(y_true)[order]
    err = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            err += (len(b) / len(p)) * abs(p[b].mean() - y_true[b].mean())
    return float(err)


def score(task: str, y_true, raw) -> dict:
    if task == "clf":
        p = 1.0 / (1.0 + np.exp(-np.asarray(raw, float)))
        return dict(metric=accuracy_score(y_true, (p > 0.5).astype(int)),
                    ece=adaptive_ece(y_true, p),
                    brier=brier_score_loss(y_true, p))
    return dict(metric=r2_score(y_true, raw), ece=np.nan, brier=np.nan)
