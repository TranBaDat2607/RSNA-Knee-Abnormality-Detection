"""Macro AUC and the Hanley-McNeil standard error used to size the annotated reference."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mean AUC across target columns; a column with only one class contributes NaN
    (excluded from the mean, matching ``np.nanmean``)."""
    return float(np.nanmean([
        roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
        for j in range(y.shape[1])
    ]))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC estimate — the reason a small annotated subset can guard
    epoch selection but shouldn't be tuned against directly."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))
