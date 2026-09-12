"""Rank-space fusion.

The metric is per-finding AUC, so only the ordering inside each column matters. Converting every
arm to percentile ranks before averaging stops a more confident (not more accurate) arm from
dominating a probability mean.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata


def rank_pct(x: np.ndarray) -> np.ndarray:
    """Column-wise average-tie percentile ranks in ``(0, 1]``."""
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, method="average", axis=0) / x.shape[0]


def weighted_rank_mean(preds: Mapping[str, np.ndarray],
                       weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Weighted mean of per-arm percentile ranks; weights are normalised, missing ones are 1."""
    names = list(preds)
    if not names:
        raise ValueError("nothing to blend")
    w = np.array([float((weights or {}).get(n, 1.0)) for n in names])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"invalid blend weights {dict(zip(names, w))}")
    w = w / w.sum()
    return sum(wi * rank_pct(preds[n]) for wi, n in zip(w, names))


def logit_blend(a: np.ndarray, b: np.ndarray, weight_b: float, eps: float = 1e-5) -> np.ndarray:
    """``(1 - w) * logit(a) + w * logit(b)`` mapped back through the sigmoid (inputs in [0, 1])."""
    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
        return np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-((1.0 - weight_b) * logit(a) + weight_b * logit(b))))
