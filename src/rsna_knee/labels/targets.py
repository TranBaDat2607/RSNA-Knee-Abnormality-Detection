"""Merging gold annotations + LLM labels into (Y, W) training arrays, and fold grouping."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def load_gold(train_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """The subset of ``train_df`` with a complete set of annotated target columns."""
    gold = train_df.set_index("StudyInstanceUID")[targets]
    return gold[gold.notna().all(axis=1)]


def build_targets(studies: list[str], train_df: pd.DataFrame, llm_labels: pd.DataFrame,
                   targets: list[str], gold_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(Y, W)`` graded-target and sample-weight arrays, one row per study.

    Priority per study: gold annotation (weight ``gold_weight``) > LLM label (weight
    scaled by distance from the 0.5 rubric midpoint, since the LLM has no explicit
    confidence output) > neither (weight 0, the study contributes nothing and callers
    should drop it via ``np.where(W.sum(1) > 0)[0]``).
    """
    gold = load_gold(train_df, targets)
    lab = llm_labels[targets]

    y = np.zeros((len(studies), len(targets)), np.float32)
    w = np.zeros_like(y)
    for i, st in enumerate(studies):
        if st in gold.index:
            y[i], w[i] = gold.loc[st].values, gold_weight
        elif st in lab.index:
            r = lab.loc[st].values.astype(np.float32)
            y[i] = r
            w[i] = 0.25 + 0.75 * (np.abs(r - 0.5) * 2.0)
    return y, w


def usable_rows(w: np.ndarray) -> np.ndarray:
    """Indices of studies with at least one non-zero-weight target."""
    return np.where(w.sum(1) > 0)[0]


def report_hash_folds(studies: list[str], train_df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Assign each study to a fold by hashing its report text.

    Some reports are byte-identical across studies (a template read for an unremarkable
    knee), and every study in such a group gets the same derived target vector. Splitting
    the group across folds would score a model on a target whose source it had already
    trained on, so the hash keeps every duplicate group whole inside one fold.
    """
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    return np.array([
        int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % n_folds
        for s in studies
    ])


def gold_positions(studies: list[str], gold: pd.DataFrame) -> np.ndarray:
    """Positions within ``studies`` of the gold-annotated ones (for out-of-fold scoring)."""
    pos = {s: i for i, s in enumerate(studies)}
    return np.array([pos[s] for s in gold.index if s in pos], dtype=int)
