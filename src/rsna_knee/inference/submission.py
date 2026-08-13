"""Submission-file writing.

A valid 0.5-benchmark submission is written before any real prediction happens, so a run
that dies mid-way (a SIGKILL from an OOM, which never reaches a Python ``except``) still
leaves a scoreable file on disk; :func:`write_submission` overwrites it once real
predictions exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_benchmark_submission(test_csv: Path, targets: list[str],
                                out_path: str = "submission.csv") -> None:
    t = pd.read_csv(test_csv)
    for c in targets:
        t[c] = 0.5
    t.to_csv(out_path, index=False)


def write_submission(rank_sum: np.ndarray, n_models: int, st_te: list[str],
                      test_df: pd.DataFrame, targets: list[str],
                      out_path: str = "submission.csv") -> pd.DataFrame:
    """Rank-mean of the finished folds, aligned onto the test index.

    Ranks rather than raw probabilities: the competition metric only rewards correct
    ordering, and rank-mean combines differently-calibrated fold models without letting
    one fold's more confident logits dominate the average.
    """
    p = rank_sum / max(n_models, 1)
    sub = pd.DataFrame(p, columns=targets)
    sub.insert(0, "StudyInstanceUID", st_te)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[targets] = sub[targets].fillna(0.5)
    sub.to_csv(out_path, index=False)
    return sub
