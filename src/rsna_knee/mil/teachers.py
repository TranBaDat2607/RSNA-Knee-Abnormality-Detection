"""Weak-label teachers: public report-label tables used as training targets for the 4,349 unlabelled
training studies.

Agreement with the 58 gold studies (A02b, macro AUC): flight0234 hybrid 0.899, top-5 mean 0.897,
stevenleehans v4 blend 0.893, yunusgmsoy 0.880, pilkwang v2 0.870, this repo's gpt-5.4-mini labels
0.855. The Raptor teacher (the labels behind the public CoAtNet checkpoints) cannot be scored — its
table omits the gold rows — and disagrees most with the best teachers on Synovitis and Fracture.

Five other public tables (barun2104, rayanbabur, tasmeemreza, zaidaliiq1000, yunusgmsoy 4-source)
copy the gold labels verbatim. Never validate on gold-58 with them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from ..config import TARGETS

TEACHERS: dict[str, tuple[str, ...]] = {
    "raptor": ("labels_llm_soft.csv",),
    "flight": ("report_labels_v4hybrid.csv",),
    "top5mean": ("report_labels_v4hybrid.csv", "llm_labels_v4_blend.csv", "llm_labels_v2.csv",
                 "yunus_llm_labels.csv", "report_labels_v2.csv"),
}

# Kaggle datasets that provide the files above.
TEACHER_DATASETS = ("dreaddevelopment/rsna-knee-labels", "flight0234/rsna-knee-hybrid-report-labels",
                    "stevenleehans/rsna-knee-llm-report-labels", "yunusgmsoy/rsna-knee-llm-report-labels",
                    "pilkwang/rsna-knee-llm-labels")


def read_table(path: str, targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    return d.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[list(targets)].astype(float)


def read_teacher(name: str, find_file: Callable[[str], str], targets: Sequence[str] = TARGETS) -> pd.DataFrame:
    """Mean of the teacher's source tables over the studies they all cover, clipped to [0, 1]."""
    tables = [read_table(find_file(f), targets) for f in TEACHERS[name]]
    common = sorted(set.intersection(*[set(t.index) for t in tables]))
    return (sum(t.reindex(common) for t in tables) / len(tables)).clip(0, 1)
