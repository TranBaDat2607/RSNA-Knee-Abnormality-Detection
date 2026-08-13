"""Locating the competition data, DINOv2 weights, and LLM labels.

Kaggle-first: the same code path finds everything whether it's running in a Kaggle
notebook (data mounted under ``/kaggle/input``) or locally (``data/`` in the repo root,
populated by hand from the competition download). No DICOM data ships in this repo — see
README.md — so the Kaggle-mount branches here are exercised on Kaggle, not by local tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from .logging_utils import log


def find_root() -> Path:
    """Locate the directory holding ``test.csv`` / ``train.csv`` and the series folders."""
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
        Path("."),
    ]
    for c in candidates:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c

    # Last resort: two-level scan, because a mounted Kaggle Dataset is sometimes nested
    # one directory deeper than the plain competition-data layout.
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError("competition mount not found")


def find_dinov2(variant: str = "small") -> Path | None:
    """Locate a mounted DINOv2 checkpoint directory (a HF ``config.json`` + weights) by
    variant name (e.g. "small", "base"). Returns None if nothing is attached."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None


def find_llm_labels() -> Path:
    """Locate ``llm_labels_full.csv``, either checked into the repo locally or attached
    as a mounted Kaggle Dataset."""
    local = Path("data/llm_labels_full.csv")
    if local.is_file():
        return local
    base = Path("/kaggle/input")
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            if "llm_labels_full.csv" in files:
                return Path(root) / "llm_labels_full.csv"
    raise FileNotFoundError("llm_labels_full.csv not found locally or under /kaggle/input")


def plan_cache(n_study: int, *, n_slot: int, img: int, cache_budget_gb: float,
               group: int, n_group_max: int) -> int:
    """Choose how many slice groups per slot the memory budget allows.

    The cache is ``n_study x n_slot x slices x img^2`` bytes. Coverage (slice count) is
    the cheap axis — linear — and resolution the expensive one, since it enters squared.
    So when the budget binds, it's the slice count that gives way, decided once from the
    training corpus size so train and test caches share one group layout.
    """
    per_slice = n_study * n_slot * img * img
    afford = int(cache_budget_gb * 1024**3 // max(per_slice, 1))
    groups = max(1, min(n_group_max, afford // group))
    if groups < n_group_max:
        log(f"cache budget {cache_budget_gb:.0f} GB allows {groups} group(s) of {group}, "
            f"not {n_group_max}")
    return groups
