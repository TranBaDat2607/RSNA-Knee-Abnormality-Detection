"""Decoding every (study, slot) once into a shared in-memory cache.

Fine-tuning revisits the same pixels every epoch; reading them from the mount each time
would make epoch count a function of I/O rather than of learning. So every slot is decoded
once into a ``uint8`` array — intensity already normalised into [0, 1] by
:func:`rsna_knee.dicom.sampling.read_slot`, so eight bits cost nothing a bilinear resize
hasn't already cost.

The cache stores ``n_group * group`` slices per slot; :func:`take_group` slices out one
group of ``group`` consecutive channels at a time — training draws a random group per
step (doubling as augmentation along the stack), inference averages over all of them.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..config import CacheConfig, SlotDef
from ..logging_utils import log
from .laterality import normalise_laterality
from .sampling import read_slot


def build_cache(slot_map: dict, plane_map: dict[str, str], lat_map: dict,
                 tag: str, cfg: CacheConfig, n_group: int, *,
                 time_budget_s: float | None = None, t0: float | None = None,
                 ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Decode every ``(study, slot)`` once. Returns ``(studies, cache, mask)``.

    ``cache`` has shape ``(n_study, n_slot, n_group*group, img, img)``, ``uint8``.
    ``mask`` has shape ``(n_study, n_slot)``, 1.0 where that slot was actually decoded.
    """
    slots = cfg.slots
    cache_slices = n_group * cfg.group
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), len(slots), cache_slices, cfg.img, cfg.img), np.uint8)
    mask = np.zeros((len(studies), len(slots)), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(slots)
            if name in slot_map[st]]
    log(f"{tag}: decoding {len(jobs)} slot-series")

    chunk = 512
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.pix_threads) as pool:
        for c0 in range(0, len(jobs), chunk):
            block = jobs[c0:c0 + chunk]
            reads = pool.map(
                lambda j: read_slot(j[3], cache_slices, cfg.img, cfg.crop_mm), block)
            for (st, k, plane, _), img in zip(block, reads):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(
                    img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < chunk:
                log(f"  {tag} {done}/{len(jobs)}")
            if time_budget_s is not None and t0 is not None and time.time() - t0 > time_budget_s:
                log(f"  {tag}: time budget reached during decode")
                break
    gc.collect()
    return studies, cache, mask


def take_group(cache_rows: torch.Tensor, g: int, group: int) -> torch.Tensor:
    """Slice ``group`` consecutive channels out of the cached slices for group index ``g``."""
    return cache_rows[:, :, g * group:(g + 1) * group]
