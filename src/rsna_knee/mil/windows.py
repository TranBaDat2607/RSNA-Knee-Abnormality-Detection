"""Window-centre selection and triplet assembly.

A window is three adjacent slices stacked into RGB, so the backbone sees a little of what lies
above and below its centre slice. Centres need both neighbours inside the filled part of the
stack. Evaluation spreads ``k`` centres evenly (deterministic); training draws them at random, so
every epoch sees a different subset of the same study.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _candidate_centres(mask: np.ndarray) -> tuple[list[int], int]:
    depth = len(mask)
    valid = np.flatnonzero(np.asarray(mask) > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    centres = list(range(lo + 1, hi))
    if not centres:
        centres = [max(1, min((lo + hi) // 2, depth - 2))]
    return centres, depth


def eval_centres(mask: np.ndarray, k: int) -> np.ndarray:
    """``k`` evenly spaced centres (repeats allowed when the stack is shorter than ``k``)."""
    centres, depth = _candidate_centres(mask)
    idx = np.linspace(0, len(centres) - 1, k).round().astype(int)
    return np.array([max(1, min(centres[i], depth - 2)) for i in idx])


def train_centres(mask: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    """``k`` random centres, cycling through every candidate before repeating one."""
    centres, depth = _candidate_centres(mask)
    pool = centres * (k // len(centres) + 1)
    rng.shuffle(pool)
    return np.array([max(1, min(c, depth - 2)) for c in pool[:k]])


def triplets(volume: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``(k, 3, H, W)`` uint8 windows from a ``(D, H, W)`` uint8 stack."""
    return np.stack([volume[centres - 1], volume[centres], volume[centres + 1]], axis=1)


def to_model_input(windows_u8: torch.Tensor, res: int, gain: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 windows (any leading dims, then 3, H, W) -> ImageNet-normalised float at ``res``.

    Runs wherever ``windows_u8`` lives, so resizing happens on the GPU. ``gain`` (one value per item
    of the first leading dim) is a training-time intensity jitter applied before normalisation.
    """
    lead = windows_u8.shape[:-3]
    x = windows_u8.reshape(-1, *windows_u8.shape[-3:]).float().div_(255.0)
    if gain is not None:
        per_item = int(np.prod(lead[1:])) if len(lead) > 1 else 1
        x = (x * gain.to(x.device).float().repeat_interleave(per_item).view(-1, 1, 1, 1)).clamp_(0, 1)
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(*lead, 3, res, res)
