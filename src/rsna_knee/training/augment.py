"""Augmentation that preserves anatomy.

A knee is not symmetric on either axis: the femur is above and the tibia below, and
``Medial OA``/``Lateral OA``/``PF OA`` are defined by which bone surface carries the
cartilage loss, so vertical flip is off by default. Horizontal flip stays forbidden
unconditionally — it would undo the laterality normalisation in
:mod:`rsna_knee.dicom.laterality`. Small rotation, scale and translation preserve anatomy
while still preventing the encoder from memorising the exact framing; for a ViT backbone
in particular, translation jitter matters more than it would for a CNN, since a
transformer's positional embeddings give it no built-in shift invariance to fall back on.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import AugConfig


def augment(imgs: torch.Tensor, cfg: AugConfig) -> torch.Tensor:
    """Small affine and intensity jitter over a whole bag. No horizontal flip, ever.

    ``imgs`` is ``(B, S, C, H, W)`` uint8 (or any dtype F.grid_sample-compatible once
    cast). One random affine draw is made *per study* (per batch item) and shared across
    that study's own slot bag, so all slices of a given study stay geometrically
    consistent with each other while different studies in the batch still get
    independent draws.
    """
    b, s, c, h, w = imgs.shape
    x = imgs.float()

    if cfg.affine:
        ang = (torch.rand(b) - 0.5) * 2 * math.radians(cfg.rot_deg)
        sc = 1.0 + (torch.rand(b) - 0.5) * 2 * cfg.scale
        tx = (torch.rand(b) - 0.5) * 2 * cfg.shift
        ty = (torch.rand(b) - 0.5) * 2 * cfg.shift
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([
            torch.stack([cos, -sin, tx], dim=1),
            torch.stack([sin, cos, ty], dim=1),
        ], dim=1).to(dtype=torch.float32, device=imgs.device)  # (b, 2, 3)
        theta = theta.repeat_interleave(s, dim=0)  # (b*s, 2, 3), shared within each study
        flat = x.reshape(b * s, c, h, w)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False).reshape(b, s, c, h, w)

    scale = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=imgs.device) - 0.5) * 2 * cfg.intensity
    x = (x * scale).clamp(0, 255)

    if cfg.use_vflip:
        flip = torch.rand(b, device=imgs.device) < 0.5
        if flip.any():
            x[flip] = torch.flip(x[flip], dims=[-2])
    return x.round().to(imgs.dtype)
