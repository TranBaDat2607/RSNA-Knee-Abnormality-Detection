"""Loading DINOv2 and opening its last few blocks for fine-tuning.

Why fine-tune rather than freeze: a frozen self-supervised encoder is the cheap option
and where this pipeline started. Resolution, encoder size, slice coverage, and slot
aggregation were each varied with everything else held fixed, and none of them moved the
score beyond what validation noise allows — a pattern that points at the representation
itself as the binding constraint, which none of those four axes touch. There's an obvious
reason to expect exactly that here: the encoder learned its features from natural images,
where nothing resembles the signal a torn meniscus makes on a proton-density sequence.

Two restraints on the fine-tune:

- **Only the last blocks move.** Early ViT blocks are generic edge/texture filters; late
  blocks are where semantics live. There isn't enough supervision here to improve the
  early ones and there's plenty to damage them, so they stay frozen.
- **The encoder learns far more slowly than the head.** The head is random at
  initialisation and has everything to learn; the encoder starts from a good solution and
  needs only to be nudged off it. A single learning rate would either leave the head
  undertrained or destroy the encoder in the first few hundred steps.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ..config import CacheConfig, TrainConfig
from ..logging_utils import log
from ..paths import find_dinov2
from .network import Model


def build_model(*, cache_cfg: CacheConfig, train_cfg: TrainConfig, n_targets: int,
                 dinov2_path: Path | None = None, variant: str = "small") -> Model:
    """Load the encoder and open its last ``train_cfg.unfreeze_last`` blocks for training."""
    from transformers import AutoModel

    p = dinov2_path if dinov2_path is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb: nn.Module = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - train_cfg.unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True

    dim = bb.config.hidden_size * 2
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {train_cfg.unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim}")
    return Model(bb, dim, n_slot=cache_cfg.n_slot, n_targets=n_targets)
