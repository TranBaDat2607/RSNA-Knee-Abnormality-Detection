"""Exponential moving average of the trainable weights.

Both validation references the training loop uses are noisy, and picking a single epoch
by either is close to picking a neighbour at random. An average over the trajectory is
more stable than any point on it, and costs one extra copy of a model that's mostly
frozen anyway.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class Ema:
    """Buffers are copied rather than averaged: normalisation constants are fixed, and
    averaging them would only accumulate floating point error."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        # Warm-up: without it the average stays anchored to the random initialisation for
        # the first few hundred steps, which is most of a short fold.
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
            else:
                v.copy_(s)

    def target(self, model: nn.Module) -> nn.Module:
        """The weights to evaluate and keep: the average, or the model if EMA is disabled."""
        return self.model if self.decay > 0 else model
