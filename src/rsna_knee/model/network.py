"""The full model: encoder plus head, trained end to end."""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import SlotHead

# Standard ImageNet normalisation constants, matching what DINOv2 was pretrained with.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Model(nn.Module):
    """A study arrives as a bag of slot images; the bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.

    ``backbone`` must accept ``pixel_values`` and return an object with
    ``.last_hidden_state`` of shape ``(B*S, tokens, dim)`` — the HF ``AutoModel`` DINOv2
    interface. Anything satisfying that (including a test stub) works here.
    """

    def __init__(self, backbone: nn.Module, dim: int, n_slot: int, n_targets: int):
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(dim, n_slot, n_targets)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1).reshape(b, s, -1)
        return self.head(feat, mask)
