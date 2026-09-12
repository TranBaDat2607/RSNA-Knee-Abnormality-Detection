"""Backbone + per-finding attention-MIL head.

Parameter names match the public Raptor checkpoints (``backbone.*``, ``norm``, ``att``, ``clsW``,
``clsb``) so those load with ``strict=True``.

The head is a softmax over windows with no positional term, so the prediction is invariant to
window order. That is why a "reversed window order" test-time view adds nothing.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


def build_backbone(arch: str = DEFAULT_ARCH, pretrained: bool = False) -> nn.Module:
    """A timm image backbone returning one pooled feature vector per image.

    Conv/attention hybrids (CoAtNet, MaxViT, ConvNeXt) contain "vit" in their names but have
    no CLS token, so they must take the average-pool path.
    """
    import timm

    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit else dict(global_pool="avg"))
    return timm.create_model(arch, **kw)


class MILClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, n_out: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(nn.Linear(feat_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n_out))
        self.clsW = nn.Parameter(torch.zeros(n_out, feat_dim))
        self.clsb = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, x: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        """``(B, K, 3, H, W)`` -> ``(B, K, F)``; ``chunk`` bounds images per backbone call."""
        b, k = x.shape[:2]
        flat = x.flatten(0, 1)
        if chunk and len(flat) > chunk:
            feats = torch.cat([self.backbone(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
        else:
            feats = self.backbone(flat)
        return feats.view(b, k, -1)

    def head(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.norm(feats)
        att = torch.softmax(self.att(h), dim=1)  # (B, K, n_out): one distribution per finding
        pooled = torch.einsum("bkn,bkf->bnf", att, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"
                    ) -> tuple[MILClassifier, int, dict]:
    """Load a Raptor-format checkpoint dict (``model``, ``arch``, ``res``, ...).

    Returns the eval-mode model, the resolution its windows must be resized to, and the
    checkpoint's non-tensor metadata.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    backbone = build_backbone(ck.get("arch", DEFAULT_ARCH))
    model = MILClassifier(backbone, backbone.num_features)
    model.load_state_dict(ck["model"], strict=True)
    meta = {k: v for k, v in ck.items() if k != "model" and not isinstance(v, torch.Tensor)}
    return model.eval().to(device), int(ck.get("res", 384)), meta
