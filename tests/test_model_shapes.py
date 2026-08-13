"""Shape/wiring tests for the model that don't need real DINOv2 weights.

``Model.forward`` only requires its backbone to accept a ``pixel_values`` kwarg and
return an object with ``.last_hidden_state`` — the HF ``AutoModel`` interface. A tiny
stub satisfying that is enough to test the study-bag flattening/folding and the head
wiring without downloading anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from rsna_knee.model.ema import Ema
from rsna_knee.model.heads import SlotHead
from rsna_knee.model.network import Model


class _StubBackbone(nn.Module):
    """Returns a fixed-dim hidden state per input image, ignoring pixel content."""

    def __init__(self, hidden_size: int = 8, tokens: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokens = tokens
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor):
        b = pixel_values.shape[0]
        pooled = self.proj(pixel_values).mean(dim=(-1, -2))  # (B, hidden)
        seq = pooled.unsqueeze(1).expand(b, self.tokens, self.hidden_size)
        return SimpleNamespace(last_hidden_state=seq)


def test_slot_head_output_shape_and_masking():
    n_slot, n_out, dim = 6, 12, 16
    head = SlotHead(dim, n_slot, n_out)
    x = torch.randn(4, n_slot, dim)
    mask = torch.ones(4, n_slot)
    out = head(x, mask)
    assert out.shape == (4, n_out)


def test_slot_head_handles_a_fully_present_and_a_partially_masked_study():
    head = SlotHead(dim=16, n_slot=6, n_out=3)
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6)
    mask[1, 3:] = 0.0  # second study only has the first 3 slots
    out = head(x, mask)
    assert torch.isfinite(out).all()


def test_model_forward_shape_with_stub_backbone():
    hidden = 8
    backbone = _StubBackbone(hidden_size=hidden)
    model = Model(backbone, dim=hidden * 2, n_slot=6, n_targets=12)
    imgs = torch.randint(0, 255, (2, 6, 3, 16, 16), dtype=torch.uint8)
    mask = torch.ones(2, 6)
    out = model(imgs, mask)
    assert out.shape == (2, 12)


def test_ema_update_moves_target_toward_source_and_target_switches_on_decay():
    backbone = _StubBackbone(hidden_size=8)
    model = Model(backbone, dim=16, n_slot=6, n_targets=12)
    ema = Ema(model, decay=0.9)

    before = ema.model.head.out.bias.clone()
    with torch.no_grad():
        model.head.out.bias.add_(1.0)
    ema.update(model)
    after = ema.model.head.out.bias
    assert not torch.allclose(before, after)          # moved
    assert not torch.allclose(after, model.head.out.bias)  # but hasn't fully caught up

    disabled = Ema(model, decay=0.0)
    assert disabled.target(model) is model            # EMA off -> use the live model directly
    assert ema.target(model) is ema.model              # EMA on -> use the averaged copy
