"""MILClassifier wiring, checkpoint key compatibility, and window-order invariance."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsna_knee.mil.model import MILClassifier
from rsna_knee.mil.recipes import PUBLIC_RAPTOR_ARMS


class _StubBackbone(nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, self.num_features, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))


def _model():
    torch.manual_seed(0)
    return MILClassifier(_StubBackbone(), _StubBackbone.num_features).eval()


def test_output_shape_and_public_checkpoint_parameter_names():
    m = _model()
    assert m(torch.randn(2, 7, 3, 12, 12)).shape == (2, 12)
    head_keys = {k for k in m.state_dict() if not k.startswith("backbone.")}
    assert head_keys == {"norm.weight", "norm.bias", "att.0.weight", "att.0.bias",
                         "att.3.weight", "att.3.bias", "clsW", "clsb"}


def test_prediction_is_invariant_to_window_order():
    # Why a "reversed window order" test-time view is a wasted pass (E1: max diff 2.6e-5).
    m = _model()
    x = torch.randn(3, 9, 3, 12, 12)
    with torch.no_grad():
        a = m(x)
        b = m(x.flip(1))
        c = m(x[:, torch.randperm(9)])
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)


def test_chunked_encoding_matches_a_single_backbone_call():
    m = _model()
    x = torch.randn(2, 11, 3, 12, 12)
    with torch.no_grad():
        assert torch.allclose(m.encode(x), m.encode(x, chunk=4), atol=1e-6)


def test_public_arm_list_has_no_duplicate_checkpoint_passes():
    files = [a.checkpoint for a in PUBLIC_RAPTOR_ARMS]
    assert len(files) == len(set(files))
