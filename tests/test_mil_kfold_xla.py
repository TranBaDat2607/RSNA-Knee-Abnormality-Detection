"""Host-side pieces of the TPU trainer (no torch_xla needed): augmentation, normalisation, eval padding."""

from __future__ import annotations

import random

import numpy as np
import torch

from rsna_knee.mil.kfold import prepare
from rsna_knee.mil.kfold_xla import _pad_rows, affine_windows, normalise


def test_affine_windows_keeps_shape_and_black_background():
    x = np.zeros((20, 3, 32, 32), np.uint8)
    x[:, :, 8:24, 8:24] = 200
    y = affine_windows(x, random.Random(0))
    assert y.shape == x.shape and y.dtype == np.uint8
    assert y[:, :, 0, 0].max() == 0  # corners stay background
    # one transform per study: every window and channel moves identically
    assert all(np.array_equal(y[0, 0], y[i, c]) for i in range(20) for c in range(3))


def test_normalise_matches_gpu_prepare():
    x = torch.randint(0, 255, (2, 4, 3, 16, 16), dtype=torch.uint8)
    g = torch.tensor([1.0, 0.9])
    assert torch.allclose(normalise(x, g), prepare(x, 16, g), atol=1e-5)


def test_pad_rows_fills_to_multiple():
    rows, n = _pad_rows([5, 6, 7], 4)
    assert n == 3 and rows == [5, 6, 7, 5]
    assert _pad_rows([1, 2, 3, 4], 4) == ([1, 2, 3, 4], 4)


def test_fleet_serves_a_slot_checkpoint_with_the_xla_trainers_eval_windows(tmp_path):
    from rsna_knee.mil.cache import shrink_volume
    from rsna_knee.mil.fleet import predict_fleet
    from rsna_knee.mil.kfold_xla import XlaWindows
    from rsna_knee.mil.model import MILClassifier, build_backbone

    torch.manual_seed(0)
    bb = build_backbone("resnet18", pretrained=False)
    model = MILClassifier(bb, bb.num_features, n_slots=5).eval()
    with torch.no_grad():
        model.slot_emb.normal_()
    torch.save({"model": model.state_dict(), "arch": "resnet18", "res": 16, "cache": 8, "n_slots": 5,
                "windows": "slot"}, tmp_path / "m.pt")
    rng = np.random.default_rng(1)
    raw = {u: rng.integers(1, 255, (44, 20, 20), dtype=np.uint8) for u in ("s0", "s1")}
    masks = np.ones((2, 44), np.uint8)
    masks[1, 20:22] = 0
    probs, _, failures = predict_fleet(["s0", "s1"], {}, "/root", [str(tmp_path / "m.pt")], torch.device("cpu"),
                                       k_eval=10, workers=0, log=lambda m: None,
                                       build_fn=lambda uid, rows, root, recipe: (raw[uid], masks[int(uid[1])]))
    assert not failures

    class _Cache:
        def __init__(self):
            self.masks = masks

        def volume(self, i):
            return shrink_volume(raw[("s0", "s1")[i]], 8)

    ds = XlaWindows(_Cache(), [0, 1], np.zeros((2, 12), np.float32), 10, False, 0, slot_aware=True)
    with torch.no_grad():
        for i in range(2):
            x, _, g, _, sl = ds[i]
            logits = model.head(model.encode(prepare(x[None], 16, g[None])), sl[None])
            assert np.allclose(torch.sigmoid(logits)[0].numpy(), probs[0, i], atol=1e-5)
