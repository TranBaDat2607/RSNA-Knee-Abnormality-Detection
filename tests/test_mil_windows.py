"""Window-centre selection must match the public Raptor checkpoints' eval windows exactly."""

from __future__ import annotations

import random

import numpy as np
import torch

from rsna_knee.mil.windows import eval_centres, to_model_input, train_centres, triplets


def _raptor_reference_centres(mask, depth, k):
    """Verbatim copy of the public inference notebook's ``_eval_centers`` + per-window clamp."""
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [max(1, min(cs[i], depth - 2)) for i in idx]


def test_eval_centres_match_the_raptor_reference_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(300):
        depth = int(rng.integers(3, 70))
        mask = (rng.random(depth) > rng.random()).astype(np.uint8)
        k = int(rng.integers(1, 70))
        assert eval_centres(mask, k).tolist() == _raptor_reference_centres(mask, depth, k)


def test_eval_centres_on_an_empty_stack_still_returns_k_valid_centres():
    c = eval_centres(np.zeros(44, np.uint8), 42)
    assert len(c) == 42 and c.min() >= 1 and c.max() <= 42


def test_train_centres_stay_inside_the_filled_band_and_cycle_before_repeating():
    mask = np.zeros(44, np.uint8)
    mask[5:30] = 1  # filled slices 5..29 -> centres 6..28 (23 candidates)
    c = train_centres(mask, 12, random.Random(1))
    assert len(c) == 12 and c.min() >= 6 and c.max() <= 28
    assert len(set(c.tolist())) == 12
    many = train_centres(mask, 46, random.Random(2))
    assert set(many.tolist()) == set(range(6, 29))


def test_triplets_stack_the_neighbouring_slices_as_channels():
    vol = np.arange(10, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(4, 2)
    w = triplets(vol, np.array([1, 5, 8]))
    assert w.shape == (3, 3, 4, 4)
    assert w[:, :, 0, 0].tolist() == [[0, 1, 2], [4, 5, 6], [7, 8, 9]]


def test_to_model_input_resizes_and_imagenet_normalises():
    x = torch.full((2, 5, 3, 8, 8), 255, dtype=torch.uint8)
    out = to_model_input(x, 16)
    assert out.shape == (2, 5, 3, 16, 16)
    expected = (1.0 - 0.485) / 0.229
    assert torch.allclose(out[0, 0, 0], torch.full((16, 16), expected), atol=1e-5)
