"""Rank-space fusion helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rsna_knee.mil.blend import logit_blend, rank_pct, weighted_rank_mean


def test_rank_pct_averages_ties_per_column():
    x = np.array([[0.1, 5.0], [0.3, 5.0], [0.3, 1.0], [0.9, 7.0]])
    r = rank_pct(x)
    assert r[:, 0].tolist() == [0.25, 0.625, 0.625, 1.0]
    assert r[:, 1].tolist() == [0.625, 0.625, 0.25, 1.0]


def test_weighted_rank_mean_ignores_calibration_and_normalises_weights():
    rng = np.random.default_rng(0)
    a, b = rng.random((50, 3)), rng.random((50, 3))
    base = weighted_rank_mean({"a": a, "b": b}, {"a": 3.0, "b": 1.0})
    squashed = weighted_rank_mean({"a": a ** 5, "b": np.log(b)}, {"a": 0.75, "b": 0.25})
    assert np.allclose(base, squashed)
    assert np.allclose(weighted_rank_mean({"a": a}), rank_pct(a))
    with pytest.raises(ValueError):
        weighted_rank_mean({"a": a}, {"a": -1.0})


def test_blending_an_arm_with_itself_keeps_its_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = y * 0.3 + rng.random(200)
    blended = weighted_rank_mean({"x": p[:, None], "y": p[:, None]})[:, 0]
    assert roc_auc_score(y, blended) == pytest.approx(roc_auc_score(y, p))


def test_logit_blend_endpoints():
    a, b = np.array([0.2, 0.7]), np.array([0.9, 0.1])
    assert np.allclose(logit_blend(a, b, 0.0), a, atol=1e-6)
    assert np.allclose(logit_blend(a, b, 1.0), b, atol=1e-6)
