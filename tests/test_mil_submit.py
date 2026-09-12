"""Submission fusion and validation (no GPU, no data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import TARGETS
from rsna_knee.mil.blend import rank_pct
from rsna_knee.mil.submit import fuse, informative, validate_submission


def _pred(seed, n=20):
    return np.random.default_rng(seed).random((n, 12))


def test_fuse_mixes_family_and_residual_gated_by_declared_weights():
    a, b, r = _pred(0), _pred(1), _pred(2)
    out = fuse({"a": a, "b": b}, {}, r, 0.4)
    expected = 0.6 * (rank_pct(a) + rank_pct(b)) / 2 + 0.4 * rank_pct(r)
    assert np.allclose(out, expected)


def test_fuse_drops_uninformative_arms_and_renormalises():
    a, r = _pred(0), _pred(2)
    dead = np.full((20, 12), 0.5)
    assert not informative(dead) and informative(a)
    assert np.allclose(fuse({"a": a, "dead": dead}, {}, r, 0.4), 0.6 * rank_pct(a) + 0.4 * rank_pct(r))
    assert np.allclose(fuse({"a": a}, {}, None, 0.4), rank_pct(a))          # residual-gated missing
    assert np.allclose(fuse({"dead": dead}, {}, r, 0.4), rank_pct(r))       # every MIL arm failed
    with pytest.raises(RuntimeError):
        fuse({"dead": dead}, {}, None, 0.4)


def test_validate_submission_contract():
    ids = [f"s{i}" for i in range(5)]
    good = pd.DataFrame({"StudyInstanceUID": ids, **{t: np.linspace(0, 1, 5) for t in TARGETS}})
    validate_submission(good, ids)
    with pytest.raises(ValueError):
        validate_submission(good.drop(columns=["ACL"]), ids)
    with pytest.raises(ValueError):
        validate_submission(good.iloc[:4], ids)
    bad = good.copy()
    bad.loc[0, "MCL"] = np.nan
    with pytest.raises(ValueError):
        validate_submission(bad, ids)
