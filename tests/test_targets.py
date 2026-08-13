import numpy as np
import pandas as pd
import pytest

from rsna_knee.labels.targets import (
    build_targets,
    gold_positions,
    load_gold,
    report_hash_folds,
    usable_rows,
)

TARGETS = ["A", "B"]


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "StudyInstanceUID": ["gold1", "llm1", "neither1"],
        "Report": ["No abnormality.", "ACL tear noted.", "Effusion present."],
        "A": [1.0, np.nan, np.nan],
        "B": [0.0, np.nan, np.nan],
    })


@pytest.fixture
def llm_labels():
    return pd.DataFrame({
        "StudyInstanceUID": ["llm1"],
        "A": [0.9],
        "B": [0.1],
    }).set_index("StudyInstanceUID")


def test_gold_study_uses_gold_labels_at_full_weight(train_df, llm_labels):
    studies = ["gold1", "llm1", "neither1"]
    y, w = build_targets(studies, train_df, llm_labels, TARGETS, gold_weight=3.0)
    assert list(y[0]) == [1.0, 0.0]
    assert list(w[0]) == [3.0, 3.0]


def test_llm_study_uses_llm_labels_weighted_by_distance_from_midpoint(train_df, llm_labels):
    studies = ["gold1", "llm1", "neither1"]
    y, w = build_targets(studies, train_df, llm_labels, TARGETS, gold_weight=3.0)
    assert np.allclose(y[1], [0.9, 0.1])
    # w = 0.25 + 0.75 * |r - 0.5| * 2
    expected_a = 0.25 + 0.75 * abs(0.9 - 0.5) * 2.0
    expected_b = 0.25 + 0.75 * abs(0.1 - 0.5) * 2.0
    assert np.allclose(w[1], [expected_a, expected_b])


def test_study_with_neither_source_gets_zero_weight_and_is_dropped(train_df, llm_labels):
    studies = ["gold1", "llm1", "neither1"]
    y, w = build_targets(studies, train_df, llm_labels, TARGETS, gold_weight=3.0)
    assert list(w[2]) == [0.0, 0.0]
    assert list(usable_rows(w)) == [0, 1]


def test_load_gold_drops_partially_annotated_rows():
    df = pd.DataFrame({
        "StudyInstanceUID": ["full", "partial"],
        "A": [1.0, 1.0],
        "B": [0.0, np.nan],
    })
    gold = load_gold(df, TARGETS)
    assert list(gold.index) == ["full"]


def test_report_hash_folds_is_deterministic_and_groups_identical_reports():
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2", "s3"],
        "Report": ["same text", "same text", "different text"],
    })
    grp = report_hash_folds(["s1", "s2", "s3"], df, n_folds=4)
    grp_again = report_hash_folds(["s1", "s2", "s3"], df, n_folds=4)
    assert list(grp) == list(grp_again)          # deterministic
    assert grp[0] == grp[1]                      # identical reports share a fold
    assert 0 <= grp[2] < 4


def test_gold_positions_maps_to_index_within_studies_list():
    gold = pd.DataFrame({"A": [1.0]}, index=pd.Index(["gold1"], name="StudyInstanceUID"))
    studies = ["neither1", "gold1", "llm1"]
    pos = gold_positions(studies, gold)
    assert list(pos) == [1]
