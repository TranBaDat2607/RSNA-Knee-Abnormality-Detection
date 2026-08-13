import numpy as np

from rsna_knee.training.metrics import hanley_mcneil_se, macro_auc


def test_macro_auc_perfect_separation():
    y = np.array([[0], [0], [1], [1]])
    p = np.array([[0.1], [0.2], [0.8], [0.9]])
    assert macro_auc(y, p) == 1.0


def test_macro_auc_ignores_single_class_columns():
    # column 0 is separable, column 1 has only one class and must be excluded, not
    # counted as 0 or crash the mean.
    y = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])
    p = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
    assert macro_auc(y, p) == 1.0


def test_hanley_mcneil_se_shrinks_with_sample_size():
    se_small = hanley_mcneil_se(0.8, n_pos=5, n_neg=5)
    se_large = hanley_mcneil_se(0.8, n_pos=500, n_neg=500)
    assert se_large < se_small
    assert se_small > 0


def test_hanley_mcneil_se_nan_on_degenerate_input():
    assert np.isnan(hanley_mcneil_se(float("nan"), 5, 5))
    assert np.isnan(hanley_mcneil_se(0.8, 0, 5))
