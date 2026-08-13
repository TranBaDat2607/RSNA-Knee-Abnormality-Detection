import numpy as np
import pandas as pd

from rsna_knee.inference.submission import write_benchmark_submission, write_submission

TARGETS = ["A", "B"]


def test_write_benchmark_submission_fills_point_five(tmp_path):
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"StudyInstanceUID": ["s1", "s2"]}).to_csv(test_csv, index=False)
    out = tmp_path / "submission.csv"

    write_benchmark_submission(test_csv, TARGETS, str(out))

    sub = pd.read_csv(out)
    assert list(sub["StudyInstanceUID"]) == ["s1", "s2"]
    assert (sub[TARGETS] == 0.5).all().all()


def test_write_submission_aligns_onto_test_index_and_fills_missing(tmp_path):
    test_df = pd.DataFrame({"StudyInstanceUID": ["s1", "s2", "s3"]})
    rank_sum = np.array([[1.0, 0.0], [0.0, 1.0]])  # only covers s1, s2
    st_te = ["s1", "s2"]
    out = tmp_path / "submission.csv"

    sub = write_submission(rank_sum, n_models=1, st_te=st_te, test_df=test_df,
                            targets=TARGETS, out_path=str(out))

    assert list(sub["StudyInstanceUID"]) == ["s1", "s2", "s3"]
    assert sub.loc[sub.StudyInstanceUID == "s3", "A"].iloc[0] == 0.5  # unseen -> filled
    assert out.exists()


def test_write_submission_averages_over_n_models(tmp_path):
    test_df = pd.DataFrame({"StudyInstanceUID": ["s1"]})
    rank_sum = np.array([[2.0, 0.0]])  # accumulated over 2 models
    out = tmp_path / "submission.csv"

    sub = write_submission(rank_sum, n_models=2, st_te=["s1"], test_df=test_df,
                            targets=TARGETS, out_path=str(out))

    assert sub.loc[0, "A"] == 1.0
