import pandas as pd
import pytest

from rsna_knee.paths import find_llm_labels, find_root, plan_cache


def test_find_root_prefers_data_dir_when_it_has_test_csv_and_series_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    (data / "test_series").mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ["s1"]}).to_csv(data / "test.csv", index=False)

    assert find_root().resolve() == data.resolve()


def test_find_root_raises_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_root()


def test_find_llm_labels_prefers_local_data_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "llm_labels_full.csv").write_text("StudyInstanceUID\n")

    assert find_llm_labels().resolve() == (data / "llm_labels_full.csv").resolve()


def test_find_llm_labels_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_llm_labels()


def test_plan_cache_shrinks_groups_under_a_tight_budget():
    groups = plan_cache(n_study=100000, n_slot=6, img=224, cache_budget_gb=1.0,
                        group=3, n_group_max=3)
    assert 1 <= groups < 3


def test_plan_cache_allows_max_groups_under_a_generous_budget():
    groups = plan_cache(n_study=10, n_slot=6, img=224, cache_budget_gb=1000.0,
                        group=3, n_group_max=3)
    assert groups == 3
