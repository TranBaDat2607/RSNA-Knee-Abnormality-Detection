"""Corpus access, teacher tables, the training dataset and argument parsing (CPU, synthetic files)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from rsna_knee.config import TARGETS
from rsna_knee.mil.corpus import SLOT_BOUNDS, Corpus
from rsna_knee.mil.orientation import IDENTITY
from rsna_knee.mil.teachers import read_teacher
from rsna_knee.mil.train import StudyWindows, parse_args
from rsna_knee.mil.windows import to_model_input


def _fake_corpus(tmp_path):
    files = {}
    for part, n in (("all", 3), ("extra", 2)):
        vols = np.random.default_rng(len(part)).integers(1, 255, (n, 44, 6, 6), dtype=np.uint8)
        np.save(tmp_path / f"{part}_vols.npy", vols)
        np.save(tmp_path / f"{part}_masks.npy", np.ones((n, 44), np.uint8))
        np.save(tmp_path / f"{part}_ids.npy", np.array([f"{part}{i}" for i in range(n)], dtype=object))
        for kind in ("vols", "masks", "ids"):
            files[f"{part}_{kind}.npy"] = str(tmp_path / f"{part}_{kind}.npy")
    return Corpus(files.__getitem__), files


def test_corpus_rows_span_both_parts_and_pickle_without_memmaps(tmp_path):
    corpus, files = _fake_corpus(tmp_path)
    assert len(corpus) == 5 and corpus.row["extra1"] == 4 and SLOT_BOUNDS == (0, 12, 22, 30, 36, 44)
    extra = np.load(files["extra_vols.npy"])
    assert np.array_equal(corpus.volume(4), extra[1])
    assert pickle.loads(pickle.dumps(corpus))._vols is None


def test_study_windows_train_eval_and_canonical(tmp_path):
    corpus, _ = _fake_corpus(tmp_path)
    y = np.random.rand(2, 12).astype(np.float32)
    flip = [IDENTITY, IDENTITY, (False, True, False, False), IDENTITY, IDENTITY]
    train = StudyWindows(corpus, [0, 3], y, k=12, train=True, seed=0)
    x, t, g = train[1]
    assert x.shape == (12, 3, 6, 6) and x.dtype == torch.uint8 and t.shape == (12,) and 0.9 <= g.item() <= 1.1
    plain = StudyWindows(corpus, [0], y[:1], k=44, train=False, seed=0)
    canon = StudyWindows(corpus, [0], y[:1], k=44, train=False, seed=0, canonical={0: flip})
    xp, xc = plain[0][0].numpy(), canon[0][0].numpy()
    assert plain[0][2].item() == 1.0
    from rsna_knee.mil.windows import eval_centres
    slots = np.searchsorted(SLOT_BOUNDS, eval_centres(corpus.masks[0], 44), side="right") - 1
    assert np.array_equal(xc[slots == 2], xp[slots == 2][..., ::-1])
    assert np.array_equal(xc[slots != 2], xp[slots != 2])


def test_gain_scales_before_normalisation():
    x = torch.full((2, 3, 3, 4, 4), 128, dtype=torch.uint8)
    out = to_model_input(x, 4, gain=torch.tensor([1.0, 0.5]))
    assert out.shape == (2, 3, 3, 4, 4)
    assert torch.all(out[0] > out[1])


def test_read_teacher_averages_its_tables(tmp_path):
    rows = ["s1", "s2", "s3"]
    a = pd.DataFrame({"StudyInstanceUID": rows, **{t: [0.2, 0.8, 1.0] for t in TARGETS}})
    b = pd.DataFrame({"StudyInstanceUID": rows[:2], **{t: [0.4, 0.6] for t in TARGETS}})
    a.to_csv(tmp_path / "report_labels_v4hybrid.csv", index=False)
    b.to_csv(tmp_path / "llm_labels_v4_blend.csv", index=False)
    import rsna_knee.mil.teachers as teachers
    teachers.TEACHERS["_test_pair"] = ("report_labels_v4hybrid.csv", "llm_labels_v4_blend.csv")
    try:
        t = read_teacher("_test_pair", lambda f: str(tmp_path / f))
    finally:
        teachers.TEACHERS.pop("_test_pair")
    assert list(t.index) == ["s1", "s2"] and np.allclose(t["ACL"].values, [0.3, 0.7])


def test_parse_args_defaults_and_flags():
    a = parse_args([])
    assert a.teacher == "flight" and a.tag == "flight" and not a.canonical and a.k == 12
    b = parse_args(["--teacher", "raptor", "--canonical", "--side_tag_only", "--train_n", "2000"])
    assert b.teacher == "raptor" and b.canonical and b.side_tag_only and b.train_n == 2000
