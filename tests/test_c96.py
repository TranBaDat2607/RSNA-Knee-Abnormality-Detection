import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rsna_knee.mil import c96
from rsna_knee.mil.train import StudyWindows


def _write_study(path, rng, filled):
    mask = np.zeros(96, np.uint8)
    mask[filled] = 1
    bufs, offs = [], [0]
    for i in range(96):
        b = b""
        if mask[i]:
            img = np.full((c96.STORED, c96.STORED), (i * 2) % 256, np.uint8)
            img[:, :16] = 255  # left edge marker, to see width flips
            b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
        bufs.append(b)
        offs.append(offs[-1] + len(b))
    np.savez(path, jpg=np.frombuffer(b"".join(bufs), np.uint8), offs=np.array(offs, np.int64), mask=mask)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "c96").mkdir()
    (tmp_path / "c96_index.csv").write_text("StudyInstanceUID\n")
    rng = np.random.default_rng(0)
    _write_study(tmp_path / "c96" / "s1.npz", rng, list(range(0, 70)))
    _write_study(tmp_path / "c96" / "s2.npz", rng, list(range(26, 96)))
    return c96.C96Corpus(lambda name: str(tmp_path / name))


def test_lazy_volume_decodes_requested_slices(corpus):
    v = corpus.volume(corpus.row["s1"])
    got = v[np.array([10, 10, 80])]
    assert got.shape == (3, c96.STORED, c96.STORED)
    assert abs(int(got[0, 100, 100]) - 20) <= 2 and (got[2] == 0).all()  # slice 80 is empty in s1


def test_mirror_reverses_sagittal_channels_and_flips_other_planes():
    w = np.arange(2 * 3 * 4 * 4, dtype=np.uint8).reshape(2, 3, 4, 4)
    m = c96.mirror(w, np.array([10, 60]))  # slot 0 (sagittal), slot 2 (coronal)
    assert (m[0] == w[0, ::-1]).all()
    assert (m[1] == w[1, ..., ::-1]).all()


def test_centre_crop_matches_nartaa_inference_crop():
    full = np.arange(384 * 384).reshape(384, 384)
    stored = full[16:368, 16:368]  # what the cache keeps
    assert (c96.crop(stored[None, None], 320)[0, 0] == full[32:352, 32:352]).all()


def test_study_windows_crop_and_mirror(corpus):
    rows = [corpus.row["s1"], corpus.row["s2"]]
    y = np.zeros((2, 12), np.float32)
    ds = StudyWindows(corpus, rows, y, k=6, train=True, seed=0, crop=320, mirror_p=1.0)
    x, t, g = ds[0]
    assert tuple(x.shape) == (6, 3, 320, 320) and x.dtype.is_floating_point is False
    ev = StudyWindows(corpus, rows, y, k=94, train=False, seed=0, crop=320)
    assert tuple(ev[1][0].shape) == (94, 3, 320, 320)
