"""Slot picking, slice spreading and stack assembly, without real DICOM files."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.mil.recipes import CORPUS44_336, SLOTS_44, SLOTS_64, Slot, VolumeRecipe
from rsna_knee.mil.volume import build_volume, pick_series, span_indices


def test_recipe_slice_counts():
    assert sum(s.n for s in SLOTS_64) == 64
    assert CORPUS44_336.n_slices == sum(s.n for s in SLOTS_44) == 44


def test_corpus_recipe_matches_the_parameters_that_reproduce_the_public_corpus():
    # Found by exhaustive search against the corpus stacks (E5c): anything else differs by 0.5-30/255.
    assert CORPUS44_336.span == (0.15, 0.85)
    assert CORPUS44_336.crop_mm == 140.0 and CORPUS44_336.img == 336


def test_span_indices_cover_the_requested_fraction():
    idx = span_indices(50, (0.02, 0.98), 18)
    assert len(idx) == 18 and idx[0] == 1 and idx[-1] == 48 and (np.diff(idx) >= 0).all()
    assert span_indices(1, (0.06, 0.94), 5).tolist() == [0] * 5


ROWS = [
    {"SeriesInstanceUID": "sag_t1", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 0},
    {"SeriesInstanceUID": "sag_pdfs", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1},
    {"SeriesInstanceUID": "cor_pdfs", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 1},
]


def test_pick_series_prefers_fluid_then_falls_back_and_never_reuses():
    assert pick_series(ROWS, "Sagittal", 1, set())["SeriesInstanceUID"] == "sag_pdfs"
    assert pick_series(ROWS, "Sagittal", 0, set())["SeriesInstanceUID"] == "sag_t1"
    assert pick_series(ROWS, "Coronal", 0, set())["SeriesInstanceUID"] == "cor_pdfs"  # fallback
    assert pick_series(ROWS, "Coronal", 0, {"cor_pdfs"}) is None
    assert pick_series(ROWS, "Axial", -1, set()) is None


def test_build_volume_fills_present_slots_and_leaves_missing_ones_empty():
    recipe = VolumeRecipe("t", 8, 140.0, (0.0, 1.0),
                          (Slot("Sagittal", 1, 3), Slot("Sagittal", 0, 2), Slot("Coronal", 0, 4),
                           Slot("Coronal", 1, 2), Slot("Axial", -1, 3)))
    opened = []

    def order_fn(series_dir):
        opened.append(series_dir.replace("\\", "/").split("/")[-1])
        return [(f"{series_dir}/{i}.dcm", 0.5) for i in range(10)], 0.5

    def read_fn(path):
        # a ramp inside every slice, offset per slice: after per-series windowing no picked
        # slice is uniformly black (an all-zero slice would count as empty, as in Raptor)
        i = int(path.replace("\\", "/").split("/")[-1].split(".")[0])
        return np.arange(400, dtype=np.float32).reshape(20, 20) + 100.0 * (i + 1)

    def resize_fn(a, spacing, crop_mm, img):
        return np.full((img, img), a.mean(), np.float32)

    vol, mask = build_volume("study", ROWS, "/root", recipe, order_fn, read_fn, resize_fn)
    assert vol.shape == (14, 8, 8)
    # sagittal fluid (3) + sagittal other (2) + coronal (4); the second coronal slot has no
    # unused series and axial has none at all -> 5 empty slices
    assert mask.tolist() == [1] * 9 + [0] * 5
    assert opened == ["sag_pdfs", "sag_t1", "cor_pdfs"]
    # intensities are windowed within each series, so the first and last picked slice of a
    # 3-slice spread over a 1..10 ramp land at the bottom and top of the range
    assert vol[0, 0, 0] < vol[1, 0, 0] < vol[2, 0, 0]


def test_crop_resize_takes_a_physical_centre_crop():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from rsna_knee.mil.volume import crop_resize

    a = np.zeros((100, 100), np.float32)
    a[25:75, 25:75] = 1.0
    out = crop_resize(a, spacing=1.0, crop_mm=50.0, img=10)
    assert out.shape == (10, 10) and np.allclose(out, 1.0)
    wide = crop_resize(a, spacing=0.5, crop_mm=200.0, img=10)  # crop bounded by the image
    assert wide.shape == (10, 10) and 0.0 < wide.mean() < 1.0
