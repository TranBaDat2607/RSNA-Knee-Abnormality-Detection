import pandas as pd

from rsna_knee.config import SLOTS_RECOVERED
from rsna_knee.dicom.slots import pick_slots


def _series(rows):
    return pd.DataFrame(rows)


def test_picks_the_matching_slot_by_plane_fluid_fatsat():
    plane_map = {"s1": "Sagittal"}
    df = _series([
        {"StudyInstanceUID": "A", "SeriesInstanceUID": "s1", "fatsat": True,
         "fluid": True, "n_slices": 20},
    ])
    out = pick_slots(df, plane_map, SLOTS_RECOVERED)
    assert "SAG_FLUID_FS" in out["A"]
    assert out["A"]["SAG_FLUID_FS"]["SeriesInstanceUID"] == "s1"


def test_tie_break_prefers_more_slices():
    plane_map = {"thin": "Sagittal", "thick": "Sagittal"}
    df = _series([
        {"StudyInstanceUID": "A", "SeriesInstanceUID": "thin", "fatsat": True,
         "fluid": True, "n_slices": 10},
        {"StudyInstanceUID": "A", "SeriesInstanceUID": "thick", "fatsat": True,
         "fluid": True, "n_slices": 30},
    ])
    out = pick_slots(df, plane_map, SLOTS_RECOVERED)
    assert out["A"]["SAG_FLUID_FS"]["SeriesInstanceUID"] == "thick"


def test_missing_slot_is_simply_absent():
    plane_map = {"s1": "Axial"}
    df = _series([
        {"StudyInstanceUID": "A", "SeriesInstanceUID": "s1", "fatsat": True,
         "fluid": True, "n_slices": 20},
    ])
    out = pick_slots(df, plane_map, SLOTS_RECOVERED)
    assert "SAG_FLUID_FS" not in out["A"]
    assert "AX_FLUID_FS" in out["A"]


def test_t1_slot_falls_back_to_any_non_fatsat_series_in_plane():
    # COR_T1 wants fluid=False, fs=False. No exact match exists, but a non-fat-sat
    # coronal series with fluid=True is present -- the scarcity fallback should still
    # pick it rather than leaving the slot empty.
    plane_map = {"s1": "Coronal"}
    df = _series([
        {"StudyInstanceUID": "A", "SeriesInstanceUID": "s1", "fatsat": False,
         "fluid": True, "n_slices": 20},
    ])
    out = pick_slots(df, plane_map, SLOTS_RECOVERED)
    assert "COR_T1" in out["A"]
    assert out["A"]["COR_T1"]["SeriesInstanceUID"] == "s1"
