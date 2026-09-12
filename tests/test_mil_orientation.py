"""Canonical anatomical orientation: synthetic DICOM geometry, no image data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.mil.orientation import (IDENTITY, apply_transform, canonical_transforms, resolve_side,
                                       slot_transform, transform_windows)

STANDARD = {"Sagittal": ((0, 1, 0), (0, 0, -1)), "Coronal": ((1, 0, 0), (0, 0, -1)), "Axial": ((1, 0, 0), (0, 1, 0))}


def test_standard_orientation_only_mirrors_by_side():
    # The only geometry seen in this competition (E5a): transforms depend on the side alone.
    assert slot_transform("Coronal", *STANDARD["Coronal"], "L") == IDENTITY
    assert slot_transform("Coronal", *STANDARD["Coronal"], "R") == (False, True, False, False)
    assert slot_transform("Axial", *STANDARD["Axial"], "L") == IDENTITY
    assert slot_transform("Axial", *STANDARD["Axial"], "R") == (False, True, False, False)
    # sagittal stack ascends along cross(+y, -z) = -x: already toward lateral for a right knee
    assert slot_transform("Sagittal", *STANDARD["Sagittal"], "R") == IDENTITY
    assert slot_transform("Sagittal", *STANDARD["Sagittal"], "L") == (False, False, False, True)
    for plane in STANDARD:
        assert slot_transform(plane, *STANDARD[plane], None) == IDENTITY


def test_non_standard_geometry_is_corrected_or_left_alone():
    assert slot_transform("Sagittal", (0, -1, 0), (0, 0, -1), None) == (False, True, False, False)
    assert slot_transform("Coronal", (1, 0, 0), (0, 0, 1), "L")[:3] == (False, False, True)
    assert slot_transform("Axial", (1, 0, 0), (0, -1, 0), "R")[:3] == (False, True, True)
    t = slot_transform("Coronal", (0, 0, -1), (1, 0, 0), "L")  # transposed acquisition
    assert t[0] and not t[1] and not t[2]
    assert slot_transform("Axial", (0, 1, 0), (0, 0, -1), "L") == IDENTITY  # geometry contradicts the plane


def test_apply_and_transform_windows_act_per_slot():
    w = np.arange(4 * 3 * 2 * 3, dtype=np.uint8).reshape(4, 3, 2, 3)
    assert np.array_equal(apply_transform(w, (False, True, False, False)), w[..., ::-1])
    assert np.array_equal(apply_transform(w, (False, False, True, False)), w[..., ::-1, :])
    assert np.array_equal(apply_transform(w, (False, False, False, True)), w[:, ::-1])
    assert apply_transform(w, (True, False, False, False)).shape == (4, 3, 3, 2)
    centres = np.array([5, 12, 25, 40])  # slots 0, 1, 2, 4 of the corpus layout
    flip_coronal = [IDENTITY, IDENTITY, (False, True, False, False), IDENTITY, IDENTITY]
    out = transform_windows(w, centres, flip_coronal)
    assert np.array_equal(out[[0, 1, 3]], w[[0, 1, 3]]) and np.array_equal(out[2], w[2, ..., ::-1])


def test_resolve_side_prefers_the_tag_and_can_ignore_geometry():
    assert resolve_side("R", 150.0) == "R"
    assert resolve_side(None, -30.0) == "R" and resolve_side(None, 30.0) == "L"
    assert resolve_side(None, 2.0) is None                     # too close to isocentre
    assert resolve_side(np.nan, -30.0, tag_only=True) is None  # untagged sites don't encode side in x


def test_canonical_transforms_from_tables():
    orientation = pd.DataFrame([
        {"StudyInstanceUID": "a", "slot": 2, "plane": "Coronal", "row_x": 1, "row_y": 0, "row_z": 0, "col_x": 0, "col_y": 0, "col_z": -1},
        {"StudyInstanceUID": "a", "slot": 0, "plane": "Sagittal", "row_x": 0, "row_y": 1, "row_z": 0, "col_x": 0, "col_y": 0, "col_z": -1},
        {"StudyInstanceUID": "b", "slot": 2, "plane": "Coronal", "row_x": 1, "row_y": 0, "row_z": 0, "col_x": 0, "col_y": 0, "col_z": -1},
    ])
    sides = pd.DataFrame([{"StudyInstanceUID": "a", "tag_side": "R", "x_median": np.nan},
                          {"StudyInstanceUID": "b", "tag_side": np.nan, "x_median": 120.0}])
    t = canonical_transforms(orientation, sides, {"a": 0, "b": 1})
    assert t[0][2] == (False, True, False, False) and t[0][0] == IDENTITY
    assert t[1][2] == IDENTITY  # geometry says left
    assert canonical_transforms(orientation, sides, {"a": 0, "b": 1}, tag_only=True)[1][2] == IDENTITY
