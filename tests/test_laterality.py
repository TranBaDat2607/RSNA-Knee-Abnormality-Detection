import pandas as pd
import torch

from rsna_knee.config import LateralityConfig
from rsna_knee.dicom.laterality import laterality_maps, normalise_laterality


def _header_df(rows):
    return pd.DataFrame(rows)


def test_tag_is_authoritative_over_position():
    # study A: tag says L, position sign would say R (x > 0 -> L actually, so make it
    # disagree deliberately) -- tag must win regardless.
    rows = [
        {"StudyInstanceUID": "A", "Laterality": "L", "ImageLaterality": None,
         "ImagePositionPatient": "-50|0|0"},  # position sign alone would say "R"
    ]
    side, info = laterality_maps(_header_df(rows), LateralityConfig())
    assert side["A"] == "L"


def test_fallback_used_when_tag_and_position_agree_enough():
    rows = []
    # Five studies with both tag and position agreeing, to clear min_agreement=0.85.
    for i in range(5):
        rows.append({"StudyInstanceUID": f"agree{i}", "Laterality": "L",
                     "ImageLaterality": None, "ImagePositionPatient": "50|0|0"})
    # One study with only position available (no tag) -- should be filled by fallback.
    rows.append({"StudyInstanceUID": "notag", "Laterality": None,
                 "ImageLaterality": None, "ImagePositionPatient": "50|0|0"})
    side, info = laterality_maps(_header_df(rows), LateralityConfig(fallback="auto"))
    assert info["fallback_used"] is True
    assert side["notag"] == "L"


def test_fallback_disabled_leaves_tagless_study_unresolved():
    rows = [{"StudyInstanceUID": "notag", "Laterality": None,
             "ImageLaterality": None, "ImagePositionPatient": "50|0|0"}]
    side, info = laterality_maps(_header_df(rows), LateralityConfig(fallback="off"))
    assert side["notag"] is None


def test_position_near_isocentre_is_not_a_usable_cue():
    rows = [{"StudyInstanceUID": "centred", "Laterality": None,
             "ImageLaterality": None, "ImagePositionPatient": "1|0|0"}]
    side, info = laterality_maps(_header_df(rows), LateralityConfig(fallback="on", min_offset_mm=5.0))
    assert side["centred"] is None


def test_normalise_laterality_flips_coronal_axial_horizontally_for_right_knee():
    img = torch.zeros(3, 4, 4)
    img[..., 0] = 1.0  # marker on the left edge
    out = normalise_laterality(img, "Coronal", "R")
    assert out[..., -1].sum() == img[..., 0].sum()
    assert out[..., 0].sum() == 0


def test_normalise_laterality_reverses_slice_order_for_sagittal_right_knee():
    img = torch.arange(3).float().view(3, 1, 1).expand(3, 2, 2).clone()
    out = normalise_laterality(img, "Sagittal", "R")
    assert torch.equal(out, torch.flip(img, dims=[0]))


def test_normalise_laterality_is_identity_for_left_knee_or_unknown():
    img = torch.rand(3, 4, 4)
    assert torch.equal(normalise_laterality(img, "Coronal", "L"), img)
    assert torch.equal(normalise_laterality(img, "Coronal", None), img)
