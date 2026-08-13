import pandas as pd

from rsna_knee.dicom.header import annotate


def _series_df(rows):
    cols = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "PixelSpacing"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


def test_fatsat_detected_from_description_keyword():
    df = _series_df([{"SeriesDescription": "SAG PD FS", "PixelSpacing": "0.3|0.3"}])
    out = annotate(df)
    assert bool(out.loc[0, "fatsat"])


def test_fatsat_detected_from_scan_options_ge_style():
    df = _series_df([{"SeriesDescription": "cor pd", "ScanOptions": "SAT_GEMS|FS"}])
    out = annotate(df)
    assert bool(out.loc[0, "fatsat"])


def test_ge_spatial_saturation_option_alone_is_not_fatsat():
    # SAT_GEMS (spatial pre-saturation) must not be confused with fat saturation --
    # matching must be on exact tokens, not a "SAT" substring.
    df = _series_df([{"SeriesDescription": "cor pd", "ScanOptions": "SAT_GEMS"}])
    out = annotate(df)
    assert not bool(out.loc[0, "fatsat"])


def test_weight_classified_from_description_keywords():
    df = _series_df([
        {"SeriesDescription": "sag t1"},
        {"SeriesDescription": "sag t2"},
        {"SeriesDescription": "sag pd"},
    ])
    out = annotate(df)
    assert list(out["weight"]) == ["T1", "T2", "PD"]


def test_weight_falls_back_to_repetition_echo_time_when_description_is_uninformative():
    df = _series_df([
        {"SeriesDescription": "series 1", "RepetitionTime": "500", "EchoTime": "15"},
        {"SeriesDescription": "series 2", "RepetitionTime": "3000", "EchoTime": "90"},
    ])
    out = annotate(df)
    assert out.loc[0, "weight"] == "T1"
    assert out.loc[1, "weight"] == "T2"


def test_fluid_is_true_only_for_pd_and_t2():
    df = _series_df([
        {"SeriesDescription": "sag t1"},
        {"SeriesDescription": "sag t2"},
        {"SeriesDescription": "sag pd"},
    ])
    out = annotate(df)
    assert list(out["fluid"]) == [False, True, True]


def test_pixel_spacing_parsed_as_first_component():
    df = _series_df([{"SeriesDescription": "sag pd", "PixelSpacing": "0.34|0.34"}])
    out = annotate(df)
    assert abs(out.loc[0, "px"] - 0.34) < 1e-9
