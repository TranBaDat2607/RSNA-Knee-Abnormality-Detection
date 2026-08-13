"""Header-only DICOM probing and sequence-type recovery.

Reads one representative slice per series (``stop_before_pixels=True`` — cheap) to recover
the tags needed downstream: fat suppression, pulse-sequence weighting, laterality, pixel
spacing. No pixel data is touched here; that's :mod:`rsna_knee.dicom.sampling`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from ..config import FATSAT_OPTS, FATSAT_RX, HDR_TAGS, PD_RX, SEP_RX, T1_RX, T2_RX


def probe(item: tuple[str, str, str, str]) -> dict:
    """Read one representative slice's header for one series.

    ``item`` is ``(split, study_uid, series_uid, series_dir)``, the unit of work handed
    to the thread pool in :func:`walk`.
    """
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                              stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the whole walk
        row["err"] = str(exc)[:120]
    return row


def walk(root: Path, split: str, *, threads: int = 16) -> pd.DataFrame:
    """Probe the header of every series under ``root / split``.

    ``split`` is ``"train_series"`` or ``"test_series"`` — a directory of
    ``study_uid/series_uid/*.dcm``.
    """
    base = root / split
    items: list[tuple[str, str, str, str]] = []
    if not base.is_dir():
        return pd.DataFrame()
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Recover fat suppression and pulse-sequence weighting from the probed header.

    Adds ``fatsat`` (bool), ``weight`` (one of T1/T2/PD/GRE/UNK), ``fluid`` (bool, True
    for the two fluid-sensitive weightings), and ``px`` (in-plane pixel spacing, mm).
    """
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(SEP_RX, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions must be matched as exact
    # tokens; a substring test on "SAT" would fire on non-fat-sat series too.
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(T1_RX), desc.str.contains(T2_RX), desc.str.contains(PD_RX)

    df["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2",
                 np.where(pdw, "PD",
                          np.where(gre, "GRE",
                                   np.where(tr < 800, "T1",
                                            np.where(te > 60, "T2",
                                                     np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df
