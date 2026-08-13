"""Choosing one series per slot per study."""

from __future__ import annotations

import pandas as pd

from ..config import SlotDef


def pick_slots(series_df: pd.DataFrame, plane_map: dict[str, str],
                slots: list[SlotDef]) -> dict[str, dict[str, pd.Series]]:
    """One series per slot per study.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, which benefits the multi-slice sampler in :mod:`sampling`.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out: dict[str, dict[str, pd.Series]] = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen: dict[str, pd.Series] = {}
        for name, plane, fluid, fs in slots:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None means "do not condition on weighting" - the SLOTS_PUBLIC scheme,
            # where the single provided flag stands in for both axes at once.
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if len(cand) == 0 and fluid is False:
                # T1 slots are the scarcest; fall back to any non-fat-sat, non-fluid
                # series in the plane before giving up on the slot entirely.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out
