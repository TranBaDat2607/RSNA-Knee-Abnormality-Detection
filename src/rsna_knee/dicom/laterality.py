"""Left/right resolution and the mirror-onto-one-convention normalisation.

Four of the twelve targets (the two menisci, medial/lateral OA) are medial/lateral pairs,
which are only meaningful once every study has been mapped onto one knee convention. The
DICOM ``Laterality``/``ImageLaterality`` tag is authoritative where present; the patient
x-coordinate sign stands in where it's absent, but only once it's shown to agree with the
tag often enough on the studies that have both — a wrong flip is worse than no flip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import LateralityConfig
from ..logging_utils import log


def _tag_side(g: pd.DataFrame) -> str | None:
    v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
    if "ImageLaterality" in g.columns:
        v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
    v = [x[0] for x in v if x and x[0] in ("L", "R")]
    return v[0] if v else None


def _position_side(g: pd.DataFrame, min_offset_mm: float) -> str | None:
    xs = []
    for s in g.get("ImagePositionPatient", pd.Series(dtype=object)).dropna():
        try:
            xs.append(float(str(s).split("|")[0]))
        except Exception:  # noqa: BLE001 - a malformed tag just contributes nothing
            pass
    if not xs:
        return None
    x = float(np.median(xs))
    if abs(x) < min_offset_mm:
        return None                      # centred in the coil: the sign carries no information
    return "R" if x < 0 else "L"         # LPS convention: the right knee sits at negative x


def laterality_maps(h: pd.DataFrame, cfg: LateralityConfig) -> tuple[dict[str, str | None], dict]:
    """Return ``(side_by_study, diagnostics)``.

    The position-based fallback is only used if it earns it: ``cfg.fallback == "auto"``
    enables it solely when it agrees with the tag on at least ``cfg.min_agreement`` of the
    studies where both are available.
    """
    tag: dict[str, str | None] = {}
    pos: dict[str, str | None] = {}
    for st, g in h.groupby("StudyInstanceUID"):
        tag[st] = _tag_side(g)
        pos[st] = _position_side(g, cfg.min_offset_mm)

    both = [st for st in tag if tag[st] and pos[st]]
    agree = float(np.mean([tag[st] == pos[st] for st in both])) if both else np.nan
    have_tag = float(np.mean([v is not None for v in tag.values()])) if tag else 0.0

    if cfg.fallback == "on":
        use = True
    elif cfg.fallback == "off":
        use = False
    else:
        use = bool(both) and np.isfinite(agree) and agree >= cfg.min_agreement

    side = {st: (tag[st] or (pos[st] if use else None)) for st in tag}
    covered = float(np.mean([v is not None for v in side.values()])) if side else 0.0
    info = {"tag_coverage": have_tag, "agreement": agree, "n_compared": len(both),
            "fallback_used": use, "final_coverage": covered}
    log(f"laterality: tag on {have_tag:.1%} of studies, x-sign agrees with it on "
        f"{agree:.1%} of {len(both)} comparable studies, fallback "
        f"{'enabled' if use else 'disabled'} -> {covered:.1%} normalised")
    return side, info


def normalise_laterality(img: torch.Tensor, plane: str, lat: str | None) -> torch.Tensor:
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks are not mirror
    images of each other — the slice order runs medial-to-lateral in opposite directions —
    so the slice (channel) order is reversed instead of the pixels.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
