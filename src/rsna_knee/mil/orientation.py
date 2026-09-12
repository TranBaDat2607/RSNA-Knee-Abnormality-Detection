"""One canonical anatomical orientation for every window.

Every series in this competition is stored in the standard DICOM orientation (E5a: 100% of the
corpus's chosen series), and patient coordinates are LPS (+x patient left, +y posterior,
+z superior). Coronal and axial columns therefore always run toward the patient's *left*, so a
right knee's lateral compartment sits on the opposite image side from a left knee's, and the
sagittal stack runs toward lateral for one side and medial for the other. The canonical
orientation used here is:

    Sagittal: columns run posterior, rows inferior, stack toward lateral
    Coronal:  columns run lateral,   rows inferior, stack toward posterior
    Axial:    columns run lateral,   rows posterior, stack toward superior

"Lateral" is +x for a left knee and -x for a right knee; when the side is unknown, lateral axes are
left as acquired. Side comes from the DICOM tag where present; patient-x geometry is only
trustworthy on sites that also write the tag (E5a: untagged sites report almost every study as
"right" once |x| is large), so ``tag_only`` exists for deployment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .corpus import SLOT_BOUNDS

Transform = tuple[bool, bool, bool, bool]  # (transpose, flip_h, flip_v, reverse_channels)
IDENTITY: Transform = (False, False, False, False)

_CANON = {"Sagittal": (("y", 1), ("z", -1), ("x", "lat")),
          "Coronal": (("x", "lat"), ("z", -1), ("y", 1)),
          "Axial": (("x", "lat"), ("y", 1), ("z", 1))}
_AX = {"x": 0, "y": 1, "z": 2}


def slot_transform(plane: str, row: Sequence[float], col: Sequence[float], side: str | None) -> Transform:
    """Transform mapping one slot's acquired slices onto the canonical orientation.

    ``row`` / ``col`` are the DICOM ImageOrientationPatient direction cosines of image rows and
    columns. Geometry that does not match the plane (oblique or mislabelled series) gets the identity.
    """
    row0, col0 = np.asarray(row, float), np.asarray(col, float)
    lat = {"L": 1, "R": -1}.get(side)
    (h_ax, h_s), (v_ax, v_s), (t_ax, t_s) = [(a, lat if s == "lat" else s) for a, s in _CANON[plane]]
    r, c = row0, col0
    ra, ca = int(np.argmax(np.abs(r))), int(np.argmax(np.abs(c)))
    transpose = False
    if ra == _AX[v_ax] and ca == _AX[h_ax]:
        r, c, ra, ca, transpose = c, r, ca, ra, True
    if ra != _AX[h_ax] or ca != _AX[v_ax]:
        return IDENTITY
    flip_h = h_s is not None and int(np.sign(r[ra])) != h_s
    flip_v = int(np.sign(c[ca])) != v_s
    normal = np.cross(row0, col0)  # stacks ascend along the *acquired* slice normal
    na = int(np.argmax(np.abs(normal)))
    reverse = t_s is not None and na == _AX[t_ax] and int(np.sign(normal[na])) != t_s
    return (transpose, bool(flip_h), bool(flip_v), bool(reverse))


def apply_transform(windows: np.ndarray, t: Transform) -> np.ndarray:
    """Apply ``t`` to uint8 windows ``(n, 3, H, W)`` of one slot."""
    transpose, flip_h, flip_v, reverse = t
    if transpose:
        windows = windows.swapaxes(-1, -2)
    if flip_h:
        windows = windows[..., ::-1]
    if flip_v:
        windows = windows[..., ::-1, :]
    if reverse:
        windows = windows[:, ::-1]
    return np.ascontiguousarray(windows)


def transform_windows(windows: np.ndarray, centres: np.ndarray, transforms: Sequence[Transform],
                      bounds: Sequence[int] = SLOT_BOUNDS) -> np.ndarray:
    """Apply each slot's transform to the windows centred inside that slot."""
    slots = np.searchsorted(bounds, centres, side="right") - 1
    out = windows.copy()
    for s, t in enumerate(transforms):
        sel = np.flatnonzero(slots == s)
        if len(sel) and t != IDENTITY:
            out[sel] = apply_transform(windows[sel], t)
    return out


def resolve_side(tag_side, x_median, tag_only: bool = False, min_offset_mm: float = 5.0) -> str | None:
    if isinstance(tag_side, str) and tag_side[:1] in ("L", "R"):
        return tag_side[:1]
    if tag_only or x_median is None or not np.isfinite(x_median) or abs(x_median) < min_offset_mm:
        return None
    return "R" if x_median < 0 else "L"


def canonical_transforms(orientation: pd.DataFrame, sides: pd.DataFrame, corpus_row: Mapping[str, int],
                         tag_only: bool = False, min_offset_mm: float = 5.0,
                         n_slots: int = len(SLOT_BOUNDS) - 1) -> dict[int, list[Transform]]:
    """``{corpus row: [transform per slot]}`` from per-slot geometry and per-study side tables.

    ``orientation`` columns: StudyInstanceUID, slot, plane, row_x/y/z, col_x/y/z (one row per filled
    slot). ``sides`` columns: StudyInstanceUID, tag_side, x_median.
    """
    side_of = {r.StudyInstanceUID: resolve_side(r.tag_side, r.x_median, tag_only, min_offset_mm)
               for r in sides.itertuples()}
    out: dict[int, list[Transform]] = {}
    for r in orientation.dropna(subset=["row_x"]).itertuples():
        i = corpus_row.get(r.StudyInstanceUID)
        if i is None:
            continue
        t = slot_transform(r.plane, (r.row_x, r.row_y, r.row_z), (r.col_x, r.col_y, r.col_z),
                           side_of.get(r.StudyInstanceUID))
        out.setdefault(i, [IDENTITY] * n_slots)[int(r.slot)] = t
    return out
