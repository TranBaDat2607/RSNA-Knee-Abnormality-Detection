"""DICOM study -> fixed ``uint8`` slice stack for a :class:`~rsna_knee.mil.recipes.VolumeRecipe`.

This reproduces the preprocessing the public Raptor checkpoints were trained on, including its
quirks (glob order for tied slice positions, truncating ``* 255`` cast, 2nd-98th percentile
window per series), because a checkpoint's gold-58 score only holds on its own preprocessing:

* series are picked per slot by plane, preferring the requested fluid sensitivity, never reusing
  a series for two slots;
* slices are ordered by position along the slice normal (``ImagePositionPatient`` projected on
  the cross product of ``ImageOrientationPatient``), falling back to ``InstanceNumber``;
* ``n`` slices are spread evenly over the recipe's span of the series;
* intensities are windowed to the series' 2nd-98th percentile, then each slice is centre-cropped
  to ``crop_mm`` millimetres using its pixel spacing and area-resized to ``img`` pixels.

Empty slots stay zero and are excluded by the returned mask.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .recipes import VolumeRecipe

SeriesRows = Sequence[Mapping]


def order_series_files(series_dir: str) -> tuple[list[tuple[str, float]], float]:
    """``[(path, pixel_spacing), ...]`` sorted along the slice normal, plus the median spacing."""
    import pydicom

    records, spacings = [], []
    for f in glob.glob(os.path.join(series_dir, "*.dcm")):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            records.append((pos, f, ps))
        except Exception:  # noqa: BLE001 - an unreadable header sorts first, it doesn't abort the study
            records.append((0.0, f, 0.5))
    records.sort(key=lambda r: r[0])
    return [(f, ps) for _, f, ps in records], (float(np.median(spacings)) if spacings else 0.5)


def read_pixels(path: str) -> np.ndarray:
    """Modality-LUT-applied float32 pixels, with MONOCHROME1 inverted."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    ds = pydicom.dcmread(path)
    a = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a: np.ndarray, spacing: float, crop_mm: float, img: int) -> np.ndarray:
    """Centre-crop ``crop_mm`` millimetres (bounded by the image) and area-resize to ``img``."""
    import cv2

    h, w = a.shape
    side = min(int(round(crop_mm / max(spacing, 1e-3))), min(h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(a[y0:y0 + side, x0:x0 + side], (img, img), interpolation=cv2.INTER_AREA)


def pick_series(rows: SeriesRows, plane: str, fluid: int, used: set[str]) -> Mapping | None:
    candidates = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def span_indices(n_files: int, span: tuple[float, float], k: int) -> np.ndarray:
    if n_files <= 1:
        return np.zeros(k, dtype=int)
    lo = int(n_files * span[0])
    hi = max(int(n_files * span[1]) - 1, lo)
    return np.linspace(lo, hi, k).round().astype(int)


def build_volume(study_uid: str, rows: SeriesRows, series_root: str, recipe: VolumeRecipe,
                 order_fn: Callable = order_series_files, read_fn: Callable = read_pixels,
                 resize_fn: Callable = crop_resize) -> tuple[np.ndarray, np.ndarray]:
    """``(stack uint8 [n_slices, img, img], mask uint8 [n_slices])``."""
    n_total = recipe.n_slices
    vol = np.zeros((n_total, recipe.img, recipe.img), np.uint8)
    idx, used = 0, set()
    for slot in recipe.slots:
        row = pick_series(rows, slot.plane, slot.fluid, used)
        if row is None:
            idx += slot.n
            continue
        used.add(row["SeriesInstanceUID"])
        files, median_spacing = order_fn(os.path.join(series_root, study_uid, str(row["SeriesInstanceUID"])))
        if not files:
            idx += slot.n
            continue
        arrays, spacings = [], []
        for p in span_indices(len(files), recipe.span, slot.n):
            path, ps = files[min(int(p), len(files) - 1)]
            try:
                arrays.append(read_fn(path))
                spacings.append(ps)
            except Exception:  # noqa: BLE001 - one undecodable slice stays blank
                arrays.append(None)
                spacings.append(median_spacing)
        valid = [a for a in arrays if a is not None]
        lo, hi = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
                  if valid else (0.0, 1.0))
        for a, ps in zip(arrays, spacings):
            if idx >= n_total:
                break
            if a is None:
                idx += 1
                continue
            a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
            a = resize_fn(a, ps if ps > 0 else median_spacing, recipe.crop_mm, recipe.img)
            vol[idx] = (a * 255).astype(np.uint8)
            idx += 1
        if idx >= n_total:
            break
    mask = (vol.reshape(n_total, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask
