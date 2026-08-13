"""Sampling one series at a fixed physical scale.

A DICOM slice of ``N x N`` pixels with spacing ``s`` mm/pixel covers ``N*s`` millimetres
of anatomy; both vary widely across this corpus, so resizing every slice to a fixed pixel
grid hands the encoder images whose physical scale differs by roughly a factor of three.
Cropping to a constant physical extent (``crop_mm``) before resizing removes that nuisance
transform instead of asking the network to learn it away.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pydicom


def read_slot(rec: pd.Series, n_slice: int, out_size: int, crop_mm: float) -> torch.Tensor | None:
    """Read ``n_slice`` physically-spread slices from one series, at ``out_size`` pixels.

    Returns a ``uint8`` tensor ``[n_slice, out_size, out_size]``, intensity-normalised
    per-series to its 1st-99th percentile (percentiles rather than min/max because MR
    intensity has no absolute scale, and a single bright vessel would otherwise compress
    the whole dynamic range). Returns ``None`` if the series has no files.
    """
    files, d, px = rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    # Spread the samples over the central 60% of the stack: the outer slices of a knee
    # series are mostly soft tissue outside the joint.
    lo, hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:  # noqa: BLE001 - a broken slice becomes a blank one, not a crash
            a = np.zeros((out_size, out_size), dtype=np.float32)
        planes.append(a)

    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    # Constant physical extent, then resize: PixelSpacing varies ~3.4x across the corpus.
    if px and np.isfinite(px) and px > 0:
        want = int(round(crop_mm / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    # uint8, not float32: these buffers queue up between reader threads and the encoder,
    # and intensity is already normalised into [0, 1] here, so eight bits cost nothing a
    # bilinear resize hasn't already cost, and the queue is a quarter the size.
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
