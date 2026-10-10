"""The d96 training cache: nartaa's 96-slice native-384 study volumes (``rsna-knee-c96-cache`` kernel).

Layout (nartaa / dreaddevelopment ``build_corpus96``): five slots of 26/22/18/12/18 slices, sagittal
fluid-sensitive, sagittal other, coronal fluid-sensitive, coronal other, axial; slices span 2-98 % of each
series, 140 mm field cropped to 384 px. The cache keeps the centre 352 px of every filled slice as JPEG
(q = 90): training crops 320 px with jitter, evaluation the centre 320 px, which is exactly nartaa's
inference crop ``[32:352]`` of the 384 px slice.

``c96/<uid>.npz`` holds ``jpg`` (all slices' bytes concatenated), ``offs`` (97 byte offsets) and ``mask``.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

import numpy as np

SLOT_SIZES = (26, 22, 18, 12, 18)
SLOT_BOUNDS = tuple(int(b) for b in np.cumsum((0,) + SLOT_SIZES))  # (0, 26, 48, 66, 78, 96)
SAGITTAL_SLOTS = (0, 1)
STORED = 352


class LazyVolume:
    """``(96, 352, 352)`` uint8 stack whose slices are decoded only when indexed."""

    def __init__(self, path: str):
        z = np.load(path)
        self.jpg, self.offs, self.mask = z["jpg"], z["offs"], z["mask"]
        self.shape = (len(self.mask), STORED, STORED)

    def slice(self, i: int) -> np.ndarray:
        import cv2

        a, b = int(self.offs[i]), int(self.offs[i + 1])
        if b <= a:
            return np.zeros(self.shape[1:], np.uint8)
        return cv2.imdecode(self.jpg[a:b], cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        cache: dict[int, np.ndarray] = {}
        out = np.empty(idx.shape + self.shape[1:], np.uint8)
        for pos, i in np.ndenumerate(idx):
            i = int(i)
            if i not in cache:
                cache[i] = self.slice(i)
            out[pos] = cache[i]
        return out


class C96Corpus:
    """Same interface as :class:`rsna_knee.mil.corpus.Corpus` (``ids``, ``masks``, ``row``, ``volume``)."""

    def __init__(self, find_file: Callable[[str], str]):
        index = find_file("c96_index.csv")
        root = os.path.join(os.path.dirname(index), "c96")
        paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.paths = paths
        self.ids = np.array([os.path.basename(p)[:-4] for p in paths])
        self.masks = np.stack([np.load(p)["mask"] for p in paths])
        self.row = {u: i for i, u in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def volume(self, i: int) -> LazyVolume:
        return LazyVolume(self.paths[i])


def sagittal(centres: np.ndarray) -> np.ndarray:
    """True where a window centre lies in a sagittal slot."""
    slot = np.searchsorted(np.asarray(SLOT_BOUNDS), np.asarray(centres), side="right") - 1
    return np.isin(slot, SAGITTAL_SLOTS)


def mirror(windows: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Anatomical left/right mirror of ``(k, 3, H, W)`` windows (nartaa's ``_mirror_tri``): sagittal windows
    reverse their three slices (the through-plane axis is left-right), coronal and axial ones flip width."""
    sag = sagittal(centres)[:, None, None, None]
    return np.ascontiguousarray(np.where(sag, windows[:, ::-1], windows[..., ::-1]))


def crop(windows: np.ndarray, size: int, offset: tuple[int, int] | None = None) -> np.ndarray:
    """``size`` px crop of ``(k, 3, H, W)`` windows at ``offset`` (row, col); centre when ``None``."""
    h, w = windows.shape[-2:]
    r, c = offset if offset is not None else ((h - size) // 2, (w - size) // 2)
    return np.ascontiguousarray(windows[..., r:r + size, c:c + size])
