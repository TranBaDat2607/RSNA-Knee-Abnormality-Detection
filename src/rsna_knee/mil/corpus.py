"""The public pre-decoded training corpus (``dreaddevelopment/knee-raptor-corpus`` + ``-ext``).

All 4,407 training studies as 44-slice x 336 px ``uint8`` stacks in the
:data:`~rsna_knee.mil.recipes.CORPUS44_336` layout, 22 GB in two memory-mappable parts. Training on
it skips DICOM decoding (0.07 s/study read vs 0.6-1.8 s decode on Kaggle), and
``build_volume(CORPUS44_336)`` reproduces it exactly (E5c), so a model trained here is served from
DICOM with identical inputs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .recipes import CORPUS44_336

SLOT_BOUNDS = tuple(int(b) for b in np.cumsum([0] + [s.n for s in CORPUS44_336.slots]))  # (0,12,22,30,36,44)
SLOT_PLANES = tuple(s.plane for s in CORPUS44_336.slots)

PARTS = (("all_vols.npy", "all_masks.npy", "all_ids.npy"),
         ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy"))


class Corpus:
    """Study stacks by row, plus ids and per-slice fill masks.

    Volumes are memory-mapped lazily in whichever process first reads them, so the object is cheap to
    fork or pickle into DataLoader workers — a pickled ``np.memmap`` would copy all 22 GB.
    """

    def __init__(self, find_file: Callable[[str], str]):
        self.vol_paths = [find_file(v) for v, _, _ in PARTS]
        ids = [np.load(find_file(i), allow_pickle=True).astype(str) for _, _, i in PARTS]
        self.part_len = [len(x) for x in ids]
        self.ids = np.concatenate(ids)
        self.masks = np.concatenate([np.load(find_file(m)) for _, m, _ in PARTS])
        self.row = {u: i for i, u in enumerate(self.ids)}
        self._vols = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_vols"] = None
        return state

    def volume(self, i: int) -> np.ndarray:
        if self._vols is None:
            self._vols = [np.load(p, mmap_mode="r") for p in self.vol_paths]
        if i < self.part_len[0]:
            return np.ascontiguousarray(self._vols[0][i])
        return np.ascontiguousarray(self._vols[1][i - self.part_len[0]])
