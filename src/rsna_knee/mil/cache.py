"""A down-sampled copy of the public corpus, small enough to stay in the page cache while training.

The corpus is 22 GB of 44 x 336 px stacks. Two training jobs reading it at small-backbone speed need
~0.4 GB/s from ``/kaggle/input``; the 256 px copy is 12.7 GB, fits in the host page cache, and costs
nothing after the first epoch. It is built once by a free CPU kernel and attached to training runs.

:func:`shrink_volume` is the only resize: the cache is built with it and test studies go through it
after ``build_volume(CORPUS44_336)``, so a model trained here sees identical inputs at inference.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

VOLS, MASKS, IDS = "{name}_vols.npy", "{name}_masks.npy", "{name}_ids.npy"


def cache_name(size: int, name: str | None = None) -> str:
    """File prefix of a cache: ``cache{size}`` for the corpus copy, or an explicit name."""
    return name or f"cache{size}"


def shrink_volume(vol: np.ndarray, size: int) -> np.ndarray:
    """``uint8 (D, H, W)`` -> ``uint8 (D, size, size)``, area-resampled slice by slice."""
    import cv2

    if vol.shape[-1] == size and vol.shape[-2] == size:
        return vol
    return np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in vol])


def build_cache(corpus, out_dir: str, size: int, log: Callable[[str], None] = print,
                log_every: int = 500, name: str | None = None) -> None:
    """Write ``cache{size}_{vols,masks,ids}.npy`` for every study of ``corpus``."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(corpus)
    depth = corpus.masks.shape[1]
    name = cache_name(size, name)
    vols = np.lib.format.open_memmap(os.path.join(out_dir, VOLS.format(name=name)), mode="w+",
                                     dtype=np.uint8, shape=(n, depth, size, size))
    for i in range(n):
        vols[i] = shrink_volume(corpus.volume(i), size)
        if log_every and (i + 1) % log_every == 0:
            log(f"cache {i + 1}/{n}")
    vols.flush()
    del vols
    np.save(os.path.join(out_dir, MASKS.format(name=name)), corpus.masks)
    np.save(os.path.join(out_dir, IDS.format(name=name)), corpus.ids)


def open_corpus(find_file: Callable[[str], str], size: int, name: str | None = None):
    """The training stacks: the public 44 x 336 corpus itself for ``name == "corpus"``, else a cache."""
    if name == "corpus":
        from .corpus import Corpus

        return Corpus(find_file)
    return CachedCorpus(find_file, size, name)


class CachedCorpus:
    """Same interface as :class:`rsna_knee.mil.corpus.Corpus` (``ids``, ``row``, ``masks``,
    ``volume(i)``), over a cache written by :func:`build_cache`. Memory-mapped lazily per process."""

    def __init__(self, find_file: Callable[[str], str], size: int, name: str | None = None):
        name = cache_name(size, name)
        self.vol_path = find_file(VOLS.format(name=name))
        self.ids = np.load(find_file(IDS.format(name=name)), allow_pickle=True).astype(str)
        self.masks = np.load(find_file(MASKS.format(name=name)))
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
            self._vols = np.load(self.vol_path, mmap_mode="r")
        return np.ascontiguousarray(self._vols[i])
