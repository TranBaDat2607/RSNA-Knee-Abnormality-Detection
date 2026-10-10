"""Inference for our k-fold models (:mod:`rsna_knee.mil.kfold`) on test studies.

Each study is decoded once with the exact public-corpus recipe (``CORPUS44_336``, E5c), shrunk with
the same :func:`~rsna_knee.mil.cache.shrink_volume` the training cache was built with, and scored by
every checkpoint. Decoding runs in a process pool (pydicom is CPU-bound and holds the GIL), the models
on one GPU; a study that fails to decode keeps 0.5 for every checkpoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .cache import shrink_volume
from .kfold import prepare
from .model import load_checkpoint
from .recipes import RECIPES
from .volume import build_volume
from .corpus import SLOT_BOUNDS
from .windows import eval_centres, slot_eval_centres, slot_of, slot_triplets, triplets


def decode_study(uid: str, rows: Sequence[Mapping], series_root: str, inputs: Sequence[tuple[str, int]],
                 build_fn: Callable = build_volume) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """One decode per recipe, shrunk to every cache size a checkpoint of that recipe was trained on."""
    out, built = {}, {}
    for recipe, size in inputs:
        if recipe not in built:
            built[recipe] = build_fn(uid, rows, series_root, RECIPES[recipe])
        vol, mask = built[recipe]
        out[recipe, size] = (shrink_volume(vol, size), mask)
    return out


def _decode_safe(args):
    uid, rows, root, inputs, build_fn = args
    try:
        return uid, decode_study(uid, rows, root, inputs, build_fn), None
    except Exception as exc:  # noqa: BLE001 - reported by the caller
        return uid, None, repr(exc)


def eval_windows(vol: np.ndarray, mask: np.ndarray, k_eval: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """uint8 ``(k, 3, H, W)`` windows and their slot ids, as the checkpoint's trainer built them
    (``"stack"``: neighbours along the whole stack; ``"slot"``: neighbours inside one series)."""
    if mode == "slot":
        c = slot_eval_centres(mask, k_eval)
        return slot_triplets(vol, mask, c, SLOT_BOUNDS), slot_of(c, SLOT_BOUNDS)
    c = eval_centres(mask, k_eval)
    return triplets(vol, c), slot_of(c, SLOT_BOUNDS)


@torch.no_grad()
def score_volume(models: Sequence[tuple], vol: np.ndarray, mask: np.ndarray,
                 k_eval: int, device: torch.device) -> np.ndarray:
    """``(n_models, 12)`` probabilities for one stack; ``models`` are ``(model, res[, window mode])``."""
    out, cache = [], {}
    for model, res, *mode in models:
        mode = mode[0] if mode else "stack"
        if (mode, res) not in cache:
            w, sl = eval_windows(vol, mask, k_eval, mode)
            cache[mode, res] = (prepare(torch.from_numpy(w)[None].to(device), res),
                                torch.from_numpy(sl.astype(np.int64))[None].to(device))
        x, sl = cache[mode, res]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            feats = model.encode(x)
        out.append(torch.sigmoid(model.head(feats.float(), sl))[0].float().cpu().numpy())
    return np.stack(out)


def predict_fleet(study_uids: Sequence[str], series_rows: Mapping[str, Sequence[Mapping]], series_root: str,
                  checkpoints: Sequence[str], device: torch.device, *, k_eval: int = 40, workers: int = 4,
                  build_fn: Callable = build_volume, load_fn: Callable = load_checkpoint,
                  log: Callable[[str], None] = print, log_every: int = 200
                  ) -> tuple[np.ndarray, list[dict], list[tuple[str, str]]]:
    """``(n_checkpoints, n_studies, 12)`` probabilities, each checkpoint's metadata, and failures."""
    models, metas, inputs = [], [], []
    for path in checkpoints:
        model, res, meta = load_fn(path, device)
        models.append((model, res, meta.get("windows", "stack")))
        metas.append({**meta, "path": path})
        inputs.append((meta.get("recipe", "corpus44_336"), int(meta.get("cache", 256))))
    groups = {key: [i for i, k in enumerate(inputs) if k == key] for key in dict.fromkeys(inputs)}
    out = np.full((len(models), len(study_uids), 12), 0.5, np.float32)
    failures: list[tuple[str, str]] = []
    index = {u: i for i, u in enumerate(study_uids)}
    t0 = time.time()
    jobs = [(u, list(series_rows.get(u, [])), series_root, list(groups), build_fn) for u in study_uids]
    pool = ProcessPoolExecutor(workers) if workers > 0 else None
    results = pool.map(_decode_safe, jobs, chunksize=1) if pool else map(_decode_safe, jobs)
    for n, (uid, decoded, err) in enumerate(results, 1):
        if decoded is None:
            failures.append((uid, err))
        else:
            try:
                for key, members in groups.items():
                    out[members, index[uid]] = score_volume([models[i] for i in members], *decoded[key],
                                                            k_eval, device)
            except Exception as exc:  # noqa: BLE001
                failures.append((uid, repr(exc)))
        if log_every and n % log_every == 0:
            log(f"fleet {n}/{len(study_uids)} studies | {(time.time() - t0) / n:.2f}s/study")
    if pool:
        pool.shutdown()
    log(f"fleet: {len(models)} checkpoint(s) x {len(study_uids)} studies in {time.time() - t0:.0f}s, "
        f"{len(failures)} failure(s)")
    return out, metas, failures
