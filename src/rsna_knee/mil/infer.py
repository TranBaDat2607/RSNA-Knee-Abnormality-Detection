"""Multi-arm inference: decode each study once, build every recipe from it, run every arm, on every GPU.

The public 0.94 notebooks run each CoAtNet arm as a separate pass that re-decodes every DICOM
study and uses one GPU. Decoding dominates: on Kaggle's 4 vCPUs the three public checkpoints took
4.8 s/study with a decode per recipe (E4), most of it reading pixels. Here each device thread walks
its shard of studies; within a study, series listings and decoded slices are memoised, so recipes
that pick the same slices (e.g. the 336 px and 384 px 64-slice recipes) decode them once, and arms
sharing a recipe share its stack.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from .model import load_checkpoint
from .recipes import ArmSpec
from .volume import build_volume, order_series_files, read_pixels
from .windows import eval_centres, to_model_input, triplets


@torch.no_grad()
def predict_study(model: torch.nn.Module, res: int, volume: np.ndarray, mask: np.ndarray,
                  k_eval: int, device: torch.device, chunk: int = 16) -> np.ndarray:
    """Sigmoid probabilities ``(12,)`` for one stack. fp16 on CUDA, retried in fp32 if a kernel
    has no fp16 implementation (T4/Turing lacks bf16 and some fp16 conv engines)."""
    windows = torch.from_numpy(triplets(volume, eval_centres(mask, k_eval))).to(device, non_blocking=True)
    x = to_model_input(windows, res)[None]
    if device.type == "cuda":
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                feats = model.encode(x, chunk=chunk)
            return torch.sigmoid(model.head(feats.float()))[0].float().cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
    return torch.sigmoid(model(x.float()))[0].float().cpu().numpy()


def predict_arms(study_uids: Sequence[str], series_rows: Mapping[str, Sequence[Mapping]],
                 series_root: str, arms: Sequence[ArmSpec], devices: Sequence[torch.device],
                 find_checkpoint: Callable[[str], str], *, build_fn: Callable = build_volume,
                 load_fn: Callable = load_checkpoint, order_fn: Callable = order_series_files,
                 read_fn: Callable = read_pixels, log: Callable[[str], None] = print,
                 chunk: int = 16, log_every: int = 200
                 ) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]]:
    """Probabilities per arm, ``{arm.name: (n_studies, 12)}``, and ``(recipe, study, error)`` failures.

    A study that fails to decode or predict keeps 0.5 for every finding of the affected arms, so
    one broken series costs that study's ranking, never the submission.
    """
    out = {a.name: np.full((len(study_uids), 12), 0.5, np.float32) for a in arms}
    failures: list[tuple[str, str, str]] = []
    lock = threading.Lock()
    recipes = list(dict.fromkeys(a.recipe for a in arms))
    by_recipe = {r: [a for a in arms if a.recipe == r] for r in recipes}
    done = [0]
    t0 = time.time()

    def worker(shard: int, device: torch.device) -> None:
        models = {}
        for recipe in recipes:
            models[recipe] = []
            for arm in by_recipe[recipe]:
                model, res = load_fn(find_checkpoint(arm.checkpoint), device)[:2]
                models[recipe].append((arm, model, res))
        for i in range(shard, len(study_uids), len(devices)):
            uid = study_uids[i]
            # per-study memo: listings and decoded slices are reused by every recipe of this study
            memo_order = functools.lru_cache(maxsize=None)(order_fn)
            memo_read = functools.lru_cache(maxsize=None)(read_fn)
            for recipe in recipes:
                try:
                    vol, mask = build_fn(uid, series_rows.get(uid, []), series_root, recipe,
                                         order_fn=memo_order, read_fn=memo_read)
                    for arm, model, res in models[recipe]:
                        out[arm.name][i] = predict_study(model, res, vol, mask, arm.k_eval, device, chunk)
                except Exception as exc:  # noqa: BLE001 - recorded and reported, never fatal
                    with lock:
                        failures.append((recipe.name, uid, repr(exc)))
            with lock:
                done[0] += 1
                if log_every and done[0] % log_every == 0:
                    el = time.time() - t0
                    log(f"{done[0]}/{len(study_uids)} studies | {el / done[0]:.2f}s/study")
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    threads = [threading.Thread(target=worker, args=(j, d)) for j, d in enumerate(devices)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"{len(arms)} arm(s) over {len(recipes)} recipe(s) x {len(study_uids)} studies on "
        f"{len(devices)} device(s) in {time.time() - t0:.0f}s ({len(failures)} failure(s))")
    return out, failures
