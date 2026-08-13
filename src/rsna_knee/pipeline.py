"""Orchestrates the full run: header pass -> laterality -> slots -> cache -> targets ->
4-fold CV -> rank-mean ensembled submission.

This is the module-per-concern replacement for ``main()`` in
``eda/rsna-knee-data-structure-eda-baseline.ipynb``. Two validation leaks the notebook
specifically guards against are preserved here (see ``training/loop.py`` and
``labels/targets.py`` docstrings): folds are grouped by report-text hash, not randomly,
and each fold's annotated-holdout reference excludes studies that fold trained on.

Call :func:`run` directly for the full result (histories, out-of-fold predictions, the
submission DataFrame); call :func:`run_safe` for the notebook's behaviour of always
leaving a scoreable ``submission.csv`` behind, even if a mid-run exception fires.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dicom.cache import build_cache
from .dicom.header import annotate, walk
from .dicom.laterality import laterality_maps
from .dicom.slots import pick_slots
from .inference.predict import predict
from .inference.submission import write_benchmark_submission, write_submission
from .labels.targets import build_targets, gold_positions, load_gold, report_hash_folds, usable_rows
from .logging_utils import log, reset_clock
from .model.backbone import build_model
from .paths import find_llm_labels, find_root, plan_cache
from .training.loop import train_one_fold
from .training.metrics import macro_auc


@dataclass
class RunResult:
    submission: pd.DataFrame | None
    n_models: int
    history: pd.DataFrame
    fold_scores: list[float]
    oof_gold: tuple[np.ndarray, np.ndarray] | None = None   # (y, p) on studies seen out-of-fold
    slot_coverage: pd.Series | None = None
    weight_table: pd.DataFrame | None = None
    laterality_info: dict = field(default_factory=dict)


def run(cfg: Config | None = None, *, out_path: str = "submission.csv") -> RunResult:
    cfg = cfg or Config()
    t0 = time.time()
    reset_clock()

    root = find_root()
    log(f"input root: {root}")

    # A scoreable file exists from the first second; overwritten once real predictions
    # are ready. A SIGKILL (OOM) never reaches the try/except in run_safe, so this is
    # the only thing that guarantees a submission survives that failure mode.
    write_benchmark_submission(root / "test.csv", cfg.targets, out_path)

    test_df = pd.read_csv(root / "test.csv")
    test_series = pd.read_csv(root / "test_series.csv")
    train_df = pd.read_csv(root / "train.csv")
    train_series = pd.read_csv(root / "train_series.csv")
    log(f"train {train_df.shape} test {test_df.shape}")

    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    log("header pass: test")
    hte = annotate(walk(root, "test_series", threads=cfg.cache.hdr_threads))
    log(f"  {len(hte)} test series")
    log("header pass: train")
    htr = annotate(walk(root, "train_series", threads=cfg.cache.hdr_threads))
    log(f"  {len(htr)} train series")

    lat_tr, lat_info_tr = laterality_maps(htr, cfg.laterality)
    lat_te, _ = laterality_maps(hte, cfg.laterality)

    slots_te = pick_slots(hte, plane_map, cfg.cache.slots)
    slots_tr = pick_slots(htr, plane_map, cfg.cache.slots)
    slot_coverage = pd.Series({
        name: float(np.mean([name in v for v in slots_tr.values()]))
        for name, *_ in cfg.cache.slots
    })
    weight_table = htr.groupby(["weight", "fatsat"]).size()

    n_group = plan_cache(
        len(train_df), n_slot=cfg.cache.n_slot, img=cfg.cache.img,
        cache_budget_gb=cfg.cache.cache_budget_gb, group=cfg.cache.group,
        n_group_max=cfg.cache.n_group_max)
    log(f"cache layout: {n_group} groups x {cfg.cache.group} slices = "
        f"{n_group * cfg.cache.group} per slot")

    st_tr, ctr, mtr = build_cache(slots_tr, plane_map, lat_tr, "train", cfg.cache, n_group,
                                   time_budget_s=cfg.train.time_budget_s, t0=t0)
    st_te, cte, mte = build_cache(slots_te, plane_map, lat_te, "test", cfg.cache, n_group,
                                   time_budget_s=cfg.train.time_budget_s, t0=t0)

    # ---- targets ---------------------------------------------------------- #
    lab = pd.read_csv(find_llm_labels()).set_index("StudyInstanceUID")
    t_lab = time.time()
    y, w = build_targets(st_tr, train_df, lab, cfg.targets, cfg.cv.gold_weight)
    log(f"derived labels for {len(lab)} studies in {time.time() - t_lab:.1f}s")

    gold = load_gold(train_df, cfg.targets)
    keep = usable_rows(w)
    log(f"supervised {len(keep)} of {len(st_tr)} studies (annotated {len(gold)})")

    grp = report_hash_folds(st_tr, train_df, cfg.cv.n_folds)
    gi_all = gold_positions(st_tr, gold)
    gold_y_all = (gold.loc[[st_tr[i] for i in gi_all]].values.astype(int)
                  if len(gi_all) else None)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank_sum = np.zeros((len(st_te), len(cfg.targets)), np.float64)
    n_models, sub = 0, None
    oof_gold = np.full((len(st_tr), len(cfg.targets)), np.nan, np.float32)
    histories: list[dict] = []
    fold_scores: list[float] = []

    max_folds = cfg.cv.n_folds if cfg.cv.max_folds == "auto" else int(cfg.cv.max_folds)
    for fold in range(min(cfg.cv.n_folds, max_folds)):
        va = np.array([i for i in keep if grp[i] == fold])
        tr = np.array([i for i in keep if grp[i] != fold])
        if len(va) == 0 or len(tr) < cfg.train.batch_studies:
            log(f"fold {fold}: not enough studies, skipped")
            continue
        gi_va = np.array([i for i in gi_all if grp[i] == fold])
        gold_y_va = (gold.loc[[st_tr[i] for i in gi_va]].values.astype(int)
                     if len(gi_va) else None)
        log(f"=== fold {fold}: train {len(tr)} / holdout {len(va)} "
            f"(annotated held out: {len(gi_va)}) ===")

        t_fold = time.time()
        state, history, best = train_one_fold(
            fold, ctr, mtr, y, w, tr, va, gi_va, gold_y_va, dev,
            cache_cfg=cfg.cache, train_cfg=cfg.train, cv_cfg=cfg.cv, aug_cfg=cfg.aug,
            n_group=n_group, n_targets=len(cfg.targets), t0=t0)
        fold_time = time.time() - t_fold
        histories.extend(history)
        fold_scores.append(best)

        model = build_model(cache_cfg=cfg.cache, train_cfg=cfg.train,
                             n_targets=len(cfg.targets)).to(dev)
        model.load_state_dict(state)
        if len(gi_va):
            oof_gold[gi_va] = predict(model, ctr, mtr, gi_va, dev, n_group=n_group,
                                       group=cfg.cache.group, n_targets=len(cfg.targets),
                                       eval_batch=cfg.train.eval_batch)
        p = predict(model, cte, mte, np.arange(len(st_te)), dev, n_group=n_group,
                     group=cfg.cache.group, n_targets=len(cfg.targets),
                     eval_batch=cfg.train.eval_batch)
        rank_sum += pd.DataFrame(p).rank(pct=True).values
        n_models += 1
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sub = write_submission(rank_sum, n_models, st_te, test_df, cfg.targets, out_path)
        log(f"fold {fold} done in {fold_time / 60:.1f} min; submission has {n_models} model(s)")

        elapsed = time.time() - t0
        if elapsed + cfg.cv.fold_time_pad * fold_time > cfg.train.time_budget_s:
            log(f"stopping after {n_models} fold(s): another would need "
                f"{cfg.cv.fold_time_pad * fold_time / 60:.0f} min and "
                f"{(cfg.train.time_budget_s - elapsed) / 60:.0f} min remain")
            break

    history_df = pd.DataFrame(histories)
    oof_result = None
    if gold_y_all is not None and len(gi_all):
        seen = np.isfinite(oof_gold[gi_all]).all(axis=1)
        if seen.sum() >= 8:
            oof_auc = macro_auc(gold_y_all[seen], oof_gold[gi_all][seen])
            log(f"out-of-fold macro AUC on {int(seen.sum())} annotated studies: {oof_auc:.4f}")
            oof_result = (gold_y_all[seen], oof_gold[gi_all][seen])
    log(f"fold selection scores: {[round(s, 4) for s in fold_scores]}")

    if sub is None:
        log("no fold completed; the 0.5 benchmark submission stands")
    else:
        log(f"submission.csv {sub.shape}; models {n_models}")

    return RunResult(submission=sub, n_models=n_models, history=history_df,
                      fold_scores=fold_scores, oof_gold=oof_result,
                      slot_coverage=slot_coverage, weight_table=weight_table,
                      laterality_info=lat_info_tr)


def run_safe(cfg: Config | None = None, *, out_path: str = "submission.csv") -> RunResult | None:
    """Run the pipeline; on any exception, fall back to a 0.5-benchmark submission so a
    failed run never leaves the caller with nothing scoreable."""
    try:
        return run(cfg, out_path=out_path)
    except Exception:
        traceback.print_exc()
        cfg = cfg or Config()
        root = find_root()
        write_benchmark_submission(root / "test.csv", cfg.targets, out_path)
        log("wrote fallback submission.csv")
        return None
