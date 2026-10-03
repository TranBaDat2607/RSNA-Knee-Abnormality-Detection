"""K-fold training on a Kaggle TPU VM (torch_xla), same model, folds, targets and outputs as :mod:`kfold`.

The GPU trainer is bound by Kaggle's 6 T4-hours a week; the 20 TPU-hours a week are otherwise unused.
This keeps every contract of :mod:`kfold` (report-grouped folds, gold-58 never trained on, best epoch
chosen against the text-only reference, ``oof.csv`` / ``gold.npy`` / ``model.pt`` per fold in the
checkpoint format :func:`rsna_knee.mil.model.load_checkpoint` reads), so checkpoints trained here are
served on T4 by :mod:`rsna_knee.mil.fleet` unchanged.

TPU-specific choices: one process per chip (``xmp.spawn``), fixed shapes everywhere (static batch,
``drop_last``, eval batches padded) so XLA compiles each graph once, bf16 autocast, and augmentation
on the host (``cv2.warpAffine``) instead of ``grid_sample`` on device. Evaluation runs sharded over
all chips and is gathered on the host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import TARGETS
from .cache import open_corpus
from .kfold import EMA, parse_args as gpu_parse_args, read_targets, report_folds
from .model import MILClassifier, build_backbone
from .train import host_memory, kaggle_find_file, macro_auc
from .corpus import SLOT_BOUNDS
from .windows import (IMAGENET_MEAN, IMAGENET_STD, eval_centres, slot_eval_centres, slot_of,
                      slot_train_centres, slot_triplets, train_centres, triplets)


def affine_windows(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """One random affine (scale 0.9-1.1, rotation +-10 deg, shift +-6 %) for all of a study's windows.

    Host-side twin of :func:`kfold.prepare`'s on-device augmentation; padding is black like the MRI
    background. ``x`` is ``uint8 (k, 3, H, W)``.
    """
    import cv2

    k, c, h, w = x.shape
    s = rng.uniform(0.9, 1.1)
    ang = rng.uniform(-10.0, 10.0)
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    m[:, 2] += (tx, ty)
    flat = x.reshape(k * c, h, w).transpose(1, 2, 0)  # H, W, k*c
    out = np.empty_like(flat)
    for i in range(0, k * c, 512):  # warpAffine takes at most 512 channels
        out[..., i:i + 512] = cv2.warpAffine(np.ascontiguousarray(flat[..., i:i + 512]), m, (w, h),
                                             flags=cv2.INTER_LINEAR, borderValue=0).reshape(h, w, -1)
    return out.transpose(2, 0, 1).reshape(k, c, h, w)


class XlaWindows(torch.utils.data.Dataset):
    """``(uint8 (k,3,H,W), targets (12,), gain, index, slot per window (k,))`` per study; augmentation
    on the host. ``slot_aware`` windows stay inside one series (:func:`slot_triplets`)."""

    def __init__(self, corpus, rows, targets: np.ndarray, k: int, train: bool, seed: int,
                 slot_aware: bool = False):
        self.corpus, self.rows, self.targets, self.k, self.train, self.seed = corpus, list(rows), targets, k, train, seed
        self.slot_aware = slot_aware

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, j: int):
        i = self.rows[j]
        mask = self.corpus.masks[i]
        rng = random.Random((self.seed * 1_000_003 + j) ^ int.from_bytes(os.urandom(4), "little"))
        if self.slot_aware:
            c = slot_train_centres(mask, self.k, rng) if self.train else slot_eval_centres(mask, self.k)
            x = slot_triplets(self.corpus.volume(i), mask, c, SLOT_BOUNDS)
        else:
            c = train_centres(mask, self.k, rng) if self.train else eval_centres(mask, self.k)
            x = triplets(self.corpus.volume(i), c)
        gain = 1.0
        if self.train:
            x = affine_windows(x, rng)
            gain = 1.0 + (rng.random() - 0.5) * 0.2
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.targets[j]),
                torch.tensor(gain, dtype=torch.float32), torch.tensor(j),
                torch.from_numpy(slot_of(c, SLOT_BOUNDS).astype(np.int64)))


def normalise(x_u8: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """uint8 ``(B, K, 3, H, W)`` on device -> ImageNet-normalised float, per-study intensity gain."""
    x = x_u8.float() / 255.0
    x = (x * gain.view(-1, 1, 1, 1, 1)).clamp(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def _pad_rows(rows: list, multiple: int) -> tuple[list, int]:
    n = len(rows)
    pad = (-n) % multiple
    return rows + rows[:1] * pad, n


def train_fold(a, fold, corpus, folds, targets, reference, gold, log) -> dict:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr

    dev = xm.xla_device()
    world, rank = xr.world_size(), xr.global_ordinal()
    master = rank == 0
    out_dir = os.path.join(a.out, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    tr_ids = [u for u in folds.index[folds != fold] if u in corpus.row]
    va_ids = [u for u in folds.index[folds == fold] if u in corpus.row] if fold >= 0 else []
    if a.limit_train:
        tr_ids, va_ids = tr_ids[:a.limit_train], va_ids[:max(a.limit_train // 4, 16)]
    y_tr = targets.reindex(tr_ids).values.astype(np.float32)
    tr_ds = XlaWindows(corpus, [corpus.row[u] for u in tr_ids], y_tr, a.k, True, a.seed + fold, a.slot_aware)
    sampler = torch.utils.data.distributed.DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                                              shuffle=True, drop_last=True, seed=a.seed + fold)
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                                         drop_last=True, persistent_workers=a.workers > 0,
                                         prefetch_factor=4 if a.workers > 0 else None)
    steps_per_epoch = len(loader)

    def eval_set(ids):
        rows, n = _pad_rows([corpus.row[u] for u in ids], world * a.eval_bs)
        ds = XlaWindows(corpus, rows, np.zeros((len(rows), 12), np.float32), a.k_eval, False, 0, a.slot_aware)
        smp = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False)
        return torch.utils.data.DataLoader(ds, batch_size=a.eval_bs, sampler=smp, num_workers=a.workers), n, len(rows)

    va_loader, n_va, n_va_pad = eval_set(va_ids) if va_ids else (None, 0, 0)
    gold_loader, n_gold, n_gold_pad = eval_set(list(gold.index))

    torch.manual_seed(a.seed + fold)
    backbone = build_backbone(a.arch, pretrained=not a.scratch)
    n_slots = len(SLOT_BOUNDS) - 1 if a.slot_aware else 0
    model = MILClassifier(backbone, backbone.num_features, drop=a.drop, n_slots=n_slots).to(dev)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.bb_lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    total = max(a.epochs * steps_per_epoch, 1)
    warm = max(int(a.warmup * total), 1)

    def lr_factor(step):
        if step < warm:
            return step / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(total - warm, 1))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    prevalence = np.clip(y_tr.mean(0), 0.03, 0.7)
    pos_w = torch.tensor(np.clip((1 - prevalence) / prevalence, 1, a.max_pos_w), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ema = EMA(model, a.ema)
    ema.shadow = {n: v.to(dev) for n, v in ema.shadow.items()}

    @torch.no_grad()
    def predict(ld, n_pad):
        model.eval()
        preds, idx = [], []
        for x, _, g, j, sl in pl.MpDeviceLoader(ld, dev):
            with torch.autocast("xla", dtype=torch.bfloat16):
                feats = model.encode(normalise(x, g))
            preds.append(torch.sigmoid(model.head(feats.float(), sl)))
            idx.append(j)
        p = xm.all_gather(torch.cat(preds), dim=0).cpu().numpy()
        i = xm.all_gather(torch.cat(idx), dim=0).cpu().numpy()
        out = np.zeros((n_pad, 12), np.float32)
        out[i] = p
        return out

    ref_va = reference.reindex(va_ids)
    ok = ref_va.notna().all(axis=1).to_numpy()
    best, history, t0 = -1.0, [], time.time()
    if master:
        log(f"fold {fold}: world {world} train {len(tr_ids)} val {len(va_ids)} gold {len(gold)} "
            f"steps/epoch {steps_per_epoch} global batch {a.bs * world} {host_memory()}")
    step = 0
    for ep in range(a.epochs):
        model.train()
        sampler.set_epoch(ep)
        te, run_loss = time.time(), []
        for it, (x, y, g, _, sl) in enumerate(pl.MpDeviceLoader(loader, dev)):
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model.head(model.encode(normalise(x, g)).float(), sl)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            if it % a.log_every == 0:
                run_loss.append(loss.detach())
            if master and it == a.log_every:
                xm.mark_step()
                ips = (it + 1) * a.bs * a.k * world / (time.time() - te)
                log(f"f{fold} ep{ep} it{it}/{steps_per_epoch} {ips:.0f} img/s (global) {host_memory()}")
        rec = {"ep": ep, "loss": float(torch.stack(run_loss).mean().cpu()) if run_loss else float("nan"),
               "train_s": round(time.time() - te, 1)}
        last = ep + 1 == a.epochs
        if (ep + 1) % a.eval_every == 0 or last:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            if va_loader is not None:
                pv = predict(va_loader, n_va_pad)[:n_va]
                rec["val_ref_macro"], rec["val_ref_per"] = macro_auc(ref_va.values[ok], pv[ok])
                score = rec["val_ref_macro"]
            else:  # full-data fit: no validation fold, keep the last epoch
                pv, score = None, float(ep)
            if score > best:
                best = score
                rec["best"] = True
                pg = predict(gold_loader, n_gold_pad)[:n_gold]
                rec["gold_macro"], rec["gold_per"] = macro_auc(gold.values, pg)
                state = {k: v.half() if v.dtype == torch.float32 else v
                         for k, v in ((k, v.detach().cpu()) for k, v in model.state_dict().items())}
                if master:
                    if pv is not None:
                        pd.DataFrame(pv, columns=TARGETS).assign(StudyInstanceUID=va_ids)[["StudyInstanceUID", *TARGETS]] \
                            .to_csv(os.path.join(out_dir, "oof.csv"), index=False)
                    np.save(os.path.join(out_dir, "gold.npy"), pg)
                    torch.save({"model": state, "arch": a.arch, "res": a.res, "cache": a.cache, "lab": TARGETS,
                                "epoch": ep, "fold": fold, "targets": a.targets, "recipe": a.recipe,
                                "n_slots": n_slots, "windows": "slot" if a.slot_aware else "stack"},
                               os.path.join(out_dir, "model.pt"))
            model.load_state_dict(live)
        history.append(rec)
        if master:
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                            if not k.endswith("_per")}))
            json.dump({"args": vars(a), "fold": fold, "history": history, "gold_ids": list(gold.index)},
                      open(os.path.join(out_dir, "run.json"), "w"), indent=1)
        xm.rendezvous(f"fold{fold}-ep{ep}")
    if master:
        log(f"fold {fold} done in {(time.time() - t0) / 60:.1f} min, best {best:.4f}")
    return {"fold": fold, "best_val": best, "minutes": (time.time() - t0) / 60}


def _mp_main(index: int, a: argparse.Namespace) -> None:
    import torch_xla.runtime as xr

    t0 = time.time()
    master = xr.global_ordinal() == 0

    def log(m: str) -> None:
        print(f"[{a.tag}][{time.time() - t0:7.1f}s] {m}", flush=True)

    corpus = open_corpus(kaggle_find_file, a.cache, a.cache_name or None)
    tr = pd.read_csv(kaggle_find_file("train.csv"), usecols=["StudyInstanceUID", "Report", *TARGETS])
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    tr = tr.set_index("StudyInstanceUID")
    gold = tr[tr[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
    targets = read_targets(kaggle_find_file(a.targets))
    reference = read_targets(kaggle_find_file(a.reference))
    pool = tr.index[~tr.index.isin(gold.index) & tr.index.isin(targets.index)]
    folds = report_folds(tr.loc[pool, "Report"], a.n_folds, a.fold_seed)
    if master:
        os.makedirs(a.out, exist_ok=True)
        folds.rename("fold").to_csv(os.path.join(a.out, "folds.csv"))
        log(f"arch={a.arch} res={a.res} targets={a.targets} pool={len(pool)} folds "
            f"{folds.value_counts().sort_index().tolist()}")
    results = []
    for fold in [int(f) for f in str(a.folds).split(",")]:
        if a.time_limit_h and results:
            per_fold = np.mean([r["minutes"] for r in results]) / 60
            if (time.time() - t0) / 3600 + per_fold > a.time_limit_h:
                if master:
                    log(f"skipping fold {fold}: time limit")
                continue
        # fold -1 = fit on every pool study (no OOF), for a final full-data model
        results.append(train_fold(a, fold, corpus, folds, targets,
                                  reference, gold, log))
        if master:
            json.dump(results, open(os.path.join(a.out, "folds_done.json"), "w"), indent=1)
    if master:
        log("done")


def parse_args(argv=None) -> argparse.Namespace:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--eval_bs", type=int, default=2)
    e, rest = extra.parse_known_args(argv)
    a = gpu_parse_args(rest)
    a.eval_bs = e.eval_bs
    return a


def main(argv=None) -> None:
    import torch_xla.distributed.xla_multiprocessing as xmp

    a = parse_args(argv)
    xmp.spawn(_mp_main, args=(a,))


if __name__ == "__main__":
    main()
