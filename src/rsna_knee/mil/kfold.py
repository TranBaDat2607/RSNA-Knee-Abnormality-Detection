"""K-fold training on the cached corpus, writing out-of-fold predictions for pseudo-label rounds.

Built for Kaggle's 6 GPU-hours a week: small backbones at 256 px on the 256 px corpus cache
(:mod:`rsna_knee.mil.cache`), fp16 + channels-last, one fold after another per GPU. Each fold
writes its validation predictions (``oof.csv``), its gold-58 predictions (``gold.npy``) and the EMA
weights of its best epoch (``model.pt``, loadable with :func:`rsna_knee.mil.model.load_checkpoint`).

Folds group studies by report text, so studies sharing a report (and therefore a label) never sit
on both sides. Gold-58 is never trained on. The best epoch is chosen on the fold's validation
studies scored against a text-only reference table, which the fold's images never trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TARGETS
from .cache import open_corpus
from .corpus import SLOT_BOUNDS
from .model import MILClassifier, build_backbone
from .train import StudyWindows, host_memory, kaggle_find_file, macro_auc
from .windows import IMAGENET_MEAN, IMAGENET_STD, slot_eval_centres, slot_of, slot_train_centres, slot_triplets


def report_folds(reports: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """Fold per study (index of ``reports``), grouping identical report texts."""
    groups = reports.fillna("").map(lambda s: hashlib.sha1(s.strip().encode("utf-8")).hexdigest())
    uniq = groups.drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return groups.map(fold_of).astype(int)


def prepare(windows_u8: torch.Tensor, res: int, gain: torch.Tensor | None = None,
            augment: bool = False) -> torch.Tensor:
    """uint8 ``(B, K, 3, H, W)`` -> normalised float ``(B, K, 3, res, res)`` on the windows' device.

    ``augment`` applies one random affine (scale, shift, small rotation) per study to all of its
    windows, before normalisation so the padding is black like the MRI background.
    """
    b, k = windows_u8.shape[:2]
    x = windows_u8.flatten(0, 1).float().div_(255.0)
    if gain is not None:
        x = (x * gain.to(x.device).float().repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    if augment:
        s = torch.empty(b, device=x.device).uniform_(0.9, 1.1)
        r = torch.empty(b, device=x.device).uniform_(-math.pi / 18, math.pi / 18)
        t = torch.empty(b, 2, device=x.device).uniform_(-0.06, 0.06)
        theta = torch.stack([torch.stack([torch.cos(r) / s, -torch.sin(r) / s, t[:, 0]], 1),
                             torch.stack([torch.sin(r) / s, torch.cos(r) / s, t[:, 1]], 1)], 1)
        theta = theta.repeat_interleave(k, 0)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).view(b, k, 3, res, res)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.state_dict().items():
            if p.dtype.is_floating_point:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[n].copy_(p)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({n: v.to(model.state_dict()[n].dtype) for n, v in self.shadow.items()})


class SlotWindows(StudyWindows):
    """:class:`StudyWindows` whose windows stay inside one series; items gain a 4th field, the slot
    of each window."""

    def __getitem__(self, j: int):
        i = self.rows[j]
        mask = self.corpus.masks[i]
        if self.train:
            rng = random.Random((self.seed * 1_000_003 + j) ^ int.from_bytes(os.urandom(4), "little"))
            centres, gain = slot_train_centres(mask, self.k, rng), 1.0 + (rng.random() - 0.5) * 0.2
        else:
            centres, gain = slot_eval_centres(mask, self.k), 1.0
        x = slot_triplets(self.corpus.volume(i), mask, centres, SLOT_BOUNDS)
        return (torch.from_numpy(x), torch.from_numpy(self.targets[j]), torch.tensor(gain, dtype=torch.float32),
                torch.from_numpy(slot_of(centres, SLOT_BOUNDS).astype(np.int64)))


@torch.no_grad()
def predict(model: nn.Module, ds: StudyWindows, res: int, device, workers: int, bs: int = 4) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _, *sl in torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        xi = prepare(x.to(device, non_blocking=True), res)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            feats = model.encode(xi.contiguous())
        slots = sl[0].to(device) if sl else None
        out.append(torch.sigmoid(model.head(feats.float(), slots)).cpu().numpy())
    return np.concatenate(out)


def read_targets(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    return d.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[TARGETS].astype(float).clip(0, 1)


def train_fold(a: argparse.Namespace, fold: int, corpus, folds: pd.Series, targets: pd.DataFrame,
               reference: pd.DataFrame, gold: pd.DataFrame, device, log: Callable[[str], None]) -> dict:
    out_dir = os.path.join(a.out, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    tr_ids = [u for u in folds.index[folds != fold] if u in corpus.row]
    va_ids = [u for u in folds.index[folds == fold] if u in corpus.row]
    if a.limit_train:
        tr_ids, va_ids = tr_ids[:a.limit_train], va_ids[:max(a.limit_train // 4, 16)]
    y_tr = targets.reindex(tr_ids).values.astype(np.float32)
    Windows = SlotWindows if a.slot_aware else StudyWindows
    tr_ds = Windows(corpus, [corpus.row[u] for u in tr_ids], y_tr, a.k, True, a.seed + fold)
    va_ds = Windows(corpus, [corpus.row[u] for u in va_ids], np.zeros((len(va_ids), 12), np.float32), a.k_eval, False, 0)
    gold_ds = Windows(corpus, [corpus.row[u] for u in gold.index], gold.values.astype(np.float32), a.k_eval, False, 0)
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                         drop_last=True, pin_memory=device.type == "cuda")

    torch.manual_seed(a.seed + fold)
    backbone = build_backbone(a.arch, pretrained=not a.scratch)
    if a.slot_aware and corpus.masks.shape[1] != SLOT_BOUNDS[-1]:
        raise ValueError(f"--slot_aware needs the {SLOT_BOUNDS[-1]}-slice corpus layout, cache has {corpus.masks.shape[1]}")
    n_slots = len(SLOT_BOUNDS) - 1 if a.slot_aware else 0
    model = MILClassifier(backbone, backbone.num_features, drop=a.drop, n_slots=n_slots).to(device)
    model = model.to(memory_format=torch.channels_last)
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.bb_lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(a.epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps,
                                                pct_start=a.warmup)
    prevalence = np.clip(y_tr.mean(0), 0.03, 0.7)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prevalence) / prevalence, 1, a.max_pos_w),
                                                         dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, a.ema)
    ref_va = reference.reindex(va_ids)
    ok = ref_va.notna().all(axis=1).to_numpy()
    best, history, t0 = -1.0, [], time.time()
    log(f"fold {fold}: train {len(tr_ids)} val {len(va_ids)} gold {len(gold)} steps {steps} {host_memory()}")
    for ep in range(a.epochs):
        model.train()
        te, total, n = time.time(), 0.0, 0
        for it, (x, y, g, *sl) in enumerate(loader):
            xi = prepare(x.to(device, non_blocking=True), a.res, g, augment=True)
            slots = sl[0].to(device) if sl else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model.head(model.encode(xi).float(), slots)
            loss = lossf(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            total += loss.item()
            n += 1
            if it % a.log_every == 0:
                ips = (it + 1) * a.bs * a.k / (time.time() - te)
                log(f"f{fold} ep{ep} it{it}/{len(loader)} loss {total / n:.4f} {ips:.0f} img/s {host_memory()}")
        rec = {"ep": ep, "loss": total / max(n, 1), "train_s": round(time.time() - te, 1)}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            pv = predict(model, va_ds, a.res, device, a.workers)
            rec["val_ref_macro"], rec["val_ref_per"] = macro_auc(ref_va.values[ok], pv[ok])
            if rec["val_ref_macro"] > best:
                best = rec["val_ref_macro"]
                rec["best"] = True
                pd.DataFrame(pv, columns=TARGETS).assign(StudyInstanceUID=va_ids)[["StudyInstanceUID", *TARGETS]] \
                    .to_csv(os.path.join(out_dir, "oof.csv"), index=False)
                pg = predict(model, gold_ds, a.res, device, a.workers)
                np.save(os.path.join(out_dir, "gold.npy"), pg)
                rec["gold_macro"], rec["gold_per"] = macro_auc(gold.values, pg)
                torch.save({"model": {k: v.half() if v.dtype == torch.float32 else v
                                      for k, v in model.state_dict().items()},
                            "arch": a.arch, "res": a.res, "cache": a.cache, "lab": TARGETS, "epoch": ep,
                            "fold": fold, "targets": a.targets, "recipe": a.recipe,
                            "n_slots": n_slots, "windows": "slot" if a.slot_aware else "stack"},
                           os.path.join(out_dir, "model.pt"))
            model.load_state_dict(live)
        history.append(rec)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                        if not k.endswith("_per")}))
        json.dump({"args": vars(a), "fold": fold, "history": history, "gold_ids": list(gold.index)},
                  open(os.path.join(out_dir, "run.json"), "w"), indent=1)
    log(f"fold {fold} done in {(time.time() - t0) / 60:.1f} min, best val {best:.4f}")
    return {"fold": fold, "best_val": best, "minutes": (time.time() - t0) / 60}


def run(rank: int, world: int, a: argparse.Namespace, find_file: Callable[[str], str] = kaggle_find_file) -> None:
    t0 = time.time()

    def log(m: str) -> None:
        print(f"[{a.tag}][{time.time() - t0:7.1f}s] {m}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    corpus = open_corpus(find_file, a.cache, a.cache_name or None)
    tr = pd.read_csv(find_file("train.csv"), usecols=["StudyInstanceUID", "Report", *TARGETS])
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    tr = tr.set_index("StudyInstanceUID")
    gold = tr[tr[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
    targets = read_targets(find_file(a.targets))
    reference = read_targets(find_file(a.reference))
    pool = tr.index[~tr.index.isin(gold.index) & tr.index.isin(targets.index)]
    folds = report_folds(tr.loc[pool, "Report"], a.n_folds, a.fold_seed)
    os.makedirs(a.out, exist_ok=True)
    folds.rename("fold").to_csv(os.path.join(a.out, "folds.csv"))
    log(f"arch={a.arch} res={a.res} cache={a.cache} targets={a.targets} pool={len(pool)} "
        f"fold sizes {folds.value_counts().sort_index().tolist()} {host_memory()}")
    results = []
    for fold in [int(f) for f in str(a.folds).split(",")]:
        if a.time_limit_h and results:
            per_fold = np.mean([r["minutes"] for r in results]) / 60
            if (time.time() - t0) / 3600 + per_fold > a.time_limit_h:
                log(f"skipping fold {fold}: {per_fold:.2f} h per fold would pass the {a.time_limit_h} h limit")
                continue
        results.append(train_fold(a, fold, corpus, folds, targets, reference, gold, device, log))
        json.dump(results, open(os.path.join(a.out, "folds_done.json"), "w"), indent=1)
    log("done")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="resnet34.a1_in1k")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--cache", type=int, default=256, help="pixel size of the corpus cache to read")
    p.add_argument("--targets", default="targets_r1.csv", help="soft training targets (file under /kaggle/input)")
    p.add_argument("--reference", default="teach4.csv", help="text-only table the validation fold is scored on")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_seed", type=int, default=2026)
    p.add_argument("--tag", default="")
    p.add_argument("--k", type=int, default=16, help="random windows per study per step")
    p.add_argument("--k_eval", type=int, default=40)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=8, help="studies per step")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--bb_lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--ema", type=float, default=0.998)
    p.add_argument("--max_pos_w", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--limit_train", type=int, default=0, help="smoke test: studies per fold")
    p.add_argument("--scratch", action="store_true", help="random init (tests)")
    p.add_argument("--time_limit_h", type=float, default=0.0)
    p.add_argument("--cache_name", default="", help='cache file prefix (default cache{size}); "corpus" reads the public 336 px corpus')
    p.add_argument("--recipe", default="corpus44_336", help="recipe the cache was built with (served at test time)")
    p.add_argument("--grad_ckpt", action="store_true", help="timm gradient checkpointing (less GPU memory)")
    p.add_argument("--slot_aware", action="store_true",
                   help="windows stay inside one series and carry a learned slot embedding")
    p.add_argument("--out", default="/kaggle/working/run")
    a = p.parse_args(argv)
    a.tag = a.tag or a.arch.split(".")[0]
    a.out = os.path.join(a.out, a.tag) if not a.out.rstrip("/").endswith(a.tag) else a.out
    return a
