"""Training a per-finding attention-MIL model on the public corpus.

Shaped by Kaggle's 2xT4 (validated there, see docs/experiment_ledger.md E3 smoke runs): fp16 autocast
+ GradScaler (T4 has no bf16), gradient checkpointing (CoAtNet-2 @384 does not fit 24 images without
it), windows resized on the GPU, few non-persistent DataLoader workers (~29 GB host RAM), and DDP with
a static graph (plain DDP rejects timm's re-entrant checkpointing).

Gold-58 is never trained on. It and a fixed weak-label hold-out are scored every ``eval_every``
epochs, and all their predictions are saved, so runs are compared offline rather than by picking an
epoch on gold. Checkpoints load with :func:`rsna_knee.mil.model.load_checkpoint` and serve through
:func:`rsna_knee.mil.infer.predict_arms` with the ``CORPUS44_336`` recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ..config import TARGETS
from .corpus import Corpus
from .model import MILClassifier, build_backbone
from .orientation import canonical_transforms, transform_windows
from .teachers import TEACHERS, read_teacher
from .windows import eval_centres, to_model_input, train_centres, triplets


def kaggle_find_file(fname: str) -> str:
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if fname in fs:
            return os.path.join(d, fname)
    raise FileNotFoundError(fname)


def host_memory() -> str:
    rss = avail = float("nan")
    try:
        with open("/proc/self/status") as f:
            rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:")) / 1e6
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:")) / 1e6
    except (OSError, StopIteration):
        pass
    gpu = f" gpu={torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_initialized() else ""
    return f"rss={rss:.1f}GB avail={avail:.1f}GB{gpu}"


class StudyWindows(torch.utils.data.Dataset):
    """``(uint8 windows (k,3,H,W), soft targets (12,), intensity gain)`` per study."""

    def __init__(self, corpus: Corpus, rows, targets: np.ndarray, k: int, train: bool, seed: int,
                 canonical: dict | None = None):
        self.corpus, self.rows, self.targets, self.k = corpus, list(rows), targets, k
        self.train, self.seed, self.canonical = train, seed, canonical

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, j: int):
        i = self.rows[j]
        mask = self.corpus.masks[i]
        if self.train:
            # forked workers inherit one RNG state and would draw identical windows
            rng = random.Random((self.seed * 1_000_003 + j) ^ int.from_bytes(os.urandom(4), "little"))
            centres = train_centres(mask, self.k, rng)
            gain = 1.0 + (rng.random() - 0.5) * 0.2
        else:
            centres = eval_centres(mask, self.k)
            gain = 1.0
        x = triplets(self.corpus.volume(i), centres)
        if self.canonical is not None and i in self.canonical:
            x = transform_windows(x, centres, self.canonical[i])
        return torch.from_numpy(x), torch.from_numpy(self.targets[j]), torch.tensor(gain, dtype=torch.float32)


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    ok = [j for j in range(y.shape[1]) if 0 < (y[:, j] > 0.5).sum() < len(y)]
    per = {TARGETS[j]: float(roc_auc_score((y[:, j] > 0.5).astype(int), p[:, j])) for j in ok}
    return float(np.mean(list(per.values()))), per


@torch.no_grad()
def predict(model: MILClassifier, res: int, ds: StudyWindows, device, workers: int) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(to_model_input(x.to(device, non_blocking=True), res))).float().cpu().numpy())
    return np.concatenate(out)


def run(rank: int, world: int, a: argparse.Namespace, find_file: Callable[[str], str] = kaggle_find_file) -> None:
    ddp = world > 1
    t0 = time.time()
    tag = f"[{a.tag}{f'/r{rank}' if ddp else ''}]"

    def log(m: str) -> None:
        print(f"{tag}[{time.time() - t0:7.1f}s] {m}", flush=True)

    if ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if ddp else "cuda:0")  # CUDA_VISIBLE_DEVICES picks the card
    torch.backends.cudnn.benchmark = True

    corpus = Corpus(find_file)
    tr = pd.read_csv(find_file("train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")[TARGETS].astype(float)
    teacher = read_teacher(a.teacher, find_file)
    reference = read_teacher("top5mean", find_file)
    pool = sorted(u for u in teacher.index if u in corpus.row and u not in gold.index and u in reference.index)
    np.random.RandomState(a.split_seed).shuffle(pool)
    holdout = pool[:a.holdout_n]
    train_ids = pool[a.holdout_n:a.holdout_n + a.train_n] if a.train_n > 0 else pool[a.holdout_n:]
    log(f"teacher={a.teacher} train={len(train_ids)} holdout={len(holdout)} gold={len(gold)} arch={a.arch} "
        f"res={a.res} k={a.k} batch={a.bs}x{world}x{a.accum} canonical={a.canonical} {host_memory()}")

    canonical = None
    if a.canonical:
        canonical = canonical_transforms(pd.read_csv(find_file(a.orientation_csv), dtype={"StudyInstanceUID": str}),
                                         pd.read_csv(find_file(a.side_csv), dtype={"StudyInstanceUID": str}),
                                         corpus.row, tag_only=a.side_tag_only)
        log(f"canonical transforms for {len(canonical)} studies")
    y_train = teacher.reindex(train_ids).fillna(teacher.mean()).values.astype(np.float32)
    train_ds = StudyWindows(corpus, [corpus.row[u] for u in train_ids], y_train, a.k, True, a.seed + rank, canonical)
    gold_ds = StudyWindows(corpus, [corpus.row[u] for u in gold.index], gold.values.astype(np.float32), a.k_eval, False, 0, canonical)
    hold_ds = StudyWindows(corpus, [corpus.row[u] for u in holdout], reference.reindex(holdout).values.astype(np.float32),
                           a.k_eval, False, 0, canonical)
    sampler = (torch.utils.data.distributed.DistributedSampler(train_ds, world, rank, shuffle=True, seed=a.seed, drop_last=True)
               if ddp else None)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.bs, sampler=sampler, shuffle=sampler is None,
                                         num_workers=a.workers, drop_last=True)

    torch.manual_seed(a.seed)
    backbone = build_backbone(a.arch, pretrained=not a.init)
    if a.grad_ckpt:
        backbone.set_grad_checkpointing(True)
    core = MILClassifier(backbone, backbone.num_features).to(device)
    if a.init:
        state = torch.load(find_file(a.init), map_location="cpu", weights_only=False)
        log(f"init from {a.init}: {core.load_state_dict(state.get('model', state), strict=False)}")
        del state
    model = nn.parallel.DistributedDataParallel(core, device_ids=[rank], static_graph=True) if ddp else core
    head = [p for n, p in core.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": core.backbone.parameters(), "lr": a.bb_lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=max(a.epochs * (len(loader) // a.accum), 1), pct_start=0.15)
    prevalence = np.clip(y_train.mean(0), 0.03, 0.7)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prevalence) / prevalence, 1, 10),
                                                         dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler("cuda")
    os.makedirs(a.out, exist_ok=True)
    history = []
    log(f"model ready {host_memory()}")
    for ep in range(a.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        total, n, te = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for it, (x, y, g) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(to_model_input(x.to(device, non_blocking=True), a.res, g))
            loss = lossf(logits.float(), y.to(device)) / a.accum
            scaler.scale(loss).backward()
            if (it + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            total += loss.item() * a.accum
            n += 1
            if it % a.log_every == 0:
                log(f"ep{ep} it{it}/{len(loader)} loss {total / n:.4f} {(time.time() - te) / (it + 1):.2f}s/it {host_memory()}")
        rec = {"ep": ep, "loss": total / max(n, 1), "train_s": time.time() - te}
        if rank == 0 and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            te = time.time()
            pg = predict(core, a.res, gold_ds, device, a.workers)
            rec["gold_macro"], rec["gold_per"] = macro_auc(gold.values, pg)
            np.save(f"{a.out}/gold_pred_ep{ep}.npy", pg)
            if holdout:
                ph = predict(core, a.res, hold_ds, device, a.workers)
                rec["holdout_top5ref_macro"], _ = macro_auc(reference.reindex(holdout).values, ph)
                np.save(f"{a.out}/holdout_pred_ep{ep}.npy", ph)
            rec["eval_s"] = time.time() - te
            if a.save_ckpt:
                torch.save({"model": core.state_dict(), "arch": a.arch, "res": a.res, "lab": TARGETS, "epoch": ep,
                            "teacher": a.teacher, "recipe": "corpus44_336", "canonical": a.canonical},
                           f"{a.out}/ckpt_ep{ep}.pt")
        history.append(rec)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "gold_per"})
            + " " + host_memory())
        if rank == 0:
            json.dump({"args": vars(a), "gold_ids": list(gold.index), "holdout_ids": holdout, "history": history},
                      open(f"{a.out}/run.json", "w"), indent=1)
        if a.time_limit_h and (time.time() - t0) / 3600 > a.time_limit_h:
            log("time limit reached")
            break
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    log("done")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--teacher", default="flight", choices=sorted(TEACHERS))
    p.add_argument("--init", default="", help="checkpoint file name under /kaggle/input to warm-start from")
    p.add_argument("--tag", default="")
    p.add_argument("--k", type=int, default=12, help="random windows per study per step")
    p.add_argument("--k_eval", type=int, default=24)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=2, help="studies per GPU")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--bb_lr", type=float, default=3e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--train_n", type=int, default=0, help="0 = every non-gold, non-hold-out study")
    p.add_argument("--holdout_n", type=int, default=500)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--canonical", action="store_true", help="canonical anatomical orientation (see orientation.py)")
    p.add_argument("--orientation_csv", default="e5a_slot_orientation.csv")
    p.add_argument("--side_csv", default="e5a_study_side.csv")
    p.add_argument("--side_tag_only", action="store_true", help="ignore patient-x geometry for side")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--time_limit_h", type=float, default=0.0)
    p.add_argument("--out", default="/kaggle/working/run")
    a = p.parse_args(argv)
    a.tag = a.tag or a.teacher
    return a
