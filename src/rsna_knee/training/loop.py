"""The per-fold training loop: two-speed AdamW, OneCycle schedule, weight EMA, and
dual-reference epoch selection.

Two validation references are tracked per epoch — AUC against the derived (gold + LLM)
labels on the fold's holdout, which covers every held-out study, and AUC against the
annotated studies held out of *this* fold specifically, which is far smaller but is the
one that resembles the actual test metric. Epoch selection uses
``min(derived_auc, annotated_auc)`` (``cv_cfg.selection == "worse_of_two"``) so an epoch
that gains on one reference while losing the other isn't preferred — it's only shown to
be *different*, not *better*.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import AugConfig, CacheConfig, CVConfig, TrainConfig
from ..dicom.cache import take_group
from ..inference.predict import predict
from ..logging_utils import log
from ..model.backbone import build_model
from ..model.ema import Ema
from .augment import augment
from .losses import rank_loss
from .metrics import macro_auc


def train_one_fold(fold: int, ctr: np.ndarray, mtr: np.ndarray, y: np.ndarray, w: np.ndarray,
                    tr: np.ndarray, va: np.ndarray, gi_va: np.ndarray,
                    gold_y_va: np.ndarray | None, dev: torch.device, *,
                    cache_cfg: CacheConfig, train_cfg: TrainConfig, cv_cfg: CVConfig,
                    aug_cfg: AugConfig, n_group: int, n_targets: int, t0: float,
                    ) -> tuple[dict, list[dict], float]:
    """Train one fold. Returns ``(best_state_dict, history, best_score)``."""
    model = build_model(cache_cfg=cache_cfg, train_cfg=train_cfg, n_targets=n_targets).to(dev)
    ema = Ema(model, train_cfg.ema_decay)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": train_cfg.lr_backbone},
        {"params": model.head.parameters(), "lr": train_cfg.lr_head},
    ], weight_decay=train_cfg.weight_decay)
    steps = max(train_cfg.epochs * (len(tr) // train_cfg.batch_studies), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[train_cfg.lr_backbone, train_cfg.lr_head],
        total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    yv = (y[va] > 0.5).astype(int)
    best, best_state, history = -1.0, None, []

    def _predict(idx: np.ndarray, eval_model: torch.nn.Module) -> np.ndarray:
        return predict(eval_model, ctr, mtr, idx, dev, n_group=n_group,
                        group=cache_cfg.group, n_targets=n_targets,
                        eval_batch=train_cfg.eval_batch)

    for ep in range(train_cfg.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot, nstep = 0.0, 0
        bs = train_cfg.batch_studies
        for b in range(0, len(perm) - bs + 1, bs):
            sel = perm[b:b + bs]
            rows = torch.from_numpy(ctr[sel]).to(dev)
            g = int(torch.randint(n_group, (1,)).item())
            imgs = augment(take_group(rows, g, cache_cfg.group), aug_cfg)
            m = torch.from_numpy(mtr[sel]).to(dev)
            yt = torch.from_numpy(y[sel]).to(dev)
            wt = torch.from_numpy(w[sel]).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(imgs, m)
                loss = (F.binary_cross_entropy_with_logits(z, yt, reduction="none") * wt).mean()
                if train_cfg.rank_loss_w > 0:
                    loss = loss + train_cfg.rank_loss_w * rank_loss(
                        z.float(), yt, wt, train_cfg.rank_pos, train_cfg.rank_neg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item()
            nstep += 1

        eval_model = ema.target(model)
        pv = _predict(va, eval_model)
        d = macro_auc(yv, pv)

        g_auc = float("nan")
        if gold_y_va is not None and len(gi_va) >= 6:
            g_auc = macro_auc(gold_y_va, _predict(gi_va, eval_model))

        history.append({"fold": fold, "epoch": ep + 1, "loss": tot / max(nstep, 1),
                         "derived": d, "annot": g_auc})
        log(f"fold {fold} epoch {ep + 1}/{train_cfg.epochs}  loss {tot / max(nstep, 1):.4f}"
            f"  derived {d:.4f}  annot(held out) {g_auc:.4f}")

        if cv_cfg.selection == "derived" or not np.isfinite(g_auc):
            score = d
        else:
            score = min(d, g_auc)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            log(f"  best so far ({cv_cfg.selection} {score:.4f})")
        if time.time() - t0 > train_cfg.time_budget_s:
            log("time budget reached inside fold")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in ema.target(model).state_dict().items()}
    del model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_state, history, best
