"""Group-averaged prediction.

Training sees one slice-group at a time (doubling as augmentation along the stack);
inference averages logits over every group of a slot, matching the aggregation the
original frozen-encoder pipeline used and measured best.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dicom.cache import take_group


@torch.no_grad()
def predict(model: torch.nn.Module, cache: np.ndarray, mask: np.ndarray, idx: np.ndarray,
            dev: torch.device, *, n_group: int, group: int, n_targets: int,
            eval_batch: int = 12) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(idx), eval_batch):
        sel = idx[b:b + eval_batch]
        rows = torch.from_numpy(cache[sel]).to(dev)
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(n_group):
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(take_group(rows, g, group), m).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / n_group).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, n_targets), np.float32)
