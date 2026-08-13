"""The pairwise ranking term supplementing the per-target BCE loss.

The competition metric is macro-averaged ROC AUC, which counts correctly ordered
positive/negative pairs. Cross-entropy is a reasonable proxy for that, but the quantity
itself can be optimised directly: inside each batch, for each target, take the studies
whose graded target is confidently positive and confidently negative and push their
logits apart. Batches are small, so this fires for common targets and stays silent for
rare ones — which is why its weight stays small and it supplements BCE rather than
replacing it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rank_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               rank_pos: float, rank_neg: float) -> torch.Tensor:
    """Pairwise ranking term on confidently-graded pairs inside the batch.

    A target with no usable pair in the batch contributes nothing rather than a constant.
    """
    parts = []
    usable = w > 0
    for j in range(logits.shape[1]):
        pos = logits[(y[:, j] > rank_pos) & usable[:, j], j]
        neg = logits[(y[:, j] < rank_neg) & usable[:, j], j]
        if len(pos) and len(neg):
            parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
    return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)
