"""Per-diagnosis attention pooling over a study's slot embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences — cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially — so pooling the
    slots identically would dilute the one that carries the evidence with the rest of
    them. Project each slot, add a learned slot identity, give every diagnosis its own
    query vector, and let it attend over the slots with absent ones masked out of the
    softmax so a missing series shifts attention onto what's present instead of feeding
    the head a zero vector.

    Deliberately this simple: richer alternatives (attention over slice groups rather
    than slots, a max instead of a mean) were measured and lost. The label is attached to
    the *study*, so nothing in the supervision says which part of a study carries the
    finding — extra attention parameters have no signal to learn that from.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
