import torch

from rsna_knee.training.losses import rank_loss


def test_rank_loss_is_lower_when_positive_scores_above_negative():
    logits = torch.tensor([[2.0], [-2.0]])
    y = torch.tensor([[1.0], [0.0]])
    w = torch.ones(2, 1)
    assert rank_loss(logits, y, w, rank_pos=0.6, rank_neg=0.4) < \
        rank_loss(-logits, y, w, rank_pos=0.6, rank_neg=0.4)


def test_rank_loss_is_zero_with_no_usable_pair():
    logits = torch.tensor([[2.0], [-2.0]])
    y = torch.tensor([[1.0], [1.0]])  # both "positive": no negative to pair against
    w = torch.ones(2, 1)
    assert float(rank_loss(logits, y, w, rank_pos=0.6, rank_neg=0.4)) == 0.0


def test_rank_loss_ignores_zero_weight_rows():
    logits = torch.tensor([[2.0], [-2.0], [-2.0]])
    y = torch.tensor([[1.0], [0.0], [0.0]])
    w = torch.tensor([[1.0], [1.0], [0.0]])
    with_dead_row = rank_loss(logits, y, w, rank_pos=0.6, rank_neg=0.4)
    without_dead_row = rank_loss(logits[:2], y[:2], w[:2], rank_pos=0.6, rank_neg=0.4)
    assert torch.allclose(with_dead_row, without_dead_row)
