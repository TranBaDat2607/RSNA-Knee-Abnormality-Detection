import torch

from rsna_knee.config import AugConfig
from rsna_knee.training.augment import augment


def test_augment_preserves_shape_and_dtype():
    cfg = AugConfig()
    t = torch.randint(0, 255, (2, 3, 3, 32, 32), dtype=torch.uint8)  # (B, S, group, H, W)
    out = augment(t, cfg)
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_augment_preserves_vertical_orientation_when_vflip_disabled():
    """A knee is not vertically symmetric: with use_vflip=False, an asymmetric marker
    must never end up mirrored top-to-bottom."""
    cfg = AugConfig(use_vflip=False)
    m = torch.zeros(1, 1, 3, 32, 32, dtype=torch.uint8)
    m[..., :6, :] = 255  # bright band along the top
    out = augment(m, cfg).float()
    assert out[..., :16, :].mean() > out[..., 16:, :].mean()


def test_augment_never_flips_horizontally():
    """Horizontal flip would undo the laterality normalisation and must never happen,
    regardless of config — there is no on-switch for it in AugConfig at all."""
    cfg = AugConfig()
    m = torch.zeros(1, 1, 3, 32, 32, dtype=torch.uint8)
    m[..., :6] = 255  # bright band on the left
    out = augment(m, cfg).float()
    assert out[..., :16].mean() > out[..., 16:].mean()


def test_augment_affine_disabled_is_near_identity_apart_from_intensity():
    cfg = AugConfig(affine=False, intensity=0.0)
    t = torch.randint(0, 255, (1, 1, 3, 8, 8), dtype=torch.uint8)
    out = augment(t, cfg)
    assert torch.equal(out, t)
