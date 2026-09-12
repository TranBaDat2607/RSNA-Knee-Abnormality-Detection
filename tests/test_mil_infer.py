"""Multi-arm inference orchestration with stub decoding and stub models (no DICOM, no weights)."""

from __future__ import annotations

import threading
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from rsna_knee.mil.infer import predict_arms
from rsna_knee.mil.model import MILClassifier
from rsna_knee.mil.recipes import ArmSpec, Slot, VolumeRecipe
from rsna_knee.mil.volume import build_volume


class _StubBackbone(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))


def _load_fn(path, device):
    torch.manual_seed(len(path))
    return MILClassifier(_StubBackbone(), 8).eval().to(device), 8, {}


R1 = VolumeRecipe("r1", 8, 140.0, (0.0, 1.0), (Slot("Sagittal", 1, 6),))
R2 = VolumeRecipe("r2", 8, 140.0, (0.0, 1.0), (Slot("Coronal", 1, 5),))


def test_arms_sharing_a_recipe_share_one_stack_and_failures_are_contained():
    calls = Counter()
    lock = threading.Lock()

    def build_fn(uid, rows, root, recipe, order_fn=None, read_fn=None):
        with lock:
            calls[(recipe.name, uid)] += 1
        if uid == "broken":
            raise ValueError("undecodable series")
        rng = np.random.default_rng(abs(hash(uid)) % 1000)
        return rng.integers(1, 255, (recipe.n_slices, 8, 8), dtype=np.uint8), np.ones(recipe.n_slices, np.uint8)

    arms = [ArmSpec("a", "a.pt", R1, 4), ArmSpec("b", "bb.pt", R1, 3), ArmSpec("c", "ccc.pt", R2, 2)]
    uids = ["s0", "s1", "broken", "s3", "s4"]
    out, failures = predict_arms(uids, {}, "/root", arms, [torch.device("cpu")] * 2,
                                 find_checkpoint=lambda f: f, build_fn=build_fn, load_fn=_load_fn,
                                 log=lambda m: None)

    assert set(out) == {"a", "b", "c"} and all(v.shape == (5, 12) for v in out.values())
    assert all(n == 1 for n in calls.values()) and len(calls) == 2 * len(uids)
    assert {(r, u) for r, u, _ in failures} == {("r1", "broken"), ("r2", "broken")}
    for v in out.values():
        assert np.all(v[2] == 0.5) and not np.allclose(v[[0, 1, 3, 4]], 0.5)
    assert not np.allclose(out["a"], out["b"])  # different models on the same stack


def test_recipes_picking_the_same_slices_decode_them_once_per_study():
    # Two recipes with identical slots/span but different pixel grids, like maxspan 336 / dense 384.
    small = VolumeRecipe("small", 8, 140.0, (0.0, 1.0), (Slot("Sagittal", 1, 4),))
    large = VolumeRecipe("large", 12, 140.0, (0.0, 1.0), (Slot("Sagittal", 1, 4),))
    rows = {u: [{"SeriesInstanceUID": "sag", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1}]
            for u in ("s0", "s1")}
    listed, decoded = Counter(), Counter()
    lock = threading.Lock()

    def order_fn(series_dir):
        with lock:
            listed[series_dir] += 1
        return [(f"{series_dir}/{i}.dcm", 0.5) for i in range(8)], 0.5

    def read_fn(path):
        with lock:
            decoded[path] += 1
        return np.arange(400, dtype=np.float32).reshape(20, 20) + float(len(path))

    def build_fn(uid, rows_, root, recipe, order_fn=None, read_fn=None):
        return build_volume(uid, rows_, root, recipe, order_fn=order_fn, read_fn=read_fn,
                            resize_fn=lambda a, s, c, img: np.resize(a, (img, img)))

    arms = [ArmSpec("s", "s.pt", small, 3), ArmSpec("l", "ll.pt", large, 3)]
    out, failures = predict_arms(["s0", "s1"], rows, "/root", arms, [torch.device("cpu")],
                                 find_checkpoint=lambda f: f, build_fn=build_fn, load_fn=_load_fn,
                                 order_fn=order_fn, read_fn=read_fn, log=lambda m: None)
    assert not failures and all(v.shape == (2, 12) for v in out.values())
    assert len(listed) == 2 and all(n == 1 for n in listed.values())      # one listing per study
    assert len(decoded) == 8 and all(n == 1 for n in decoded.values())    # 4 slices x 2 studies, once each
