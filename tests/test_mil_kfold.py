"""Corpus cache, report-grouped folds, augmentation and an end-to-end k-fold run (CPU, synthetic files)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from rsna_knee.config import TARGETS
from rsna_knee.mil.cache import CachedCorpus, build_cache, shrink_volume
from rsna_knee.mil.corpus import Corpus
from rsna_knee.mil.kfold import parse_args, prepare, report_folds, run
from rsna_knee.mil.model import load_checkpoint


def _fake_corpus(tmp_path, n_all=14, n_extra=6, px=12):
    files = {}
    for part, n in (("all", n_all), ("extra", n_extra)):
        vols = np.random.default_rng(len(part)).integers(1, 255, (n, 44, px, px), dtype=np.uint8)
        np.save(tmp_path / f"{part}_vols.npy", vols)
        np.save(tmp_path / f"{part}_masks.npy", np.ones((n, 44), np.uint8))
        np.save(tmp_path / f"{part}_ids.npy", np.array([f"{part}{i}" for i in range(n)], dtype=object))
        for kind in ("vols", "masks", "ids"):
            files[f"{part}_{kind}.npy"] = str(tmp_path / f"{part}_{kind}.npy")
    return Corpus(files.__getitem__)


def test_cache_matches_shrink_volume(tmp_path):
    corpus = _fake_corpus(tmp_path)
    build_cache(corpus, str(tmp_path / "c"), 8, log=lambda m: None)
    cached = CachedCorpus(lambda f: str(tmp_path / "c" / f), 8)
    assert len(cached) == len(corpus) and list(cached.ids) == list(corpus.ids)
    assert np.array_equal(cached.volume(15), shrink_volume(corpus.volume(15), 8))
    assert cached.volume(0).shape == (44, 8, 8) and cached.masks.shape == (20, 44)


def test_report_folds_keep_identical_reports_together():
    reports = pd.Series(["a", "b", "a", "c", "d", "b", "e", None], index=[f"s{i}" for i in range(8)])
    folds = report_folds(reports, 3, seed=0)
    assert folds["s0"] == folds["s2"] and folds["s1"] == folds["s5"]
    assert set(folds) <= {0, 1, 2} and len(folds) == 8
    assert folds.equals(report_folds(reports, 3, seed=0))


def test_prepare_shapes_and_augment_keeps_background_black():
    x = torch.zeros((2, 3, 3, 16, 16), dtype=torch.uint8)
    out = prepare(x, 8, torch.tensor([1.0, 1.0]), augment=True)
    assert out.shape == (2, 3, 3, 8, 8)
    black = prepare(torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8), 8)
    assert torch.allclose(out[:, :, 0], black[0, 0, 0].expand_as(out[:, :, 0]), atol=1e-5)


def test_kfold_run_end_to_end(tmp_path):
    corpus = _fake_corpus(tmp_path)
    build_cache(corpus, str(tmp_path), 8, log=lambda m: None)
    ids = list(corpus.ids)
    rng = np.random.default_rng(0)
    labels = pd.DataFrame(rng.integers(0, 2, (len(ids), 12)).astype(float), columns=TARGETS, index=ids)
    labels.iloc[:, 0] = [0, 1] * (len(ids) // 2)
    train = labels.copy()
    train.iloc[4:] = np.nan  # the first four studies are "gold"
    train.insert(0, "Report", [f"report {i % 9}" for i in range(len(ids))])
    train.rename_axis("StudyInstanceUID").reset_index().to_csv(tmp_path / "train.csv", index=False)
    soft = labels.clip(0.1, 0.9).rename_axis("StudyInstanceUID").reset_index()
    soft.to_csv(tmp_path / "targets.csv", index=False)
    soft.to_csv(tmp_path / "ref.csv", index=False)

    a = parse_args(["--arch", "resnet18", "--scratch", "--cache", "8", "--res", "16", "--targets", "targets.csv",
                    "--reference", "ref.csv", "--folds", "0,1", "--n_folds", "2", "--k", "4", "--k_eval", "6",
                    "--epochs", "2", "--bs", "2", "--workers", "0", "--eval_every", "1", "--out", str(tmp_path / "out")])
    run(0, 1, a, find_file=lambda f: str(tmp_path / f))

    folds = pd.read_csv(tmp_path / "out" / "resnet18" / "folds.csv", dtype={"StudyInstanceUID": str})
    assert len(folds) == len(ids) - 4 and not folds.StudyInstanceUID.isin(ids[:4]).any()
    oof = pd.concat([pd.read_csv(tmp_path / "out" / "resnet18" / f"fold{f}" / "oof.csv") for f in (0, 1)])
    assert sorted(oof.StudyInstanceUID) == sorted(folds.StudyInstanceUID)
    assert ((oof[TARGETS] >= 0) & (oof[TARGETS] <= 1)).all().all()
    gold = np.load(tmp_path / "out" / "resnet18" / "fold0" / "gold.npy")
    assert gold.shape == (4, 12)
    model, res, meta = load_checkpoint(str(tmp_path / "out" / "resnet18" / "fold1" / "model.pt"))
    assert res == 16 and meta["cache"] == 8 and meta["arch"] == "resnet18"
    assert os.path.exists(tmp_path / "out" / "resnet18" / "folds_done.json")


def test_fleet_matches_kfold_validation_predictions(tmp_path):
    from rsna_knee.mil.fleet import predict_fleet
    from rsna_knee.mil.kfold import predict
    from rsna_knee.mil.model import MILClassifier, build_backbone
    from rsna_knee.mil.train import StudyWindows

    torch.manual_seed(0)
    bb = build_backbone("resnet18", pretrained=False)
    model = MILClassifier(bb, bb.num_features).eval()
    torch.save({"model": model.state_dict(), "arch": "resnet18", "res": 16, "cache": 8}, tmp_path / "m.pt")
    rng = np.random.default_rng(1)
    raw = {u: rng.integers(1, 255, (44, 20, 20), dtype=np.uint8) for u in ("s0", "s1", "bad")}

    def build_fn(uid, rows, root, recipe):
        if uid == "bad":
            raise ValueError("no series")
        return raw[uid], np.ones(44, np.uint8)

    probs, metas, failures = predict_fleet(["s0", "bad", "s1"], {}, "/root", [str(tmp_path / "m.pt")],
                                           torch.device("cpu"), k_eval=6, workers=0, build_fn=build_fn,
                                           log=lambda m: None)
    assert probs.shape == (1, 3, 12) and [u for u, _ in failures] == ["bad"] and np.all(probs[0, 1] == 0.5)

    class _Cache:
        masks = np.ones((2, 44), np.uint8)

        def volume(self, i):
            return shrink_volume(raw[("s0", "s1")[i]], 8)

    expected = predict(model, StudyWindows(_Cache(), [0, 1], np.zeros((2, 12), np.float32), 6, False, 0),
                       16, torch.device("cpu"), workers=0)
    assert np.allclose(probs[0, [0, 2]], expected, atol=1e-5) and metas[0]["arch"] == "resnet18"


def test_ours_combine_and_blend(tmp_path):
    from rsna_knee.mil.ours import blend, combine, find_checkpoints

    for tag in ("r1_a", "r1_b", "other"):
        for f in (0, 1):
            (tmp_path / "in" / tag / f"fold{f}").mkdir(parents=True)
            (tmp_path / "in" / tag / f"fold{f}" / "model.pt").write_bytes(b"")
    found = find_checkpoints(["r1_a", "r1_b"], str(tmp_path / "in"))
    assert sorted(found) == ["r1_a", "r1_b"] and all(len(v) == 2 for v in found.values())

    rng = np.random.default_rng(0)
    probs = rng.random((4, 6, 12))
    metas = [{"path": p} for t in ("r1_a", "r1_b") for p in found[t]]
    ours = combine(probs, metas, ["r1_a", "r1_b"])
    assert ours.shape == (6, 12)
    public = pd.DataFrame(rng.random((6, 12)), columns=TARGETS)
    public.insert(0, "StudyInstanceUID", [f"s{i}" for i in range(6)])
    kept = blend(public, ours, 0.0)
    assert np.allclose(kept[TARGETS].rank().values, public[TARGETS].rank().values)
    mixed = blend(public, ours, 0.5)
    assert list(mixed.columns) == list(public.columns) and mixed[TARGETS].max().max() <= 1.0


def test_fleet_decodes_each_recipe_once_per_study(tmp_path):
    from rsna_knee.mil.fleet import predict_fleet
    from rsna_knee.mil.model import MILClassifier, build_backbone

    torch.manual_seed(0)
    bb = build_backbone("resnet18", pretrained=False)
    sd = MILClassifier(bb, bb.num_features).eval().state_dict()
    paths = []
    for i, (recipe, cache) in enumerate((("corpus44_336", 8), ("wide44_256", 8), ("wide44_256", 12))):
        torch.save({"model": sd, "arch": "resnet18", "res": 16, "cache": cache, "recipe": recipe}, tmp_path / f"{i}.pt")
        paths.append(str(tmp_path / f"{i}.pt"))
    calls = []

    def build_fn(uid, rows, root, recipe):
        calls.append((uid, recipe.name))
        return np.full((44, 20, 20), 7 if recipe.name == "wide44_256" else 200, np.uint8), np.ones(44, np.uint8)

    probs, _, failures = predict_fleet(["s0", "s1"], {}, "/root", paths, torch.device("cpu"), k_eval=6,
                                       workers=0, build_fn=build_fn, log=lambda m: None)
    assert not failures and sorted(calls) == sorted((u, r) for u in ("s0", "s1") for r in ("corpus44_336", "wide44_256"))
    assert not np.allclose(probs[0], probs[1])  # different recipes -> different inputs


def test_slot_aware_kfold_run_and_fleet_parity(tmp_path):
    from rsna_knee.mil.fleet import predict_fleet
    from rsna_knee.mil.kfold import SlotWindows, predict

    corpus = _fake_corpus(tmp_path)
    build_cache(corpus, str(tmp_path), 8, log=lambda m: None)
    ids = list(corpus.ids)
    labels = pd.DataFrame(np.random.default_rng(0).integers(0, 2, (len(ids), 12)).astype(float), columns=TARGETS, index=ids)
    labels.iloc[:, 0] = [0, 1] * (len(ids) // 2)
    train = labels.copy()
    train.iloc[4:] = np.nan
    train.insert(0, "Report", [f"report {i % 9}" for i in range(len(ids))])
    train.rename_axis("StudyInstanceUID").reset_index().to_csv(tmp_path / "train.csv", index=False)
    labels.clip(0.1, 0.9).rename_axis("StudyInstanceUID").reset_index().to_csv(tmp_path / "t.csv", index=False)
    a = parse_args(["--arch", "resnet18", "--scratch", "--cache", "8", "--res", "16", "--targets", "t.csv",
                    "--reference", "t.csv", "--folds", "0", "--n_folds", "2", "--k", "4", "--k_eval", "6",
                    "--epochs", "1", "--bs", "2", "--workers", "0", "--slot_aware", "--tag", "s",
                    "--out", str(tmp_path / "out")])
    run(0, 1, a, find_file=lambda f: str(tmp_path / f))
    path = str(tmp_path / "out" / "s" / "fold0" / "model.pt")
    model, res, meta = load_checkpoint(path)
    assert meta["windows"] == "slot" and meta["n_slots"] == 5 and meta["recipe"] == "corpus44_336"

    raw = {u: np.random.default_rng(3).integers(1, 255, (44, 20, 20), dtype=np.uint8) for u in ("x0",)}
    probs, _, failures = predict_fleet(["x0"], {}, "/root", [path], torch.device("cpu"), k_eval=6, workers=0,
                                       build_fn=lambda *args: (raw["x0"], np.ones(44, np.uint8)), log=lambda m: None)

    class _Cache:
        masks = np.ones((1, 44), np.uint8)

        def volume(self, i):
            return shrink_volume(raw["x0"], 8)

    expected = predict(model, SlotWindows(_Cache(), [0], np.zeros((1, 12), np.float32), 6, False, 0), 16,
                       torch.device("cpu"), workers=0)
    assert not failures and np.allclose(probs[0], expected, atol=1e-5)
