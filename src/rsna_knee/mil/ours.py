"""Our leg of a submission: k-fold checkpoints on the test studies, blended with a public submission.

Run as ``python -m rsna_knee.mil.ours --public <csv> --out submission.csv`` after a public notebook
has written its ``submission.csv``. Checkpoints are every ``fold*/model.pt`` under the attached
inputs whose run tag is in ``--tags``; per tag the folds are averaged in probability space, tags
are averaged in rank space, and the result is rank-blended with the public file at weight ``--w``:

    final = (1 - w) * rank(public) + w * rank(ours)

The weight is fixed in advance, not fitted to the leaderboard. Any failure (no checkpoints, a
missing study, a constant column) leaves the public file untouched, so this leg can only add a vote.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from ..config import TARGETS
from .blend import rank_pct


def find_checkpoints(tags: list[str], root: str = "/kaggle/input") -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "fold*", "model.pt"), recursive=True)):
        tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if tag in tags:
            found.setdefault(tag, []).append(path)
    return found


def combine(probs: np.ndarray, metas: list[dict], tags: list[str]) -> np.ndarray:
    """Fold mean per tag (probability space), then the mean of the tags' ranks, ``(n, 12)``."""
    per_tag = []
    for tag in tags:
        idx = [i for i, m in enumerate(metas) if os.path.basename(os.path.dirname(os.path.dirname(m["path"]))) == tag]
        if idx:
            per_tag.append(rank_pct(probs[idx].mean(0)))
    return np.mean(per_tag, axis=0)


def blend(public: pd.DataFrame, ours: np.ndarray, w: float) -> pd.DataFrame:
    out = public.copy()
    out[TARGETS] = (1 - w) * rank_pct(public[TARGETS].to_numpy(np.float64)) + w * rank_pct(ours)
    out[TARGETS] = rank_pct(out[TARGETS].to_numpy(np.float64))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--public", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tags", default="r1_resnet34,r1_convnext_nano")
    p.add_argument("--w", type=float, default=0.4)
    p.add_argument("--k_eval", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save", default="/kaggle/working/ours_probs.npz")
    a = p.parse_args(argv)

    import torch

    from .fleet import predict_fleet
    from .submit import find_competition_root

    t0 = time.time()
    public = pd.read_csv(a.public, dtype={"StudyInstanceUID": str})
    tags = [t for t in a.tags.split(",") if t]
    ckpts = find_checkpoints(tags)
    print(f"[ours] checkpoints: { {t: len(v) for t, v in ckpts.items()} }", flush=True)
    if not ckpts:
        print("[ours] no checkpoints found -> public file kept", flush=True)
        public.to_csv(a.out, index=False)
        return 0
    root = find_competition_root()
    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    rows = {u: g.to_dict("records") for u, g in series.groupby("StudyInstanceUID")}
    uids = public["StudyInstanceUID"].tolist()
    assert set(uids) == set(test["StudyInstanceUID"]), "public submission does not cover test.csv"
    paths = [p for t in tags for p in ckpts.get(t, [])]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probs, metas, failures = predict_fleet(uids, rows, str(root / "test_series"), paths, device,
                                           k_eval=a.k_eval, workers=a.workers)
    np.savez(a.save, probs=probs, uids=np.array(uids), paths=np.array(paths))
    ours = combine(probs, metas, tags)
    ok = np.isfinite(ours).all() and ours.std(axis=0).min() > 1e-9 and len(failures) <= 0.02 * len(uids)
    report = {"checkpoints": len(paths), "failures": len(failures), "ok": bool(ok), "w": a.w,
              "minutes": round((time.time() - t0) / 60, 1)}
    print(f"[ours] {json.dumps(report)}", flush=True)
    if not ok:
        print(f"[ours] leg rejected -> public file kept; first failures {failures[:3]}", flush=True)
        public.to_csv(a.out, index=False)
        return 0
    out = blend(public, ours, a.w)
    assert list(out.columns) == list(public.columns) and np.isfinite(out[TARGETS].to_numpy()).all()
    out.to_csv(a.out, index=False)
    print(f"[ours] wrote {a.out} ({len(out)} studies)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
