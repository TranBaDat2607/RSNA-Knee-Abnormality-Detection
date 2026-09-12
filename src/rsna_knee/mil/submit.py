"""Offline submission: CoAtNet MIL arms + the residual-gated CoAtNet -> rank fusion -> ``submission.csv``.

Every stage is guarded so a failing arm costs its vote, never the submission:

1. a 0.5 benchmark file is written first (a SIGKILL mid-run still leaves something scoreable);
2. MIL arms (the public Raptor checkpoints plus any of ours) via :func:`~rsna_knee.mil.infer.predict_arms`
   — one decode per study shared by every recipe, studies sharded over both GPUs;
3. the residual-gated CoAtNet through its own packaged dual-T4 runtime in a subprocess (it pins its
   own OpenCV build and preprocessing, so it is run as shipped rather than re-implemented);
4. fixed-weight rank fusion: ``(1 - w) * mean_rank(MIL arms) + w * rank(residual-gated)``;
5. schema validation before the final write.

Fusion weights are declared, not searched: equal weights inside the MIL family matched the public
notebooks' hand-tuned inner weights on gold-58 (E1), and the residual-gated arm keeps the 0.4 share
the public stack gives it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TARGETS
from .blend import rank_pct
from .recipes import PUBLIC_RAPTOR_ARMS, ArmSpec


@dataclass
class SubmitConfig:
    mil_arms: tuple[ArmSpec, ...] = PUBLIC_RAPTOR_ARMS
    mil_weights: dict[str, float] = field(default_factory=dict)  # missing -> 1.0
    resgated_weight: float = 0.4
    out_path: str = "submission.csv"
    work_dir: str = "/kaggle/working"


def kaggle_find_file(fname: str) -> str:
    for d, dirs, fs in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
        if fname in fs:
            return os.path.join(d, fname)
    raise FileNotFoundError(fname)


def find_competition_root() -> Path:
    for c in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection")):
        if (c / "test.csv").is_file() and (c / "test_series.csv").is_file():
            return c
    raise FileNotFoundError("competition data not mounted")


def informative(pred: np.ndarray, min_std: float = 1e-6) -> bool:
    """False for an arm that produced (almost) constant output, e.g. its weights failed to load."""
    return bool(np.isfinite(pred).all() and np.all(pred.std(axis=0) > min_std))


def fuse(mil_preds: Mapping[str, np.ndarray], mil_weights: Mapping[str, float],
         resgated: np.ndarray | None, resgated_weight: float) -> np.ndarray:
    """Rank fusion; uninformative arms are dropped and the remaining weights renormalised."""
    usable = {k: v for k, v in mil_preds.items() if informative(v)}
    parts, weights = [], []
    if usable:
        w = np.array([float(mil_weights.get(k, 1.0)) for k in usable])
        family = sum(wi * rank_pct(usable[k]) for wi, k in zip(w / w.sum(), usable))
        parts.append(family)
        weights.append(1.0 - resgated_weight if resgated is not None and informative(resgated) else 1.0)
    if resgated is not None and informative(resgated):
        parts.append(rank_pct(resgated))
        weights.append(resgated_weight if usable else 1.0)
    if not parts:
        raise RuntimeError("no informative arm")
    weights = np.array(weights) / sum(weights)
    return sum(wi * p for wi, p in zip(weights, parts))


def validate_submission(frame: pd.DataFrame, test_ids: Sequence[str], targets: Sequence[str] = TARGETS) -> None:
    if list(frame.columns) != ["StudyInstanceUID", *targets]:
        raise ValueError(f"columns differ from the competition contract: {list(frame.columns)}")
    if len(frame) != len(test_ids) or not frame["StudyInstanceUID"].is_unique:
        raise ValueError("row count or StudyInstanceUID uniqueness")
    if set(frame["StudyInstanceUID"].astype(str)) != set(map(str, test_ids)):
        raise ValueError("StudyInstanceUID set differs from test.csv")
    values = frame[list(targets)].to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("non-finite or out-of-range prediction")


def run_resgated(competition_root: Path, work_dir: str, find_file: Callable[[str], str],
                 test_ids: Sequence[str], log: Callable[[str], None]) -> np.ndarray | None:
    """Residual-gated CoAtNet e4/e6/e8 via its packaged runtime; ``None`` if anything is missing or fails."""
    try:
        art = os.path.dirname(find_file("coat_resgated_ep10_top3_manifest.json"))
        wheel = next(os.path.join(d, f) for d, _, fs in os.walk("/kaggle/input") for f in fs
                     if f.startswith("opencv_python_headless-4.12.0.88-") and f.endswith(".whl"))
    except (FileNotFoundError, StopIteration) as exc:
        log(f"residual-gated arm unavailable: {exc!r}")
        return None
    envd = os.path.join(work_dir, "_coat_env")
    out = os.path.join(work_dir, "_resgated.csv")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "--target", envd, wheel], check=True)
        child = ("import sys\n"
                 f"sys.path.insert(0, {envd!r}); sys.path.insert(0, {art!r})\n"
                 "import coatnet_resgated_ep10_top3_inference as rt\n"
                 "from pathlib import Path\n"
                 f"rt.run_submission(competition_root=Path({str(competition_root)!r}), artifact_root=Path({art!r}), "
                 f"output_path=Path({out!r}), gpu_batch_studies=2, backbone_micro_images=8)\n")
        env = {**os.environ, "PYTHONPATH": f"{envd}:{art}:" + os.environ.get("PYTHONPATH", "")}
        t = time.time()
        proc = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            log(f"residual-gated arm failed: {proc.stderr[-1500:]}")
            return None
        pred = pd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        log(f"residual-gated arm done in {time.time() - t:.0f}s")
        return pred.reindex(list(map(str, test_ids)))[list(TARGETS)].to_numpy(np.float64)
    except Exception as exc:  # noqa: BLE001 - the submission must survive any arm
        log(f"residual-gated arm failed: {exc!r}")
        return None


def main(cfg: SubmitConfig = SubmitConfig(), find_file: Callable[[str], str] = kaggle_find_file,
         log: Callable[[str], None] = print) -> pd.DataFrame:
    import torch

    from ..inference.submission import write_benchmark_submission
    from .infer import predict_arms

    t0 = time.time()
    root = find_competition_root()
    write_benchmark_submission(root / "test.csv", TARGETS, cfg.out_path)
    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    test_ids = test["StudyInstanceUID"].tolist()
    series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    rows = {k: v.to_dict("records") for k, v in series.groupby("StudyInstanceUID", sort=False)}
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    log(f"{len(test_ids)} test studies | {len(devices)} device(s) | MIL arms {[a.name for a in cfg.mil_arms]}")

    mil_preds, failures = predict_arms(test_ids, rows, str(root / "test_series"), cfg.mil_arms, devices,
                                       find_checkpoint=find_file, log=log)
    torch.cuda.empty_cache()
    resgated = run_resgated(root, cfg.work_dir, find_file, test_ids, log) if cfg.resgated_weight > 0 else None
    fused = fuse(mil_preds, cfg.mil_weights, resgated, cfg.resgated_weight)

    frame = pd.DataFrame(fused, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", test_ids)
    validate_submission(frame, test_ids)
    frame.to_csv(cfg.out_path, index=False)
    receipt = {"minutes": round((time.time() - t0) / 60, 1), "studies": len(test_ids),
               "mil_arms": {k: informative(v) for k, v in mil_preds.items()},
               "resgated": resgated is not None and informative(resgated),
               "study_failures": len(failures), "weights": {"mil": cfg.mil_weights, "resgated": cfg.resgated_weight}}
    with open(os.path.join(cfg.work_dir, "submit_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    log(f"submission written: {receipt}")
    return frame
