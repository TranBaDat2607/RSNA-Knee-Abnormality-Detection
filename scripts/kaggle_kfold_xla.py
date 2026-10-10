#!/usr/bin/env python3
"""Kaggle TPU-VM script-kernel entry point for ``rsna_knee.mil.kfold_xla`` (k-fold training with OOF).

Attach the code dataset (``tranbadat/rsna-knee-code``), the corpus cache kernel output
(``tranbadat/rsna-knee-cache256``) and the competition data; machine shape ``TpuV5E8``, internet on
(timm weights). Each ``RUNS`` entry trains in its own subprocess, one after another, over all chips.
"""
import json
import os
import subprocess
import sys

RUNS: list = [
    ["--arch", "convnext_nano.in12k_ft_in1k", "--tag", "x1_convnext_nano", "--epochs", "16",
     "--bs", "4", "--workers", "6", "--bb_lr", "4e-4", "--head_lr", "2e-3", "--targets", "targets_r2.csv", "--time_limit_h", "8.5"],
]


def _code_root() -> str:
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
        if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
            return os.path.dirname(d)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


if __name__ == "__main__":
    try:
        import timm  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
    env = {**os.environ, "PYTHONPATH": _code_root()}
    import timm

    for argv in RUNS:  # fetch weights once, before eight processes race on the hub cache
        timm.create_model(argv[argv.index("--arch") + 1], pretrained=True, num_classes=0)
    results = []
    for argv in RUNS:
        rc = subprocess.call([sys.executable, "-m", "rsna_knee.mil.kfold_xla", *argv], env=env)
        results.append({"argv": argv, "exit_code": rc})
        print(f"[plan] {argv} -> exit {rc}", flush=True)
        json.dump(results, open("/kaggle/working/plan_results.json", "w"), indent=1)
