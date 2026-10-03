#!/usr/bin/env python3
"""Kaggle script-kernel entry point for ``rsna_knee.mil.kfold`` (k-fold training with OOF output).

Same pattern as ``kaggle_mil_train.py``: the attached code dataset (``tranbadat/rsna-knee-code``) goes
on ``sys.path`` and ``PLAN`` runs as isolated stages (``rsna_knee.mil.plan``). Attach the code
dataset, the corpus cache kernel output (``tranbadat/rsna-knee-cache256``) and the competition data.
"""
import os
import sys

PLAN: list = [{"parallel": [
    ["--arch", "resnet34.a1_in1k", "--tag", "r1_resnet34", "--time_limit_h", "3.6"],
    ["--arch", "convnext_nano.in12k_ft_in1k", "--tag", "r1_convnext_nano", "--time_limit_h", "3.6"],
]}]


def _add_code_to_path() -> None:
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
        if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
            sys.path.insert(0, os.path.dirname(d))
            return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


_add_code_to_path()

from rsna_knee.mil import kfold, plan  # noqa: E402

if __name__ == "__main__":
    if not plan.child_main(sys.argv, run=kfold.run, parse_args=kfold.parse_args):
        plan.run_plan(PLAN, os.path.abspath(__file__), parse_args=kfold.parse_args)
