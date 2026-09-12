#!/usr/bin/env python3
"""Kaggle script-kernel entry point for ``rsna_knee.mil.train``.

A script kernel uploads a single file, so this launcher puts the attached code dataset
(``tranbadat/rsna-knee-code``) on ``sys.path`` and runs ``PLAN`` — a list of stages, see
``rsna_knee.mil.plan``. Copy it into a kernel directory and fill in ``PLAN``, e.g.::

    PLAN = [{"parallel": [["--tag", "rawA", "--teacher", "flight", "--train_n", "2000", "--grad_ckpt"],
                          ["--tag", "canonB", "--canonical", "--teacher", "flight", "--train_n", "2000", "--grad_ckpt"]]}]

Attach: the code dataset, both corpus parts, the teacher datasets (``rsna_knee.mil.teachers.TEACHER_DATASETS``),
the competition data, and — for ``--canonical`` — the orientation/side tables.
"""
import os
import sys

PLAN: list = []


def _add_code_to_path() -> None:
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
        if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
            sys.path.insert(0, os.path.dirname(d))
            return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


_add_code_to_path()

from rsna_knee.mil import plan, train  # noqa: E402

if __name__ == "__main__":
    if not plan.child_main(sys.argv, run=train.run, parse_args=train.parse_args):
        plan.run_plan(PLAN, os.path.abspath(__file__), parse_args=train.parse_args)
