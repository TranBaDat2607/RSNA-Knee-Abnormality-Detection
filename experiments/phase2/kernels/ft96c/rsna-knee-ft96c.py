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

# Phase 3 FT96: fine-tune on the d96 cache (nartaa recipe input) with teach4 labels.
# ft96c: CLEAN ftC recipe on teach4opus labels (teach4 + Opus 4.5, gold agreement 0.901), 2 epochs, v10 and v5 starts.
PLAN: list = [
 {
  "parallel": [
   [
    "--tag",
    "ftCo_v10",
    "--init",
    "raptor_ft_coatnet_v10_full.pt",
    "--out",
    "/kaggle/working/ftCo",
    "--seed",
    "42",
    "--corpus",
    "c96",
    "--crop",
    "320",
    "--res",
    "320",
    "--teacher",
    "teach4opus",
    "--k",
    "12",
    "--k_eval",
    "94",
    "--k_eval_holdout",
    "24",
    "--eval_chunk",
    "16",
    "--mirror_tta",
    "--mirror_p",
    "0.5",
    "--epochs",
    "2",
    "--bs",
    "2",
    "--accum",
    "8",
    "--workers",
    "2",
    "--bb_lr",
    "1.5e-5",
    "--head_lr",
    "1.5e-4",
    "--wd",
    "0.05",
    "--holdout_n",
    "300",
    "--grad_ckpt",
    "--save_ckpt",
    "--time_limit_h",
    "10.5",
    "--log_every",
    "200"
   ],
   [
    "--tag",
    "ftC5o_v5",
    "--init",
    "raptor_ft_coatnet_v5_full_swa.pt",
    "--out",
    "/kaggle/working/ftC5o",
    "--seed",
    "5",
    "--corpus",
    "c96",
    "--crop",
    "320",
    "--res",
    "320",
    "--teacher",
    "teach4opus",
    "--k",
    "12",
    "--k_eval",
    "94",
    "--k_eval_holdout",
    "24",
    "--eval_chunk",
    "16",
    "--mirror_tta",
    "--mirror_p",
    "0.5",
    "--epochs",
    "2",
    "--bs",
    "2",
    "--accum",
    "8",
    "--workers",
    "2",
    "--bb_lr",
    "1.5e-5",
    "--head_lr",
    "1.5e-4",
    "--wd",
    "0.05",
    "--holdout_n",
    "300",
    "--grad_ckpt",
    "--save_ckpt",
    "--time_limit_h",
    "10.5",
    "--log_every",
    "200"
   ]
  ]
 }
]


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
