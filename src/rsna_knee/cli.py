"""Command-line entrypoint: ``python -m rsna_knee.cli`` or the ``rsna-knee`` console script.

Only exposes the knobs worth changing from a shell without editing :mod:`rsna_knee.config`
— everything else takes its default from ``Config()``. Run from the competition data root
(or on Kaggle, where the path is auto-detected; see :mod:`rsna_knee.paths`).
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import run_safe


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsna-knee",
        description="Train the 4-fold CV imaging model and write submission.csv.")
    p.add_argument("--out", default="submission.csv", help="output submission path")
    p.add_argument("--epochs", type=int, default=None, help="override Config.train.epochs")
    p.add_argument("--folds", type=int, default=None, help="override Config.cv.n_folds")
    p.add_argument("--time-budget-hours", type=float, default=None,
                    help="override Config.train.time_budget_s")
    p.add_argument("--slot-scheme", choices=["recovered", "public"], default=None,
                    help="override Config.cache.slot_scheme")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = Config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.folds is not None:
        cfg.cv.n_folds = args.folds
    if args.time_budget_hours is not None:
        cfg.train.time_budget_s = args.time_budget_hours * 3600
    if args.slot_scheme is not None:
        cfg.cache.slot_scheme = args.slot_scheme

    result = run_safe(cfg, out_path=args.out)
    return 0 if result is not None and result.submission is not None else 1


if __name__ == "__main__":
    sys.exit(main())
