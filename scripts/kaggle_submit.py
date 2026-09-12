#!/usr/bin/env python3
"""Kaggle submission-notebook entry point (internet off): runs ``rsna_knee.mil.submit.main``.

Attach: the code dataset ``tranbadat/rsna-knee-code``; the public Raptor checkpoints
(``dreaddevelopment/raptor-knee-maxspan``, ``-native384dense``, ``-native384``); the residual-gated
CoAtNet (``mattiaangeli/rsna-knee-coat-resgated-ep10-top3``) and its pinned OpenCV wheel
(``mattiaangeli/opencv-python-headless-4120088-x86``); and the competition data.
"""
import os
import sys


def _add_code_to_path() -> None:
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
        if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
            sys.path.insert(0, os.path.dirname(d))
            return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


_add_code_to_path()

from rsna_knee.mil.submit import SubmitConfig, main  # noqa: E402

if __name__ == "__main__":
    main(SubmitConfig())
