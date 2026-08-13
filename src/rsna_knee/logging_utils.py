"""A single relative-time logger shared by every module.

Kept deliberately tiny (``print`` under the hood, not the ``logging`` module) because the
notebook this package replaces used exactly this, and the runs are long, single-process,
and mostly interesting for their timing — a Kaggle run's elapsed-seconds prefix is what
you actually want to eyeball while it works through 4,407 studies.
"""

from __future__ import annotations

import time


class Clock:
    """Wall-clock start point plus a ``log`` that prefixes messages with elapsed seconds."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def elapsed(self) -> float:
        return time.time() - self.t0


# A process-wide default clock. Most call sites just want "log this with elapsed time"
# without threading a Clock instance through every function signature; pass an explicit
# Clock (e.g. from Config-driven code that wants a fresh one per run) where that matters.
_default_clock = Clock()


def log(msg: str) -> None:
    _default_clock.log(msg)


def reset_clock() -> None:
    """Restart the module-level clock's zero point (e.g. at the start of a pipeline run)."""
    global _default_clock
    _default_clock = Clock()
