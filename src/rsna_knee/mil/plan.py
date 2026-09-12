"""Run training as a plan of isolated stages.

Each stage runs in its own subprocess, so one stage dying (a host-RAM OOM ends in SIGKILL, with no
Python traceback) never takes the others down:

    {"parallel": [argv_gpu0, argv_gpu1]}  independent single-GPU jobs side by side (an A/B pair)
    {"ddp": argv}                         one job data-parallel over every visible GPU

Pretrained weights are fetched once before any stage starts, so concurrent jobs never race on the
hub cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

Plan = Sequence[dict]


def plan_archs(plan: Plan, parse_args: Callable) -> set[str]:
    return {parse_args(list(argv)).arch for stage in plan for argv in (stage.get("parallel") or [stage["ddp"]])}


def child_main(argv: Sequence[str], run: Callable, parse_args: Callable) -> bool:
    """Handle ``--child <json argv>`` / ``--ddp <json argv>``; returns False for a plan invocation."""
    if len(argv) < 3 or argv[1] not in ("--child", "--ddp"):
        return False
    args = parse_args(json.loads(argv[2]))
    if argv[1] == "--child":
        run(0, 1, args)
    else:
        import torch
        import torch.multiprocessing as mp

        world = torch.cuda.device_count()
        mp.spawn(run, args=(world, args), nprocs=world, join=True)
    return True


def run_plan(plan: Plan, script: str, parse_args: Callable, out_dir: str = "/kaggle/working",
             log: Callable[[str], None] = print, prefetch: bool = True) -> list[dict]:
    if prefetch:
        import timm

        for arch in sorted(plan_archs(plan, parse_args)):
            timm.create_model(arch, pretrained=True, num_classes=0)
    results = []
    for i, stage in enumerate(plan):
        t = time.time()
        if "parallel" in stage:
            procs = [subprocess.Popen([sys.executable, script, "--child", json.dumps(list(argv))],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
                     for g, argv in enumerate(stage["parallel"])]
            codes = [p.wait() for p in procs]
        else:
            codes = [subprocess.call([sys.executable, script, "--ddp", json.dumps(list(stage["ddp"]))])]
        results.append({"stage": i, "exit_codes": codes, "minutes": round((time.time() - t) / 60, 2)})
        log(f"[plan] stage {i} exit codes {codes} after {(time.time() - t) / 60:.1f} min")
        with open(os.path.join(out_dir, "plan_results.json"), "w") as f:
            json.dump(results, f, indent=1)
    return results
