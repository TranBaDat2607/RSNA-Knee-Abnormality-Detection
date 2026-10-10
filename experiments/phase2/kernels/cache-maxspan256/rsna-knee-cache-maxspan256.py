#!/usr/bin/env python3
"""CPU kernel: build the ``maxspan64_256`` training cache (Raptor v5 slots: 64 slices over 2-98 %) from the DICOMs."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

for d, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
    if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
        sys.path.insert(0, os.path.dirname(d))
        break

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rsna_knee.mil.cache import CachedCorpus  # noqa: E402
from rsna_knee.mil.recipes import MAXSPAN64_256  # noqa: E402
from rsna_knee.mil.submit import find_competition_root  # noqa: E402
from rsna_knee.mil.volume import build_volume  # noqa: E402

NAME, OUT = "maxspan256", "/kaggle/working"
root = find_competition_root()
series = pd.read_csv(root / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
rows = {u: g.to_dict("records") for u, g in series.groupby("StudyInstanceUID")}
ids = pd.read_csv(root / "train.csv", usecols=["StudyInstanceUID"], dtype=str).StudyInstanceUID.drop_duplicates().tolist()
print("studies", len(ids), "cpus", os.cpu_count(), flush=True)


def job(u):
    try:
        return build_volume(u, rows.get(u, []), str(root / "train_series"), MAXSPAN64_256), None
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)


t0 = time.time()
n, depth, px = len(ids), MAXSPAN64_256.n_slices, MAXSPAN64_256.img
vols = np.lib.format.open_memmap(f"{OUT}/{NAME}_vols.npy", mode="w+", dtype=np.uint8, shape=(n, depth, px, px))
masks = np.zeros((n, depth), np.uint8)
fails = []
with ProcessPoolExecutor(os.cpu_count()) as pool:
    for i, (res, err) in enumerate(pool.map(job, ids, chunksize=4)):
        if res is None:
            fails.append((ids[i], err))
        else:
            vols[i], masks[i] = res
        if (i + 1) % 200 == 0:
            print(f"[{time.time() - t0:6.0f}s] {i + 1}/{n} failures {len(fails)}", flush=True)
vols.flush()
del vols
np.save(f"{OUT}/{NAME}_masks.npy", masks)
np.save(f"{OUT}/{NAME}_ids.npy", np.array(ids, dtype=object))
c = CachedCorpus(lambda f: os.path.join(OUT, f), px, NAME)
for i in (0, n // 2, n - 1):
    v, m = build_volume(ids[i], rows.get(ids[i], []), str(root / "train_series"), MAXSPAN64_256)
    assert np.array_equal(c.volume(i), v) and np.array_equal(c.masks[i], m), i
json.dump({"failures": fails, "seconds": round(time.time() - t0), "mean_filled": float(masks.mean())},
          open(f"{OUT}/{NAME}_build.json", "w"), indent=1)
print("verified; failures", len(fails), fails[:5], "done in", round(time.time() - t0), "s", flush=True)
