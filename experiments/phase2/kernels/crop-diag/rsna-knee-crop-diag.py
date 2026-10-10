#!/usr/bin/env python3
"""CPU diagnostic: is the knee off-centre in the corpus' fixed centre 140 mm crop?

For 400 training studies x every slot series of CORPUS44_336: the middle slice's tissue mask (Otsu), its centroid
offset from the image centre in mm, and the fraction of tissue that falls outside the centred 140 mm square."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
for d, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [x for x in dirs if x not in ("competitions", "train_series", "test_series")]
    if os.path.basename(d) == "rsna_knee" and "__init__.py" in files:
        sys.path.insert(0, os.path.dirname(d)); break
import numpy as np, pandas as pd, cv2  # noqa: E402
from rsna_knee.mil.recipes import CORPUS44_336  # noqa: E402
from rsna_knee.mil.submit import find_competition_root  # noqa: E402
from rsna_knee.mil.volume import order_series_files, pick_series, read_pixels  # noqa: E402

root = find_competition_root()
series = pd.read_csv(root / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
rows = {u: g.to_dict("records") for u, g in series.groupby("StudyInstanceUID")}
ids = sorted(rows)[:: max(1, len(rows) // 400)][:400]


def one(u):
    out, used = [], set()
    for slot in CORPUS44_336.slots:
        r = pick_series(rows[u], slot.plane, slot.fluid, used)
        if r is None:
            continue
        used.add(r["SeriesInstanceUID"])
        try:
            files, sp = order_series_files(os.path.join(root, "train_series", u, str(r["SeriesInstanceUID"])))
            a = read_pixels(files[len(files) // 2][0])
        except Exception as e:  # noqa: BLE001
            out.append({"plane": slot.plane, "err": repr(e)}); continue
        lo, hi = np.percentile(a, [2, 98]); g = (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        _, m = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = m.shape; ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        side = min(int(round(140 / max(sp, 1e-3))), h, w); y0, x0 = (h - side) // 2, (w - side) // 2
        inside = m[y0:y0 + side, x0:x0 + side].sum()
        out.append({"plane": slot.plane, "fluid": slot.fluid, "spacing": sp, "fov_mm": w * sp,
                    "dx_mm": (xs.mean() - w / 2) * sp, "dy_mm": (ys.mean() - h / 2) * sp,
                    "outside_frac": 1 - inside / m.sum(), "crop_covers_image": side >= min(h, w)})
    return u, out


t0 = time.time()
with ProcessPoolExecutor(os.cpu_count()) as pool:
    res = dict(pool.map(one, ids, chunksize=4))
recs = [dict(study=u, **r) for u, rr in res.items() for r in rr if "err" not in r]
df = pd.DataFrame(recs); df.to_csv("/kaggle/working/crop_diag.csv", index=False)
df["off_mm"] = np.hypot(df.dx_mm, df.dy_mm)
summ = df.groupby("plane").agg(n=("study", "size"), fov_med=("fov_mm", "median"), off_med=("off_mm", "median"),
                               off_p90=("off_mm", lambda s: s.quantile(.9)), outside_med=("outside_frac", "median"),
                               outside_p90=("outside_frac", lambda s: s.quantile(.9)),
                               crop_full=("crop_covers_image", "mean"))
print(summ.round(3).to_string(), flush=True)
print("errors", sum("err" in r for rr in res.values() for r in rr), "seconds", round(time.time() - t0), flush=True)
