"""S6T pixel cache, one shard: heliosli's own inference path builds the training rows.

For every training study of this shard, compact_infer.CStudies.rows(uid) runs DICOM -> native render -> geometry bank ->
SKM-TEA anchors -> px_cache.StudySlices.build (views v4, size 320), i.e. the code that renders the rows s6fall scores at
inference.  The rows are written in px_cache's format (px.u8.npy, meta.f32.npy, index.json) so compact_train reads them.

Code tree = s6fall's scored file set (nartaa/rsna-knee-s6fall-weights s6_extras.bin + heliosli's dataset), every file
checked against code_manifest.json.  No network, no model training, no label reading beyond the gold exclusion.
"""
import hashlib, io, json, os, shutil, sys, tarfile, time, types
from pathlib import Path
import numpy as np
import pandas as pd

SHARD, NSHARD, LIMIT = 0, 1, 12
SIZE, VIEWS = 320, "v4"
WORKERS = 4
COMP = "rsna-knee-abnormality-detection"
S6F_SLUG, HELIOS_SLUG = "rsna-knee-s6fall-weights", "rsna-orthofoundation-fold0-epoch7-20261004"
ANATOMY_FILE, ANATOMY_SHA256 = "anatomy.pt", "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
PAYLOAD_MAGIC = b"K1PAYLOAD1\n"
SOURCE_SHA = {
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
    "compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
}
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
TMP = Path("/tmp/s6tc")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_mount(slug, marker):
    cands = [INPUT / slug, INPUT / "competitions" / slug]
    cands += list(INPUT.glob("datasets/*/" + slug)) + list(INPUT.glob("datasets/*/" + slug + "/*"))
    cands += list(INPUT.glob("*/" + slug)) + list(INPUT.glob(slug + "/*"))
    found = []
    for p in cands:
        if (p / marker).is_file() and p.resolve() not in [f.resolve() for f in found]:
            found.append(p)
    if len(found) != 1:
        raise SystemExit("cannot resolve %s (marker %s): %s" % (slug, marker, found))
    return found[0]


def safe_rel(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise SystemExit("unsafe payload member %r" % name)
    return p


def assemble_code(ds, helios, root):
    """s6fall's prepare_inputs(): extras from the payload, every other file from heliosli's dataset, sha-checked."""
    man = json.loads((ds / "code_manifest.json").read_text())
    extras = {n: man["members"][n] for n in man["from_this_dataset"]}
    if root.exists():
        shutil.rmtree(root)
    with open(ds / "s6_extras.bin", "rb") as f:
        if f.read(len(PAYLOAD_MAGIC)) != PAYLOAD_MAGIC:
            raise SystemExit("payload magic mismatch")
        with tarfile.open(fileobj=f, mode="r:") as tar:
            ms = tar.getmembers()
            if {m.name for m in ms} != set(extras):
                raise SystemExit("payload member set differs from the manifest")
            for m in ms:
                dst = root / safe_rel(m.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dst, "xb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
    for n, rec in man["members"].items():
        dst = root / safe_rel(n)
        if n not in extras:
            src = helios / safe_rel(n)
            if not src.is_file() or src.stat().st_size != rec["bytes"]:
                raise SystemExit("heliosli file %s missing or size differs" % n)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        if sha256_file(dst) != rec["sha256"]:
            raise SystemExit("code file %s sha differs" % n)
    for n, want in SOURCE_SHA.items():
        if sha256_file(root / "heliosli" / n) != want:
            raise SystemExit("heliosli/%s differs from s6fall's SOURCE_SHA" % n)
    return root / "heliosli", root / "cloud_code"


def _strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value %r" % (val,))


def install_paths(src, code):
    """= s6fall's launcher: sys.path, the training.download_data stub and the distutils guard (vendored MONAI 0.6.0)."""
    sys.dont_write_bytecode = True
    for p in (str(src), str(code)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    if "distutils.util" not in sys.modules:
        try:
            import distutils.util  # noqa: F401
        except Exception:
            d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
            d.__path__ = []
            u.strtobool = _strtobool
            d.util = u
            sys.modules["distutils"], sys.modules["distutils.util"] = d, u


class RowsDS:
    """One item = CStudies.rows(uid) for one study (its own CStudies per worker process)."""

    def __init__(self, uids, comp, src, code, anatomy):
        self.uids, self.comp, self.src, self.code, self.anatomy = uids, comp, src, code, anatomy
        self.cs = None

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        t = time.time()
        try:
            if self.cs is None:
                install_paths(self.src, self.code)
                import torch
                torch.set_num_threads(1)
                import probe_infer as PI
                orig, cache = PI.install_native, {}

                def install_native(lean=True):        # s6fall's memoise_install_native
                    if lean not in cache:
                        cache[lean] = orig(lean)
                    return cache[lean]
                PI.install_native = install_native
                import compact_infer as CI
                tmp = TMP / ("w%d" % os.getpid())
                tmp.mkdir(parents=True, exist_ok=True)
                self.cs = CI.CStudies(str(self.comp), "train", [], str(tmp), str(self.code), SIZE, eval_stride=1, views=VIEWS,
                                      slices=3, anchor_weights=str(self.anatomy), phys_mm=3.5)
            item, sides = self.cs.rows(uid)
            if item.get("error"):
                return {"uid": uid, "error": item["error"], "seconds": time.time() - t}
            return {"uid": uid, "error": None, "px": item["px"].numpy(), "meta": item["meta"].numpy(), "blocks": item["blocks"],
                    "removed": int(item.get("removed", 0)), "skipped": item.get("skipped_series", []), "sides": sorted(map(str, sides)),
                    "seconds": time.time() - t}
        except Exception as exc:
            return {"uid": uid, "error": repr(exc)[:400], "seconds": time.time() - t}


def identity(x):
    return x


def npy_header(shape, dtype):
    b = io.BytesIO()
    np.lib.format.write_array_header_1_0(b, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False,
                                             "shape": tuple(int(s) for s in shape)})
    return b.getvalue()


PLACEHOLDER_N = 10 ** 8                 # header written first with this row count, rewritten in place with the real one


def main():
    import torch
    comp = find_mount(COMP, "train.csv")
    ds = find_mount(S6F_SLUG, "code_manifest.json")
    helios = find_mount(HELIOS_SLUG, ANATOMY_FILE)
    if sha256_file(helios / ANATOMY_FILE) != ANATOMY_SHA256:
        raise SystemExit("anatomy.pt differs from s6fall's anchor checkpoint")
    src, code = assemble_code(ds, helios, TMP / "code")
    log("code tree ok:", src, code)

    train = pd.read_csv(comp / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[train[LABELS].notna().any(axis=1)]
    rh = train["Report"].fillna("").map(lambda s: hashlib.sha1(s.strip().encode()).hexdigest())
    gold_hash = set(rh[train["StudyInstanceUID"].isin(gold["StudyInstanceUID"])])
    keep = train[~train["StudyInstanceUID"].isin(gold["StudyInstanceUID"]) & ~rh.isin(gold_hash)]["StudyInstanceUID"].tolist()
    log("train %d | gold %d | gold-report duplicates %d | pool %d" % (len(train), len(gold), len(train) - len(gold) - len(keep), len(keep)))
    uids = sorted(u for u in keep if int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % NSHARD == SHARD)
    if LIMIT:
        uids = uids[:LIMIT]
    log("shard %d of %d: %d studies" % (SHARD, NSHARD, len(uids)))

    out = WORK / ("px_s%d" % SHARD)
    out.mkdir(parents=True, exist_ok=True)
    h0 = npy_header((PLACEHOLDER_N, SIZE, SIZE), np.uint8)
    raw_px = open(out / "px.u8.npy", "wb")
    raw_px.write(h0)
    metas = []
    index = {"order": uids, "done": {}, "blocks": {}, "errors": {}, "cursor": 0, "removed_series": 0, "sides": {},
             "meta_cols": None, "seconds": {},
             "config": {"size": SIZE, "views": VIEWS, "builder": "compact_infer.CStudies.rows", "shard": SHARD, "nshard": NSHARD,
                        "policy": "series_companded", "local_mm": 100.0}}
    loader = torch.utils.data.DataLoader(RowsDS(uids, comp, src, code, helios / ANATOMY_FILE), batch_size=None, shuffle=False,
                                         num_workers=WORKERS, prefetch_factor=2, persistent_workers=False, collate_fn=identity)
    n_done = 0
    for item in loader:
        uid = item["uid"]
        n_done += 1
        index["seconds"][uid] = round(item["seconds"], 2)
        if item["error"]:
            index["errors"][uid] = item["error"]
            log("ERROR", uid, item["error"][:200])
            continue
        px, meta = item["px"], item["meta"]
        cur, n = index["cursor"], len(px)
        if n:
            assert px.dtype == np.uint8 and px.shape[1:] == (SIZE, SIZE), px.shape
            raw_px.write(np.ascontiguousarray(px).tobytes())
            metas.append(np.ascontiguousarray(meta, dtype=np.float32))
        blocks = item["blocks"]
        for b in blocks:
            b["views"] = [[int(v), cur + int(off)] for v, off in b["views"]]
        index["done"][uid] = [cur, n, len(blocks)]
        index["blocks"][uid] = blocks
        index["sides"][uid] = item["sides"]
        index["cursor"] = cur + n
        index["removed_series"] += item["removed"]
        if n_done % 25 == 0 or n_done == len(uids):
            el = time.time() - T0
            log("%d/%d studies | rows %d (%.1f/study) | %.1f GB | errors %d" % (
                n_done, len(uids), index["cursor"], index["cursor"] / max(1, len(index["done"])),
                index["cursor"] * SIZE * SIZE / 2 ** 30, len(index["errors"])))
    raw_px.close()
    N = index["cursor"]
    # px_cache's format: px.u8.npy = the rows behind an .npy header; the placeholder header is rewritten in place
    h1 = npy_header((N, SIZE, SIZE), np.uint8)
    if len(h1) != len(h0):
        raise SystemExit("npy header length changed (%d vs %d)" % (len(h1), len(h0)))
    with open(out / "px.u8.npy", "r+b") as fo:
        fo.write(h1)
    meta_all = np.concatenate(metas) if metas else np.zeros((0, 12), np.float32)
    assert len(meta_all) == N, (len(meta_all), N)
    np.save(out / "meta.f32.npy", meta_all)
    chk = np.load(out / "px.u8.npy", mmap_mode="r")
    assert chk.shape == (N, SIZE, SIZE), chk.shape
    assert os.path.getsize(out / "px.u8.npy") == len(h1) + N * SIZE * SIZE
    ncol = int(meta_all.shape[1])
    index["meta_cols"] = ncol
    index["n_upper"] = N
    secs = list(index["seconds"].values())
    index["summary"] = {"studies": len(index["done"]), "errors": len(index["errors"]), "rows": N,
                        "gb": round(N * SIZE * SIZE / 2 ** 30, 2), "mean_s_per_study": round(float(np.mean(secs)), 2) if secs else None,
                        "wall_min": round((time.time() - T0) / 60, 1)}
    json.dump(index, open(out / "index.json", "w", encoding="utf-8"))
    log("DONE", json.dumps(index["summary"]))


if __name__ == "__main__":
    main()
