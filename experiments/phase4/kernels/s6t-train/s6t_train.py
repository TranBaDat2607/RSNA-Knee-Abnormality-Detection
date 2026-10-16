"""S6T training: heliosli's compact_train.py (s6fall's sha) on our teach4opus labels, two seeds, one per T4.

Patches applied to the trainer source (each asserted to match exactly once):
  1. fp16 autocast + GradScaler instead of bf16 (the T4 has no bf16).
  2. all-data training (S6T_ALL_DATA=1): the single model trains on the whole pool, nothing is held out (as s6fall).
  3. --px takes a comma-separated list of cache shards (px_cache format); rows keep their shard as the file index.
Everything else (model, loss, sampling, schedule) is compact_train unchanged.  Pre-registered in docs/phase4_notes.md.
"""
import hashlib, json, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd

EPOCHS = int(os.environ.get("S6T_EPOCHS", "16"))
SEEDS = (7, 8)
DEV = int(os.environ.get("S6T_DEV", "0"))   # 1 = dev run: 1 epoch on whatever the attached cache holds, then stop
STOP_AFTER_H = 11.4                     # never start an epoch that cannot finish before this (checkpoint kept for a resume)
COMP = "rsna-knee-abnormality-detection"
S6F_SLUG, HELIOS_SLUG, CODE_SLUG = "rsna-knee-s6fall-weights", "rsna-orthofoundation-fold0-epoch7-20261004", "rsna-knee-code"
TRAINER_SHA = "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80"
PAYLOAD_MAGIC = b"K1PAYLOAD1\n"
INPUT, WORK, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/s6t")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def walk_inputs(name, max_depth=6):
    """Files called `name` under /kaggle/input, never descending into the DICOM trees (hundreds of thousands of files)."""
    hits = []
    for root, dirs, files in os.walk(INPUT):
        depth = len(Path(root).relative_to(INPUT).parts)
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series") and depth < max_depth]
        if name in files:
            hits.append(Path(root) / name)
    return sorted(hits)


def find_dirs(name):
    return sorted({p.parent for p in walk_inputs(name)})


def find_mount(slug, marker):
    hits = [p for p in walk_inputs(marker) if slug in str(p)]
    if not hits:
        raise SystemExit("cannot find %s/%s under /kaggle/input" % (slug, marker))
    return hits[0].parent


def trainer_source():
    ds = find_mount(S6F_SLUG, "code_manifest.json")
    with open(ds / "s6_extras.bin", "rb") as f:
        if f.read(len(PAYLOAD_MAGIC)) != PAYLOAD_MAGIC:
            raise SystemExit("payload magic mismatch")
        with tarfile.open(fileobj=f, mode="r:") as tar:
            src = tar.extractfile("heliosli/compact_train.py").read()
    if sha256_bytes(src) != TRAINER_SHA:
        raise SystemExit("compact_train.py differs from s6fall's")
    return src.decode()


def sub_once(text, old, new):
    n = text.count(old)
    if n != 1:
        raise SystemExit("patch anchor found %d times: %r" % (n, old[:80]))
    return text.replace(old, new)


JPEG_READER = '''

class JpegRows:
    """Rows of one s6t-cache shard: px[rows] -> uint8 [len(rows), S, S] decoded from the JPEG blob (memory-mapped)."""

    def __init__(self, d):
        self.blob = np.load(os.path.join(d, "rows.off.npy"))
        self.data = np.memmap(os.path.join(d, "rows.jpg.bin"), dtype=np.uint8, mode="r")
        size = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))["config"]["size"]
        self.shape = (len(self.blob) - 1, size, size)

    def __getitem__(self, rows):
        import cv2
        rows = np.atleast_1d(np.asarray(rows, np.int64))
        out = np.empty((len(rows),) + self.shape[1:], np.uint8)
        for j, r in enumerate(rows):
            out[j] = cv2.imdecode(np.asarray(self.data[self.blob[r]:self.blob[r + 1]]), cv2.IMREAD_GRAYSCALE)
        return out


def _px_open(d):
    if os.path.exists(os.path.join(d, "rows.off.npy")):
        return JpegRows(d)
    return np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r")

'''


def to_jpeg(src_dir, dst_dir, q=90):
    """Dev only: a raw px_cache shard (px.u8.npy) re-written as a JPEG shard, so the dev run exercises JpegRows."""
    import cv2
    dst_dir.mkdir(parents=True, exist_ok=True)
    px = np.load(src_dir / "px.u8.npy", mmap_mode="r")
    off, pos = [0], 0
    with open(dst_dir / "rows.jpg.bin", "wb") as f:
        for row in px:
            ok, b = cv2.imencode(".jpg", np.ascontiguousarray(row), [cv2.IMWRITE_JPEG_QUALITY, q])
            f.write(b.tobytes())
            pos += len(b)
            off.append(pos)
    np.save(dst_dir / "rows.off.npy", np.asarray(off, np.int64))
    shutil.copyfile(src_dir / "meta.f32.npy", dst_dir / "meta.f32.npy")
    ix = json.load(open(src_dir / "index.json"))
    ix["config"]["codec"] = "jpeg%d" % q
    json.dump(ix, open(dst_dir / "index.json", "w"))
    return dst_dir


def patch(src):
    # 1. fp16 + GradScaler
    n = src.count('torch.autocast("cuda", dtype=torch.bfloat16)')
    assert n == 3, n
    src = src.replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cuda", dtype=torch.float16)')
    src = sub_once(src, "            opt = torch.optim.AdamW(groups + [{\"params\": head, \"lr\": a.head_lr}], weight_decay=a.wd)\n",
                   "            opt = torch.optim.AdamW(groups + [{\"params\": head, \"lr\": a.head_lr}], weight_decay=a.wd)\n"
                   "            scaler = torch.amp.GradScaler(\"cuda\")\n")
    src = sub_once(src, "                    loss.backward()\n", "                    scaler.scale(loss).backward()\n")
    src = sub_once(src, "                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)\n                    opt.step()\n",
                   "                    scaler.unscale_(opt)\n"
                   "                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)\n"
                   "                    scaler.step(opt)\n                    scaler.update()\n")
    # 2. all-data training (run() only; reeval() keeps its fold logic)
    src = sub_once(src, "            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)\n"
                        "            ho = np.flatnonzero(pool_fold == k)\n            torch.manual_seed(1000 + 17 * seed)",
                   "            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)\n"
                   "            ho = np.flatnonzero(pool_fold == k)\n"
                   "            if os.environ.get('S6T_ALL_DATA') == '1':\n"
                   "                tr, ho = np.arange(len(pool)), np.zeros(0, np.int64)\n"
                   "                print('  all-data: %d training studies, nothing held out' % len(tr), flush=True)\n"
                   "            torch.manual_seed(1000 + 17 * seed)")
    # epoch-time guard: stop cleanly (checkpoint written) before the session limit
    src = sub_once(src, "            for ep in range(start_ep, a.epochs):\n                ds.epoch = ep\n",
                   "            for ep in range(start_ep, a.epochs):\n"
                   "                _lim = float(os.environ.get('S6T_DEADLINE', '0'))\n"
                   "                if _lim and ep > start_ep and time.time() + (time.time() - _t_ep0) / (ep - start_ep) > _lim:\n"
                   "                    print('  S6T deadline: stopping before epoch %d (checkpoint of epoch %d kept)' % (ep, ep - 1), flush=True)\n"
                   "                    sys.exit(4)\n"
                   "                ds.epoch = ep\n")
    src = sub_once(src, "                print(\"  resumed %s seed %d fold %d after epoch %d (%s)\" % (arm, seed, k, ck[\"epoch\"], ckpt_path), flush=True)\n",
                   "                print(\"  resumed %s seed %d fold %d after epoch %d (%s)\" % (arm, seed, k, ck[\"epoch\"], ckpt_path), flush=True)\n"
                   "            _t_ep0 = time.time()\n")
    src = sub_once(src, "            start_ep = 0\n", "            start_ep = 0\n            _t_ep0 = time.time()\n")
    # 4. JPEG row shards (rows.jpg.bin + rows.off.npy, from s6t-cache) next to px_cache's raw px.u8.npy
    src = sub_once(src, "C = len(LABELS)\n", "C = len(LABELS)\n" + JPEG_READER)
    src = sub_once(src, "            self.px = [np.load(os.path.join(d, \"px.u8.npy\"), mmap_mode=\"r\") for d in self.dirs]\n",
                   "            self.px = [_px_open(d) for d in self.dirs]\n")
    # 3. multi-shard cache: --px a,b,c
    src = sub_once(src, "        index = json.load(open(os.path.join(px_dir, \"index.json\"), encoding=\"utf-8\"))\n"
                        "        # blocks carry (file, view_id, start)",
                   "        _shards = px_dir.split(',')\n"
                   "        _idx = [json.load(open(os.path.join(d, 'index.json'), encoding='utf-8')) for d in _shards]\n"
                   "        _where = {u: f for f, ix in enumerate(_idx) for u in ix['done']}\n"
                   "        index = {'blocks': {u: _idx[_where[u]]['blocks'][u] for u in uids}}\n"
                   "        # blocks carry (file, view_id, start)")
    src = sub_once(src, "        self.blocks = {u: [dict(b, views=[(0, v, st) for v, st in b[\"views\"]]) for b in index[\"blocks\"][u]] for u in uids}\n"
                        "        self.dirs = [px_dir]\n",
                   "        self.blocks = {u: [dict(b, views=[(_where[u], v, st) for v, st in b[\"views\"]]) for b in index[\"blocks\"][u]] for u in uids}\n"
                   "        self.dirs = list(_shards)\n")
    src = sub_once(src, "    index = json.load(open(os.path.join(a.px, \"index.json\"), encoding=\"utf-8\"))\n"
                        "    done = {u for u, v in index[\"done\"].items() if v[1] > 0}\n",
                   "    index = {'done': {}}\n"
                   "    for _d in a.px.split(','):\n"
                   "        index['done'].update(json.load(open(os.path.join(_d, 'index.json'), encoding='utf-8'))['done'])\n"
                   "    done = {u for u, v in index[\"done\"].items() if v[1] > 0}\n")
    return src


XLA_HELPERS = '''

def _xla_dev():
    import torch_xla.core.xla_model as xm
    try:                                   # XLA-safe gradient checkpointing for timm (optimisation barriers, no CSE):
        import torch.utils.checkpoint as _tc   # timm's checkpoint_seq calls torch.utils.checkpoint.checkpoint(..., use_reentrant=False),
        import torch_xla.utils.checkpoint as _xc   # whose non-reentrant path fails on XLA ("module 'torch' has no attribute 'xla'")

        def _xla_ckpt(function, *args, use_reentrant=None, **kwargs):
            return _xc.checkpoint(function, *args, use_reentrant=True, **kwargs)
        _tc.checkpoint = _xla_ckpt
    except Exception as _e:
        print("S6T_XLA: checkpoint not redirected:", repr(_e)[:200], flush=True)
    return xm.xla_device()


def _mark_step():
    import torch_xla.core.xla_model as xm
    xm.mark_step()


def _xsave(obj, path):
    import torch_xla.core.xla_model as xm
    xm.save(obj, path)

'''


def patch_xla(src):
    """TPU (torch_xla, one chip per process) on top of patch(): bf16 autocast (no GradScaler), static shapes, crop on CPU,
    mark_step per optimiser step, loss summed on device, xm.save checkpoints. ConvNeXt has no BatchNorm and the head gives
    padded windows -inf attention, so running all B x W windows through the encoder leaves every real window's maths unchanged."""
    src = sub_once(src, "C = len(LABELS)\n", "C = len(LABELS)\n" + XLA_HELPERS)
    src = sub_once(src, 'torch.device("cpu") if a.check_targets else torch.device("cuda")',
                   'torch.device("cpu") if a.check_targets else _xla_dev()')
    n = src.count('torch.autocast("cuda", dtype=torch.float16)')
    assert n == 3, n
    src = src.replace('torch.autocast("cuda", dtype=torch.float16)', 'torch.autocast("xla", dtype=torch.bfloat16)')
    src = sub_once(src, 'scaler = torch.amp.GradScaler("cuda")', 'scaler = torch.amp.GradScaler("cuda", enabled=False)')
    n = src.count(".to(dev).to(memory_format=torch.channels_last)")
    assert n == 2, n
    src = src.replace(".to(dev).to(memory_format=torch.channels_last)", ".to(dev)")
    src = sub_once(src, "xs = ((xs - self.mean) / self.std).contiguous(memory_format=torch.channels_last)",
                   "xs = ((xs - self.mean) / self.std).contiguous()")
    src = sub_once(src, "z, sl, bl = model(augment(img.to(dev, non_blocking=True), gen), meta.to(dev), mask.to(dev), return_aux=True)",
                   "z, sl, bl = model(augment(img, gen).to(dev), meta.to(dev), mask.to(dev), return_aux=True)")
    src = sub_once(src, "                    sched.step()\n", "                    sched.step()\n                    _mark_step()\n")
    src = sub_once(src, "                    tot += float(loss.detach()) * len(idx)\n", "                    tot = tot + loss.detach() * len(idx)\n")
    src = sub_once(src, "                log.append(round(tot / max(1, n), 5))\n", "                log.append(round(float(tot) / max(1, n), 5))\n")
    src = sub_once(src, '                torch.cuda.set_rng_state_all(ck["cuda_rng"])\n',
                   '                if torch.cuda.is_available() and ck.get("cuda_rng"):\n'
                   '                    torch.cuda.set_rng_state_all(ck["cuda_rng"])\n')
    src = sub_once(src, '"cuda_rng": torch.cuda.get_rng_state_all()}', '"cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}')
    src = sub_once(src, '                    torch.save({"epoch": ep,', '                    _xsave({"epoch": ep,')
    src = sub_once(src, "        x = img[mask]                                               # only real windows go through the encoder\n",
                   "        x = img.flatten(0, 1)                                       # [S6T XLA] all B*W windows: static shapes\n")
    n = src.count("            t[mask] = f\n")
    assert n == 2, n
    src = src.replace("            t[mask] = f\n", "            t = f.reshape(t.shape)\n")
    src = sub_once(src, "    return (loss.sum(0)[ok] / den[ok]).mean()\n",
                   "    return ((loss.sum(0) / den.clamp_min(1e-12)) * ok).sum() / ok.sum().clamp_min(1)\n")
    src = sub_once(src, "    W = max(len(b[0]) for b in batch)\n",
                   "    W = max(int(os.environ.get('S6T_PAD_W', '0')), max(len(b[0]) for b in batch))\n")
    return src


def tpu_env(chip):
    """PJRT: this process sees one TPU chip only (several independent processes on one TPU VM)."""
    port = 8476 + chip
    return {"PJRT_DEVICE": "TPU", "TPU_VISIBLE_CHIPS": str(chip), "TPU_VISIBLE_DEVICES": str(chip),
            "TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1", "TPU_PROCESS_PORT": str(port),
            "TPU_PROCESS_ADDRESSES": "localhost:%d" % port, "CLOUD_TPU_TASK_ID": "0", "S6T_PAD_W": "64"}


def targets(path):
    """teach4opus soft labels as the E_prior arm's 'prior' column; gold studies and gold-report duplicates excluded."""
    code = find_mount(CODE_SLUG, "teach4opus.csv")
    t4o = pd.read_csv(code / "teach4opus.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    comp = find_mount(COMP, "train.csv")
    train = pd.read_csv(comp / "train.csv", dtype={"StudyInstanceUID": str})
    gold = set(train.loc[train[LABELS].notna().any(axis=1), "StudyInstanceUID"])
    rh = train["Report"].fillna("").map(lambda s: hashlib.sha1(s.strip().encode()).hexdigest())
    gold_hash = set(rh[train["StudyInstanceUID"].isin(gold)])
    drop = gold | set(train.loc[rh.isin(gold_hash), "StudyInstanceUID"])
    rows = []
    for u in train["StudyInstanceUID"]:
        if u in drop:
            continue
        r = {"StudyInstanceUID": u, "is_gold": "0", "review_holdout": "0", "fold": "0"}
        for L in LABELS:
            v = float(t4o.loc[u, L])
            assert 0.0 <= v <= 1.0
            r[L + "|prior"] = "%.6f" % v
            r[L + "|strict"] = "%.6f" % v
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)
    log("targets: %d studies (dropped %d gold / gold-report duplicates)" % (len(rows), len(drop)))


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    xla = os.environ.get("S6T_XLA") == "1"             # TPU VM: one seed per TPU chip (torch_xla), see patch_xla()
    trainer = TMP / "compact_train_s6t.py"
    trainer.write_text(patch_xla(patch(trainer_source())) if xla else patch(trainer_source()))
    log("patched trainer written", "(XLA/TPU)" if xla else "(CUDA)")
    tpath = TMP / "targets_s6t.csv"
    targets(tpath)
    shards = [p for p in find_dirs("meta.f32.npy") if (p / "index.json").is_file()]
    if not shards:
        raise SystemExit("no px cache shard attached")
    if DEV:
        shards = [to_jpeg(p, TMP / "devjpg" / p.name) if (p / "px.u8.npy").is_file() else p for p in shards]
    shards = [str(p) for p in shards]
    n_st = sum(len(json.load(open(Path(d) / "index.json"))["done"]) for d in shards)
    log("cache shards %d, studies %d: %s" % (len(shards), n_st, shards))
    import torch
    if xla:
        n_gpu = len(SEEDS)
        log("TPU mode: %d processes, one chip each; torch %s" % (n_gpu, torch.__version__))
    else:
        n_gpu = torch.cuda.device_count()
        log("GPUs", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
    deadline = T0 + STOP_AFTER_H * 3600
    if os.environ.get("KG_DEADLINE"):                  # kgkit gpu wrapper: graceful-stop time of the session
        deadline = min(deadline, float(os.environ["KG_DEADLINE"]))
    procs = []
    for gi, seed in enumerate(SEEDS[:max(1, n_gpu)]):
        out = WORK / ("s6t_s%d" % seed)
        out.mkdir(parents=True, exist_ok=True)
        # resume from an attached earlier run of this kernel family when one is present
        prev = [p for p in walk_inputs("E_prior_s%d_f0.pt" % seed, 8) if "/ckpt/" in str(p)]
        if prev:                                       # several attached runs: resume from the most advanced checkpoint
            ep_of = {p: int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) for p in prev}
            best = max(prev, key=lambda p: ep_of[p])
            (out / "ckpt").mkdir(exist_ok=True)
            shutil.copyfile(best, out / "ckpt" / best.name)
            log("seed %d resumes after epoch %d from %s (candidates %s)" % (seed, ep_of[best], best, {str(k): v for k, v in ep_of.items()}))
        cmd = [sys.executable, "-u", str(trainer), "--px", ",".join(shards), "--targets", str(tpath), "--out", str(out),
               "--arms", "E_prior", "--backbone", "convnext_small.fb_in22k_ft_in1k", "--init", "timm", "--seeds", "1",
               "--seed-base", str(seed), "--folds", "1", "--epochs", str(1 if DEV else EPOCHS), "--batch", "2",
               "--train-windows", "64", "--crop", "288", "--slices", "3", "--phys-mm", "3.5", "--spatial-attn", "--grad-ckpt",
               "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "6" if xla else "2", "--eval-workers", "1", "--aug", "crop",
               "--train-sampling", "uniform", "--eval-stride", "3", "--resume", "--ckpt-every", "1",
               "--i-have-user-authorisation-to-finetune"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), S6T_ALL_DATA="1", S6T_DEADLINE="%.0f" % deadline,
                   OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        if xla:
            env.update(tpu_env(gi))
            env.pop("CUDA_VISIBLE_DEVICES", None)
        lf = open(WORK / ("train_s%d.log" % seed), "w")
        procs.append((seed, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(TMP)), lf))
        log("seed %d started on GPU %d" % (seed, gi))
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(300)
        for seed, p, _ in procs:
            tail = (WORK / ("train_s%d.log" % seed)).read_text(errors="replace").strip().splitlines()[-1:]
            log("seed %d %s | %s" % (seed, "running" if p.poll() is None else "exit %s" % p.returncode, tail[0][:200] if tail else ""))
    rec = {}
    for seed, p, lf in procs:
        lf.close()
        m = WORK / ("s6t_s%d" % seed) / "models" / ("E_prior_s%d_f0.pt" % seed)
        rec[seed] = {"exit": p.returncode, "model": m.is_file(), "sha256": sha256_bytes(m.read_bytes()) if m.is_file() else None}
    json.dump(rec, open(WORK / "s6t_receipt.json", "w"), indent=1)
    log("DONE", rec)


if __name__ == "__main__":
    main()
