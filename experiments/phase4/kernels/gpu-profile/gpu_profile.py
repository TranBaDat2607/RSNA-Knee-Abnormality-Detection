"""Why is the S6T recipe ~4.9x slower per step on a T4 than on one TPU v5e chip?  Measures each part on 2xT4 (diagnostic,
no model is trained or kept):

  env      CPUs (physical / logical), RAM, GPU clocks and power limit, torch / cuDNN
  decode   one core: ms per JPEG row of the S6T cache (a step reads 2 studies x 64 windows x 3 slices = 384 rows)
  loader   the real DataLoader (Studies + collate, batch 2, 64 windows), no GPU: 2 workers alone, 2 workers x 2 processes
           (as in s6t-train: one per T4), 4 workers alone
  compute  synthetic batches already on the GPU, the real model and step (ConvNeXt-small, spatial attention, fp16 autocast +
           GradScaler, channels_last, gradient checkpointing, AdamW, clip 5): alone on one T4, then on both T4s at once;
           variants: forward only, no gradient checkpointing, torch.compile
  full     the real loop (loader with 2 workers + step) on both T4s at once, as in s6t-train: time waiting for data vs compute

nvidia-smi is sampled every second (SM clock, power, temperature, utilisation, throttle reasons) and /proc/stat gives CPU use.
Results: /kaggle/working/gpu_profile.json (+ the log).  GPROF_SMOKE=1: tiny CPU run for a local check.
"""
import importlib.util, json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
SMOKE = os.environ.get("GPROF_SMOKE") == "1"
OUT = Path(os.environ.get("GPROF_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("GPROF_TMP", "/tmp/gprof"))
WIN, BATCH, SIZE, CROP = (4, 2, 72, 64) if SMOKE else (64, 2, 320, 288)
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def load_s6t():
    spec = importlib.util.spec_from_file_location("s6t_train", HERE.parents[1] / "s6t-train" / "s6t_train.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def trainer():
    """The CUDA-patched trainer s6t-train ran (same sha check), imported as a module for its Net / Studies / collate."""
    path = TMP / "ct.py"
    if not path.is_file():
        S = load_s6t()
        src = Path(os.environ["GPROF_TRAINER"]).read_text() if os.environ.get("GPROF_TRAINER") else S.trainer_source()
        TMP.mkdir(parents=True, exist_ok=True)
        path.write_text(S.patch(src))
    spec = importlib.util.spec_from_file_location("ct", path)
    ct = importlib.util.module_from_spec(spec)
    sys.modules["ct"] = ct
    spec.loader.exec_module(ct)
    return ct


def shards():
    if os.environ.get("GPROF_SHARDS"):
        return os.environ["GPROF_SHARDS"].split(",")
    S = load_s6t()
    return [str(p) for p in S.find_dirs("meta.f32.npy") if (p / "index.json").is_file()]


def uids_of(sh):
    done = {}
    for d in sh:
        done.update(json.load(open(Path(d) / "index.json"))["done"])
    return sorted(u for u, v in done.items() if v[1] > 0)


def make_loader(ct, sh, workers, seed=0):
    import torch
    ds = ct.Studies(",".join(sh), uids_of(sh), WIN, "train", seed, eval_stride=3, train_sampling="uniform", cover_stride=3,
                    slices=3, phys_mm=3.5)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=workers, collate_fn=ct.collate,
                                       drop_last=True, persistent_workers=False, pin_memory=not SMOKE, prefetch_factor=2 if workers else None,
                                       generator=torch.Generator().manual_seed(seed))


def cpu_times():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]                       # total, idle + iowait


def cpu_busy(a, b):
    return round(100.0 * (1 - (b[1] - a[1]) / max(1, b[0] - a[0])), 1)


# ---------------------------------------------------------------- worker modes (run in subprocesses, one GPU each)

def mode_loader(workers, steps):
    ct = trainer()
    dl = make_loader(ct, shards(), workers)
    it = iter(dl)
    for _ in range(3):
        next(it)
    c0, t = cpu_times(), time.time()
    nwin = 0
    for _ in range(steps):
        img, meta, mask, idx = next(it)
        nwin += int(mask.sum())
    dt = time.time() - t
    return {"s_per_batch": round(dt / steps, 3), "windows_per_batch": round(nwin / steps, 1), "cpu_busy_pct": cpu_busy(c0, cpu_times())}


def build(ct, ckpt=True, compile_=False):
    import torch
    dev = torch.device("cpu" if SMOKE else "cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    model = ct.Net("convnext_small.fb_in22k_ft_in1k", None, crop=CROP, in_chans=3, spatial=True).to(dev).to(memory_format=torch.channels_last)
    if ckpt:
        model.enc.set_grad_checkpointing(True)
    if compile_:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(32, _dynamo.config.cache_size_limit)
        model.enc.compile(dynamic=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=not SMOKE)
    return model, opt, scaler, dev


def step_fn(model, opt, scaler, dev, train=True):
    import torch
    import torch.nn.functional as F
    gen = torch.Generator().manual_seed(1)
    amp = torch.autocast("cpu", dtype=torch.bfloat16) if SMOKE else torch.autocast("cuda", dtype=torch.float16)

    def crop(img):
        oy = int(torch.randint(0, SIZE - CROP + 1, (1,), generator=gen))
        ox = int(torch.randint(0, SIZE - CROP + 1, (1,), generator=gen))
        return img[..., oy:oy + CROP, ox:ox + CROP]

    def step(img, meta, mask):
        y = torch.rand((img.shape[0], 12), device=dev)
        if not train:
            with torch.no_grad(), amp:
                z = model(crop(img), meta, mask, return_aux=True)[0]
            return float(z.float().sum())
        with amp:
            z = model(crop(img), meta, mask, return_aux=True)[0]
        loss = F.binary_cross_entropy_with_logits(z.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt)
        scaler.update()
        return float(loss.detach())                  # one host sync per step, as in compact_train
    return step


def synthetic(dev):
    import torch
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (BATCH, WIN, 3, SIZE, SIZE), dtype=torch.uint8, generator=g).to(dev)
    meta = torch.rand((BATCH, WIN, 6), generator=g).to(dev)
    meta[..., 0] = 0                                 # slot ids / view ids are small integers in the real cache
    meta[..., 1] = 0
    return img, meta, torch.ones((BATCH, WIN), dtype=torch.bool, device=dev)


def mode_compute(variant, steps):
    import torch
    ct = trainer()
    t_build = time.time()
    model, opt, scaler, dev = build(ct, ckpt=variant != "no_ckpt", compile_=variant == "compile")
    model.train()
    step = step_fn(model, opt, scaler, dev, train=variant != "fwd_only")
    img, meta, mask = synthetic(dev)
    res = {}
    try:
        t = time.time()
        for _ in range(3):
            step(img, meta, mask)
        res["warmup_s"] = round(time.time() - t, 1)
        if not SMOKE:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(steps):
            step(img, meta, mask)
        if not SMOKE:
            torch.cuda.synchronize()
        dt = (time.time() - t) / steps
        res.update(s_per_step=round(dt, 3), images_per_s=round(BATCH * WIN / dt, 1),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if not SMOKE else None,
                   build_s=round(t - t_build - res["warmup_s"], 1))
    except torch.OutOfMemoryError as e:
        res["error"] = "OOM: " + str(e)[:200]
    return res


def mode_full(workers, steps):
    import torch
    ct = trainer()
    model, opt, scaler, dev = build(ct)
    model.train()
    step = step_fn(model, opt, scaler, dev)
    it = iter(make_loader(ct, shards(), workers, seed=int(os.environ.get("GPROF_SEED", "0"))))
    for _ in range(3):
        img, meta, mask, _ = next(it)
        step(img.to(dev, non_blocking=True), meta.to(dev), mask.to(dev))
    wait = comp = 0.0
    nwin = 0
    c0, t_all = cpu_times(), time.time()
    for _ in range(steps):
        t = time.time()
        img, meta, mask, _ = next(it)
        t1 = time.time()
        step(img.to(dev, non_blocking=True), meta.to(dev), mask.to(dev))   # ends in a host sync (float(loss))
        t2 = time.time()
        wait += t1 - t
        comp += t2 - t1
        nwin += int(mask.sum())
    tot = time.time() - t_all
    return {"s_per_step": round(tot / steps, 3), "data_wait_s": round(wait / steps, 3), "compute_s": round(comp / steps, 3),
            "windows_per_step": round(nwin / steps, 1), "cpu_busy_pct": cpu_busy(c0, cpu_times())}


# ---------------------------------------------------------------- orchestration

class Smi:
    """nvidia-smi sampled every second in the background; summary per phase."""

    def __init__(self):
        self.rows, self.on = [], not SMOKE
        if not self.on:
            return
        q = "clocks_event_reasons.active"
        if subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv"], capture_output=True).returncode:
            q = "clocks_throttle_reasons.active"
        self.fields = ["index", "clocks.sm", "clocks.max.sm", "power.draw", "power.limit", "temperature.gpu", "utilization.gpu", q]
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=" + ",".join(self.fields), "--format=csv,noheader,nounits", "-lms", "1000"],
                                  stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.rows.append((time.time(), [x.strip() for x in line.split(",")]))

    def summary(self, t0, t1):
        if not self.on:
            return {}
        out = {}
        for gpu in ("0", "1"):
            r = [v for t, v in self.rows if t0 <= t <= t1 and v[0] == gpu]
            if not r:
                continue
            num = lambda j: [float(v[j]) for v in r if v[j].replace(".", "", 1).isdigit()]
            reasons = {}
            for v in r:
                reasons[v[7]] = reasons.get(v[7], 0) + 1
            out["gpu" + gpu] = {"sm_mhz_mean": round(np.mean(num(1)), 0) if num(1) else None, "sm_mhz_max": max(num(2)) if num(2) else None,
                                "power_w_mean": round(np.mean(num(3)), 1) if num(3) else None, "power_limit_w": max(num(4)) if num(4) else None,
                                "temp_c_max": max(num(5)) if num(5) else None, "util_pct_mean": round(np.mean(num(6)), 1) if num(6) else None,
                                "throttle_reasons": reasons, "samples": len(r)}
        return out


def run_parallel(jobs, timeout):
    """jobs: [(gpu, argv)] started together; returns their JSON results."""
    procs = []
    for gpu, argv in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", GPROF_SEED=str(gpu))
        procs.append(subprocess.Popen([sys.executable, "-u", str(HERE)] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, env=env))
    out = []
    for p in procs:
        try:
            so, se = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            so, se = p.communicate()
            out.append({"error": "timeout %ds" % timeout})
            continue
        res = [l for l in so.splitlines() if l.startswith("GPROF_RESULT ")]
        out.append(json.loads(res[-1][13:]) if res else {"error": "exit %s: %s" % (p.returncode, se.strip()[-600:])})
    return out


def env_info():
    import torch
    info = {"cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}
    try:
        lc = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        info["lscpu"] = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in lc.splitlines() if ":" in l)
                         if k.strip() in ("Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "CPU MHz", "CPU max MHz")}
    except OSError:
        pass
    mem = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo") if l.startswith(("MemTotal", "MemAvailable"))}
    info["mem"] = mem
    info["torch"], info["cuda"] = torch.__version__, torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    if not SMOKE:
        info["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit,memory.total",
                                       "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    return info


def mode_decode(n):
    ct = trainer()
    sh = shards()
    px = ct._px_open(sh[0])
    rng = np.random.default_rng(0)
    rows = rng.integers(0, px.shape[0], n)
    t = time.time()
    for r in rows:
        px[[int(r)]]
    ms = (time.time() - t) / n * 1000
    jpeg_kb = float(np.diff(px.blob).mean()) / 1024 if hasattr(px, "blob") else None
    return {"ms_per_row": round(ms, 3), "rows_per_step": BATCH * WIN * 3, "s_per_step_one_core": round(ms * BATCH * WIN * 3 / 1000, 3),
            "jpeg_kb_mean": round(jpeg_kb, 1) if jpeg_kb else None, "size": px.shape[-1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trainer()                                        # writes the patched trainer once (sha checked)
    R = {"config": {"batch": BATCH, "windows": WIN, "cache_size": SIZE, "crop": CROP, "smoke": SMOKE}}
    smi = Smi()
    R["env"] = env_info()
    log("env", json.dumps(R["env"]))
    n_gpu = 1 if SMOKE else 2
    st = 3 if SMOKE else 25
    lst = 4 if SMOKE else 40
    T = 600 if SMOKE else 1800

    def phase(name, jobs, timeout=T):
        t0 = time.time()
        res = run_parallel(jobs, timeout)
        R[name] = {"results": res, "wall_s": round(time.time() - t0, 1), "smi": smi.summary(t0, time.time())}
        log(name, json.dumps(R[name]))
        json.dump(R, open(OUT / "gpu_profile.json", "w"), indent=1)

    R["decode"] = mode_decode(50 if SMOKE else 1500)
    log("decode", json.dumps(R["decode"]))
    phase("loader_2w_alone", [(0, ["loader", "2", str(lst)])])
    phase("loader_2w_x2", [(g, ["loader", "2", str(lst)]) for g in range(n_gpu)])
    phase("loader_4w_alone", [(0, ["loader", "4", str(lst)])])
    phase("compute_real_alone", [(0, ["compute", "real", str(st)])])
    phase("compute_real_x2", [(g, ["compute", "real", str(st)]) for g in range(n_gpu)])
    phase("compute_fwd_only", [(0, ["compute", "fwd_only", str(st)])])
    phase("compute_no_ckpt", [(0, ["compute", "no_ckpt", str(st)])])
    phase("full_2w_x2", [(g, ["full", "2", str(lst)]) for g in range(n_gpu)])
    phase("compute_compile", [(0, ["compute", "compile", str(st)])], timeout=900 if not SMOKE else 600)
    R["tpu_reference"] = {"s_per_step": 0.43, "source": "tranbadat/s6t-tpu-train (v5e, 1 chip per seed, bf16, padded to 64 windows)"}
    json.dump(R, open(OUT / "gpu_profile.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        m, a1, a2 = sys.argv[1], sys.argv[2], int(sys.argv[3])
        r = {"loader": lambda: mode_loader(int(a1), a2), "compute": lambda: mode_compute(a1, a2),
             "full": lambda: mode_full(int(a1), a2)}[m]()
        print("GPROF_RESULT " + json.dumps(r), flush=True)
    else:
        main()
