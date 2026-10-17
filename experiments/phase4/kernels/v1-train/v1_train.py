"""V1 training: heliosli's compact_train.py (s6fall's sha, as S6T) with the OrthoFoundation-L ViT encoder (--backbone ofl_l).

Pre-registered in docs/phase4_notes.md (2026-10-17). Everything S6T patches (fp16 + GradScaler on T4, all-data training,
multi-shard JPEG cache, epoch deadline guard; on TPU the torch_xla patch) is reused from ../s6t-train/s6t_train.py.
Only what ofl_l forces differs: the encoder starts from the public OrthoFoundation-L checkpoint (official sha256, converted
to timm's layout exactly as heliosli's orthofoundation.convert_state does), no spatial attention, no gradient checkpointing,
32 train windows.

env: V1_SEEDS (GPU "7,8", TPU "9,10"), V1_EPOCHS (8), V1_DEV (n > 0: dev run, 1 epoch on n studies), S6T_XLA=1 for TPU,
     V1_STOP_AFTER_H (11.4 on GPU; set 8.3 on TPU), V1_INIT ("" = random encoder, local smoke tests only).
"""
import hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s6t_train", Path(__file__).resolve().parents[1] / "s6t-train" / "s6t_train.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

EPOCHS = int(os.environ.get("V1_EPOCHS", "8"))
SEEDS = tuple(int(s) for s in os.environ.get("V1_SEEDS", "7,8").split(","))
DEV = int(os.environ.get("V1_DEV", "0"))
STOP_AFTER_H = float(os.environ.get("V1_STOP_AFTER_H", "11.4"))
CROP, WINDOWS, FREEZE = 288, 32, 16
OFL_SLUG, OFL_FILE = "orthofoundation-l-dinov3", "OrthoFoudation-L.pth"      # pritom2357's copy of the official release (MIT)
OFL_SHA = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"   # official Git-LFS sha256 (ytrsk/OrthoFoundation)
WORK, TMP, log = S.WORK, Path("/tmp/v1"), S.log


def convert_state(source):
    """Official OrthoFoundation-L (DINOv3 layout) -> timm vit_large_patch16_dinov3_qkvb, as heliosli's
    training/orthofoundation.py: masked QKV bias split into q/v bias (the key bias must be zero), storage tokens ->
    reg_token, ls*.gamma -> gamma_*, mask token / RoPE periods / bias masks dropped."""
    import torch
    if not all(k.startswith("backbone.") for k in source):
        raise SystemExit("unexpected OFL-L checkpoint namespace")
    raw = {k[len("backbone."):]: v for k, v in source.items()}
    masks = [v for k, v in raw.items() if k.endswith(".bias_mask")]
    if len(masks) != 24 or any(torch.count_nonzero(v) for v in masks):
        raise SystemExit("OFL-L attention-bias policy differs from the verified all-disabled one")
    out = {}
    for key, value in raw.items():
        if key in ("mask_token", "rope_embed.periods") or key.endswith(".bias_mask"):
            continue
        if key.endswith(".attn.qkv.bias"):
            q, k, v = (value * raw[key.replace(".bias", ".bias_mask")]).chunk(3)
            if torch.count_nonzero(k):
                raise SystemExit("nonzero key bias")
            out[key.replace("qkv.bias", "q_bias")] = q.contiguous()
            out[key.replace("qkv.bias", "v_bias")] = v.contiguous()
            continue
        out[key.replace("storage_tokens", "reg_token").replace("ls1.gamma", "gamma_1").replace("ls2.gamma", "gamma_2")] = value.contiguous()
    return out


def ofl_init():
    """The public OFL-L encoder in timm's layout, written as a torch file (OFLEncoder torch.loads any non-.safetensors init)."""
    if "V1_INIT" in os.environ:
        return os.environ["V1_INIT"]
    import torch
    src = [p for p in S.walk_inputs(OFL_FILE) if OFL_SLUG in str(p)]
    if not src:
        raise SystemExit("cannot find %s/%s under /kaggle/input" % (OFL_SLUG, OFL_FILE))
    h = hashlib.sha256()
    with open(src[0], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != OFL_SHA:
        raise SystemExit("OFL-L checkpoint differs from the official release: %s" % h.hexdigest())
    sd = convert_state(torch.load(src[0], map_location="cpu", weights_only=True))
    dst = TMP / "ofl_l_timm.pt"
    torch.save(sd, dst)
    log("OFL-L encoder: %s (official sha256 %s..) -> %s, %d tensors" % (src[0], OFL_SHA[:8], dst, len(sd)))
    return str(dst)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    xla = os.environ.get("S6T_XLA") == "1"
    trainer = TMP / "compact_train_v1.py"
    src = S.patch(S.trainer_source())
    trainer.write_text(S.patch_xla(src) if xla else src)
    log("patched trainer written", "(XLA/TPU)" if xla else "(CUDA)")
    tpath = TMP / "targets_v1.csv"
    S.targets(tpath)
    shards = [p for p in S.find_dirs("meta.f32.npy") if (p / "index.json").is_file()]
    if not shards:
        raise SystemExit("no px cache shard attached")
    if DEV:                                            # dev: the first DEV studies the cache holds
        import pandas as pd
        have = set()
        for d in shards:
            have |= set(json.load(open(d / "index.json"))["done"])
        t = pd.read_csv(tpath, dtype={"StudyInstanceUID": str})
        t[t["StudyInstanceUID"].isin(have)].head(DEV).to_csv(tpath, index=False)
        log("dev run: %d studies, 1 epoch" % DEV)
    shards = [str(p) for p in shards]
    init = ofl_init()
    import torch
    n_dev = len(SEEDS) if xla else torch.cuda.device_count()
    log("TPU mode" if xla else "GPUs %s" % [torch.cuda.get_device_name(i) for i in range(n_dev)], "seeds", SEEDS[:max(1, n_dev)])
    deadline = S.T0 + STOP_AFTER_H * 3600
    if os.environ.get("KG_DEADLINE"):
        deadline = min(deadline, float(os.environ["KG_DEADLINE"]))
    procs = []
    for gi, seed in enumerate(SEEDS[:max(1, n_dev)]):
        out = WORK / ("v1_s%d" % seed)
        out.mkdir(parents=True, exist_ok=True)
        prev = [p for p in S.walk_inputs("E_prior_s%d_f0.pt" % seed, 8) if "/ckpt/" in str(p) and "/v1_s%d/" % seed in str(p)]
        if prev:                                       # resume from the most advanced attached V1 checkpoint of this seed
            ep_of = {p: int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) for p in prev}
            best = max(prev, key=lambda p: ep_of[p])
            (out / "ckpt").mkdir(exist_ok=True)
            shutil.copyfile(best, out / "ckpt" / best.name)
            log("seed %d resumes after epoch %d from %s" % (seed, ep_of[best], best))
        cmd = [sys.executable, "-u", str(trainer), "--px", ",".join(shards), "--targets", str(tpath), "--out", str(out),
               "--arms", "E_prior", "--backbone", "ofl_l", "--init", init, "--freeze-blocks", str(FREEZE),
               "--seeds", "1", "--seed-base", str(seed), "--folds", "1", "--epochs", str(1 if DEV else EPOCHS),
               "--batch", "2", "--train-windows", str(WINDOWS), "--crop", str(CROP), "--slices", "3", "--phys-mm", "3.5",
               "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "6" if xla else "2", "--eval-workers", "1",
               "--aug", "crop", "--train-sampling", "uniform", "--eval-stride", "3", "--resume", "--ckpt-every", "1",
               "--i-have-user-authorisation-to-finetune"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), S6T_ALL_DATA="1", S6T_DEADLINE="%.0f" % deadline,
                   OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        if xla:
            env.update(S.tpu_env(gi))
            env["S6T_PAD_W"] = str(WINDOWS)
            env.pop("CUDA_VISIBLE_DEVICES", None)
        lf = open(WORK / ("train_v1_s%d.log" % seed), "w")
        procs.append((seed, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(TMP)), lf))
        log("seed %d started on %s %d" % (seed, "TPU chip" if xla else "GPU", gi))
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(60 if DEV else 300)
        for seed, p, _ in procs:
            tail = (WORK / ("train_v1_s%d.log" % seed)).read_text(errors="replace").strip().splitlines()[-1:]
            log("seed %d %s | %s" % (seed, "running" if p.poll() is None else "exit %s" % p.returncode, tail[0][:200] if tail else ""))
    rec = {}
    for seed, p, lf in procs:
        lf.close()
        m = WORK / ("v1_s%d" % seed) / "models" / ("E_prior_s%d_f0.pt" % seed)
        rec[seed] = {"exit": p.returncode, "model": m.is_file(), "sha256": S.sha256_bytes(m.read_bytes()) if m.is_file() else None}
    json.dump(rec, open(WORK / "v1_receipt.json", "w"), indent=1)
    log("DONE", rec)


if __name__ == "__main__":
    main()
