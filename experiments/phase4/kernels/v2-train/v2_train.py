"""V2 training: heliosli's compact_train.py (s6fall's sha, as S6T) with a fully fine-tuned DINOv2 ViT-S encoder.

Pre-registered in docs/phase4_notes.md (2026-10-17). Everything S6T patches (fp16 + GradScaler on T4, all-data training,
multi-shard JPEG cache, epoch deadline guard; on TPU the torch_xla patch) is reused from ../s6t-train/s6t_train.py.
Only what the ViT forces differs from S6T: backbone vit_small_patch14_dinov2.lvd142m (public timm weights), crop 280
(patch 14 x 20), layer-wise LR decay 0.8.

env: V2_SEEDS (GPU "11,12", TPU "13,14"), V2_EPOCHS (12), V2_DEV (n > 0: dev run, 1 epoch on n studies), S6T_XLA=1 for TPU,
     V2_STOP_AFTER_H (11.4 on GPU; set 8.3 on TPU), V2_INIT ("" = random encoder, local smoke tests only; default timm).
"""
import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s6t_train", Path(__file__).resolve().parents[1] / "s6t-train" / "s6t_train.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

EPOCHS = int(os.environ.get("V2_EPOCHS", "12"))
SEEDS = tuple(int(s) for s in os.environ.get("V2_SEEDS", "11,12").split(","))
DEV = int(os.environ.get("V2_DEV", "0"))
STOP_AFTER_H = float(os.environ.get("V2_STOP_AFTER_H", "11.4"))
BACKBONE, INIT = "vit_small_patch14_dinov2.lvd142m", os.environ.get("V2_INIT", "timm")
CROP, WINDOWS, LLRD = int(os.environ.get("V2_CROP", "280")), int(os.environ.get("V2_WINDOWS", "64")), "0.8"
WORK, TMP, log = S.WORK, Path("/tmp/v2"), S.log


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    xla = os.environ.get("S6T_XLA") == "1"
    trainer = TMP / "compact_train_v2.py"
    src = S.patch(S.trainer_source())
    trainer.write_text(S.patch_xla(src) if xla else src)
    log("patched trainer written", "(XLA/TPU)" if xla else "(CUDA)")
    tpath = TMP / "targets_v2.csv"
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
    import torch
    n_dev = len(SEEDS) if xla else torch.cuda.device_count()
    log("TPU mode" if xla else "GPUs %s" % [torch.cuda.get_device_name(i) for i in range(n_dev)], "seeds", SEEDS[:max(1, n_dev)],
        "| backbone", BACKBONE, "init", INIT or "random", "crop", CROP, "windows", WINDOWS, "epochs", 1 if DEV else EPOCHS)
    deadline = S.T0 + STOP_AFTER_H * 3600
    if os.environ.get("KG_DEADLINE"):
        deadline = min(deadline, float(os.environ["KG_DEADLINE"]))
    procs = []
    for gi, seed in enumerate(SEEDS[:max(1, n_dev)]):
        out = WORK / ("v2_s%d" % seed)
        out.mkdir(parents=True, exist_ok=True)
        prev = [p for p in S.walk_inputs("E_prior_s%d_f0.pt" % seed, 8) if "/ckpt/" in str(p) and "/v2_s%d/" % seed in str(p)]
        if prev:                                       # resume from the most advanced attached V2 checkpoint of this seed
            ep_of = {p: int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) for p in prev}
            best = max(prev, key=lambda p: ep_of[p])
            (out / "ckpt").mkdir(exist_ok=True)
            shutil.copyfile(best, out / "ckpt" / best.name)
            log("seed %d resumes after epoch %d from %s" % (seed, ep_of[best], best))
        cmd = [sys.executable, "-u", str(trainer), "--px", ",".join(shards), "--targets", str(tpath), "--out", str(out),
               "--arms", "E_prior", "--backbone", BACKBONE, "--init", INIT, "--llrd", LLRD,
               "--seeds", "1", "--seed-base", str(seed), "--folds", "1", "--epochs", str(1 if DEV else EPOCHS),
               "--batch", "2", "--train-windows", str(WINDOWS), "--crop", str(CROP), "--slices", "3", "--phys-mm", "3.5",
               "--spatial-attn", "--grad-ckpt",
               "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "6" if xla else "2", "--eval-workers", "1",
               "--aug", "crop", "--train-sampling", "uniform", "--eval-stride", "3", "--resume", "--ckpt-every", "1",
               "--i-have-user-authorisation-to-finetune"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), S6T_ALL_DATA="1", S6T_DEADLINE="%.0f" % deadline,
                   OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        if xla:
            env.update(S.tpu_env(gi))
            env["S6T_PAD_W"] = str(WINDOWS)
            env.pop("CUDA_VISIBLE_DEVICES", None)
        lf = open(WORK / ("train_v2_s%d.log" % seed), "w")
        procs.append((seed, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(TMP)), lf))
        log("seed %d started on %s %d" % (seed, "TPU chip" if xla else "GPU", gi))
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(60 if DEV else 300)
        for seed, p, _ in procs:
            tail = (WORK / ("train_v2_s%d.log" % seed)).read_text(errors="replace").strip().splitlines()[-1:]
            log("seed %d %s | %s" % (seed, "running" if p.poll() is None else "exit %s" % p.returncode, tail[0][:200] if tail else ""))
    rec = {}
    for seed, p, lf in procs:
        lf.close()
        m = WORK / ("v2_s%d" % seed) / "models" / ("E_prior_s%d_f0.pt" % seed)
        rec[seed] = {"exit": p.returncode, "model": m.is_file(), "sha256": S.sha256_bytes(m.read_bytes()) if m.is_file() else None}
    json.dump(rec, open(WORK / "v2_receipt.json", "w"), indent=1)
    log("DONE", rec)


if __name__ == "__main__":
    main()
