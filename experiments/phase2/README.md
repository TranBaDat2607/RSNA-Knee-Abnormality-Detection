# Phase 2 experiments — Kaggle launchers and analysis scripts

Results and decisions are written up in [`docs/phase2_notes.md`](../../docs/phase2_notes.md); this folder keeps
what is needed to rerun them. Large outputs (checkpoints, `oof.csv`, `gold.npy`, caches) are not in git — they
are the outputs of the Kaggle kernels named below (`kaggle kernels output tranbadat/<kernel> -p <dir>`).

Each `kernels/<name>/` folder is a Kaggle script kernel: `kernel-metadata.json` + the script, pushed with
`kaggle kernels push -p kernels/<name>`. They all import the package from the private dataset
`tranbadat/rsna-knee-code` (a zip of `src/rsna_knee` plus the label tables `teach4.csv`, `targets_r1/r2/r4.csv`).

| Folder | Kaggle kernel | What | Result |
|---|---|---|---|
| `cache-wide256` | `rsna-knee-cache-wide256` (CPU) | `WIDE44_256` training cache from DICOM | 4,407 studies, 0 failures, 46 min |
| `gpu-ab` | `rsna-knee-gpu-ab` | fold-0 A/B: stack vs slot-aware windows, wide span, 336 px | slot neutral, wide −0.0055, 336 +0.0025 |
| `gpu-ab2` | `rsna-knee-gpu-ab2` | ConvNeXt-tiny (CoAtNet-1 arm OOM'd) | tiny 0.8711 < nano 0.8755 |
| `gpu-ab3` | `rsna-knee-gpu-ab3` | CoAtNet-RMLP-1 224 (grad-ckpt) + ResNet-50 224 | CoAtNet = nano, ρ 0.97; ResNet-50 −0.024 |
| `gpu-ab4` | `rsna-knee-gpu-ab4` | 40 training windows vs 20 epochs | k40 worse; 20 epochs +0.0045 |
| `r3a/b/c` | `rsna-knee-r3a/b/c` | R3 production: nano 336 px, 18 ep, `targets_r2`, 5 folds | OOF 0.880, gold 0.898, **LB 0.920** |
| `ours-solo-r3` | `rsna-knee-ours-solo` v2 | R3 alone on the test set | submission 56787175: 0.920 |
| `ours-blend-r3` | `rsna-knee-ours-blend` v2 | jiweiliu 0.943 public notebook + R3 at w = 0.10 | submission 56787692: 0.942 |
| `r4a/b/c` | `rsna-knee-r4a/b/c` | R4: R3 recipe on heavy-OOF `targets_r4` | running |
| `ours-solo-r4` | `rsna-knee-ours-solo` (next version) | R4 alone on the test set | pending |
| `r5ta/b/c` | `rsna-knee-r5ta/b/c` | R5t control: R3 recipe on `teach4.csv` only | running |
| `tpu-xla-ab` | (not run) | the same A/B on TPU v5e-8 — the TPU queue was hours long, GPU was used instead | — |

`analysis/` (run locally; paths point at the session scratchpad where kernel outputs were downloaded):

- `analyze_run.py <run_dir> <tag>` — per-run gold-58 (fold-mean) and OOF vs teach4.
- `ab_table.py <dir>...` — per-finding table and paired bootstrap of fold-0 A/B arms vs `gab_slot`.
- `build_r3.py`, `build_r4.py [out.csv]` — soft targets = W_TEXT·teach4 + image OOF mix (quantile-mapped).
- `search.py`, `score_of.py`, `topic.py` — kagglesdk search (notebook public scores, discussion threads).
