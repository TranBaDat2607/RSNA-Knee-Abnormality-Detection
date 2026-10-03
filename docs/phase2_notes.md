# Phase 2 notes — pushing past 0.943 toward 0.95 (2026-09-29)

Goal: public LB ≥ 0.95 (current best of this repo: **0.939**, submission `56184308`). Kaggle GPU only
(6 h/week, resets Saturdays 00:00 UTC; ~4 h of the week ending 2026-10-03 already used). Deadline 2026-10-22.

## 1. What the research found

**Leaderboard (2026-09-28):** #1 0.960; ~20 teams ≥ 0.955; 56 teams ≥ 0.95 on the Efficiency LB, whose
#1 (Scott Willis, 0.958) is a fast, small model → 0.95+ does not need a huge ensemble.

**Public notebooks** (`best_public_score` via the kagglesdk search API — the CLI can't show scores):
- Best *public-only* stacks: **0.943** (jiweiliu `rsna-knee-fast-2xt4-inference`, mattiaangeli
  "Speedy Raptors" v34, evgendvorkin `rsna-versia-5`, ...). Recipe = 20 DINOv2 + A5 + RadImageNet +
  4 Raptor CoAtNet views + 4-member CoAt family (resgated, D4, Global96, Repair-v1).
- Best overall public: **0.945** (`pjmathematician/rsna-knee-d4-lite`, `aastikrajan15/knee-s75-w50`) =
  the 0.943 stack rank-blended at 0.45–0.5 with the author's **own private model** (aastikrajan: a
  ConvNeXt-tiny 320×24 fleet that scores 0.927 alone). Titles like "0.957" / "speedy947" are not LB scores.
- ~200 public forks sit at 0.941 by weight tweaks only; Mattia Angeli / DreadDevelopment warn that
  blend-weight tweaking overfits the public split.

**Discussions** (read via `discussion_api_client.get_topic`/`list_comments`; saved in the session scratchpad):
- *Best single-model score* (topic 735304): single small models at 0.94–0.954 (5-fold ResNet 224 px
  0.954; single-fold CoAtNet 224 px 0.950; Qwen-3.5-2B VLM 0.950; small ResNet 0.949 in 5 min).
  The lever everyone names is **labels**: soft labels, several label sources, and mixing
  **out-of-fold** model predictions into the labels (pseudo-labels; in-fold ones just parrot).
  Resolution 224–288 is enough; bigger backbones add little.
- *"Not addressed" is a label too* (733932): report silence means "absent" for some findings
  (Baker's) but "unknown" for others (Synovitis 34% positive when silent).

## 2. Label work (local, no GPU)

- Scored ~20 public label tables on the 58 gold studies **with a leak check**: 9 tables contain the
  gold labels verbatim (mohammadsaidulislam, noisyislands, shingo257 ×3, tonylin1026, yehezkielhaganta,
  riadmohamed42 gold74) and one is gold-smoothed (narisettichaitanya, 0.996) — all excluded.
- Clean best: riadmohamed42 HYBRID 0.905, flight0234 0.899, nartaa v0 0.893, stevenleehans v4 0.893
  (tsuyu122 0.912 mixes MRNet/OAI with undocumented construction — not used).
- `teach4.csv` = mean of those four (0.897, text only; used as the validation reference).
- `targets_r1.csv` = 0.75·teach4 + 0.25·mean(our Aug DINOv3/RadImageNet OOF) → **0.9145** gold.
- `targets_r2.csv` = 0.35·teach4 + 0.15·old OOF + 0.5·R1 OOF (ConvNeXt where available, else ResNet),
  each OOF quantile-mapped onto teach4's per-finding distribution → **0.923** gold (raw mix 0.926).

All tables are in the code dataset `tranbadat/rsna-knee-code` (and the session scratchpad).

## 3. Code added (`src/rsna_knee/mil/`, 89 tests passing)

| Module | Purpose |
|---|---|
| `cache.py` | 256 px copy of the public 44×336 corpus (`shrink_volume` = the only resize, used at train and test) |
| `kfold.py` | 5-fold training grouped by report hash, GPU affine aug, EMA, fp16/channels-last; writes `oof.csv`, `gold.npy`, `model.pt` per fold |
| `fleet.py` | test inference: process-pool DICOM decode with the exact corpus recipe → shrink → all checkpoints (parity test vs training predictions) |
| `ours.py` | submission leg: our checkpoints → rank blend with a public `submission.csv` at fixed weight; any failure keeps the public file |
| `scripts/kaggle_kfold_train.py` | Kaggle launcher (two single-GPU jobs side by side) |

## 4. Kaggle runs

| Kernel | What | Result |
|---|---|---|
| `tranbadat/rsna-knee-cache256` (CPU) | build the 256 px cache | done, 615 s, parity-verified |
| `tranbadat/rsna-knee-kfold-r1` (2×T4, ~3.7 h) | R1: resnet34.a1_in1k + convnext_nano.in12k_ft_in1k, 12 epochs, `targets_r1` | see below |
| `tranbadat/rsna-knee-ours-blend` (2×T4) | jiweiliu 0.943 notebook unchanged + our leg (ConvNeXt, w = 0.35) | dry run OK: 3 checkpoints, 0 failures, blended `submission.csv` written. **Submitted 2026-09-29 as `56654316`, score pending** (grading takes ~2.5 h). |

R1 results (gold-58 = fold-mean predictions; OOF scored against text-only teach4):

| Model | Folds | Speed (T4) | OOF vs teach4 | Gold-58 |
|---|---|---|---|---|
| resnet34 | 5/5 (31 min/fold) | ~260 img/s | 0.842 | 0.871 |
| convnext_nano | 3/5 (55 min/fold; time cap) | ~175 img/s | 0.866 | **0.905** |
| rank blend of both | — | — | 0.862 | 0.893 (ResNet drags it down) |

Both were still improving at the last epoch (best epoch = 11 of 12 in every fold) → undertrained.
For reference, public CoAtNet checkpoints score 0.912–0.920 on gold-58.

## 5. Next steps

1. **Read the LB score of submission `56654316`** (`rsna-knee-ours-blend` v1; check with
   `kaggle competitions submissions rsna-knee-abnormality-detection`) — first LB read of whether our
   leg adds to the 0.943 base. Decision rule (ledger): keep if LB ≥ 0.945; halve the weight if < 0.943.
2. **R2 (after the 2026-10-03 quota reset):** convnext_nano on `targets_r2.csv`, ~14–16 epochs, all 5
   folds split across both T4s (`kernel tranbadat/rsna-knee-kfold-r2`, staged in the scratchpad;
   needs `targets_r2.csv` in the code dataset). Don't retrain R1 models — reuse them.
3. Then update the submission leg to the R2 folds and re-submit; consider a second pseudo-label round
   if R2's gold-58 rises.

## 6. 2026-10-01 update

**LB reads.** `56654316` (0.943 public stack + R1 ConvNeXt at w = 0.35) scored **0.938**, below the
base (decision rule: our leg out at this strength). Diagnostic: the R1 ConvNeXt-nano 3-fold leg
**alone** (`tranbadat/rsna-knee-ours-solo` v1) scored **0.917** — the pipeline transfers
(gold-58 0.905 → LB 0.917), the model is just too weak to add to a 0.943 stack.

**Public notebooks (all 861 of the competition, scored via the search API):** still nothing ≥ 0.95.
Top: 0.946 `pjmathematician/rsna-knee-d4-blend`, 0.945 `pjmathematician/rsna-knee-d4-lite` and
`aastikrajan15/knee-s75-w5{0,60}` — all the 0.943 stack rank-blended at 0.45–0.5 with the author's own
model from **private** datasets (`rsna-knee-eff6-assets`, `rsna-knee-ens14-assets`,
`aastikrajan15/knee-ckpt`: 403). Not reusable.

**New label tables** (gold-58, leak-checked): ctogaurav v4 blend 0.893 (clean, below teach4 0.896);
jaygautam007 posterior-calibrated and narisettichaitanya teacher soft labels contain gold verbatim /
smoothed (1.000 / 0.996) — excluded.

**Discussions:** 0.949–0.954 single models are small (ResNet / CoAtNet at 224–288 px); the lever named
by all of them is labels — soft labels, several sources, and **out-of-fold** predictions mixed heavily
(≥ 50 %) into the report labels. bf16 on T4 is emulated (~50× slower) — use fp16.

**Compute.** `TpuV5E8` / `TpuV6E8` / `NvidiaL4` are valid kernel machine shapes; TPU quota (20 h/week)
is unused. `src/rsna_knee/mil/kfold_xla.py` is the TPU twin of `kfold.py` (same folds, outputs and
checkpoint format; host-side augmentation, bf16, fixed shapes).

**Running:** `tranbadat/rsna-knee-kfold-r2` — ConvNeXt-nano on `targets_r2`, 5 folds over 2×T4,
11 epochs (spends the GPU hours left before the 2026-10-03 reset).

## 7. 2026-10-02 update

**State.** Best LB still **0.939** (E8), rank ≈ 1,760 / 4,874. LB top 0.961; 95 teams ≥ 0.95, 248 ≥ 0.945.
Team-merge deadline 2026-10-15, final 2026-10-22. GPU quota 5.6 / **30 h** used (resets 2026-10-03 00:00 UTC;
the quota API misreports the allowance as 6 h — earlier notes saying "6 h/week" are wrong);
TPU 0.03 / 20 h used.

**Research refresh.** No public notebook above 0.946 (all of 2026-10-01/02's new ones are 0.934–0.943
forks of the public stack). "Best single-model score" thread: ResNet-50 5-fold @224 → 0.954; CoAtNet
single fold @224 → 0.950 (gold-58 OOF 0.930); small ResNet single fold → 0.949 (gold-58 ≈ 0.930);
140–160 mm field of view; all name labels (soft, OOF-pseudo-label mixed ≥ 50 %) as the lever.
Others report gold-58 ≈ 0.93 ↔ LB ≈ 0.95; ours is 0.905 ↔ 0.917. The gap is the image model.

**R2 (ConvNeXt-nano on `targets_r2`, 5 folds, 11 epochs, 2×T4 151 min):** gold-58 **0.899** (R1: 0.905),
OOF vs teach4 0.872 (R1: 0.866). Best epoch = last in every fold again → still undertrained.
R3 targets (`build_r3.py`: 0.3·teach4 + image OOF mix incl. R2) score 0.918 on gold-58 — no better than
R2 targets (0.923). **Labels have plateaued at ≈ 0.92 gold; don't spend more on label mixing.**

**TPU probe 2 failed** (`Expected 8 worker addresses, got 1`): the parent process called `jax.devices()`
(and imported torch_xla) before `xmp.spawn`, so it held the TPU. The launcher
(`scripts/kaggle_kfold_xla.py`) never touches the TPU in the parent; smoke test
`tranbadat/rsna-knee-xla-smoke` (fold 0, 400 studies, 2 epochs) queued to confirm.

**Model changes (opt-in, 95 tests):**
- `--slot_aware`: windows stay inside one series (`windows.slot_triplets`; the old triplets straddle
  slot boundaries, mixing two series in one window) and a zero-initialised learned slot embedding
  (plane × sequence) is added to each window's features (`MILClassifier(n_slots=5)`). Checkpoint meta
  `windows: "slot"`, `n_slots`; `fleet` serves both window modes.
- `WIDE44_256` recipe: corpus slots over 4–96 % of each series instead of 15–85 %, built at 256 px from
  DICOM (CPU kernel `tranbadat/rsna-knee-cache-wide256`, cache prefix `wide256`). Named caches
  (`--cache_name`, `--recipe`); `fleet` decodes once per recipe and serves mixed-recipe checkpoints.

**Planned A/B (TPU, `tranbadat/rsna-knee-xla-ab`):** fold 0, 12 epochs, ConvNeXt-nano, `targets_r2`:
stack windows vs slot-aware. Read on the fold's ≈ 870 validation studies vs teach4 (far more power than
gold-58) plus gold-58 as a guard. Then wide vs corpus span on the winner.

### A/B results, fold 0 (2026-10-02, GPU; ConvNeXt-nano unless noted, `targets_r2`, 12 epochs, eval vs teach4 on 872 val studies)

Kernels `tranbadat/rsna-knee-gpu-ab` (layout/resolution), `-ab2` (ConvNeXt-tiny), `-ab3` (CoAtNet-1, ResNet-50).
Paired bootstrap (300×) of the val macro difference against `gab_slot`:

| Arm | Val macro | Gold-58 | Δ vs slot (90 % CI) | Verdict |
|---|---|---|---|---|
| `gab_stack` (old windows) | 0.8753 | 0.8955 | −0.0002 [−0.0023, +0.0018] | slot-awareness is neutral |
| `gab_slot` (256 px) | 0.8755 | 0.8915 | — | reference |
| `gab_wide` (4–96 % span) | 0.8701 | 0.8821 | −0.0055 [−0.0079, −0.0030] | **rejected** |
| `gab_336` (raw 336 px corpus) | **0.8781** | 0.8952 | **+0.0025 [+0.0002, +0.0044]**, P>0 0.97 | small real gain: MCL, lateral meniscus, medial OA, contusion; loses fracture |
| ConvNeXt-tiny 256 (bs 6) | 0.8711 | 0.8891 | (not paired) | rejected — slower, worse |
| ResNet-50 224 | 0.8514 | 0.8590 | (not paired) | rejected |

Every run was still improving at its last epoch. Wall time per fold on one T4: nano 256 56 min, nano 336 96 min,
tiny 72 min, ResNet-50 42 min. GPU sessions: max 2 concurrent.

More arms (same fold / targets / reference; `-ab3`, `-ab4`):

| Arm | Val macro | Gold-58 | Δ vs slot (90 % CI) | Verdict |
|---|---|---|---|---|
| CoAtNet-RMLP-1 224 (grad-ckpt, 12 min/epoch) | 0.8754 | 0.9017 | −0.0002 [−0.0028, +0.0022] | equal accuracy at 3× the cost; Spearman 0.967 with nano-336, rank blend 0.8783 ≈ nano-336 alone → **no diversity, rejected** |
| nano, k = 40 training windows (bs 3) | 0.8675 @ ep 9 | 0.894 | behind at every eval | **rejected** (2.4× cost) |
| nano, 20 epochs | **0.8800** @ ep 19 | 0.898 | +0.0045 over 12 epochs | **adopted** (≈ 18–20 epochs) |

**Reading.** Every variant lands within ±0.005 of 0.875 on val and 0.89–0.90 on gold-58, and different
backbones agree at ρ ≈ 0.97: the models are label-bound. Resolution (336) and length (18–20 epochs) are the
only image-side levers that moved anything.

**Production run R3** (`tranbadat/rsna-knee-r3a/b/c`, tag `r3_nano336`): ConvNeXt-nano, raw 336 px corpus,
slot-aware, 18 epochs, `targets_r2`, 5 folds over three 2×T4 sessions (≈ 2.4 h per fold). Next: solo LB read
with `rsna-knee-ours-solo` (tags `r3_nano336`), then the blend decision against the 0.943 public stack.

### R3 result and the blend rule (pre-registered 2026-10-03 02:05 UTC, before the solo LB read)

R3 (5 folds, all best at epoch 17 of 18): OOF vs teach4 **0.8801** (R2 0.8719, R1 0.866); gold-58 fold-mean
0.8977 (R2 0.8986, R1 0.905 on 3 folds). Solo submission `56787175` (`rsna-knee-ours-solo` v2, dry run: 5
checkpoints, 0 failures, 6.1 min).

Blend kernel `rsna-knee-ours-blend` (jiweiliu 0.943 public notebook unchanged + `rsna_knee.mil.ours`, tag
`r3_nano336`). R1 (solo 0.917) at w = 0.35 scored 0.938 — too much weight for its strength — so the weight
scales with the solo read:

| R3 solo LB | blend weight w |
|---|---|
| ≥ 0.935 | 0.35 |
| 0.925 – 0.934 | 0.20 |
| < 0.925 | 0.10 |

Keep the blend as a final candidate only if it scores ≥ 0.944 (above the 0.943 base); otherwise R3 is not
strong enough to add and the next lever is a different label source (diversity), not more image-side tuning.

**R3 solo LB: 0.920** (`56787175`; R1 solo 0.917). OOF vs teach4 rose +0.014 (0.866 → 0.880) but the LB only
+0.003: image-side gains measured against report-derived labels barely transfer to the radiologist-read test
labels. Per the rule above, the blend runs at **w = 0.10** (`rsna-knee-ours-blend` v2).

### R4 — heavy OOF pseudo-labels (pre-registered 2026-10-03 ~09:00 UTC)

**Hypothesis:** the LB is label-bound (R1 → R3 image changes: OOF +0.014, LB +0.003). Participants at 0.95
single-model report mixing ≥ 50 % out-of-fold model predictions into the report labels. R2 targets used the
weaker R1 OOF; R3's OOF is the best we have.
**Targets** `targets_r4.csv` = 0.3·teach4 + 0.7·image mix (R3 0.7, R2 0.2, R1 0.1; each quantile-mapped onto
teach4), built by `build_r4.py` (scratchpad). Gold-58 of the recipe 0.913 (R2 targets 0.923 — within noise).
**Training:** identical to R3 (nano, 336 px, slot-aware, 18 epochs), tag `r4_nano336`, kernels
`tranbadat/rsna-knee-r4a/b/c`. Only the targets differ, so the LB difference is the label effect.
**Decision rule (solo LB vs R3 0.920):** ≥ 0.925 → labels are the lever: run another round (R5 from R4 OOF)
and use R4 in the blend at the weight table above; 0.918–0.924 → no measurable label effect, stop pseudo-label
rounds; < 0.918 → heavy mixing hurts, revert to R2-style targets.
(OOF vs teach4 is not comparable across R3/R4: R4 targets are mostly R3's own OOF.)

**R3 blend LB: 0.942** (`56787692`: jiweiliu 0.943 public notebook unchanged + R3 at w = 0.10; graded in ≈ 7 h).
−0.001 vs the base = no measurable effect; below the 0.944 bar, so **R3 does not go into the final blend**.
Our best LB is now 0.942 (from the public stack, not from R3). R4 (above) is the direct test of the
label-bound hypothesis.

**R4 progress.** Folds 0–3 done (all exit 0); gold-58 fold-mean over folds 0–3: R4 0.8922 vs R3 0.8969 (noise).
Fold 4 (`r4c`) running; solo kernel staged (`rsna-knee-ours-solo` next version, tag `r4_nano336`).

**R5t — control: report labels only** (`tranbadat/rsna-knee-r5ta/b/c`, tag `r5t_nano336`): the R3 recipe
trained on `teach4.csv` (no OOF mixing). Gives the label axis three LB points at fixed image recipe —
0 % OOF (R5t), R2 mix (R3, 0.920), heavy mix (R4) — i.e. whether OOF pseudo-labelling moves our LB at all.

**R4 trained** (5 folds, all best at epoch 17): OOF vs teach4 0.8817 (inflated — targets contain R3's OOF),
gold-58 fold-mean 0.8910 (R3 0.8977, noise). Solo kernel `rsna-knee-ours-solo` v3: 5 checkpoints, 0 failures.
Submitted as `56799230`; LB pending.

**R5t fold 0** (report labels only): best val vs teach4 0.8718 (ep 15) vs R3 fold 0 0.8805 — the model trained
directly on teach4 scores *lower against teach4* than the one trained on the OOF-mixed R2 targets; gold-58
0.895 vs 0.891. Folds 1–4 running (`r5ta` done, `r5tb`, `r5tc`).

**R4 solo LB: 0.917** (`56799230`) vs R3 0.920. Pre-registered rule: < 0.918 → heavy OOF mixing does not help
our LB (−0.003 is at noise level; read it as "no gain"). **Stop pseudo-label rounds.**

**Where this leaves us.** Every model we have built from the public 44-slice corpus lands at LB 0.917–0.920,
regardless of resolution, length, backbone or label mix; the public CoAtNets trained on the same corpus sit at
≈ 0.924 each. Participants report single models at 0.949–0.954 from their own preprocessing. The remaining
untested common factor is the corpus preprocessing itself (slot choice keyed on plane × FS only — which merges
T1/PD/T2 contrasts — the 15–85 % span, per-series 2–98 % window, centre 140 mm crop).

**Label calibration from gold-58 — rejected (local, `analysis/calib_loo.py`).** Idea: experts mark findings the
reports omit (gold Fracture 31 % vs 7 % of reports), so learn P(expert label | all 12 report labels) per
finding. Leave-one-out on gold-58 with a ridge logistic: macro AUC **0.835 vs 0.896** for the raw teach4 column
(C = 0.05; worse at weaker regularisation). 58 studies cannot support a cross-finding mapping; the matching
report label stays the best single signal. Also checked: `train_series.csv` only carries plane and FS (no
T1/PD/T2), and studies average 5.5 series, so the corpus' 5 slots already cover most of each study.

**Direction (2026-10-03, owner's decision):** no team merge — the solution stays independent. Public ideas and
techniques may be used; effort goes into our own preprocessing, models and ensemble. (Outreach draft removed.)

### Per-finding gold-58 across runs → the maxspan hypothesis (2026-10-03)

| | teach4 labels | R1 (LB .917) | R3 (.920) | R4 (.917) | Raptor v5 |
|---|---|---|---|---|---|
| MCL | 0.971 | 0.902 | 0.889 | 0.884 | **0.982** |
| PF OA | 0.898 | 0.830 | 0.817 | 0.819 | 0.848 |
| Effusion | 0.880 | **0.985** | 0.939 | 0.914 | 0.983 |
| macro | 0.896 | 0.905 | 0.898 | 0.891 | **0.920** |

Our models fall *below their own report labels* on fine structures (MCL −0.08, PF OA −0.08, ACL, lateral
OA/meniscus) and beat them on diffuse findings; effusion degrades monotonically as more OOF is mixed in.
Raptor v5 differs from our input in slices: 64 over 2–98 % (coronal 20 slices) vs the corpus' 44 over 15–85 %
(coronal 14) — the same slice density, wider coverage. WIDE44 (44 slices over 4–96 %, i.e. sparser) lost
−0.0055, which does not test this.

**A/B `gab_max16` / `gab_max24`** (`rsna-knee-gpu-ab5`, after `rsna-knee-cache-maxspan256` builds the 64-slice
cache from DICOM): nano 256 px stack windows, 12 epochs, `targets_r2`, k_eval 60, k = 16 and 24, vs `gab_stack`
(0.8753 val, gold 0.8955). Read: val vs teach4 and gold-58 MCL / PF OA / macro.

### Crop diagnostic → tissue-centred crop (2026-10-03)

`rsna-knee-crop-diag` (CPU, 400 training studies, middle slice of every slot series, Otsu tissue mask;
`experiments/phase2/crop_diag.csv`): field of view median 160 mm; tissue centroid offset from the image centre
median 13 mm sagittal / 9 mm coronal / 8 mm axial (p90 26 / 21 / 18 mm); tissue outside the centred 140 mm crop
median 12 % sagittal / 10 % coronal (p90 22 % / 20 %); 67 % of series lose > 5 % of their tissue. The MCL sits at
the medial edge on coronal slices (~50 mm from the knee centre), so a 20 mm offset puts it at the crop border.

`CORPUS44C_256` = the corpus layout with the crop centred on each series' tissue centroid (middle slice;
`volume.tissue_centre`, clamped inside the image); test-time decode uses the same `build_volume`.
Cache `rsna-knee-cache-ctr256`; A/B `rsna-knee-gpu-ab6`: `gab_ctr` (nano 256 stack, 12 ep, `targets_r2`) vs
`gab_stack` (0.8753), plus `gab_stack_s7` = the baseline with seed 7 to measure training noise (the bootstrap
CIs so far cover study sampling only).

**R5t trained (report labels only):** OOF vs teach4 **0.8679** (lower), gold-58 fold-mean **0.9054** (higher;
best epochs 13–15). Effusion 0.980, lateral meniscus 0.873, synovitis 0.799 recover.

| Run | targets | OOF vs teach4 | gold-58 | solo LB |
|---|---|---|---|---|
| R5t | teach4 only | 0.868 | **0.905** | pending |
| R3 | R2 mix (≈ 65 % OOF) | 0.880 | 0.898 | 0.920 |
| R4 | 70 % OOF (mostly R3) | 0.882 | 0.891 | 0.917 |

Gold-58 falls monotonically as more OOF is mixed in, and the LB moves the same way on R3→R4. OOF pseudo-labels
make the model fit the *report* labels better and the *expert* labels worse — the opposite of what forum
posts report. R5t's solo LB (`rsna-knee-ours-solo`, tag `r5t_nano336`) is the test; if ≥ 0.921 the next
production run uses teach4-only targets.

**`ab6` result (fold 0, vs `gab_stack`, paired bootstrap):**

| Arm | Val | Gold-58 | Δ val (90 % CI) |
|---|---|---|---|
| `gab_stack_s7` (same as baseline, seed 7) | 0.8735 | 0.8937 | −0.0019 [−0.0039, +0.0002] |
| `gab_ctr` (tissue-centred crop) | 0.8726 | 0.8957 | −0.0027 [−0.0044, −0.0007] |

**Training noise ≈ 0.002 on final val** (larger mid-training: 0.006 at epoch 5). The tissue-centred crop is at
that floor → **rejected** (no gain). Re-read of earlier A/Bs with this floor: 336 px (+0.0025) is borderline —
kept for its MCL / lateral-meniscus gains; wide span (−0.0055) and ResNet-50 (−0.024) remain real losses;
slot-aware, ConvNeXt-tiny, CoAtNet-1 are noise.

**R5t solo LB: 0.922** (`56803647`) — best own model. Label axis at fixed image recipe:
R5t (teach4 only) 0.922 > R3 (R2 mix) 0.920 > R4 (heavy OOF) 0.917; gold-58 ranks them identically
(0.905 > 0.898 > 0.891). **OOF pseudo-labelling hurts our LB; train on report labels.** Rule fired (≥ 0.921):
the next production run uses `teach4.csv`. Note the target tables' own gold-58 (teach4 0.896, R2 0.923,
R4 0.913) does *not* predict the trained model's LB — the model trained on the "worse" table is best.
Gold-58 of the *trained model* does track the LB at this 0.007–0.014 scale.
