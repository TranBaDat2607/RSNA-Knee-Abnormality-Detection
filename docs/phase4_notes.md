# Phase 4: last week (2026-10-15 to 10-22)

Driven by the `kaggle-grandmaster` plugin loop (`kg-ideas` → backlog → pre-registered rule → kernel → ledger).
Backlog: `reports/backlog.md` (`python -m kgkit backlog list`).

## Where things stood on 2026-10-15 (evening)

- Our best: A12 `57197656` public **0.957** (rank ~211 / 6,043). Gold cutoff ≈ rank 22 ≈ 0.963; top 0.967.
- Public notebooks top out at 0.957 (search API `best_public_score`); no public leg we do not already use.
- Gold-58 Spearman between the public legs (mean over 12 findings):

  | | raptor | R7 | s6fall | S6 | reader |
  |---|---|---|---|---|---|
  | raptor (0.65 n384 + 0.35 n224) | 1 | 0.867 | 0.884 | 0.866 | 0.895 |
  | R7 | | 1 | 0.871 | **0.931** | 0.847 |
  | s6fall | | | 1 | 0.866 | 0.869 |

  S6 sits at 0.931 with R7 (same author and pipeline), so it is not a fourth leg; the reader is weak (gold 0.915).
- **Host statement (thread 743416, Chris Carr for RSNA, 10-14):** no dataset list; external data must be "publicly
  available and equally accessible ... at no cost". Datasets that need only registration and acknowledgement of
  licence terms clearly qualify; ones that need an institutional agreement or affiliation clearly do not. Teams should
  disclose what they used. This bears on the final picks (backlog #2).
- **Forum (thread 735304):** clean single models at public 0.953–0.959 (2.5D ConvNeXt-T 5-fold 0.959, ResNet-50 0.954,
  DINOv2 0.952); the lever they name is OOF pseudo-labels mixed into report labels. nartaa (thread 749359) retrained
  heliosli's S6 on their own labels: fold-0 model 0.948, all-data model 0.950 (= s6fall).
- `kgkit discussions` cannot read an active competition's forum (Meta Kaggle hides `ForumId`): UAT issue TranBaDat2607/claude-code-kaggle-grandmaster#14. Threads
  were read through the Kaggle search API instead (`experiments/phase2/analysis/topic.py`).

### Runtime budget of A12 on the hidden test (estimated from the gold-58 logs, 2×T4, 1,322 studies)

nartaa 384 ≈ 45 min, nartaa 224 ≈ 20 min, R7 ≈ 3.5–4.5 h (606 windows/study), s6fall ≈ 1.5 h → A12 ≈ 6.5–8 h of 9 h.
A new leg must therefore cost ≲ 30 min, or ride inside an existing pass.

## Final picks after the host statement (backlog #2; evidence for the user, who ticks the picks)

The 10-10 plan was one OAI-track pick (A12) plus a clean hedge (C3, public 0.946). The host statement of 10-14 changes
that hedge's value:

- OAI registration (login.gov plus the NDA data-use terms, "Institutional Approval Required: No") matches "registration and
  acknowledgement of licensing terms", which the hosts call clearly allowed. The open points are NIH's country blocks and
  fastMRI/SKM-TEA's application forms. Hosts will not rule dataset by dataset and ask teams to disclose.
- We use only **public weights** that other teams trained. We never touched OAI, fastMRI or SKM-TEA data.
- C3 sits 0.011 below A12 and 0.010 below today's bronze cutoff (0.956). It can only matter if an OAI ban applied to
  public-weight users across the whole board (about 1,000 teams), and that ban would also remove most of the teams around us.
- **Recommendation:** two OAI-track picks: A12 plus the best different OAI-track submission (A13 if it is strictly above
  0.957, otherwise A6 or A10 at 0.956, the less-tuned form). Keep C3 only if the user still wants the 10-10 hedge. This
  remains the user's decision.

## S6T: heliosli's S6 recipe retrained on our labels (pre-registered 2026-10-15, before any training or gold read)

**Why.** s6fall (S6 recipe, nartaa's labels) added +0.001 as an equal third leg. A label change alone gave
ρ 0.866 between S6 and s6fall. `compact_infer` can score a second model group on the same windows (`--models2`), so
a model with s6fall's geometry costs only its GPU forward pass inside A12's existing s6fall stage.

**Data.** Pixel cache built on Kaggle with heliosli's own inference path, `compact_infer.CStudies.rows()`
(DICOM → native render → geometry bank → SKM-TEA anchors → `px_cache.StudySlices.build`), so training and inference
rows come from the same code. Pool: the 4,407 training studies minus the 58 gold studies and the one non-gold study
whose report text equals a gold report (4,348).

**Labels.** `teach4opus` soft labels (mean of teach4 and the Opus 4.5 v1 public table; gold agreement 0.9014) as the
`L|prior` column of the `E_prior` arm. Not tuned on gold.

**Model (fixed).** `compact_train.py` (sha as in s6fall's manifest) with one change: bf16 autocast → fp16 autocast +
GradScaler (T4 has no bf16). convnext_small.fb_in22k_ft_in1k, size 320, crop 288, views v4, slices 3, phys_mm 3.5,
spatial attention, 64 train windows, batch 2, lr 1e-4 / head 1e-3, wd 0.05, aug crop, all data (no folds).
Two seeds (7 and 8), one per T4; the leg is the mean of the seeds that finish (compact_infer group mean).
Epochs: 16 if the measured step time fits 11.5 h, otherwise the largest count that fits (decided from the dev run's
step time, not from any score).

**Dev runs (2026-10-15).** Cache dev (12 studies, CPU): 0 errors, 641 rows/study, 61 MB/study raw (265 GB for the pool),
73 s/study/worker with the SKM-TEA V-Net on CPU. The shards therefore run on GPU kernels and store rows as JPEG q90
(13.6 MB/study, mean |error| 1.1 grey levels; 4 shards < 20 GB each). Trainer dev (`kgkit gpu build/push`, 12 studies,
1 epoch, both seeds): exit 0, models saved, peak GPU memory 6.4 GB of 15 GB. A 12-study epoch cannot time a full
epoch. **Epoch rule, amended before any score:** keep 16 epochs. If they do not fit one 12 h session, the run stops
at an epoch boundary with its checkpoint and resumes in a second session (the cosine schedule spans the 16 epochs).

**Cache shards (GPU kernels `rsna-knee-s6t-cache-s0..s3`), all 4,348 studies, 0 errors:**

| shard | studies | rows | JPEG GB | minutes |
|---|---|---|---|---|
| s0 | 1,028 | 605,643 | 11.8 | 58 |
| s1 | 1,135 | 666,721 | 13.1 | 83 |
| s2 | 1,118 | 668,886 | 13.1 | 80 |
| s3 | 1,067 | 640,396 | 12.5 | 63 |

About 5 GPU-h in total (two kernels at a time).

**Training run `tranbadat/s6t-train` v1** (`kgkit gpu build/push`, 2×T4, both seeds, 12 h cap), pushed 2026-10-15
19:32 UTC. `kgkit gpu push` first refused it, because finished cache kernels were still reserved at 12 h each
(UAT TranBaDat2607/claude-code-kaggle-grandmaster#16). The live quota (13.67 h) covered the cap, so it went out with `--force`.

**Session 1 result (`s6t-train` v1, 10.42 GPU-h).** About 77 min per epoch (≈ 2.1 s/step on a T4; each seed on its own
T4). The epoch guard stopped both seeds cleanly after **epoch 7 of 16**, keeping their checkpoints. Training loss
s7 0.534 → 0.431 and s8 0.536 → 0.429 (s6fall at the same epoch: 0.552 → 0.433). No score was read. The remaining
8 epochs run in two resume sessions: **2a** now, on the 3.24 h of this week's quota (≈ 2 epochs), and **2b** after
the 10-17 00:00 UTC reset (≈ 6 epochs, ≈ 8 h).

**Session 2a (`s6t-r1-train`, 2.51 GPU-h):** both seeds resumed after epoch 7 and trained epochs 8–9 (loss s7
0.4250 → 0.4187, s8 0.4229 → 0.4170), then stopped with epoch-9 checkpoints. Epochs 10–15 remain: on TPU if
`s6t-tpu-train` starts first, otherwise GPU session 2b (`s6t-r2-train`, built, pushed right after the 10-17 reset).

**TPU path (2026-10-16, GPU quota spent, TPU 20 h unused).** `s6t_train.py` now has a TPU mode (`S6T_XLA=1`): one
seed per v5e chip (PJRT `TPU_VISIBLE_CHIPS`), bf16 autocast, static shapes (W padded to 64, all B×W windows through the
encoder, crop on CPU), `mark_step` per optimiser step, `xm.save` checkpoints, and torch_xla's reentrant checkpoint in
place of timm's. On CPU, the patched forward equals the original (max |Δlogit| 4.8e-7 with padded windows). A local
torch_xla 2.8 (PJRT CPU) run trained, checkpointed, resumed and saved, and the CUDA epoch-7 checkpoint loads on `xla:0`
(AdamW state, scheduler step 17,392). No shape avoids the TPU queue (300+ deep per the forum): `TpuV5E1`, `TpuV5E4`
and `TpuV6E1` are silently stored as `TpuV5E8`; `TpuV6E8` and `Tpu1VmV38` are kept as their own pools, but queued too.
Kaggle allows one batch TPU session per user. The real job `s6t-tpu-train` (resumes from the most advanced attached
checkpoint) was queued at 08:37 UTC. If it has not started by the 10-17 00:00 GPU reset, GPU session 2b goes ahead.

**Gate (all must hold; read once, after training).**
1. Leg gold-58 macro AUC ≥ 0.925 (both public S6-recipe models are ≥ 0.9267).
2. Mean Spearman on gold-58 ≤ 0.90 against each of raptor, R7 and s6fall.
3. A13 gold-58 ≥ A12's 0.9406 − 0.003 (regression guard only; gold-58 cannot resolve ±0.003).
4. Measured S6T GPU cost projects A13 to ≤ 8.5 h on 1,322 studies; otherwise one seed only, then re-check.

**Inference cost.** In A12, S6T rides in s6fall's `compact_infer` call (`--models2 <S6T> --w2 0`: `predictions.csv` stays
s6fall's own, and the per-model logits come out of `--dump-logits`). S6 took ≈ 4.7 GPU-s per model per study at eval
stride 1, so each S6T seed adds ≈ 0.85 h on 1,322 studies, or ≈ 1.7 h for two seeds. A12's own hidden-test runtime has
never been measured.

**D1 runtime probe (pre-registered and submitted 2026-10-15 16:37 UTC).** A12 (`rsna-knee-oai-957s6f` v1) is resubmitted unchanged
with one of today's unused submissions, and its status is polled every 3 min. PENDING → COMPLETE wall time is an upper
bound on A12's hidden-test runtime (it includes queueing). Its score is **not** a decision input (expected 0.957).
It sets the S6T seed count: if A12 ≤ 6.3 h, two seeds; if ≤ 7.2 h, one seed; otherwise S6T does not go into A12 as a
fourth leg.

**D1 result:** `57237639` went PENDING → COMPLETE between 16:37:09 and 21:23:32 UTC, so **A12 takes ≤ 4.77 h** of the
9 h on the hidden test (queueing included). Public 0.957, as expected. → **S6T goes in with two seeds.** Earlier
estimates from the gold-58 per-study timings (6.5–8 h) were far too pessimistic: about 4 h of runtime is free, enough
for another leg of ≤ 2–3 h.

**Form (fixed now, not tuned).** A13 = four equal legs: final = 3/4 × A12 value + 1/4 × rank(S6T), inside A12's
time guard (any failure, timeout or < 90 % coverage keeps A12's file). Submit A13 if the gate holds and the dev run on
the 3-row test passes. **A13 replaces A12 as the OAI pick only if its public score is strictly above 0.957.**

## 2026-10-16: the board moved; heliosli's R6 (pre-registered before any R6 run or read)

Leaderboard 12:30 UTC (6,084 teams): rank 22 = 0.963, rank 101 = 0.959, 416 teams ≥ 0.958, 764 ≥ 0.957. We are rank
517 at 0.957. Silver (top 5 % ≈ 304) now needs 0.958, and the bronze line (top 10 % ≈ 608) is inside the 0.957 tie block.

`watch_public.py`: two new public 0.958 notebooks (heliosli `blend-mine-in-private-lb-luck-to-all-upvoters`, goodpjw v26).
Both use nartaa 384 + 224 plus heliosli's **R6** (bundle `fe055406fad6a12f`, 4 folds, `gold_in_training 0`, flip TTA) in
the compact slot, plus per-finding Raptor weights (Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35). goodpjw's code
carries a single-class `PROBE_CLASS` switch, so those weights were most likely read off the public LB. heliosli's 2-leg
history: R7 0.9573 → R6+flip 0.9576 → per-finding weights 0.958. The R6 stage is A12's R7 stage with three lines changed
(bundle id, dataset `heliosli/rsna-knee-r6-frozen-fe055406-w1`, `FLIP_TTA = True`). Same code, so A12's audit covers it.
R6 and R7 share size 384 / crop 352 / views v4 / stride 1 and the trainer md5, but R6 adopts its own Effusion fluid and
Baker's state stackers. bliverrigpdoge's "knee-centred ConvNeXt" leg (0.957) reads a private dataset: not usable.
We were never able to test R6 itself: in phase 3 the `rsna-knee-r6-frozen-fe055406` dataset already held R7.

**Per-finding weights: not adopted.** They are public-LB fits on 5 findings. Our rule is no weight tuning on gold or LB.

**R6 gold-58 diagnostic (`kernels/r6-gold58`, expiring quota of this week).** R6 plain, then R6 + flip, on the 58 gold
studies (stand-in root), with per-study time. **Rules, read once:**
1. *Fourth leg* (same gate as S6T): R6 (flip) gold ≥ 0.925, mean Spearman ≤ 0.90 vs each of raptor, R7 and s6fall,
   A14 = 3/4 × A12 + 1/4 × rank(R6) ≥ A12 − 0.003, and runtime fits.
2. *Otherwise a swap* (R6 is the R7 family): A14 = A12 with R7 replaced by R6 (the flip form if its measured cost keeps A13 +
   A14 ≤ 8.5 h on 1,322 studies, else plain). Built only if R6 gold ≥ 0.925 and A14 gold ≥ A12 − 0.003. It is an LB test
   of a same-family swap: expected gain about +0.0003 (heliosli's 2-leg numbers). **A14 replaces A12 only if public is
   strictly above 0.957.** S6T keeps priority on runtime: if both go in, R6 drops to plain.

**R6 gold-58 result (`rsna-knee-r6-gold58` v1, 12:16–12:46 UTC, `analysis/r6_gate.py`, read once):**

| | gold-58 macro | ρ raptor | ρ R7 | ρ s6fall | GPU s/study/worker |
|---|---|---|---|---|---|
| R7 plain (phase 3) | 0.9312 | 0.867 | 1 | 0.871 | 12.6 |
| **R6 plain** | **0.9371** | 0.873 | **0.980** | 0.878 | 12.3 |
| R6 flip | 0.9366 | 0.874 | 0.979 | 0.879 | 23.7 |
| A12 | 0.9404 | | | | |
| A14 fourth leg (3/4 A12 + 1/4 R6) | 0.9399 (−0.0006) | | | | |
| A14 swap (R7 → R6) | 0.9401 (−0.0003) | | | | |

R6 is a stronger R7 (+0.006 single leg: Synovitis 0.859 vs 0.814, Baker's 0.969 vs 0.942, Medial Meniscus 0.971 vs 0.958;
weaker on Lateral Meniscus/OA and Fracture) but the same family (ρ 0.98). Rule 1 fails (1b); **rule 2 passes → A14 = swap**.
Plain vs flip: ρ 0.998, flip costs 2× GPU (+≈1.2–1.5 h on 1,322 studies). With S6T's ≈1.7 h, A13 + A14 would project to
≈8.0 h against the 8.5 h cap, and the rule makes R6 plain whenever S6T joins. **A14 therefore uses R6 plain**, so A14's
public score is also the base that A13 builds on. Compute time is unchanged from A12 (R6 and R7 have the same geometry and cost).

A14 (`kernels/oai-a14`, `build.py <A12 nb> plain`): only the compact cell changes (bundle id, dataset, `FLIP_TTA = False`).
The dev run on the 3-row test was pushed at 12:47 UTC on this week's last 0.25 GPU-h.
**Dev run v1 passed** (532 s): Raptor 384/224, R6 (`fe055406-w1`, flip off, 3/3 scored, 0 fallback), s6fall 3/3,
three-way blend written, `kgkit validate` OK. **Submitted 2026-10-16 12:58 UTC as `57269763` (ledger `0011_a14`).**

**A14 result: public 0.957** (`57269763`, PENDING 12:58 → COMPLETE by 17:14 UTC, so ≤ 4.3 h on the hidden test). It ties
A12 → **A12 stays the OAI pick** (rule: strictly above 0.957), and A13 is built on A12. A14 counts as the "different
OAI-track" candidate at 0.957 for the final-pick table. The same-family swap gave nothing measurable on the LB, as the
phase-3 statistics predicted (R6 is a stronger R7 on gold-58 but ρ 0.98).

**Base for A13 (fixed before A14's score):** if A14 scores strictly above 0.957, A13 = `oai-a13/build.py <A14 nb> 2`.
Otherwise it is built on A12 as planned. The A13 builder runs on the A14 notebook unchanged (tested: 884 KB, R6 bundle kept).
A13's metadata attaches both the R7 and R6 datasets, so it works on either base.
