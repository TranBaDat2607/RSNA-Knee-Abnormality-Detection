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
- `kgkit discussions` cannot read an active competition's forum (Meta Kaggle hides `ForumId`): UAT issue #9. Threads
  were read through the Kaggle search API instead (`experiments/phase2/analysis/topic.py`).

### Runtime budget of A12 on the hidden test (estimated from the gold-58 logs, 2×T4, 1,322 studies)

nartaa 384 ≈ 45 min, nartaa 224 ≈ 20 min, R7 ≈ 3.5–4.5 h (606 windows/study), s6fall ≈ 1.5 h → A12 ≈ 6.5–8 h of 9 h.
A new leg must therefore cost ≲ 30 min, or ride inside an existing pass.

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

**Form (fixed now, not tuned).** A13 = four equal legs: final = 3/4 × A12 value + 1/4 × rank(S6T), inside A12's
time guard (any failure, timeout or < 90 % coverage keeps A12's file). Submit A13 if the gate holds and the dev run on
the 3-row test passes. **A13 replaces A12 as the OAI pick only if its public score is strictly above 0.957.**
