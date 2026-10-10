# Phase 3: endgame (2026-10-10 to 10-22)

## Where things stood on 2026-10-10

| Submission | What | Public LB |
|---|---|---|
| `56787692` | jiweiliu 0.943 community stack + our R3 at w = 0.10 | **0.942** (our best) |
| `56814752` | R5t solo + 2 full-data R5t models (`r5tfull`) | 0.924 (R5t folds alone 0.922) |

`rsna-knee-gpu-ab5` (64-slice maxspan input, fold 0): `gab_max24` 0.8727 and `gab_max16` 0.8683 val vs the
`gab_stack` reference 0.8753, so **rejected**.

### The leaderboard moved: OAI external data

- Top public LB was 0.964 on 10-10. 891 of 5,660 teams were at ≥ 0.950, bronze ≈ 0.950, silver ≈ 0.951, top 20 ≈ 0.961.
- Every public notebook at ≥ 0.950 depends on `nartaa/rsna-knee-publication-swa-weights-20261007`: one
  CoAtNet-RMLP-2 (Raptor MIL recipe) trained on 4,349 report-labelled studies **plus 2,399 OAI knees** (masked
  external labels for PF OA, Lateral OA, Synovitis). It scores 0.949 alone (native 320 crop, all 94 windows,
  anatomical L/R mirror TTA). A second checkpoint (224 crop, 0.945) is in the same dataset.
- The reproducible 0.950–0.951 notebooks are all nartaa + goodpjw2008's clean 2.5D ConvNeXt reader
  (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, 0.929 alone, 3 folds, per-finding rank weights). heliosli's 0.954
  (nartaa 0.7 + an OrthoFoundation/SKM-TEA compact model 0.3) cannot be reproduced: its weights were retired.
  Notebook titles that say "0.954" score 0.950 per the search API.
- Hosts have not ruled on OAI. They banned KneeCoT for unequal access, and users in China report they can't
  register with NDA. **User decision (10-10): two tracks.** Final pick 1 = best blend with the public OAI weights.
  Final pick 2 = best clean blend (competition data + previously used public weights). We do not download OAI.
- Other forum findings: nartaa's clean (no-OAI) all-data CoAtNet scored 0.939 → 0.940 (96 slices native 384, 94
  windows) → 0.942 (+ JEV labels). Inference with 24 windows instead of all 94 cost 0.006 on their model. A
  forum post reports the public baseline's slice order (sorted by SOP UID) is random; IPP order is worth +0.028 on
  that baseline (our corpus pipeline orders by position). Local CV on ~4,300 weak labels tracked one team's public
  LB better (r 0.87) than gold-58 (r 0.66).

## Kernels

| Folder | Kaggle kernel | Track | What |
|---|---|---|---|
| `kernels/oai-blend` | `rsna-knee-oai-blend` | OAI | goodpjw 0.950 notebook (nartaa + reader) + [A2: nartaa 224 checkpoint] + C: our R5t leg (`build.py`) |
| `kernels/oai-gold58` | `rsna-knee-oai-gold58` | diag | nartaa acc/eff + reader on gold-58 (stand-in root): leg correlations, regression guard |
| `kernels/clean-blend` | `rsna-knee-clean-blend` | CLEAN | community stack (as in `ours-blend-r3`) + goodpjw reader at a flat 0.30 (goodpjw measured 0.944 for this pair) |
| `kernels/c96-cache` | `rsna-knee-c96-cache` (CPU) | both | nartaa's d96 input (96 slices, 384 px, their `fastread` builder) for all 4,407 training studies, JPEG centre 352 |
| `kernels/ft96` | `rsna-knee-ft96` | both | fine-tune on d96 with `teach4` labels, 3 epochs, 320 crop + mirror aug: ftO from nartaa (OAI), ftC from Raptor v10 (CLEAN) |

### Gold-58 of the public legs (`rsna-knee-oai-gold58`, `analysis/gold_legs.py`)

| Leg / variant | gold-58 macro | vs AB parent [95 % CI] |
|---|---|---|
| nartaa acc (selected on gold, optimistic) | 0.9229 | |
| nartaa eff 224 | 0.9150 | |
| goodpjw reader (clean, gold never trained on) | 0.9153 | |
| our R5t (5 folds + 2 full) | 0.9063 | |
| AB = goodpjw 0.950 notebook | 0.9248 | |
| v1: AB + ours 0.10 | 0.9257 | +0.0009 [−0.0008, +0.0028] |
| v2: eff 0.25 in A, then B | 0.9275 | +0.0027 [−0.0005, +0.0069] |
| v3: v2 + ours 0.10 | 0.9280 | +0.0031 [−0.0007, +0.0079] |

Mean Spearman between legs on gold: ours–acc 0.89, ours–reader 0.88, reader–acc 0.89, eff–acc 0.94. No variant
regresses; the LB decides.

### FT96 decision rule (pre-registered)

A fine-tuned model becomes a blend leg only if its gold-58 (94 windows, mirror TTA) is at least its start
checkpoint's gold-58 minus 0.005 and its rank correlation with the start is below 0.97 (otherwise it adds nothing).
ftO enters the OAI blend at 0.25 inside Part A (same slot as the eff checkpoint); ftC enters the clean blend at 0.20.

### FT96 results (`rsna-knee-ft96`, 126 min for both jobs on 2×T4)

| Job | start gold-58 | ep0 | ep1 | ep2 | holdout (top5mean ref) | decision |
|---|---|---|---|---|---|---|
| ftO (nartaa start, OAI) | 0.9229 | 0.9164 | 0.9101 | 0.9099 | 0.921 | **rejected**: below start − 0.005, Spearman with nartaa 0.972 |
| ftC (Raptor v10 start, clean) | 0.9174 (own recipe) | 0.9171 | 0.9198 | **0.9211** | 0.909 | **accepted** for the clean blend at 0.20 |

ftC Spearman: with the reader 0.894, with nartaa 0.926. Gold-58: reader + ftC 50/50 = 0.9236 vs the reader alone
0.9153. In the OAI blend ftC adds only about +0.0007 on gold, so it is not used there.
Fine-tuning nartaa on teach4 pulls it away from the expert labels. nartaa's own labels (their report labels +
three public tables + JEV) appear better aligned with gold than teach4.

### ft96b (pre-registered)

Same recipe from the public Raptor v8 and v5 checkpoints (seeds 8, 5). A sibling is accepted if its ep2 gold-58 is
≥ 0.912. The clean FT leg becomes the rank-mean of the accepted ftC models if that mean's gold-58 ≥ ftC alone −
0.002 (otherwise ftC alone stays). Weight in the clean blend unchanged (0.20).

### ft96b results (`rsna-knee-ft96b` v2, 166 min; v1 sat in QUEUED and was replaced)

| Model | gold-58 ep0 / ep1 / ep2 | holdout-300 vs teach4 | decision |
|---|---|---|---|
| ftC8 (v8 start) | 0.9219 / 0.9146 / 0.9142 | 0.8991 | accepted (≥ 0.912) |
| ftC5 (v5 start) | 0.9231 / 0.9191 / 0.9195 | 0.9100 | accepted |
| family rank-mean (ftC, ftC8, ftC5) | 0.9212 (ftC 0.9211) | 0.9113 (ftC 0.9115) | passes the rule, **not used** |

Holdout Spearman between siblings is 0.94–0.95, so the mean adds nothing measurable (+0.0001 gold, −0.0002 holdout).
The clean notebook already takes ≈ 7.5 h of 9 h, and two extra CoAtNet-2 legs would cost ≈ 40–60 min. By the
code-competition rule (drop the worst Δscore/runtime members first), clean C2 keeps ftC alone (`rsna-knee-clean-blend`
v5). This overrides the pre-registered rule on runtime grounds only; no score was looked at beyond what the rule
named. Note v8/v5 starts peak at ep0 on gold (0.922/0.923) and then decline. More epochs on teach4 move all the
clean starts toward the weak labels, so ep0-ep1 checkpoints may be the better leg if this is revisited.

### Multi-scale TTA on nartaa (`rsna-knee-oai-tta-gold58`): rejected

Pre-registered: add a view if mean(320, view) gold-58 ≥ 320 − 0.001 and Spearman < 0.99. gold-58: 320 0.9229, 352
0.9214, 384 0.9215; mean(320,352) 0.9220 (ρ 0.996), mean(320,384) 0.9226 (ρ 0.990), all three 0.9224. No
diversity, no gain. Matches nartaa's own 320 ≈ 384 finding.

## Pre-registered blend weights (fixed before any LB read)

- Our R5t leg: 0.10 (rule from phase 2: solo < 0.925 → 0.10).
- nartaa 224 checkpoint inside Part A: 0.25 (second checkpoint of the same run, weaker by 0.004 public).
- Reader in the clean stack: flat 0.30 (goodpjw's own measured setting).
- A variant is kept only if it beats its parent on the public LB; gold-58 is a regression guard only (nartaa
  was selected on it).

## Results

Submitted 2026-10-10 19:35 UTC (Claude submits since 10-11, ≤ ~3/day): `57051391` CLEAN C2 (clean-blend v5),
`57051394` OAI A3 (oai-blend v3), `57051397` OAI A2 (oai-blend v2). Queued for the next day: OAI A1 (v1), CLEAN C1 (v1).
