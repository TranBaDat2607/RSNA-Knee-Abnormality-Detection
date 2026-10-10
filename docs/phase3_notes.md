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

## Pre-registered blend weights (fixed before any LB read)

- Our R5t leg: 0.10 (rule from phase 2: solo < 0.925 → 0.10).
- nartaa 224 checkpoint inside Part A: 0.25 (second checkpoint of the same run, weaker by 0.004 public).
- Reader in the clean stack: flat 0.30 (goodpjw's own measured setting).
- A variant is kept only if it beats its parent on the public LB; gold-58 is a regression guard only (nartaa
  was selected on it).

## Results

(filled in as submissions are scored)
