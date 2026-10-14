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

### Label tables vs gold-58 (clean, LLM-from-report)

| Table | gold-58 macro | note |
|---|---|---|
| teach4 (ours, phase 2) | 0.8963 | |
| JEV FINAL2 / HYBRID | 0.8745 / 0.9025 | HYBRID blend and thresholds were **tuned on gold studies**, so optimistic |
| Opus 4.5 v1 (NTejas-1 GitHub) | 0.8919 | no gold copying |
| Opus 4.5 v2 "repaired" | 0.8941 | self-distilled from model OOF; its author found it hurt training, not used |
| **teach4 + Opus v1 mean (`teach4opus`)** | **0.9014** | +0.005 over teach4 |

### ft96c (pre-registered): ftC recipe on `teach4opus`, 2 epochs, v10 and v5 starts

One replaces ftC in the clean blend only if its final gold-58 ≥ 0.9241 (ftC + 0.003). Otherwise ftC stays.

Result (`rsna-knee-ft96c`, 86 min): ftC5o (v5 start) ep0 0.9270, ep1 **0.9242**, passes by 0.0001. ftCo (v10 start)
ep0 0.9251, ep1 0.9226, fails. The teach4opus labels lifted every ep0 (0.925–0.927 vs 0.917–0.923 on teach4).
**Not executed** (user stopped the session here): the next step is clean-blend with `--ft-glob ftC5o/ckpt_ep1.pt`
(kernel source `tranbadat/rsna-knee-ft96c`) at the same 0.20. The margin is 0.0001 on 58 studies, so expect ≈ 0 on the LB.

## Pre-registered blend weights (fixed before any LB read)

- Our R5t leg: 0.10 (rule from phase 2: solo < 0.925 → 0.10).
- nartaa 224 checkpoint inside Part A: 0.25 (second checkpoint of the same run, weaker by 0.004 public).
- Reader in the clean stack: flat 0.30 (goodpjw's own measured setting).
- A variant is kept only if it beats its parent on the public LB; gold-58 is a regression guard only (nartaa
  was selected on it).

## Results

Submitted 2026-10-10 19:35 UTC (Claude submits since 10-11, ≤ ~3/day): `57051391` CLEAN C2 (clean-blend v5),
`57051394` OAI A3 (oai-blend v3), `57051397` OAI A2 (oai-blend v2).

| Submission | Variant | Public LB | vs parent |
|---|---|---|---|
| (public) | goodpjw notebook = nartaa + reader (parent) | 0.950 | |
| `57051397` | A2: + nartaa 224 checkpoint 0.25 | **0.950** | 0.000 |
| `57051394` | A3: A2 + our R5t leg 0.10 | 0.949 | −0.001 |
| `57051391` | C2 clean: stack + reader 0.30 + ftC 0.20 | **0.945** | clean best (was 0.942; stack + reader public 0.944) |

The gold-58 gains (+0.0027 / +0.0031) did not transfer. Our R5t leg costs ≈ 0.001 again, the same as on the clean
stack (0.943 → 0.942), so it is retired from both tracks. A1 (R5t-only leg) is not submitted, since both its
pieces are already answered. Our best is now 0.950 (A2), the public-fork plateau where ≈ 890 teams sit.

## Final-pick candidates (as of 2026-10-11)

- OAI track: `57051397` (A2, 0.950).
- Clean hedge: `57051391` (C2, 0.945).
- Standing: rank 699 / 5,710 at 0.950. The top-10 % cutoff is 0.950, with 1,035 teams tied at or above it.

## 2026-10-12: the public plateau moved to 0.954

- heliosli restored `heliosli/rsna-knee-compact-v1` (S6: ConvNeXt-small, 4 folds, knee-joint anchors, side
  classifier; code in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`). 18 public notebooks now score a real
  0.954 (search API `best_public_score`). All are nartaa (OAI) + S6 rank-blended 70/30, some with goodpjw's reader
  on top at per-finding weights (0.03-0.35). The S6 leg is external-weights, so it belongs to the OAI track only.
- OAI forum state (10-12): still no host ruling. NIH has barred NDA access from China, Russia, Iran and others since
  2025-04-04 (thread 748276), the same uneven-access argument as the KneeCoT ban. The OAI model's author is rank ~182
  at 0.955; no top-40 team (>= 0.960) has disclosed its data.
- **A5 (OAI track)** = `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` unchanged, as private kernel
  `rsna-knee-oai-954` (`kernels/oai-954`). Its source was audited before the push (embedded base64 = a DICOM reader,
  embedded heliosli sources, offline wheel installs; no network, no secrets). Rule fixed before the run: submit if the
  dev run passes schema/NaN checks; it replaces A2 as the OAI final pick if its public score is >= 0.951.
  Our own legs are not added: every one so far cost ~0.001 on OAI blends.

**A5 result (`57110710`): public 0.954** (+0.004 over A2). It passes the rule (≥ 0.951) and replaces A2 as the OAI
final pick. Final-pick candidates are now OAI `57110710` (A5, 0.954) and clean hedge `57051391` (C2, 0.945).
Standing (2026-10-14): rank 1,387 / 5,980 at 0.954. 1,088 teams are above 0.954, the top-10 % cutoff is 0.956,
and the top of the board is 0.966.

## 2026-10-14: the public plateau moved to 0.956

- heliosli published R7 (`heliosli/rsna-knee-compact-r7cnxfsss`: ConvNeXt-small, 4 folds, crop 352, knee-joint
  anchors, **fastMRI auxiliary training**, refitted Effusion/Baker's stackers; code in the same
  `rsna-orthofoundation-fold0-epoch7-20261004` dataset). nartaa + R7 50/50 rank blend = public 0.956
  (`heliosli/0-957-latest-dataset-score-in-one-click`). ~25 public notebooks now score 0.956 (search API).
- varshithakolupuri added nartaa's 224 checkpoint at 0.35 of the Raptor slot (0.956, ranked higher), then gave that
  checkpoint nartaa's own anatomical mirror TTA (`rsna-0-956-one-free-flip-68-ranks-up`, 0.956, rank 357 → 289).
- Top public notebook 0.957 (`bliverrigpdoge/r7-raptor-knee-centred-convnext`) = the 0.956 base + a private
  knee-centred-crop ConvNeXt/EffNetV2 leg at 0.2; the leg's weights are private, a fork gives 0.956.
- goodpjw's notebook v11/v12 moved to the same base (reader at 1/10 weight; they measured that the reader at full
  weight no longer adds on R7: 0.956 base + reader → 0.954).
- R7 uses fastMRI (application-gated) and nartaa uses OAI, so every 0.956 notebook belongs to the OAI track only.
- Forum (10-14): still no host ruling on OAI (thread 749145 asks for one before the deadline).

### A6 (OAI track, pre-registered before any run)

`kernels/oai-956` = `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` unchanged, as private kernel
`rsna-knee-oai-956`. Audit: internet off, wheels installed with `--no-index` from attached datasets, the only embedded
blob is the DICOM fast-read module already audited for A5. Rule: submit if the dev run passes the schema/NaN checks;
A6 replaces A5 as the OAI final pick if its public score is ≥ 0.955. Our own legs are not added.

### A7 (OAI track, pre-registered before any run)

`kernels/oai-956s6` = `goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` v12 unchanged (private kernel
`rsna-knee-oai-956s6`): the A6 base (nartaa 384 + 224 at 0.35 of the Raptor slot, R7) plus heliosli's S6 at 0.3 of
the compact slot and the goodpjw reader at 1/10 of its v6 weights, both time-guarded. Audit as for A6 (internet off,
`--no-index` wheels, no network calls). Rule: submit if the dev run passes; A7 replaces A6 as the OAI pick only if
its public score is strictly higher (displayed 3 decimals); on a tie A6 stays (fewer members, less runtime risk).

A6 dev run (v1): 3 rows, no NaN, 224 stage ran (25 s), `NOTEBOOK_COMPLETE`. Submitted 2026-10-14 14:21 UTC as `57185288`.

A7 dev run (v1): 3 rows, no NaN; 224, S6 (unscored 0) and reader (3 folds) all ran, `NOTEBOOK_COMPLETE`. Submitted 2026-10-14 14:26 UTC as `57185425`.

### C3 (CLEAN track): ftC5o replaces ftC in clean C2 (rule from ft96c, already passed)

`kernels/clean-c3` = clean-blend v5 with the FT leg glob `ftC/ckpt_ep2.pt` → `ftC5o/ckpt_ep1.pt` (kernel source
`rsna-knee-ft96 → rsna-knee-ft96c`); weight unchanged at 0.20. Submit if the dev run shows the FT leg ran; C3 replaces
C2 as the clean pick only if its public score is strictly higher than 0.945.

### gold-58 diagnostic for the compact legs (`kernels/oai-gold58-r7`, `rsna-knee-oai-gold58-r7`)

R7's manifest says `gold_in_training: 0` and `flip_tta: false`; its worker implements `--flip-tta` with the trainer's
own mirror transform (coronal/axial width flip, sagittal slice reversal; MCL from the plain pass). The kernel runs R7
plain, R7 + flip TTA and S6 on the 58 gold studies (stand-in root), and records per-study time.

**A8 rule (fixed before the run):** A8 = A6 with `--flip-tta` added to R7's worker args. Built and submitted only if
(1) gold-58 macro of R7 flip ≥ R7 plain − 0.002, (2) the A6-style gold blend (0.5 × [0.65 nartaa384 + 0.35 nartaa224]
+ 0.5 × R7, ranks) with flip ≥ the same with plain − 0.001, and (3) the measured flip cost projects the full A6
notebook to ≤ 7.5 h on 1,322 studies. A8 replaces A6 only if its public score is strictly higher.

C3 dev run (v1): FT leg loaded `rsna-knee-ft96c/ftC5o/ckpt_ep1.pt`, blended at 0.2; 3 rows, no NaN. Submitted 2026-10-14 14:45 UTC as `57186035`.

### gold-58 compact-leg results (`rsna-knee-oai-gold58-r7` v1, `analysis/gold_compact.py`)

| Leg / blend | gold-58 macro | note |
|---|---|---|
| nartaa 384 | 0.9229 | selected on gold (optimistic) |
| nartaa 224 | 0.9150 | |
| goodpjw reader | 0.9153 | gold never trained |
| **R7 plain** | **0.9312** | gold never trained; best single leg (Lateral OA 0.884 vs nartaa 0.776, PF OA 0.904 vs 0.860) |
| R7 flip TTA | 0.9317 | +0.0004 [−0.0025, +0.0035], Spearman with plain 0.998 |
| S6 | 0.9267 | |
| A6 blend (R7 plain) | 0.9365 | |
| A6 blend with R7 flip | 0.9366 | +0.0001 [−0.0018, +0.0019] |
| A6 + S6 0.3 in compact slot | 0.9351 | −0.0014 [−0.0050, +0.0015] (A7's S6 part; A7 also has the reader) |

Timing on 58 real studies (2×T4, compile included): R7 plain 757 s, R7 flip 996 s (gpu 12.6 → 24.2 s/study per
worker), S6 675 s. Flip TTA adds ≈ 4.1 s/study wall ≈ +1.5 h on 1,322 studies.

**A8 decision: not submitted.** Rules (1) and (2) pass but the flip pass is a near-copy of the plain pass (ρ 0.998,
+0.0001 in the blend), and +1.5 h puts rule (3) (≤ 7.5 h) at risk; no expected gain for real timeout risk. The
staged notebook (`kernels/oai-956flip`) is kept unpushed.
R7 weight in the A6 blend (information only, not used for tuning): 0.3 → 0.9328, 0.5 → 0.9365, 0.6/0.7 → 0.9366,
0.8 → 0.9348, 1.0 → 0.9312. Flat around 0.5-0.7, so no weight probe is submitted.

Our clean ftC5o (ft96c ep1, gold 0.9242) in the A6 blend, gold-58: Spearman with nartaa 0.931, with R7 0.860 (R7 vs
nartaa 0.865). A6 + ftC5o at 0.10 / 0.15 / 0.20: 0.9365 / 0.9365 / 0.9373 vs 0.9365 (CI at 0.20: [−0.0025, +0.0042]).
No measurable gain, and our legs cost ≈ 0.001 on the LB twice before (A3, clean stack), so it is not submitted.

**A6 result (`57185288`): public 0.956** (+0.002 over A5 0.954). It passes the rule (≥ 0.955) and replaces A5 as the
OAI final pick. A7 (`57185425`) and C3 (`57186035`) are still grading as of 2026-10-14 19:00 UTC.

| Submission | Variant | Public LB | decision |
|---|---|---|---|
| `57185288` | A6: public 0.956 recipe (nartaa 384 + 224 mirror, R7) | **0.956** | OAI final pick (was A5 0.954) |
| `57185425` | A7: A6 + S6 0.3 + reader 1/10 | 0.956 | tie → A6 stays (rule: strictly higher only) |
| `57186035` | C3: clean C2 with ftC5o replacing ftC | **0.946** | clean final pick (was C2 0.945) |

Final-pick candidates (2026-10-14): OAI `57185288` (A6, 0.956) and clean hedge `57186035` (C3, 0.946).

## 2026-10-14 (evening): new public legs

- heliosli `0-9573-latest-public-dataset-score-in-one-click` (public 0.957): nartaa 384 + nartaa 224 (**one view**, no mirror)
  at 0.35 of the Raptor slot + R7 (the same bundle, re-published as `heliosli/rsna-knee-r6-frozen-fe055406`), 50/50. It is A6
  without the 224 mirror view. The gold-58 "A6[r7_plain]" row above used the plain 224 predictions, so it is this recipe: 0.9365.
- nartaa `rsna-knee-s6fall-weights` (public 0.950 alone, `rsna-knee-s6fall-solo`): heliosli's S6 architecture retrained on
  4,263 studies with nartaa's JEV-blend report labels, **gold-58 excluded, no OAI**; it uses heliosli's SKM-TEA anatomy model for
  the knee-centred views, so it goes to the OAI track (as S6 did).

### A10 (OAI, pre-registered): `kernels/oai-957` = heliosli 0.9573 notebook unchanged (`rsna-knee-oai-957`)

Audit: internet off, only `--no-index` installs. Submit if the dev run passes; A10 replaces A6 as the OAI pick only if its
public score is strictly higher than 0.956.

### s6fall gold-58 diagnostic (`kernels/oai-gold58-s6fall`) and A11 rule (fixed before the run)

nartaa's script runs unchanged as a module through its own `K1_KAGGLE_ROOT` hook, with the gold-58 stand-in as the
competition folder; only its `prepare_inputs()` + `run_b()` (B = s6fall) run.
**A11** = A10 + s6fall at 0.3 of the compact slot (compact = 0.7 rank(R7) + 0.3 rank(s6fall); the slot design goodpjw used for
S6). Built only if (1) gold-58 of A11 ≥ A10's 0.9365, (2) mean Spearman(s6fall, R7) on gold < 0.95, and (3) the measured
s6fall cost keeps the notebook ≤ 7.5 h on 1,322 studies. A11 replaces the OAI pick only if strictly higher on the public LB.

A10 v1 submit was refused ("non-versioned dataset heliosli/rsna-knee-r6-frozen-fe055406"). v2 points `C_SLUG` at
`heliosli/rsna-knee-compact-r7cnxfsss`, which holds the same bundle (`8c7b666e27dc0e8f`, checked by the notebook's own
manifest + sha256 asserts); nothing else changes.

### s6fall gold-58 result (`rsna-knee-oai-gold58-s6fall` v1, 274 s for 58 studies on 2×T4)

s6fall gold-58 **0.9314** (R7 0.9312). Mean Spearman: s6fall–R7 0.871, s6fall–nartaa 0.879, s6fall–S6 0.866 (R7–nartaa 0.865):
a third leg as strong as R7 and as far from both legs as they are from each other.
**A11 (s6fall at 0.3 of the compact slot): 0.9362 vs A10 0.9365 → rule (1) fails, not built.**
After seeing that, the equal three-way form was computed: final = 2/3 × A10 + 1/3 × rank(s6fall) = 0.9406 (+0.0042,
CI [−0.0020, +0.0105], 9/12 findings up; s6fall weight 0.20 / 0.25 / 0.33 / 0.40 / 0.50 → 0.9369 / 0.9381 / 0.9406 / 0.9406 /
0.9401). **This form was chosen after the gold read, so gold does not validate it**; its rationale is structural (three legs of
equal gold strength and equal pairwise correlation → equal weights), and the LB decides.

### A12 (OAI): `kernels/oai-957s6f` = A10 v2 + s6fall as a third equal leg (`rsna-knee-oai-957s6f`)

nartaa's s6fall script (unchanged, embedded) is loaded as a module after A10's blend; only `prepare_inputs()` + `run_b()` run.
Final = 2/3 × A10 value + 1/3 × rank(s6fall); s6fall starts only if ≥ 70 min remain and is capped to leave 25 min; any failure
keeps A10's file; a study s6fall did not score keeps its A10 value; < 90 % scored → A10 kept. Submit if the dev run shows the
s6fall stage and blend completed. A12 becomes the OAI pick only if its public score is strictly higher than the best of A6/A10.

A10 v2 and A12 v1 dev runs passed (A12: s6fall scored 3/3 in 64 s, three-way blend written). Submitted 2026-10-14 23:23 UTC: A10 `57197655`, A12 `57197656`.
