# Beating the ~0.94 public ensemble — study, experiments, and the path forward

*Session of 2026-09-12. Full per-experiment detail, pre-registered rules and raw numbers are in
[`experiment_ledger.md`](experiment_ledger.md); this report is the synthesis. Stopped because the
Kaggle GPU quota ran out — see "Status and what is left". **Updated 2026-09-17: E8's public LB score
came back at 0.939** and is filled in below.*

## TL;DR

- **The 0.94 public stack is, underneath, one strong representation plus diversity.** The
  CoAtNet-RMLP-2 @384 2.5-D attention-MIL family ("Raptor" checkpoints + a residual-gated CoAtNet)
  scores 0.91–0.92 per checkpoint on the 58 gold studies, far above the DINOv2 (0.84) and RadImageNet
  (0.85) arms. Everything else in the stack adds diversity on top of it.
- **The kind of diversity that pays is pipeline/label diversity, not more backbones or checkpoints.**
  Public checkpoints of the same Raptor pipeline are near-duplicates (Spearman 0.91–0.94) and adding
  one more changes nothing; the residual-gated CoAtNet — different author, labels and preprocessing —
  is the only strong model that adds on gold-58 (+0.007). The public author's own panel found that
  swapping backbones on the same labels/views is worth +0.001 on the LB.
- **Several public-notebook practices are wasteful or risky:** the "v5-reverse" arm is a mathematical
  no-op (window order doesn't matter to the pooling; max difference 2.6e-5) costing ~45 min of the
  9 h budget; the latest 0.939→0.941 gains are per-target weights probed on the public LB; five public
  label tables copy the gold labels verbatim, silently invalidating any gold-58 check that uses them.
- **Tested and rejected by pre-registered rules:** an unused public checkpoint (v4); end-to-end
  retraining on better report labels (the teacher effect is real but narrow — Synovitis +0.067 — and
  the existing family absorbs almost all of it); canonical anatomical orientation (E5b: far worse on a
  2,000-study subset, −0.097 hold-out macro — though mostly on findings orientation cannot affect, so
  it reads as a training-dynamics side effect of that transform rather than a clean verdict).
- **Built, validated on Kaggle, and in the repo:** a clean CoAtNet inference engine that reproduces the
  public notebooks' predictions exactly and decodes each study once for all arms; the exact recipe of
  the public training corpus; a corpus trainer (fp16, grad-checkpointing, single-GPU A/B pairs or DDP)
  proven on 2×T4; and an offline submission pipeline. 82 unit tests.
- **The clean baseline scores 0.939 on the public LB (E8, submission `56184308`)** — level with the
  0.939–0.941 public notebooks while dropping the entire DINO/RadImageNet chain and their LB-probed
  per-target weights, and using ≈ 2.5–3 h of the 9 h limit. Per the pre-registered rule (≥ 0.935),
  **the chain stays out**; 0.939 is now the reference every later addition is measured against.
- **Not yet shown: a score above 0.94.** The new-model stage (E6) — the one lever with evidence
  behind it — was not started because the GPU quota ran out.

## Why the 0.94 stack works

| Evidence | Finding |
|---|---|
| Gold-58, clean hold-out predictions (A01, E1) | CoAtNet checkpoints 0.912–0.920; DINOv2 OOF 0.840; RadImageNet OOF 0.854 |
| Correlation between arms (E1) | Raptor checkpoints 0.91–0.94 with each other; residual-gated ≈ 0.85; DINO/Rad 0.74–0.79 |
| Fusion on gold-58 (E1) | Raptor family 0.923 → + residual-gated 0.930; equal weights = the hand-tuned weights (−0.001) |
| Public LB history | one CoAtNet 0.914 → CoAtNet family ≈ 0.92 → + DINO/Rad chain 0.937–0.941 |
| Gold-58 limits (A01, A04) | the chain's LB gain is invisible on gold-58 (Δ −0.003, 90% CI [−0.014, +0.008]) → gold-58 can guard against large regressions, not tune ±0.01 |
| Efficiency LB | the #1 *most efficient* team is also at 0.954 → 0.95+ does not need a 9-hour mega-ensemble |

The remaining headroom is concentrated in a few findings (family, gold-58): Synovitis 0.83, PF OA 0.85,
Lateral OA 0.86, Lateral Meniscus 0.88 — while the medial counterparts sit at 0.97–0.98.

## Experiments

| ID | Hypothesis | Data | Result | Conclusion | Next stage? |
|---|---|---|---|---|---|
| A01–A04 | (analysis) what each arm is worth, how correlated | gold-58, public OOF files | see above | CoAtNet family dominates; pipeline diversity is what adds | — |
| A02b | better report labels exist | 12 public label tables vs gold-58 | 5 tables leak gold; best clean 0.899 vs ours 0.855 (+0.042, CI [+0.026, +0.059]) | label quality varies a lot, most on Synovitis | yes → E2 |
| E1 | an unused public checkpoint (v4) strengthens the family; equal weights lose nothing | gold-58 from DICOM, 6 checkpoints | v4 0.914 but ρ 0.94 with v5, no gain; v5-reverse identical to v5; equal weights −0.001 | H3 rejected; drop v4 and v5-reverse | no |
| E2 | the teacher changes what a frozen CoAtNet predicts | v8 features of 4,407 studies; head retrained 3 ways, gold excluded | best teacher +0.006 macro (P>0 0.86); Synovitis +0.067; OA −0.015 | pre-registered rule not met | no (E3 not launched) |
| A05 | relabelled-head diversity is real and survives fusion | E2 predictions + family | different-teacher head +0.0069 vs same-teacher head (P>0 0.98); inside the family only +0.0016 (P>0 0.76) | real but absorbed | no |
| E5a | orientation varies; knee side is recoverable | headers of 20,792 slot series | orientation 100% standard (right knees are always mirror images); side: tag on 50%, geometry agrees 0.924 but is unreliable on untagged sites | canonicalisation = mirror right knees; side noisy for ~half | gate passed → E5b |
| E5c | the public corpus has a recoverable exact recipe | 640 preprocessing variants × 29 slots | span 15–85%, per-series 2–98th pct, 140 mm, area resize → mean diff 0.00 | corpus-trained models can be served from DICOM exactly | — |
| E5b | one canonical anatomical orientation lifts the lateral findings | 2,000 studies × 3 epochs, CoAtNet-2, raw vs canonical, same teacher | canonical − raw: hold-out macro −0.097 (CI [−0.114, −0.080]); lateral pair −0.021 hold-out / −0.070 gold; side-agnostic findings −0.12 / −0.15 | rejected (pre-registered "clearly negative"); damage pattern points to a training side effect, not anatomy | no |
| E8 | clean pipeline without the DINO/Rad chain stays near the public stack | public LB | **0.939** (submission `56184308`) vs 0.939–0.941 for the public chain notebooks | top pre-registered band (≥ 0.935) → chain stays out; 0.939 is the reference baseline | yes → E6 measured against it |

Engineering checks (not hypotheses): **E3 smoke** (v1 host-OOM SIGKILL → fixed; v2 passed on 2×T4),
**E4** (repo inference engine reproduces the public predictions to ≤ 5e-4, same gold-58 AUCs),
**E7** (residual-gated arm reproduces its shipped ranks, Spearman 0.9996), **E8 dry run** (full
offline submission passes on the public test stub).

GPU used: ≈ 5 h of Kaggle T4 sessions (E1 16 min, E2a 29 min, smokes ~12 min, E4 24 min, E5b 98 min,
E7 3 min, submission dry run 2 min, plus session start-up), and ≈ 40 min of free CPU kernels.

## What was built

All under `src/rsna_knee/mil/` (see `src/rsna_knee/README.md` for the module map), tested by 34 new
unit tests (82 total), and shipped to Kaggle as the private dataset `tranbadat/rsna-knee-code`:

| Piece | What it does | Validated |
|---|---|---|
| `recipes`, `volume`, `windows`, `model` | DICOM → slice stacks → windows → CoAtNet MIL, state-dict compatible with the public checkpoints | E4 parity ≤ 5e-4 |
| `infer` | all arms, both GPUs, one decode per study shared by every recipe | E4 + unit tests |
| `corpus` | lazy access to the public 44×336 corpus; `CORPUS44_336` is its exact recipe | E5c exact match |
| `orientation` | canonical anatomical orientation per slot, tag-first side resolution (kept as an option; E5b rejected it for training) | unit tests; E5b |
| `teachers` | public report-label tables, with measured gold agreement and the leak list | A02b |
| `train`, `plan` | corpus trainer (fp16, grad-ckpt, single-GPU A/B pairs or DDP), stage isolation | E3 smoke v2, E5b on 2×T4 |
| `submit` + `scripts/kaggle_submit.py` | offline submission: benchmark-first, fixed-weight rank fusion, arms that fail are dropped | E8 dry run |
| `scripts/kaggle_mil_train.py` | one-file Kaggle launcher for training plans | compile + unit tests |

Runtime on 2×T4 for ≈ 1,322 test studies: public CoAtNet family ≤ 106 min (measured before shared
decode), residual-gated ≈ 64 min → the clean pipeline uses ≈ 2.5–3 h of the 9 h limit, leaving room for
one or two new models.

## The recommended path past 0.94

1. **Base — settled:** the clean CoAtNet pipeline (public v5 / v10 / v8 at equal weight +
   residual-gated at 0.4). Gold-58 0.930, **public LB 0.939** (E8), ~3 h, no LB-fitted weights. The
   pre-registered reading fired in the top band, so the DINO/Rad chain does **not** come back: it and
   the LB-probed per-target weights together are worth ≤ ~0.002 over this family, at several hours of
   runtime. That leaves ≈ 6 h of the 9 h limit for new arms.
2. **Main lever:** one new strong CoAtNet-class model whose *pipeline* differs from the Raptor family —
   the only kind of addition that has measurably helped. Train it on the full public corpus with the
   validated trainer (≈ 5–7 h on 2×T4, one session), serve it with the exact corpus recipe, and add it
   at a fixed equal member weight: gold-58 as a regression guard, one LB submission as the test. The
   corpus trainer's raw-window recipe with the best report-label teacher is the ready-to-run
   configuration; a different slot layout / crop is the next view to test on a subset first.
   Canonical orientation is off the table unless a follow-up isolates the sagittal channel-order change
   (e.g. mirror coronal/axial only) on a subset.
3. **Don't:** add more Raptor-pipeline checkpoints, run the v5-reverse view, swap backbones on the same
   labels and views, fit per-target weights on the public LB or on gold-58, or validate on the public
   label tables that contain gold labels.

## Status and what is left (quota exhausted)

- **E8** scored **0.939** on the public LB (read 2026-09-17, submission `56184308`) — recorded above,
  in the ledger and in `README.md`. Decision taken: no DINO/Rad chain.
- **E6** (the full-data new model) was not started: it needs GPU quota. With E5b negative, the
  pre-registered branch is the view-diversity model (raw windows, best teacher, subset-first for any
  new view).
- Kaggle kernels created this session (all private): e1 coat panel, e2a features, e2b relabel heads,
  e3 smokes, e4 infer check, e5a orientation, e5b canon A/B, e5c corpus recipe, e7 resgated check,
  clean-submit (submitted as E8), plus a trivial auth probe that can be deleted.
