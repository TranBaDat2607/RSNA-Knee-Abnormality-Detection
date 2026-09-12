# Experiment ledger — beating the ~0.94 public ensemble

Goal: find the highest-impact, simplest, most defensible path past the ~0.94 public
ensemble (`manderson240/cohezion-rsna-knee-sota-ensemble` and its 0.939–0.941 siblings)
without tuning against the public leaderboard or the 58-study gold set.

Each entry: hypothesis → experiment → data → result → conclusion → next stage?

---

## 0. Situation (2026-09-12)

- Public LB top: **0.955** (several teams 0.95+). Best public notebooks: 0.939–0.941. Ours: 0.910.
- Final submission deadline 2026-10-22. Submissions are inference-only notebooks, ≤9 h on 2×T4,
  no internet. Test set ≈1,300 studies. Kaggle GPUs available to us: T4 (and P100, unsupported
  by this image's PyTorch).
- Validation reality: 58 gold studies (macro-AUC SE ≈ 0.02–0.03). Weak labels exist for all
  4,349 other training studies (LLM report labels, ~0.85–0.87 macro agreement with gold).
- **Efficiency leaderboard (public, 2026-09-12):** the #1 *most efficient* team (runtime-weighted)
  is also at **0.954** public; #7 efficient is 0.952, #10 is 0.950. → 0.95+ does **not** require a
  9-hour mega-ensemble. The gap above the public 0.94 stack is a better core model, not more members.

### What the 0.94 stack is (verified by reading the notebooks and logs)

| Stage | Component | Input view | Labels it was trained on |
|---|---|---|---|
| 1 | DINOv2-S, 20 members (5 folds × 4 seeds), per-finding window pooling | 6 slots × 12 slices, 336 px, 130 mm | rule-based multilingual report extractor |
| 2 | DINOv3-S "A5" 5 folds, blended at 0.45 | 6 slots × 16 slices, 336 px | community LLM labels |
| 3 | RadImageNet ResNet-50 frozen encoder + 3 head families (v15, v52, E13), logistic calibrator on 7 findings | 3–4 slots × 8 slices, 224 px | "three public report-label teachers" |
| 4 | CoAtNet-RMLP-2 @384, 2.5D triplet windows, per-finding attention MIL ("Raptor"): v5 (+reverse view), v8, v10; plus residual-gated CoAtNet e4/e6/e8 at 0.40 inside the family | 5 slots × 44–64 slices, 140 mm | dreaddevelopment's LLM soft labels / mattiaangeli's labels |
| out | final blend: CoAtNet family 0.60 (0.75–1.00 on ACL, menisci, lateral OA, fracture — LB-probed) | | |

### Why it works (evidence, not guesses)

- **A01 (gold-58, local, free):** residual-gated CoAtNet top-3 = **0.909–0.912** macro; DINOv2 OOF
  0.840; RadImageNet v15 OOF 0.854; E11 0.826. The CoAtNet 2.5D/384/many-window/per-finding-
  attention recipe is by far the strongest single representation.
- **dreaddevelopment's own panel (public notebook notes):** CoAtNet-384 alone = 0.914 LB; adding
  Swin-B-384 / EffNetV2-L-480 trained on the *same labels and views* moved LB by only **+0.001**
  (the +0.010 seen on a 45-study gold set was noise). → *Backbone diversity with shared labels
  and views is nearly worthless.*
- The public chain went 0.914 (one CoAtNet) → ~0.92 (CoAtNet family) → **0.94** by fusing with the
  weak (0.84–0.85 gold) DINO/Rad chain, which differs in **label source, slot layout, crop and
  resolution**. → *The diversity that paid was pipeline/label diversity, not architecture.*
- On gold-58 that same chain fusion is invisible (coat 0.909 vs coat 0.6 + chain 0.4 → 0.906,
  90% CI [−0.014, +0.008]). → *Gold-58 cannot resolve ±0.01 blend effects; it is a guard, not a
  tuner.*
- The latest public gains (0.939 → 0.941) are single-target outer-routing weights probed on the
  public LB. → *Not a direction we will follow.*

### A02 — label source agreement with gold (57 studies with all sources)

| Source | macro | weakest findings |
|---|---:|---|
| ours (gpt-5.4-mini) | 0.853 | Synovitis 0.63, Lat OA 0.78, Fracture 0.81, Contusion 0.82 |
| pilkwang v2 | 0.870 | Synovitis 0.69, Contusion 0.77, Lat OA 0.79 |
| rank-average of both | 0.873 | Synovitis 0.63 |

Image models beat every report-label source on Effusion (0.95 vs 0.83), Synovitis (0.80 vs 0.69),
Contusion (0.92 vs 0.82) and Medial OA (0.98 vs 0.94) → gold labels are image-based; report labels
carry systematic (not random) noise on those findings.

**A02b — every public label table (free, local).** Five public tables (barun2104, rayanbabur,
tasmeemreza, zaidaliiq1000, yunusgmsoy-4-source) **copy the 58 gold labels verbatim** (AUC 1.000,
exact match 1.000) → unusable for any gold-58 evaluation and a leakage trap for anyone validating
on gold with them. Among clean sources (gold agreement, macro):

| Source | macro | Synovitis | Lat OA | PF OA | Effusion | Contusion | Fracture |
|---|---:|---:|---:|---:|---:|---:|---:|
| flight0234 hybrid v4 | **0.899** | 0.79 | 0.83 | 0.90 | 0.88 | 0.86 | 0.87 |
| rank-ens top-5 | 0.897 | 0.79 | 0.82 | 0.90 | 0.87 | 0.86 | 0.84 |
| stevenleehans v4 blend | 0.893 | 0.79 | 0.83 | 0.90 | 0.88 | 0.86 | 0.79 |
| yunusgmsoy | 0.880 | 0.68 | 0.84 | 0.88 | 0.84 | 0.83 | 0.88 |
| pilkwang v2 | 0.870 | 0.69 | 0.79 | 0.89 | 0.83 | 0.77 | 0.87 |
| ours (gpt-5.4-mini) | 0.855 | **0.62** | 0.79 | 0.85 | 0.83 | 0.83 | 0.82 |
| Qwen2.5-32B (laymond) | 0.843 | 0.66 | 0.85 | 0.71 | 0.84 | 0.72 | 0.75 |
| GPT-5.6-sol (lixin73) | 0.835 | 0.68 | 0.82 | 0.81 | 0.70 | 0.81 | 0.80 |

Best clean source − ours = **+0.042** macro (bootstrap 90% CI [+0.026, +0.059], P>0 = 1.00) — the one
label-quality difference on gold-58 that is well outside noise. Rank-ensembling sources does not beat
the best single source (they share lineage).

**A03 — the Raptor CoAtNets' own training labels** (dreaddevelopment soft labels, gold rows
removed) versus the best teachers on the 4,349 non-gold studies (per-target Spearman): 0.82–0.95 on
most findings, but **Synovitis 0.52, Fracture 0.55–0.61, Contusion 0.75, MCL 0.78, Baker's 0.79**.
The strongest image family was trained on labels that disagree most with the best-agreement
teachers on exactly the weakest gold findings.

### A03 — remaining headroom on gold-58 (best single CoAtNet, per finding)

Synovitis 0.80 · PF OA 0.84 · Lateral OA 0.85 · MCL 0.89 · Lateral Meniscus 0.90 · Fracture 0.91 ·
Contusion 0.92 · the rest ≥ 0.95.

---

## Hypotheses, ranked by expected impact × testability

| ID | Hypothesis | Why plausible | Cheap test |
|---|---|---|---|
| H1 | A **strong (CoAtNet-class) model with an independent pipeline** — different label teachers *and* a different view — adds the same kind of decorrelated signal the weak chain adds, but from a 0.90-class model | the only large historical gains came from pipeline/label diversity | subset training A/B, correlation + blend gain vs existing CoAtNet family on gold-58 |
| H2 | A **higher-effective-resolution / joint-centred view** improves the resolution-limited findings (menisci, lateral/PF OA, MCL) | frontier routes exactly those findings to the highest-res family; centre 140 mm crops at 336 px discard native detail | same-labels, same-backbone, same-budget A/B on a subset; weak-label hold-out (~500 studies) + gold-58 |
| H3 | **Unused independent checkpoint** (dreaddevelopment v4 widedense, 0.914 LB alone) strengthens the CoAtNet family at a fixed equal weight | independent training run of the strongest recipe | gold-58 panel of every public CoAtNet checkpoint (inference only) |
| H4 | Better **label targets on the systematically-noisy findings** (Synovitis, Effusion, Contusion, OA) | A02 | label-agreement AUC (free) then a head-only probe |
| H5 | Principled (equal/structural) fusion loses nothing vs LB-probed weights | gold-58 can't see LB-probed differences | falls out of the H3 panel |

Rejected up front: more backbones on the same labels/views (dreaddevelopment's evidence); per-target
weight probing on the public LB; per-target weight fitting on gold-58.

---

## Experiments

### E1 — CoAtNet-family gold-58 panel + runtime + T4 training throughput  *(queued)*

- **Hypotheses:** H3, H5; feasibility numbers for H1/H2.
- **Experiment:** Kaggle GPU (T4) script. (a) Run every public CoAtNet checkpoint (v5, v5-reverse,
  v8, v10, v4, v4-swa) on the 58 gold studies straight from DICOM with each checkpoint's exact
  preprocessing; time DICOM build vs model per study. (b) Re-run v4/v8 from the public 336 px
  corpus to measure what the corpus loses vs DICOM. (c) Benchmark fwd+bwd fp16 throughput at 384 px
  for candidate backbones.
- **Data:** gold-58 only (all checkpoints were trained on the other 4,349 studies → clean hold-out).
- **Decision rule:** v4 enters the final family only if it is not worse than the median public
  checkpoint on gold-58 and its correlation with v5 is < 0.95. Throughput decides whether a
  CoAtNet-2 full training (H1) fits in ≤ 2 Kaggle GPU sessions.
- **Result (kernel `tranbadat/rsna-knee-e1-coat-panel` v1, 16 min on T4):**

  | Checkpoint (gold-58 macro) | v5 maxspan | v5 *reverse view* | v10 dense | v8 native384 | v8 via 336 corpus | v4 widedense | v4 swa | resgated e4/6/8 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | macro AUC | **0.920** | 0.920 | 0.917 | 0.912 | 0.917 | 0.913 | 0.914 | 0.912 |

  - **The frontier's "v5-reverse" arm is a no-op.** Max |p(v5) − p(v5-reverse)| = 2.6e-5: the
    per-finding attention pool is permutation-invariant, so reversing window order changes
    nothing. The 0.939–0.941 notebooks spend a full CoAtNet pass (~20–40 min of test runtime)
    re-computing v5; their "0.60 v5 + 0.10 reverse" is simply 0.70 on v5.
  - Public Raptor checkpoints are near-duplicates of each other (mean per-target Spearman
    0.91–0.94; v4 vs v5 0.936). The residual-gated CoAtNet — a different author, labels and
    preprocessing — is the only strong model that is meaningfully decorrelated (ρ ≈ 0.85), and it
    is the only addition gold-58 can see: Raptor family 0.923 → +resgated 0.930 (+0.007).
  - Equal-weight families match the frontier's hand-tuned inner weights (−0.001, 90% CI
    [−0.005, +0.003]); adding v4 changes nothing (−0.0006). Adding single-fold DINO/Rad OOF at 0.4
    costs −0.014 (CI [−0.023, −0.005]) on gold-58 (note: OOF fold models are weaker than the
    5-fold/20-member ensembles used at test time).
  - Corpus parity: v8 from the public 336 px corpus scores 0.917 vs 0.912 from DICOM (Spearman
    0.96) → the corpus is a valid, 25× faster substrate for experiments (0.07 s/study read vs
    0.6–1.8 s/study DICOM build).
  - T4 fp16 throughput @384 (fwd+bwd, img/s, grad-ckpt on / off): CoAtNet-2 13.1 / OOM ·
    ConvNeXt-S 36 / 45 · ConvNeXt-T 59 / 75 · EffNetV2-S 51 / 68 · MaxViT-T 21 / 28 ·
    EffNetV2-M 31 / 41. Inference CoAtNet-2: 55 img/s. → a 12-epoch CoAtNet-2 run over 4,349
    studies × 12 windows ≈ 7 h on 2×T4 (fits one 12 h session); ConvNeXt-S ≈ 2.2 h.
- **Conclusion:** H3 **rejected** (v4 is redundant). H5 **supported** (principled equal weights lose
  nothing gold-58 can see). The evidence for H1 strengthens: among strong models, *pipeline*
  diversity (resgated) is what adds; checkpoint diversity within one pipeline does not.
- **Next stage:** no. Drop v5-reverse and v4 from any final pipeline; spend that runtime on a
  genuinely different strong model.

### Runtime budget (from measured per-study timings, not yet a hidden-test measurement)

Hidden test ≈ 1,322 studies, 9 h on 2×T4.

| Step | Measured | Implication |
|---|---|---|
| DICOM → stack (one recipe) | 0.6–1.8 s/study, CPU (E1) | the larger half of a CoAtNet arm's cost |
| CoAtNet-2 eval, 42–62 windows | 0.85–1.4 s/study, one T4 (E1) | |
| v8 features from the pre-decoded corpus | 2.58 studies/s on 2×T4 (E2a, 4,407 studies in 29 min) | model-only cost when decode is shared |
| public 0.94 notebooks' Raptor block | 4 sequential single-GPU passes, each re-decoding (v5, v10, **v5-reverse**, v8) | ≈ 3 h, ≈ 45 min of it the no-op reverse pass |
| same 3 useful checkpoints via `rsna_knee.mil.infer` | one decode per recipe, studies sharded over both GPUs | ≈ 1 h 10 min estimated |

→ Deduplicating the Raptor block frees roughly 1.5–2 h of the 9 h budget: enough for one or two
new CoAtNet-class arms without dropping anything useful.

### E2 — does the label teacher change what a strong CoAtNet representation predicts?  *(running)*

- **Hypothesis (H4 → H1):** training targets from the best-agreement teacher (flight0234 hybrid, or
  the top-5 mean) improve gold-58 on the findings where the Raptor teacher disagrees with it
  (Synovitis, Fracture, Contusion, MCL), even with a frozen backbone.
- **Experiment:** E2a (GPU) extracts frozen Raptor-v8 features for all 4,407 studies from the corpus
  (42 windows × 1024-d). E2b (CPU) trains the identical per-finding attention head three times —
  A Raptor teacher (control), B flight hybrid, C top-5 mean — 5 seeds × 15 epochs, window dropout,
  gold excluded from training; scored on gold-58 with paired bootstrap.
- **Bias:** the backbone was fine-tuned on A, so the comparison favours A; a B/C gain is conservative.
- **Decision rule:** B or C beats A by ≥ +0.01 macro with P>0 ≥ 0.9, or by ≥ +0.03 on ≥ 2 of the
  targeted findings without losing > 0.02 elsewhere → labels are a real lever → next stage
  trains a CoAtNet end-to-end on the winning teacher (H1). Otherwise labels are not the lever.
- **Result (E2a `rsna-knee-e2a-v8-features` 29 min GPU; E2b `rsna-knee-e2b-relabel-heads` 23 min CPU):**

  | Head on frozen v8 features | macro | Synovitis | Contusion | Fracture | MCL | Lat OA | PF OA | Effusion |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | original checkpoint head | 0.917 | 0.803 | 0.937 | 0.915 | 0.986 | 0.805 | 0.842 | 0.990 |
  | A Raptor teacher (recipe control) | 0.916 | 0.771 | 0.949 | 0.907 | 0.986 | 0.832 | 0.861 | 0.974 |
  | B flight hybrid | **0.922** | **0.838** | 0.958 | 0.900 | 0.989 | 0.816 | 0.846 | 0.988 |
  | C top-5 mean | 0.921 | 0.834 | 0.953 | 0.906 | 0.984 | 0.820 | 0.844 | 0.992 |

  The head recipe reproduces the checkpoint (A − original = −0.001, CI [−0.009, +0.007]). B − A =
  +0.006 (90% CI [−0.003, +0.016], P>0 0.86); C − A = +0.005 (P>0 0.83). Only Synovitis moves by
  ≥ 0.03 (+0.067); Lateral/PF OA lose 0.015.
- **Conclusion: the pre-registered rule is not met → E3 is not launched.** The teacher is a strong
  lever for one finding (Synovitis — exactly where A03 showed the Raptor teacher disagrees most), not
  a broad one.

**A05 — is the relabelled head's diversity real, and does it survive fusion? (free, local)**

| Gold-58 macro, paired bootstrap | Δ | 90% CI | P>0 |
|---|---:|---|---:|
| original head + 2nd head, **same** teacher (A) − original | +0.0008 | [−0.004, +0.006] | 0.59 |
| original head + head on **different** teacher (B) − original | +0.0077 | [+0.002, +0.014] | 0.99 |
| (orig + B) − (orig + A) | +0.0069 | [+0.001, +0.013] | 0.98 |
| CoAtNet family + B as 4th Raptor member − family | +0.0016 | [−0.002, +0.005] | 0.76 |
| CoAtNet family + B − family + A | +0.0009 | [−0.002, +0.004] | 0.70 |

→ Teacher diversity is genuine signal on a single backbone (a second same-teacher head is worth
nothing, a different-teacher head is worth +0.007), but the existing family (3 Raptor checkpoints +
residual-gated CoAtNet, gold-58 0.929) already absorbs almost all of it. **Relabelled heads are not
the path past 0.94.** Kept as a near-free optional member (it reuses the v8 features the family
already computes), not as a strategy.

### E5 — one canonical anatomical orientation for every window  *(pre-registered)*

- **Hypothesis (H6):** the CoAtNet family's weakest findings after Synovitis are the *lateral* ones
  (gold-58 family: Lateral OA 0.857, Lateral Meniscus 0.883 vs Medial OA 0.981, Medial Meniscus
  0.969). The public corpus keeps slices as acquired: a right knee's coronal/axial image is the
  mirror of a left knee's, and in-plane directions vary with the acquisition. With an
  order-invariant window pool and ~4,300 weakly labelled studies, the network must learn every
  finding in every orientation, and the rarer lateral findings (fewer positives to learn both
  mirror images from) suffer most. Mapping every window onto one canonical orientation — lateral
  always on the same image side, anterior and superior fixed — should lift the lateral findings
  without hurting the rest.
- **Supporting hint (not evidence):** the residual-gated CoAtNet canonicalises sagittal
  orientation and is the family's best Lateral Meniscus model (0.898 vs 0.850–0.878).
- **E5a (CPU, free):** header pass over the corpus's chosen series: in-plane direction cosines per
  slot, knee side (DICOM tag vs patient-x sign), and a 25-study parity check of the repo's
  `build_volume` against the corpus. **Gate:** E5b runs only if (i) orientation actually varies (a
  meaningful share of studies would be flipped) and (ii) side is resolvable for ≥ 90% of studies
  with tag/geometry agreement ≥ 0.9.
- **E5a result (`rsna-knee-e5a-orientation`, 4 min CPU):**
  - In-plane orientation **never varies**: 100% of chosen series are stored in the standard DICOM
    orientation (sagittal rows→posterior/cols→inferior, coronal →patient-left/inferior, axial
    →patient-left/posterior). So coronal/axial columns always run toward the patient's *left*: a
    right knee's lateral compartment is on the opposite image side from a left knee's, for every
    study. Canonicalisation reduces to mirroring coronal/axial windows of right knees and reversing
    the sagittal channel order of left knees — about half of all studies are transformed.
  - Side: DICOM tag on 49.9% of studies (R 1,110 / L 1,091); patient-x geometry on 95.0%;
    tag/geometry agreement 0.924 on 2,075 studies → combined coverage 0.979. **Gate passed.**
    Caveat found: geometry is reliable for tagged right knees (0.99) but not for tagged left knees
    (0.85); the disagreements sit near the scanner isocentre (median |x| 21 mm vs 112 mm when they
    agree), and untagged studies skew R (1,390 vs 722). E5b uses the 5 mm threshold as
    pre-registered, so a few percent of studies are mirrored the wrong way — noise that biases the
    test *against* canonicalisation. A deployed version should require a larger |x| margin.
  - Parity check **failed**: the repo's `build_volume` (a port of the public *inference*
    preprocessing) does not reproduce the public corpus stacks — mean |diff| 26.7/255 (range 18–33),
    though exactly the same slots are filled. The corpus was built with different parameters. This
    does not affect E5b (both jobs read the corpus), but any model trained on the corpus must be
    served with the corpus's own preprocessing → **E5c** (CPU) searches ordering × span ×
    percentiles × per-slice/series window × crop × interpolation on 6 studies for the exact recipe.
  - **E5c result (`rsna-knee-e5c-corpus-recipe`, 12 min CPU):** exhaustive search over 640
    preprocessing variants on 29 slots of 6 studies. One variant reproduces the corpus **exactly**
    (mean |diff| 0.00 on every slot): geometry slice ordering, **span 15–85%**, per-series 2–98th
    percentile window, 140 mm crop, area resize. Next best: linear resize 0.47; per-slice window 7.2;
    1–99th percentile 9.4. The corpus is the older narrow-span build, not the 6–94% layout of the
    native44 checkpoint. `rsna_knee.mil.recipes.CORPUS44_336` is fixed accordingly (and pinned by a
    unit test), so any model trained on the corpus can be served from DICOM with identical inputs.
  - **Side-reliability follow-up (local, before E5b's result):** raising the patient-x threshold
    barely improves agreement on tagged studies (0.924 at 5 mm → 0.967 at 60 mm → 0.982 at 80 mm)
    while untagged calls become almost all "R" (66% → 95% → 99%). The untagged half of the corpus
    comes from sites whose x coordinate does not encode side, so **only the tagged ~50% of studies
    get a trustworthy side**; E5b's canonical arm mirrors many untagged studies essentially at
    random. Amendment to E5b's interpretation, recorded before its result: a *positive* result is
    strong evidence (it survives heavy side noise); a *null* result is **inconclusive**, not a
    rejection — the next step would be image-based side estimation, not dropping the idea.
- **E5b (GPU, subset A/B):** two identical CoAtNet-2 jobs side by side on the two T4s — same
  2,000 studies, same flight-hybrid teacher, 3 epochs — differing only in canonicalised vs raw
  windows. Metrics: gold-58 and the 500-study weak-label hold-out, macro and the four
  medial/lateral findings.
- **Decision rule:** canonicalisation is adopted for the new model if the mean of Lateral Meniscus +
  Lateral OA improves by ≥ +0.02 on the hold-out *and* is not worse on gold-58, with hold-out macro
  not lower by more than 0.005. Otherwise it is rejected.
- **E5b result (`rsna-knee-e5b-canon-ab`, 98 min on 2×T4, both jobs exit 0):** canonical transforms
  touched 33% of slots (horizontal mirror) and 17% (sagittal channel reversal), as designed.

  | Arm (flight teacher, 2,000 studies) | loss ep0→ep2 | gold-58 macro ep0 / ep1 / ep2 | hold-out macro ep0 / ep1 / ep2 |
  |---|---|---|---|
  | raw windows | 0.949 → 0.851 | 0.657 / 0.711 / 0.732 | 0.679 / 0.741 / 0.751 |
  | canonical windows | 0.958 → 0.902 | 0.607 / 0.600 / 0.602 | 0.580 / 0.625 / 0.655 |

  **A06 (local):** hold-out canonical − raw at epoch 2: macro −0.097 (90% CI [−0.114, −0.080]);
  lateral pair −0.021 (CI [−0.046, +0.004]) on hold-out and −0.070 on gold-58; medial pair −0.081 /
  −0.106; the eight side-agnostic findings −0.119 / −0.150 (ACL −0.19, Contusion −0.15, Fracture
  −0.15, Synovitis −0.13, Effusion −0.10 on hold-out). Prediction Spearman between arms 0.64.
- **Conclusion: rejected** under the pre-registered "clearly negative" branch (lateral findings worse on
  both hold-out and gold-58; macro far below the −0.005 tolerance). **Caveat, stated plainly:** the
  damage is largest on findings that mirroring cannot affect and the canonical arm also fits its own
  training labels worse, which the hypothesised mechanism does not explain. The result is therefore
  evidence that *this transform slows early fine-tuning badly* (plausibly the side-dependent reversal of
  the sagittal triplet channel order against the pretrained RGB filters, compounded by ~25% wrong sides
  on untagged studies), not a clean verdict on anatomical canonicalisation. Both arms are far from
  converged (a fully trained CoAtNet reaches ~0.91 gold). Not followed up: the GPU quota is exhausted.
- **Next stage:** no.

### E8 — leaderboard baseline of the clean pipeline  *(pre-registered)*

- **Question:** where does the clean, efficient pipeline stand on the public LB *without* the DINO/Rad
  chain and without any new model? The public stack's history says the chain added ~+0.02 on top of a
  single CoAtNet, but gold-58 cannot see it (E1/A04); only the LB can. This one submission also proves
  the offline path end to end (no internet, runtime, schema).
- **Pipeline (`scripts/kaggle_submit.py` → `rsna_knee.mil.submit`):** public Raptor v5 / v10 / v8 at
  equal weight (no v5-reverse, no v4) + residual-gated CoAtNet at 0.4, rank fusion, fixed weights.
- **Reading:** ≥ 0.935 → the chain is worth ≤ ~0.005 over this family and stays out (simpler, faster);
  0.925–0.935 → integrate the chain at the public stack's global 0.4 outer weight and measure once;
  < 0.925 → something is wrong with the pipeline itself (debug before anything else). This score is the
  reference every later addition (e.g. the E6 model) is compared against — one submission per change.
- **Dry run (`rsna-knee-clean-submit` v1, 3-study public test stub):** passed — 3 MIL arms (shared
  decode, 2 GPUs, 46 s) and the residual-gated arm (48 s) all informative, 0 study failures,
  schema-valid `submission.csv`, 1.6 min end to end. Submitted to the competition as the E8 baseline.

### E7 — engineering check of the residual-gated CoAtNet arm  *(done)*

- Not a hypothesis test. The residual-gated CoAtNet (mattiaangeli e4/e6/e8) is the only strong public
  model that adds on gold-58 (+0.007 over the Raptor family), so any final pipeline runs it. Its
  packaged dual-T4 runtime has its own preprocessing and a pinned OpenCV wheel; its hidden-test cost
  is unknown.
- Run it through its own `run_submission` on the 58 gold studies (a stand-in competition root whose
  "test" studies are the gold training studies) and record: wall time per study (model loading
  excluded) extrapolated to 1,322 studies; agreement of its ranks with the gold-58 predictions shipped
  in its artifact.
- **Result (`rsna-knee-e7-resgated-check`, 3 min GPU):** parity **passed** — rank Spearman 0.9996 with
  the shipped predictions, max rank difference one position of 58, gold-58 macro 0.9093 vs 0.9095,
  0 failures. Runtime 2.9 s/study on 2×T4 (≈ 11 s model load per shard) → **≈ 64 min for 1,322
  studies**. Budget so far: Raptor block ≤ 106 min (E4, before shared decode) + residual-gated 64 min
  ≈ 2.5–3 h of the 9 h limit.

### E6 — the next stage for each E5b outcome  *(pre-registered before E5b's result)*

- **If E5b meets its adoption rule:** full-data training (≈ 4,349 studies, DDP on 2×T4, one session
  with `--time_limit_h`) of one new CoAtNet-2 arm — flight-hybrid teacher + canonical orientation, the
  same side policy as E5b's canonical arm — saving a checkpoint every epoch. It enters the final
  family only through gold-58 fusion with the existing arms at a fixed equal member weight (no
  weight search), and is then checked once on the public LB against the same pipeline without it.
- **If E5b is null (inconclusive, per the amendment above):** no full-data run on this idea yet.
  Next step is an image-based knee-side estimate (a small head on frozen features, trained on the
  ~2,200 tagged studies, validated on held-out tagged studies) so untagged studies can be mirrored
  correctly; E5b is re-run only if that estimate reaches ≥ 0.97 held-out accuracy.
- **If E5b is clearly negative** (canonical arm worse on the lateral findings on both hold-out and
  gold-58): canonical orientation is rejected, and the new-model effort moves to the remaining
  pipeline-diversity lever — a different view (slot layout / crop) rather than orientation.
- **Outcome:** E5b was clearly negative → the third branch applies. **Not started**: the Kaggle GPU
  quota ran out on 2026-09-12. Ready-to-run configuration when quota returns: `rsna_knee.mil.train`
  on the full corpus with raw windows and the flight-hybrid teacher (DDP, `--save_ckpt`,
  `--time_limit_h`), with any new view (slot layout / crop) validated on a 2,000-study subset first,
  exactly as E5b was.

### E3 — end-to-end CoAtNet on the winning teacher, subset first  *(pre-registered before E2b's result; not launched — E2 failed its rule)*

- **Runs only if E2 passes its rule.** Otherwise H1 moves to view/pipeline diversity (H2) instead.
- **Hypothesis:** a CoAtNet-2 trained end-to-end on the better-agreement teacher is (i) at least as
  strong as a same-budget run on the Raptor teacher and (ii) less correlated with the public Raptor
  family, so it adds more when fused with the existing CoAtNet family.
- **Experiment:** two identical runs (`train_mil.py`, 2×T4 DDP, fp16, grad-ckpt): CoAtNet-2 @384
  from ImageNet-12k weights, 2,000 random non-gold training studies, 3 epochs, 12 random windows
  per study, effective batch 8 studies; only the teacher differs (E2 winner vs Raptor teacher).
  Gold-58 and a fixed 500-study weak-label hold-out scored after every epoch. ≈ 1 h GPU per run.
- **Why a subset is enough:** the question is relative (teacher X vs teacher A at equal budget, and
  how either correlates with / adds to the Raptor family), not the absolute score of a final model.
- **Metrics:** gold-58 macro and targeted findings; mean per-target Spearman with v5/v10/v8 on
  gold-58; gold-58 gain when rank-blended into the frontier CoAtNet family (paired bootstrap).
- **Engineering pre-check (not an experiment):** `rsna-knee-e3-smoke` — the same trainer on 40
  training / 10 hold-out studies for one epoch, only to prove the 2×T4 DDP + gradient-checkpointing
  + fp16 path runs end to end before either ~1 h run is launched.
  - v1 **failed**: a DDP rank was SIGKILLed before its first logged iteration (the signature of
    host-RAM OOM on the ~29 GB 2×T4 VM; no Python traceback). Fixes, all in one retry
    (`rsna-knee-e3-smoke2`): a plan runner that executes each stage in its own subprocess; the A/B
    pair as two independent single-GPU jobs (one per T4, no DDP needed for a comparison); lazily
    opened memmaps so DataLoader workers never copy the corpus; fewer, non-persistent workers;
    pretrained weights fetched once before any job starts; RSS/MemAvailable logged from every
    process. DDP (needed later for full-data training) is retried as a separate, isolated stage.
  - v2 **passed** (2026-09-12): both stages exit 0. Parallel single-GPU pair: 2.4 GB RSS each,
    27 GB host RAM still free, 11.3 GB GPU, ≈ 1.65 s per 24-image step (≈ 14 img/s per T4), gold-58
    + 10-study hold-out scored in ≈ 42 s. DDP stage (2 ranks, static graph + grad-ckpt) also runs
    end to end. → an E3 job (2,000 studies × 3 epochs + 3 evals) ≈ 1 h 40 min per T4, both jobs
    in one session.

### E4 — engineering check of `rsna_knee.mil` on real DICOM  *(done)*

- Not a hypothesis test. The repo package is shipped to Kaggle as the private dataset
  `tranbadat/rsna-knee-code` (the final offline submission will load code the same way).
- Checks: (1) the three public Raptor arms via `rsna_knee.mil.infer.predict_arms` reproduce E1's
  gold-58 predictions (E1 ran the public notebooks' verbatim code); (2) wall time of the 3-arm block
  with shared decode on both GPUs, on 300 training studies, extrapolated to 1,322 test studies.
- **Result (`rsna-knee-e4-mil-infer-check`, 24 min GPU):** parity **passed** — max |p − p_E1| ≤ 5e-4
  for all three arms; gold-58 macro identical (0.9199 / 0.9171 vs 0.9169 / 0.9116), 0 failures.
  Runtime: 597 s (maxspan 336) + 508 s (dense 384) + 332 s (native44 384) for 300 studies =
  **4.8 s/study → ≈ 106 min for 1,322 test studies** — slower than the per-study estimate because
  DICOM decoding on 4 vCPUs is the bottleneck, not the GPUs. maxspan and dense pick exactly the
  same slices (same slots and span, different pixel grid), so decoding once per study and sharing
  it across recipes removes roughly a third of the decode work → implemented in `predict_arms`.
- **Decision rule:** full-data training of the winner-teacher model (≈ 7 h, one session) only if its
  fusion gain over the CoAtNet family is ≥ +0.004 with P>0 ≥ 0.8 **and** at least the control run's
  gain. If the control adds as much, the gain is "another independent run", not the teacher — then
  the full run uses whichever teacher scored higher on gold-58, and labels are not credited.
