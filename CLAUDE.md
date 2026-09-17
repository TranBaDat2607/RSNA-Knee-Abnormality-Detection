# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Work on Kaggle's RSNA Knee Abnormality Detection competition: predict twelve clinically
important findings (ACL/MCL injury, meniscus tears, three-compartment osteoarthritis,
effusion, synovitis, Baker's cyst, contusion, fracture) from knee MRI studies, scored as
macro-averaged ROC AUC across the twelve targets. Only ~1.3% of training studies carry
ground-truth labels; every study carries a free-text radiology report instead. This
repo's contribution is an **LLM-based report labeler** that turns those reports into
training targets for the imaging model.

The EDA and baseline imaging pipeline (`eda/rsna-knee-data-structure-eda-baseline.ipynb`)
is built on top of Roman Rozen's public baseline notebook for this competition
(https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline). This repo
keeps that pipeline (physical-scale MRI sampling, partially fine-tuned DINOv2 backbone
with a per-diagnosis attention head, 4-fold CV) and swaps in an LLM-generated label set as
an alternative source of training targets.

## Repo layout

```
eda/                    EDA + baseline imaging model notebook (narrative, one-off analysis)
src/rsna_knee/          module-per-concern package version of the same imaging pipeline,
                         plus mil/ — the CoAtNet 2.5-D attention-MIL pipeline (see below)
tests/                  unit tests for src/rsna_knee (DICOM-free modules run locally
                         without Kaggle data; see src/rsna_knee/README.md)
scripts/llm_label_gold.py   LLM report-labeling pipeline
scripts/kaggle_submit.py    offline Kaggle submission entry (rsna_knee.mil.submit)
scripts/kaggle_mil_train.py Kaggle launcher for training plans (rsna_knee.mil.train/plan)
data/                   gold labels, LLM-generated labels, usage logs (train.csv/test.csv
                         and raw DICOM are NOT included — see competition page)
docs/report.md          2026-09-12 study: why the 0.94 stack works, experiments, path forward
docs/experiment_ledger.md  every hypothesis/experiment/result/decision rule, in order
docs/requirements.md    competition task description
EDA_BASELINE_RESULTS.md summary of the baseline notebook's results
environment.yml         conda env `rsna-knee` (CPU-only; all GPU work runs on Kaggle)
```

`data/train.csv`, `data/test.csv`, and the DICOM directories are gitignored (large /
redistribution-restricted by Kaggle rules) — they must be downloaded from the competition
page into `data/` (or mounted, on Kaggle) before the imaging pipeline can run.

## Commands

### `src/rsna_knee` package (imaging model)

```bash
conda env create -f environment.yml && conda activate rsna-knee   # or: pip install -e ".[dev,viz,mil]"
pip install -e ".[dev]"     # install package + pytest
pytest                       # run all tests (82 passing locally; see below for scope)
pytest tests/test_targets.py                    # single file
pytest tests/test_targets.py::test_gold_positions   # single test

# from the competition data root (or on Kaggle, path auto-detected):
rsna-knee                          # full run, notebook defaults -> submission.csv
rsna-knee --epochs 6 --folds 2     # quicker iteration
rsna-knee --slot-scheme public     # alternative slot definition (see config.py)
rsna-knee --time-budget-hours 2 --out my_submission.csv
# equivalently: python -m rsna_knee.cli ...
```

Tests cover everything that doesn't require Kaggle's mounted DICOM data or downloaded
DINOv2 weights: metrics, augmentation, laterality resolution, header sequence-typing,
slot picking, gold/LLM label merging and fold grouping, model shape wiring (via a stub
backbone), EMA, losses, submission writing, path resolution. `dicom/sampling.py` (actual
pixel decoding) and `model/backbone.py` (actual DINOv2 loading) are only exercised on
Kaggle.

### `scripts/llm_label_gold.py` (report labeling)

```bash
cp .env.example .env        # fill in OPENAI_API_KEY and OPENAI_MODEL

python scripts/llm_label_gold.py --limit 6           # quick smoke test
python scripts/llm_label_gold.py                     # real-time, resumable, all rows
python scripts/llm_label_gold.py --batch-size 10      # fewer, bigger calls
python scripts/llm_label_gold.py --in-csv X --out-csv Y --token-budget N

python scripts/llm_label_gold.py --submit-batch       # OpenAI Batch API: submit and exit
python scripts/llm_label_gold.py --check-batch        # Batch API: poll / collect
```

No dedicated test suite or lint config for this script or the notebook; `pytest` only
covers `src/rsna_knee`.

### Notebook

`eda/rsna-knee-data-structure-eda-baseline.ipynb` needs the full dependency set
(`pandas numpy torch transformers pydicom scikit-learn matplotlib seaborn python-dotenv
openai`) plus the downloaded competition data; it is not exercised by `pytest`.

## Architecture

Two parallel implementations of the same imaging pipeline exist, at different stages of
maturity — know which one you're changing:

- **`eda/rsna-knee-data-structure-eda-baseline.ipynb`**: the original, narrative
  notebook. This is where the EDA and the reasoning behind each pipeline decision (why
  physical-scale sampling, why laterality normalization, why no flips, etc. — see
  `EDA_BASELINE_RESULTS.md`) live. Includes a rule-based multilingual report extractor
  (notebook section 2) that is **not** ported to the package below — it's kept only as
  the historical baseline (0.814 mean AUC) that the LLM labeler is compared against.
- **`src/rsna_knee/`**: module-per-concern rewrite of the notebook's `main()`, the
  actively-developed target for future modeling changes. Pipeline stages, one module
  each: DICOM header pass (`dicom/header.py`) -> laterality resolution
  (`dicom/laterality.py`) -> slot picking (`dicom/slots.py`, one series per
  plane x weighting/fat-sat combination) -> physical-scale decode cache (`dicom/cache.py`,
  `dicom/sampling.py`) -> target merging (`labels/targets.py`, gold `train.csv` rows +
  LLM labels -> graded `(Y, W)` arrays) -> model (`model/backbone.py` DINOv2,
  `model/heads.py` `SlotHead` masked-softmax attention, `model/network.py`) -> 4-fold CV
  training (`training/loop.py`, `training/augment.py`, `training/losses.py`,
  `training/metrics.py`) -> rank-mean submission (`inference/predict.py`,
  `inference/submission.py`). `pipeline.py` (`run()` / `run_safe()`) orchestrates all of
  it; `config.py` holds every tunable as a dataclass (`Config()` with no args reproduces
  the notebook's run); `cli.py` is the entrypoint. Full module-by-module map in
  `src/rsna_knee/README.md`.

Two validation-leak guards run through both implementations and are worth preserving in
any change: **folds are grouped by report-text hash**, not randomly (so studies sharing
an identical report — and therefore an identical LLM/derived label — stay in one fold),
and **each fold's annotated-holdout reference excludes studies that fold trained on**.

**Labels feeding the imaging model**: every study except the ~58 gold-annotated ones uses
graded 0-1 labels from `data/llm_labels_full.csv` (generated by
`scripts/llm_label_gold.py`, see below); the gold-annotated studies use the real
`train.csv` labels at a higher sample weight (`gold_weight` in `CVConfig`, default 3.0).
The rule-based extractor is not in this path at all.

`scripts/llm_label_gold.py` independently prompts an LLM to read each study's free-text
`Report` and output a graded 0-1 label for each of the twelve targets — a drop-in
replacement for the notebook's rule-based extractor as a target source, validated against
the 58 gold-annotated studies before being trusted for the full corpus. It is resumable
(skips `StudyInstanceUID`s already in `--out-csv`) and token-budget-aware (stops cleanly
before a configurable per-run cap rather than erroring out mid-run) — see the "LLM
report-labeling pipeline" section below for the full history and current status.

## LLM report-labeling pipeline (full corpus DONE)

`scripts/llm_label_gold.py` uses an LLM (`OPENAI_MODEL` in `.env`) to generate
graded per-finding labels (0-1 for each of the 12 targets in `TARGETS`) from
the free-text `Report` column, as an alternative to the rule-based extractor
described in `EDA_BASELINE_RESULTS.md`.

**Current model: `gpt-5.4-mini`** (switched from `gpt-5.6-sol`, see "Model
history" below for why).

### Results on the 58 gold studies so far

| Model | Mean agreement AUC | Tokens for 58 studies | Est. tokens for full 4,407 corpus |
|---|---:|---:|---:|
| Rule-based extractor (baseline) | 0.814 | -- | -- |
| `gpt-5.6-sol` | 0.869 | not instrumented (heavy hidden-reasoning waste) | ~2.77M+ (needed multiple days) |
| **`gpt-5.4-mini`** | **0.869** | **34,783 (~600/study)** | **~2.64M -- fits in ~1 run** |

Same mean AUC, same per-target ranking (Synovitis weakest at ~0.68, ACL
strongest at ~0.97-0.99), but `gpt-5.4-mini` is ~80x cheaper per study and
doesn't have the hidden-reasoning-token problem `gpt-5.6-sol` had. **Use
`gpt-5.4-mini` for the full-corpus run.**

Output files:
- `data/gold_annotated.csv` -- the 58 fully-annotated studies + their real labels.
- `data/llm_labels.csv` -- `gpt-5.6-sol` result on the 58 gold studies (0.869 AUC).
- `data/llm_labels_gpt54mini.csv` -- `gpt-5.4-mini` result on the 58 gold studies (0.869 AUC).
- `data/llm_labels_full_gpt56sol_STALE_726of4407.csv` -- **stale, do not use.**
  Partial full-corpus run (726/4407) made with `gpt-5.6-sol` before switching
  models. Renamed rather than deleted so the work isn't lost, but it should
  NOT be merged with a `gpt-5.4-mini` full-corpus run -- different model,
  would mix label distributions.

**Full corpus: DONE.** `data/llm_labels_full.csv` has all 4,407 studies
labeled with `gpt-5.4-mini` (graded 0-1 scores for the 12 `TARGETS`, same
column layout as `gold_annotated.csv` but without the ground-truth columns).
Took 2 runs: first stopped cleanly at the 2.4M-token budget with 4,374/4,407
done, a top-up run finished the remaining 33. Total: **2,415,243 tokens**.

This is the actual deliverable -- the 58-sample runs were quality checks
against ground truth only, this full run is what should feed the imaging
model's training targets next.

**Bug fixed along the way:** `load_done()` originally returned pandas Series
for previously-labeled rows (from `df.iterrows()`), which broke when mixed
with the plain dicts used for newly-labeled rows in the same
`pd.DataFrame(rows)` call (`AttributeError: 'dict' object has no attribute
'dtype'`). Only surfaced on the first resume with a large prior batch (4,374
rows) -- earlier resumes in testing were empty or tiny. Fixed by converting
to `.to_dict()` in `load_done()`. No data was lost when it crashed; the one
in-flight batch just got recomputed on retry.

**Done (this note previously said "not started" -- corrected 2026-08-13):**
`data/llm_labels_full.csv` is hooked up as training targets for the imaging
model. The notebook's `main()` (`eda/rsna-knee-data-structure-eda-baseline.ipynb`)
loads it via `find_llm_labels()` and uses it for every study except the ~58
gold-annotated ones (which use the real `train.csv` labels at a higher sample
weight, `GOLD_WEIGHT = 3.0`); the rule-based extractor from section 2 is no
longer in the training-target path at all -- it's kept only as the
comparison baseline the "Results" table above measures against. This is the
run that produced the `f7fc901` "first submission: 0.813" commit (OOF macro
AUC 0.7675 on the 58 annotated studies -- see `EDA_BASELINE_RESULTS.md` §3).

**Done:** a module-per-concern Python package under `src/rsna_knee/` now
exists, porting this same pipeline (DICOM ingest -> laterality/slots ->
physical-scale cache -> gold+LLM target merge -> DINOv2+SlotHead model ->
4-fold CV training loop -> rank-mean submission) out of the notebook into
plain `.py` files with unit tests (`tests/`, 48 passing locally on the
DICOM-free modules -- the DICOM/model modules need Kaggle's mounted data and
DINOv2 weights to exercise for real). The rule-based extractor was
deliberately *not* ported, since it's no longer on the path that produces
the submission. Run via `pip install -e .` then `rsna-knee` (or
`python -m rsna_knee.cli`) from the competition data root. The notebook
stays as-is for narrative EDA; the package is what should be iterated on for
future modeling changes (resolution/crop tuning, a second backbone for
ensembling -- see `EDA_BASELINE_RESULTS.md` §5 "Where the notebook flags
future work").

### Daily token budget

**The 2,500,000 token/day limit is shared across a whole family of models**,
not just one -- confirmed via the OpenAI account limits page: it covers
`gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.1-codex-mini`, `gpt-5-mini`,
`gpt-5-nano`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o1-mini`, `o3-mini`, `o4-mini`,
and `codex-mini-latest`. `gpt-5.6-sol` (the original model) is NOT in this
list, which may be why it behaved so differently (heavier reasoning, no
temperature support) -- it's likely a different tier/pool entirely. If any
other tool or process is also calling one of the listed mini/nano models
against the same account on the same day, it eats into this same 2.5M pool.

`scripts/llm_label_gold.py` enforces a **2,400,000-token stop threshold**
per run (`--token-budget`, default `DEFAULT_TOKEN_BUDGET` in the script) --
it checks projected usage before each batch (using a rolling average of
recent real batches) and stops cleanly once the next batch would cross that
line, rather than erroring out mid-run.

**The script is resumable by design**: it skips any `StudyInstanceUID`
already present in `--out-csv`, so re-running the exact same command on a
later day (once the quota resets) picks up where it left off. No manual
bookkeeping needed between runs -- just re-run the same command.

Each run appends one line to `data/llm_label_usage.log` (JSON lines) with
tokens used, studies labelled, and studies remaining.

**Sizing with `gpt-5.4-mini`:** measured at ~600 tokens/study on the 58 gold
studies. Full corpus (4,407 studies) -> ~2.64M tokens estimated total, i.e.
it should mostly fit in a single ~2.4M-token run, with a small top-up run
likely needed to finish the last few hundred studies. Much better than the
`gpt-5.6-sol` estimate (~2.77M+, needing 2+ days) since gpt-5.4-mini doesn't
burn tokens on hidden reasoning for this task.

### Model history / known quirks

**`gpt-5.6-sol`** (original model, now abandoned for this task): a
reasoning-family model with non-standard Chat Completions behavior,
discovered by trial and error:
- Rejects `temperature` entirely (any value).
- Needs `max_completion_tokens`, not `max_tokens`.
- Defaulted to spending 60-100% of the completion budget on hidden reasoning
  tokens for this simple classification task, occasionally consuming the
  *entire* budget and returning empty content. Fixed with
  `"reasoning_effort": "none"`, cutting tokens ~3x with no visible quality
  loss -- but even with that fix it was still far more expensive per study
  than gpt-5.4-mini turned out to be, and not part of the free 2.5M/day pool.

**`gpt-5.4-mini`** (current model): worked cleanly out of the box with the
same request body (including `reasoning_effort: "none"` and
`max_completion_tokens`) -- no adaptation needed, no errors, no truncation.
Matched gpt-5.6-sol's AUC at a fraction of the token cost.

`request_body()` in the script still sends `reasoning_effort: "none"`
unconditionally -- if a future model rejects that parameter,
`_adapt_body_for_error()` already has a fallback to drop it and retry (same
pattern used for temperature/max_tokens).

### Batch API path exists but is separate

`--submit-batch` / `--check-batch` use OpenAI's async Batch API (~50%
cheaper, up to 24h turnaround) instead of the real-time path. **The token
budget guard above only applies to the real-time path** -- not confirmed
whether Batch API jobs draw from the same daily token pool as the real-time
family list above. Given gpt-5.4-mini's real-time cost is now cheap enough
that the full corpus likely fits in ~1 real-time run anyway, Batch API is
probably not needed for this corpus size, but worth reconsidering if the
per-study cost estimate turns out higher in practice.

## Second backbone arm (EfficientNet-B3) -- built, gold-only smoke test inconclusive

`src/rsna_knee/model/backbone.py` now has two backbone builders, both feeding the same
`Model`/`SlotHead` wiring (`build_model()` unchanged in signature): `build_model()`
(DINOv2) and `build_efficientnet_model()` (EfficientNet-B3, the Melas-Kyriazi
`efficientnet_pytorch` implementation). `training/loop.py`'s `train_one_fold()` takes an
optional `model_builder` callable so either arm trains through the same loop/OOF
machinery. `training/metrics.py` gained `per_target_auc` and `paired_bootstrap_auc_diff`
for comparing two arms target-by-target on a small annotated set without over-trusting
noise (see git history for the reasoning -- Hanley-McNeil SE on 58 gold studies gives
~+/-0.10-0.16 per-target, so a paired bootstrap on the AUC *difference* is the only
statistically sound way to ask "is arm A actually better than arm B here").

**Bug fixed along the way:** `build_model()`'s DINOv2 loader assumed an HF-format
directory (`config.json` + weights via `AutoModel.from_pretrained`), but the actual
public Kaggle dataset for this competition
(`girishbose/dinov2-vitb14-rsna-knee`) is a bare `facebookresearch/dinov2` torch.hub
checkpoint -- no `config.json`, different key names, fused `qkv` instead of separate
query/key/value. `find_dinov2_raw()` (`paths.py`) + `_remap_dinov2_raw_state_dict()`
(`backbone.py`) handle that format now; the remap was verified **byte-identical** to the
reference `facebookresearch/dinov2` architecture's own output at the checkpoint's native
518px resolution before being trusted (the two only diverge in position-embedding
interpolation at other resolutions, which end-to-end fine-tuning already adapts to).
`build_model()` tries the HF path first, falls back to the raw-checkpoint path
automatically -- no config change needed to use either.

### `scripts/compare_backbones_gold.py` -- gold-only complementarity check

Trains both arms on **the 58 gold-annotated studies only** (not the ~4,400-study
LLM-labeled corpus), 4-fold, to cheaply check whether DINOv2 and EfficientNet-B3 make
complementary errors before committing to the expensive full-corpus run for both arms.
Outputs per-target AUC for each arm, the paired-bootstrap CI on their difference,
Spearman correlation between their raw predictions (the actual complementarity signal,
independent of which one's more accurate), and a rank-blend AUC.

**Run 2 (2026-08-13, `data/backbone_compare_gold/`): DONE, converged, weak/mixed
result.** Ran on Kaggle; full details in `data/backbone_compare_gold/README.md`
(run 1's CPU/P100 misfire is kept there too, archived under
`run1_cpu_undertrained/`, as the record of the GPU issue and its fix -- the assigned
Tesla P100 (`sm_60`) isn't supported by this Kaggle image's PyTorch build (`sm_70`+
only); the fix was `machine_shape: "NvidiaTeslaT4"` in `kernel-metadata.json`, named
explicitly in the kaggle-cli docs (`kernels_metadata.md`), which also flags
`NvidiaTeslaP100` as having known compatibility issues -- not discoverable from the
installed `kaggle`/`kagglesdk` packages alone).

Run 2 (T4, full 10 epochs, 8.7 min) converged properly (DINOv2 per-fold AUCs 0.54-0.68,
none below chance). **Macro AUC: DINOv2 0.531, EfficientNet-B3 0.444, naive rank-blend
0.496.** Complementarity signal is weak and mixed: per-target Spearman correlation
between the two arms' raw predictions is low almost everywhere (mean `|rho|` 0.19,
several near zero, Lateral OA -0.46) -- the right precondition for combining two
models -- but EfficientNet-B3 is also just less accurate overall (DINOv2 wins the point
estimate on 9/12 targets), and only Medial OA has a paired-bootstrap CI that excludes
zero (favoring DINOv2). **Rank-blending does not help at this scale**: naive 50/50
rank-mean underperforms DINOv2 alone on 9/12 targets and only beats *both* arms
individually on 1/12 (Lateral OA). Recommendation: don't commit to the expensive
full-corpus dual-arm run on this evidence -- if pursuing EfficientNet-B3 further, tune
its own LR/unfreeze schedule first (it's currently using DINOv2's untouched
hyperparameters) and/or try an OOF-weighted blend instead of naive rank-mean.

## External ensemble notebook (`tranbadat/rsna-knee-abnormality`, Kaggle) -- 0.910
public LB, blend hardened to 0.909 (flat), real gains still open

A separate, **private, multi-person Kaggle notebook** -- `tranbadat/rsna-knee-abnormality`,
not part of this git repo -- scored **0.910 public LB** on 2026-08-14, well above this
repo's own single-arm pipeline (0.813, the `f7fc901` submission described above). Pulled
via `kaggle kernels pull tranbadat/rsna-knee-abnormality -m` for inspection; the patched
copy and both runs' logs/outputs live in a session scratchpad, not checked into this repo.

**What it actually is**: inference-only (no training happens in-kernel -- `enable_internet:
false`), blending three pretrained arms loaded as teammates' Kaggle datasets rather than
trained by this repo's pipeline: a 24-member DINOv2 TTA ensemble (`pilkwang/rsna-knee-weights`,
two physical-scale crop configs, jittered multi-window TTA), a 5-fold DINOv3-small arm
(`mattiaangeli/knee-mri-fold-weights`, `vit_small_patch16_dinov3.lvd1689m`, timm, custom
"CodexResidualPool" head), and a 5-head RadImageNet-pretrained ResNet-50 arm ("Rad15",
`marwanmath/resnet-50-radimagenet-marwan`). Combined via fixed, LB-validated weights
(0.65/0.35 DINOv2:DINOv3, then 0.75/0.25 adding RadImageNet), then a further "V11" layer
blends several candidate recipes (`safe`, `challenger`, `d3_heavy`, `target_stability`,
`uncertainty`, `diversity`) -- `challenger` was the one actually submitted for the 0.910
score.

**Diagnosis**: `challenger` put up to 30% of its weight on three terms
(`target_stability`, `uncertainty`, `diversity`) whose per-target alpha weights were
computed from fold-correlation/std-dev over Kaggle's **3-row public dry-run `test.csv`**
-- statistically indefensible at n=3 (confirmed directly in the notebook's own
`v11_ensemble_diagnostics.json`: correlations only land on a handful of discrete values,
the fingerprint of a 3-point sample). The two base weights it built on (0.65/0.35,
0.75/0.25) are the actually-evidence-backed part, validated across many real prior public-LB
submissions, not this run's 3 rows.

**Fix tried and validated**: patched the notebook's blend cell so `submission.csv`
defaults to `grounded = 0.55*exact + 0.45*rankfold` instead of `challenger` -- both
`exact` and `rankfold` are built purely from the two fixed, LB-validated weights, zero
exposure to the n=3-fit terms (`challenger`/`safe`/etc. are still computed and written as
candidate files, just no longer the default). Pushed as kernel version 2, validated end
to end before submitting: schema match, no nulls/non-finite, deterministic vs the v1 run
(confirms nothing upstream broke), and diffed against the known-0.910 `challenger` output
(identical on 11/12 targets on the 3-row dry-run set, only MCL differs).

**Result (submission `55523770`, 2026-08-15): 0.909** -- flat vs 0.910 (delta 0.001,
noise-level). Reads as a real, if modest, confirmation: removing 30% of `challenger`'s
weight (the n=3-fit terms) cost essentially nothing, consistent with those terms adding
no real signal. But it also means blend-recipe reshuffling among the *same* three arms
has hit its ceiling -- further gains need one of: multi-scale TTA added to this repo's
own `inference/predict.py` (currently single-crop, group-averaged only, no multi-window
jitter or multi-scale ensembling, unlike the notebook's DINOv2 arm), LLM label-quality
work on the weakest targets (Synovitis 0.709, Effusion 0.777, Lateral OA 0.839, PF OA
0.833 gold agreement-AUC), or a genuinely new architecture/pretraining-domain arm --
specifically a **RadImageNet-style, radiology-pretrained CNN**, not EfficientNet-B3
(natural-image-pretrained, same failure-mode family as DINOv2's ViT, which is why the
gold-only smoke test above found weak/mixed complementarity) -- trained on the full
corpus and validated on gold-58 first.

**Kaggle CLI operational notes** (non-obvious, cost real time to work out):
- `kaggle competitions submit -f <local-path>` fails with an opaque, bodyless 400 for
  this competition because it's a **Code Competition** -- submissions must reference a
  kernel's own output, not a raw file upload. Correct form:
  `kaggle competitions submit <comp> -k <kernel> -v <version> -f <output-filename>
  -m <message>` (the `-f` value is the *output filename the kernel wrote*, e.g.
  `submission.csv`, not a local path).
- Grading a submitted kernel version (rerun against the hidden test set) took **~2.5
  hours** for this notebook, vs. ~5-9 min for a dev run against the 3-row public
  dry-run `test.csv` -- the RadImageNet arm alone ran at ~90-120s/study on 3 studies,
  and the cache-sizing log line references a real hidden test set around 1,322 studies,
  so per-study cost likely doesn't amortize away at scale. Budget hours, not minutes,
  before assuming a pending submission has failed; the CLI gives no progress signal or
  error detail while `SubmissionStatus.PENDING`.

## CoAtNet MIL pipeline and the "beat 0.94" experiment program (2026-09-12)

Read `docs/report.md` first (synthesis), then `docs/experiment_ledger.md` (every hypothesis,
pre-registered decision rule, run name and result, in order). Stopped because the Kaggle GPU
quota ran out.

**State when stopped:** E5b (canonical-orientation A/B, kernel `rsna-knee-e5b-canon-ab`) finished
and was **rejected**: canonical windows were far worse than raw on a 2,000-study subset (hold-out
macro −0.097), mostly on findings orientation cannot affect — a training side effect of the
transform, not a verdict on anatomy (details in the ledger). E6 (full-data new CoAtNet model; with
E5b negative, the raw-window / best-teacher / new-view branch) was not started — no GPU quota.

**E8 scored 0.939 on the public LB** (read 2026-09-17; submission `56184308` of kernel
`rsna-knee-clean-submit` v1: public Raptor v5/v10/v8 equal weight + residual-gated CoAtNet 0.4, no
DINO/Rad chain). This is the repo's best submission (previous: 0.910/0.909 from the external ensemble
notebook below) and it is level with the public 0.939–0.941 chain notebooks while using ≈ 2.5–3 h of
the 9 h limit. Per E8's pre-registered rule (≥ 0.935) **the DINO/Rad chain stays out** — it plus the
public notebooks' LB-probed per-target weights are worth ≤ ~0.002 over this family. 0.939 is now the
reference baseline: every later addition is one submission compared against it. Nothing above 0.94
has been demonstrated yet.

**What the evidence says (details in the report):**
- The 0.94 public stack rests on the CoAtNet-RMLP-2 @384 2.5-D per-finding attention-MIL family
  (gold-58 0.912–0.920 per checkpoint vs DINOv2 0.840 / RadImageNet 0.854). Pipeline/label
  diversity is what adds (residual-gated CoAtNet +0.007 on gold-58); more checkpoints of the same
  pipeline (e.g. `raptor-knee-widedense` v4) and backbone swaps on the same labels/views do not.
- The public notebooks' "v5-reverse" arm is a no-op (the attention pool ignores window order) and
  their 0.939→0.941 gains are LB-probed per-target weights. Gold-58 cannot resolve ±0.01 fusion
  effects (SE ~0.02): use it as a regression guard, never to tune weights.
- Five public label tables copy the gold-58 labels verbatim: barun2104 stratified folds,
  rayanbabur calibrated targets, tasmeemreza refined labels, zaidaliiq1000, yunusgmsoy
  4-source-merged. Best clean teacher: flight0234 hybrid (0.899 gold agreement); ours 0.855.
- Relabelling a frozen CoAtNet's head with a better teacher helps Synovitis (+0.067) but the
  existing family absorbs almost all of it (+0.0016); the pre-registered rule for end-to-end
  relabelled training failed.

**`src/rsna_knee/mil/` (82 tests total, package shipped to Kaggle as private dataset
`tranbadat/rsna-knee-code`):** inference engine reproducing the public checkpoints to ≤ 5e-4
(E4) with one DICOM decode per study shared across recipes; `CORPUS44_336` = the exact recipe of
the public pre-decoded corpus `dreaddevelopment/knee-raptor-corpus(-ext)` (span **0.15–0.85**, not
0.06–0.94; mean diff 0.00, E5c); canonical orientation (every series is stored in standard DICOM
orientation, so right knees are mirror images; side = DICOM tag on only ~50% of studies, patient-x
geometry does not encode side on untagged sites); corpus trainer + plan runner; offline submission.

**Runtime budget (2×T4, ≈ 1,322 test studies):** public CoAtNet v5+v10+v8 ≤ 106 min (measured
before shared decode, decode-bound on 4 vCPUs), residual-gated ≈ 64 min (runs its own packaged
runtime with a pinned OpenCV 4.12 wheel in a subprocess) → ≈ 2.5–3 h of the 9 h limit.

**Kaggle operations learned this session:**
- Auth: Kaggle CLI 2.x needs a new-style token in `~/.kaggle/access_token`. The legacy
  `kaggle.json` key can read public kernels/datasets but cannot push kernels, list your own
  kernels, or use competition endpoints ("Authentication required").
- Windows: `kaggle datasets create|version -p <absolute path>` crashes building its upload-cache
  path; `cd` into the parent directory and pass the folder name relatively.
- Script kernels upload one file: ship the package as a zip of `src/rsna_knee` in the code dataset
  (Kaggle extracts it server-side) and have the launcher walk `/kaggle/input` for
  `rsna_knee/__init__.py` (see `scripts/kaggle_submit.py`, `scripts/kaggle_mil_train.py`).
- GPUs: T4×2 only (P100 unsupported). T4 has no bf16 → fp16 autocast + GradScaler. CoAtNet-2 @384
  trains at ≈ 13–14 img/s per T4 and needs gradient checkpointing for 24 images; ≈ 29 GB host RAM.
- A host-RAM OOM shows up only as `process ... terminated with signal SIGKILL`. Run stages in
  subprocesses, open memmaps lazily (a pickled `np.memmap` copies the whole array into workers), use
  ≤ 2 non-persistent DataLoader workers, fetch pretrained weights before spawning jobs. Two
  single-GPU jobs side by side (`CUDA_VISIBLE_DEVICES`) is the efficient A/B pattern; DDP with timm
  gradient checkpointing needs `static_graph=True`.
- Monitoring: poll `kaggle kernels status <kernel>`; fetch only logs/receipts with
  `kaggle kernels output <kernel> -p <dir> --file-pattern '(\.log$|\.json$)'` (logs are JSON
  streams of stdout/stderr events).
- The residual-gated arm's runtime needs `test.csv`, `test_series.csv`, `sample_submission.csv`
  (same study order) and `test_series/`; a stand-in root with symlinked training studies lets it run
  on gold-58 (E7).
