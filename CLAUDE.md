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
src/rsna_knee/          module-per-concern package version of the same imaging pipeline
tests/                  unit tests for src/rsna_knee (DICOM-free modules run locally
                         without Kaggle data; see src/rsna_knee/README.md)
scripts/llm_label_gold.py   LLM report-labeling pipeline
data/                   gold labels, LLM-generated labels, usage logs (train.csv/test.csv
                         and raw DICOM are NOT included — see competition page)
docs/requirements.md    competition task description
EDA_BASELINE_RESULTS.md summary of the baseline notebook's results
```

`data/train.csv`, `data/test.csv`, and the DICOM directories are gitignored (large /
redistribution-restricted by Kaggle rules) — they must be downloaded from the competition
page into `data/` (or mounted, on Kaggle) before the imaging pipeline can run.

## Commands

### `src/rsna_knee` package (imaging model)

```bash
pip install -e ".[dev]"     # install package + pytest
pytest                       # run all tests (48 passing locally; see below for scope)
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
