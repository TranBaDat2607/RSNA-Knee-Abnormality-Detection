# RSNA Knee Abnormality Detection

Work on Kaggle's [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition: predict twelve clinically important findings (ACL/MCL injury, meniscus tears,
three-compartment osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture)
from knee MRI studies, scored as macro-averaged ROC AUC across the twelve targets.

Only ~1.3% of training studies carry ground-truth labels; every study carries a free-text
radiology report instead. This repo's contribution is an **LLM-based report labeler** that
turns those reports into training targets for the imaging model, evaluated against the
same baseline and gold-label set as the original notebook this builds on.

## Credit

The EDA and baseline imaging pipeline (`eda/rsna-knee-data-structure-eda-baseline.ipynb`)
is built on top of [Roman Rozen's public baseline notebook](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline)
for this competition. This repo keeps that pipeline (multilingual report extractor,
physical-scale MRI sampling, partially fine-tuned DINOv2 backbone with a per-diagnosis
attention head, 4-fold CV) and swaps in an LLM-generated label set as an alternative
source of training targets.

## Result

| Label source | Out-of-fold macro AUC (58 held-out annotated studies) |
|---|---:|
| Original (rule-based report extractor, Roman Rozen baseline) | 0.809 |
| **This repo (`gpt-5.4-mini`-labeled targets)** | **0.813** |

## How the labels were generated

`scripts/llm_label_gold.py` prompts an LLM (model set via `OPENAI_MODEL` in `.env`) to
read each study's free-text `Report` and output a graded 0–1 label for each of the twelve
targets, as a drop-in replacement for the notebook's rule-based clause-level extractor.

- Validated against the 58 gold-annotated studies before committing to a full-corpus run
  (mean agreement AUC 0.869 vs. the rule-based extractor's 0.814 — see `CLAUDE.md` for the
  full model comparison).
- All 4,407 training studies are labeled in `data/llm_labels_full.csv`, which is what
  feeds the imaging model above.
- Resumable and token-budget-aware: it skips studies already present in the output CSV and
  stops cleanly before exceeding a configurable per-run token budget, so a multi-day label
  run can be split across several invocations of the same command.

```bash
# copy the env template and fill in your key
cp .env.example .env

# quick smoke test
python scripts/llm_label_gold.py --limit 6

# full run (resumable — re-run the same command to continue after a stop)
python scripts/llm_label_gold.py
```

## Repo layout

```
eda/                    EDA + baseline imaging model notebook (narrative, one-off analysis)
src/rsna_knee/          module-per-concern package version of the same imaging pipeline
tests/                  unit tests for src/rsna_knee (the DICOM-free modules run locally
                         without Kaggle data; see src/rsna_knee/README.md)
scripts/llm_label_gold.py   LLM report-labeling pipeline
data/                   gold labels, LLM-generated labels, usage logs
docs/requirements.md    competition task description
EDA_BASELINE_RESULTS.md summary of the baseline notebook's results
CLAUDE.md               working notes on the labeling pipeline
```

`data/train.csv` (Kaggle's original competition data) is not included in this repo — see
the competition page to download it.

## Setup

```bash
pip install pandas numpy torch transformers pydicom scikit-learn matplotlib seaborn python-dotenv openai

# to work on src/rsna_knee itself:
pip install -e ".[dev]"
pytest
```

Requires an OpenAI API key in `.env` (see `.env.example`) to run the labeling script; the
notebook itself only needs the packages above plus the downloaded competition data.

## The `src/rsna_knee` package

The notebook (`eda/rsna-knee-data-structure-eda-baseline.ipynb`) is where the EDA and the
narrative reasoning behind each pipeline decision live. `src/rsna_knee/` is the same
training/inference pipeline reorganized into a plain-`.py`, module-per-concern package —
DICOM ingestion, laterality/slot resolution, physical-scale caching, gold+LLM label
merging, the DINOv2+SlotHead model, the 4-fold CV training loop, and rank-mean submission
writing each get their own module, with unit tests for everything that doesn't require
Kaggle's mounted DICOM data. See `src/rsna_knee/README.md` for the module map and how to
run it.

## License

No license has been chosen yet for the code in this repo. Competition data is subject to
Kaggle's competition rules and is not redistributed here.
