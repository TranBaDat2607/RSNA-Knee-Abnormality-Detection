# RSNA Knee Abnormality Detection

Work on Kaggle's [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition: predict twelve clinically important findings (ACL/MCL injury, meniscus tears,
three-compartment osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture)
from knee MRI studies, scored as macro-averaged ROC AUC across the twelve targets.

Only ~1.3% of training studies carry ground-truth labels; every study carries a free-text
radiology report instead. The repo has two parts:

1. an **LLM-based report labeler** that turns those reports into training targets (the original
   contribution, below);
2. a **CoAtNet 2.5-D attention-MIL pipeline** (`src/rsna_knee/mil/`) — the representation behind the
   ~0.94 public ensembles — rebuilt as tested, Kaggle-validated code, together with a hypothesis-driven
   study of what it would take to beat that ensemble. Start with
   [`docs/report.md`](docs/report.md); every experiment is logged in
   [`docs/experiment_ledger.md`](docs/experiment_ledger.md).

## Credit

The EDA and baseline imaging pipeline (`eda/rsna-knee-data-structure-eda-baseline.ipynb`)
is built on top of [Roman Rozen's public baseline notebook](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline)
for this competition. This repo keeps that pipeline (multilingual report extractor,
physical-scale MRI sampling, partially fine-tuned DINOv2 backbone with a per-diagnosis
attention head, 4-fold CV) and swaps in an LLM-generated label set as an alternative
source of training targets.

The CoAtNet MIL pipeline runs public checkpoints and data by other competitors, loaded as Kaggle
datasets and credited to their authors: the "Raptor" CoAtNet checkpoints and the pre-decoded training
corpus (dreaddevelopment), the residual-gated CoAtNet (mattiaangeli), and public report-label tables
(flight0234, stevenleehans, yunusgmsoy, pilkwang).

## Results

### Report labels → DINOv2 baseline (original contribution)

| Label source | Out-of-fold macro AUC (58 held-out annotated studies) |
|---|---:|
| Original (rule-based report extractor, Roman Rozen baseline) | 0.809 |
| **This repo (`gpt-5.4-mini`-labeled targets)** | **0.813** |

### CoAtNet MIL pipeline (2026-09 study)

| Model / pipeline | Gold-58 macro AUC | Public LB |
|---|---:|---:|
| DINOv2 slot model (public 20-member OOF) | 0.840 | — |
| Single public CoAtNet checkpoint (v5 / v10 / v8) | 0.920 / 0.917 / 0.912 | — |
| Residual-gated CoAtNet (e4/e6/e8) | 0.909 | — |
| **Clean pipeline: v5 + v10 + v8 (equal) + residual-gated (0.4)** | **0.930** | **0.939** (E8) |
| Public 0.94 notebooks (adds DINO/RadImageNet chain, LB-probed weights) | — | 0.939–0.941 |

The clean pipeline matches the public 0.94 notebooks (0.939 vs 0.939–0.941) with four arms, fixed
weights and ≈ 2.5–3 h of the 9 h runtime limit — no DINO/RadImageNet chain and no LB-probed per-target
weights. Per E8's pre-registered rule the chain stays out, and 0.939 is the reference score every
later change is measured against.

Gold-58 numbers are clean hold-out predictions (every checkpoint was trained without those studies).
Gold-58 has a macro-AUC standard error of ~0.02, so it guards against regressions rather than ranking
close candidates — see the report for how each claim was tested.

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
- Later comparison (docs/report.md, A02b): the best public report-label table agrees with gold
  better (0.899 macro) than these labels (0.855), mostly on Synovitis; five other public tables copy
  the gold labels verbatim and must not be used for gold-58 validation.

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
eda/                        EDA + baseline imaging model notebook (narrative, one-off analysis)
src/rsna_knee/              module-per-concern package: the DINOv2 pipeline, and mil/ — the CoAtNet
                            MIL pipeline (inference, corpus trainer, submission)
tests/                      unit tests for src/rsna_knee (run locally without Kaggle data or weights)
scripts/llm_label_gold.py   LLM report-labeling pipeline
scripts/kaggle_submit.py    offline Kaggle submission entry point (rsna_knee.mil.submit)
scripts/kaggle_mil_train.py Kaggle launcher for training plans (rsna_knee.mil.train / plan)
data/                       gold labels, LLM-generated labels, usage logs
docs/report.md              the 2026-09 study: why the 0.94 stack works, experiments, recommended path
docs/experiment_ledger.md   every hypothesis, experiment, result and decision rule, in order
docs/requirements.md        competition task description
EDA_BASELINE_RESULTS.md     summary of the baseline notebook's results
CLAUDE.md                   working notes (labeling pipeline, Kaggle operations, experiment state)
environment.yml             conda environment for local development
```

`data/train.csv` (Kaggle's original competition data) is not included in this repo — see
the competition page to download it.

## Setup

```bash
conda env create -f environment.yml   # CPU-only local env; GPU work runs on Kaggle
conda activate rsna-knee
pytest                                # 82 tests
```

or, without conda:

```bash
pip install -e ".[dev,viz,mil]" python-dotenv openai kaggle
pytest
```

Requires an OpenAI API key in `.env` (see `.env.example`) to run the labeling script. Kaggle work
needs an API token in `~/.kaggle/access_token`; the repo package is attached to Kaggle kernels as a
dataset (see `CLAUDE.md`).

## The `src/rsna_knee` package

The notebook (`eda/rsna-knee-data-structure-eda-baseline.ipynb`) is where the EDA and the
narrative reasoning behind each pipeline decision live. `src/rsna_knee/` is the same
training/inference pipeline reorganized into a plain-`.py`, module-per-concern package —
DICOM ingestion, laterality/slot resolution, physical-scale caching, gold+LLM label
merging, the DINOv2+SlotHead model, the 4-fold CV training loop, and rank-mean submission
writing each get their own module.

`src/rsna_knee/mil/` adds the CoAtNet attention-MIL pipeline: DICOM → slice stacks that reproduce the
public checkpoints' preprocessing exactly, multi-arm inference sharing one decode per study across all
arms and both GPUs, the exact recipe of the public pre-decoded training corpus, canonical anatomical
orientation, a corpus trainer (fp16, gradient checkpointing, single-GPU A/B pairs or DDP), and the
offline submission. See `src/rsna_knee/README.md` for the module map.

## License

No license has been chosen yet for the code in this repo. Competition data is subject to
Kaggle's competition rules and is not redistributed here. Public checkpoints and label tables used by
the MIL pipeline keep their authors' licenses (e.g. RadImageNet-derived artifacts are CC-BY-NC-SA-4.0).
