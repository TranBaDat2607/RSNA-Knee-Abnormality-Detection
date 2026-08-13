# `rsna_knee`

Module-per-concern rewrite of `eda/rsna-knee-data-structure-eda-baseline.ipynb`'s imaging
pipeline. The notebook stays the place for narrative EDA and one-off analysis; this
package is what actually trains and predicts, in plain `.py` files with unit tests.

Not ported here: the notebook's multilingual rule-based report extractor (section 2).
It's no longer what feeds the model — training targets come from `train.csv`'s own
annotated rows plus the LLM-generated labels in `data/llm_labels_full.csv` (see
`CLAUDE.md`) — so it stays notebook-only, kept there as the historical comparison
baseline `EDA_BASELINE_RESULTS.md` measures against.

## Module map

| Module | What it does |
|---|---|
| `config.py` | Every tunable constant (targets, slot definitions, hyperparameters), grouped into dataclasses. `Config()` with no arguments reproduces the notebook's run. |
| `paths.py` | Finds the competition data root, DINOv2 weights, and `llm_labels_full.csv` — on Kaggle (`/kaggle/input/...`) or locally (`data/`). |
| `logging_utils.py` | The one relative-time `log()` every module shares. |
| `dicom/header.py` | Header-only DICOM probing (`walk`) and sequence-type recovery (`annotate`: fat-sat, T1/T2/PD weighting). |
| `dicom/laterality.py` | Left/right resolution (DICOM tag, with a patient-position fallback that only activates once it's shown to agree with the tag), and `normalise_laterality` (mirrors every study onto one knee convention). |
| `dicom/slots.py` | Picks one series per (plane x fat-sat/weighting) slot per study. |
| `dicom/sampling.py` | Reads one series at a fixed physical scale (`crop_mm` -> `img` px) with 1st/99th-percentile intensity normalisation. |
| `dicom/cache.py` | Decodes every chosen series once into a shared `uint8` in-memory array (`build_cache`), and `take_group` for slicing out one group of slices at a time. |
| `labels/targets.py` | Merges gold annotations (`train.csv`) and LLM labels (`llm_labels_full.csv`) into `(Y, W)` graded-target/sample-weight arrays, plus report-hash fold grouping. |
| `model/heads.py` | `SlotHead` — per-diagnosis masked-softmax attention over a study's slot embeddings. |
| `model/network.py` | `Model` — flattens the slot bag for the encoder, folds it back for the head. |
| `model/backbone.py` | Loads DINOv2, freezes all but the last `unfreeze_last` transformer blocks. |
| `model/ema.py` | `Ema` — exponential moving average of the trainable weights, used for epoch selection. |
| `training/augment.py` | Small affine (rotate/scale/shift) + intensity jitter. No horizontal flip, ever; vertical flip is off by default — both would destroy anatomy the targets depend on. |
| `training/losses.py` | `rank_loss` — a pairwise ranking term on confidently-graded pairs, supplementing BCE. |
| `training/metrics.py` | `macro_auc`, `hanley_mcneil_se`. |
| `training/loop.py` | `train_one_fold` — two-speed AdamW, OneCycle schedule, EMA, dual-reference (`min(derived, annotated)`) epoch selection. |
| `inference/predict.py` | Group-averaged, sigmoided prediction. |
| `inference/submission.py` | Writes the 0.5 benchmark file up front and the rank-mean-ensembled real submission once folds finish. |
| `pipeline.py` | `run()` / `run_safe()` — orchestrates all of the above into the same 4-fold CV run `main()` performs in the notebook. |
| `cli.py` | `python -m rsna_knee.cli` / the `rsna-knee` console script. |

## Running it

Needs the competition's DICOM data mounted (Kaggle) or populated under `data/` locally —
this repo does not ship it. From the data root:

```bash
pip install -e .
rsna-knee                          # full run, notebook defaults
rsna-knee --epochs 6 --folds 2     # quicker iteration
rsna-knee --slot-scheme public     # the alternative slot definition (see config.py)
```

Or from Python:

```python
from rsna_knee.config import Config
from rsna_knee.pipeline import run_safe

cfg = Config()
cfg.train.epochs = 6
result = run_safe(cfg)
print(result.fold_scores, result.oof_gold)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Everything that doesn't require Kaggle's mounted DICOM data or downloaded DINOv2 weights
is covered: metrics, augmentation (shape/dtype/orientation preservation), laterality
resolution, header sequence-typing, slot picking, gold/LLM label merging and fold
grouping, the model's shape wiring (via a stub backbone satisfying the HF `AutoModel`
interface), EMA, losses, submission writing, and path resolution. `dicom/sampling.py`
(actual pixel decoding) and `model/backbone.py` (actual DINOv2 loading) are exercised on
Kaggle, not by local tests.
