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

### `mil/` — the 2.5-D CoAtNet attention-MIL family

The strongest representation available for this competition (0.91–0.92 gold-58 macro AUC per
public checkpoint vs 0.84 for the DINOv2 slot model above; see `docs/experiment_ledger.md`).

| Module | What it does |
|---|---|
| `mil/recipes.py` | `Slot` / `VolumeRecipe` (which series, how many slices, span, crop, pixel grid) and `ArmSpec` (checkpoint + recipe + eval windows). `PUBLIC_RAPTOR_ARMS` lists the public checkpoints worth running — without the no-op reversed-order view and the redundant v4. |
| `mil/volume.py` | `build_volume` — DICOM study → `uint8` slice stack + mask, reproducing the public checkpoints' preprocessing exactly (geometry ordering, span sampling, 2–98th percentile window, physical centre crop, area resize). |
| `mil/windows.py` | Eval / train window-centre selection and triplet assembly; `to_model_input` resizes and normalises on the GPU. |
| `mil/model.py` | `MILClassifier` — backbone + per-finding attention head, state-dict compatible with the public checkpoints; `load_checkpoint`. |
| `mil/blend.py` | `rank_pct`, `weighted_rank_mean`, `logit_blend`. |
| `mil/infer.py` | `predict_arms` — walks each GPU's shard of studies, decodes a study's slices once for every recipe (memoised listings/pixels), runs every arm, and contains per-study failures to that study. |
| `mil/corpus.py` | `Corpus` — the public pre-decoded 44 × 336 training corpus (both parts), memory-mapped lazily so DataLoader workers never copy it. `build_volume(CORPUS44_336)` reproduces it exactly. |
| `mil/orientation.py` | `slot_transform` / `canonical_transforms` / `transform_windows` — maps every window onto one anatomical orientation (right knees mirrored so lateral is always on the same side); `resolve_side` with a tag-only mode, since patient-x geometry does not encode side on untagged sites. |
| `mil/teachers.py` | `read_teacher` — the public report-label tables used as weak targets (flight hybrid, top-5 mean, Raptor's own), with their measured gold agreement and the list of tables that leak gold labels. |
| `mil/train.py` | `run` / `parse_args` — corpus training loop: fp16 + GradScaler, gradient checkpointing, single-GPU or DDP (static graph), random train windows / even eval windows, gold-58 and a weak-label hold-out scored and saved every eval, checkpoints loadable by `load_checkpoint`. |
| `mil/plan.py` | `run_plan` / `child_main` — executes a plan of stages in isolated subprocesses (two single-GPU jobs side by side, or one DDP job) after fetching pretrained weights once. `scripts/kaggle_mil_train.py` is the one-file Kaggle launcher. |
| `mil/submit.py` | `main(SubmitConfig)` — offline submission: 0.5 benchmark written first, MIL arms via `predict_arms`, the residual-gated CoAtNet through its own packaged runtime, fixed-weight rank fusion that drops arms with constant output, schema validation. `scripts/kaggle_submit.py` is the notebook entry point. |

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

For `mil/` (`pip install -e ".[dev,mil]"`): eval window centres are checked against a verbatim
copy of the public checkpoints' reference implementation on random masks; the head's parameter
names and its invariance to window order; slot picking, span sampling and stack assembly with
injected readers; the physical centre crop (skipped without OpenCV); rank fusion; and multi-arm
orchestration (one decode per recipe per study, failures contained) with stub models. Real DICOM
decoding and checkpoint loading are exercised on Kaggle.
