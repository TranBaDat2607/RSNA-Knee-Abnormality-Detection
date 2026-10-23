# RSNA Knee Abnormality Detection

Work on Kaggle's [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition: predict twelve clinically important findings (ACL/MCL injury, meniscus tears,
three-compartment osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture)
from knee MRI studies, scored as macro-averaged ROC AUC across the twelve targets.

## Final result

| | |
|---|---|
| **Private leaderboard** | **0.943**, rank **340 / 6,425** teams (top 5.3 %) — **🥉 Bronze medal** |
| Public leaderboard | 0.958, rank 443 |
| Scored submission | `57490753` (A20, OAI track): the heliosli public 0.9586 recipe + nartaa s6fall leg + **our S6T** model (heliosli's S6 recipe retrained on our own report labels) sharing the S6 slot |
| Team | `nammovuivui` (solo, Kaggle user `tranbadat`) |
| Competition | Code competition, 2026-08-05 → 2026-10-22, 7,683 competitors, $77,000 |

Medal cutoffs on the private leaderboard: gold 0.954 (rank 22), silver 0.943 (rank 321), bronze 0.942 (rank 642).
We tie the silver-cutoff score at three decimals but rank 340 on the tie-break. The winning score is 0.960.

The leaderboard was read on 2026-10-23, the day after the deadline. Kaggle has not finalized it yet: removing
disqualified teams can still move the rank, but the 0.943 score will not change.

Every submission we made dropped about 0.015 from public to private, and their order mostly held. Our best
private score on any submission was 0.943 (A14, A17, A20, A22), and the scored pick is one of them. The clean
track (no OAI-trained weights) finished at 0.931 (C3).

## What this repo contains

Only ~1.3% of training studies carry ground-truth labels; every study carries a free-text
radiology report instead. The work came in five parts:

1. an **LLM-based report labeler** that turns those reports into training targets (the original
   contribution, below);
2. a **CoAtNet 2.5-D attention-MIL pipeline** (`src/rsna_knee/mil/`) — the representation behind the
   ~0.94 public ensembles — rebuilt as tested, Kaggle-validated code, together with a hypothesis-driven
   study of what it would take to beat that ensemble. Start with
   [`docs/report.md`](docs/report.md); every experiment is logged in
   [`docs/experiment_ledger.md`](docs/experiment_ledger.md);
3. **Phase 2 — our own k-fold models** (`experiments/phase2/`): ConvNeXt-nano models trained on report labels,
   scored alone and as a small leg blended into the public stack. Results in
   [`docs/phase2_notes.md`](docs/phase2_notes.md); how to submit in [`docs/handoff_2026-10-04.md`](docs/handoff_2026-10-04.md).
4. **Phase 3, endgame** (`experiments/phase2/kernels/{oai-blend,clean-blend,c96-cache,ft96*}`): two submission
   tracks (public OAI-trained weights vs clean), plus our own CoAtNet fine-tuned on the 96-slice native-384 input.
   Results in [`docs/phase3_notes.md`](docs/phase3_notes.md) and [`docs/phase3_report.md`](docs/phase3_report.md).
5. **Phase 4, final week** (`experiments/phase4/`): a search for one more strong, decorrelated leg (S6T, R6,
   OrthoFoundation ViT-L, V-JEPA 2.1, DINOv2 ViT-S), and the final A16–A26 submissions. Results in
   [`docs/phase4_notes.md`](docs/phase4_notes.md); the final-pick reasoning in
   [`reports/final-selection.md`](reports/final-selection.md).

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
(flight0234, stevenleehans, yunusgmsoy, pilkwang). The 0.942 blend in phase 2 uses the public
`jiweiliu/rsna-knee-fast-2xt4-inference` notebook (0.943) unchanged and adds our own model as a
small-weight leg. The final OAI-track submissions build on public notebooks and weights by heliosli (R6/R7/S6
legs and the 0.9586 recipe), nartaa (OAI-trained CoAtNet and the s6fall leg), goodpjw2008 (2.5-D ConvNeXt
reader) and varshithakolupuri.

## Results

All scores are from the Kaggle submissions list, read 2026-10-23 after the competition closed.
Gold-58 is the macro AUC on the 58 expert-labeled training studies, held out from every model scored on it.

### Report labels → DINOv2 baseline (original contribution)

| Label source | Public LB | Private LB |
|---|---:|---:|
| Original (rule-based report extractor, Roman Rozen baseline, version forked here) | 0.809 | — |
| **This repo (`gpt-5.4-mini`-labeled targets)**, submission `55429575` | **0.813** | **0.804** |

The same run's out-of-fold macro AUC on the 58 held-out annotated studies is 0.7675
(`EDA_BASELINE_RESULTS.md` §3). The baseline notebook has since been updated upstream and now scores 0.894.

### External ensemble notebook (2026-08)

| Submission | What | Public LB | Private LB |
|---|---|---:|---:|
| `55503088` | teammate notebook `tranbadat/rsna-knee-abnormality` (DINOv2 TTA + DINOv3 + RadImageNet, `challenger` blend) | 0.910 | 0.889 |
| `55523770` | same, `grounded` blend (drops the weights fit on the 3-row dry-run test) | 0.909 | 0.889 |
| `55569274` | same, per-target weights fit on LLM-labelled OOF | 0.910 | 0.890 |

### CoAtNet MIL pipeline (2026-09 study)

| Model / pipeline | Gold-58 | Public LB | Private LB |
|---|---:|---:|---:|
| DINOv2 slot model (public 20-member OOF) | 0.840 | — | — |
| Single public CoAtNet checkpoint (v5 / v10 / v8) | 0.920 / 0.917 / 0.912 | — | — |
| Residual-gated CoAtNet (e4/e6/e8) | 0.909 | — | — |
| **Clean pipeline E8: v5 + v10 + v8 (equal) + residual-gated (0.4)**, `56184308` | **0.930** | **0.939** | **0.921** |
| Public 0.94 notebooks (adds DINO/RadImageNet chain, LB-probed weights) | — | 0.939–0.941 | — |

The clean pipeline matches the public 0.94 notebooks with four arms, fixed weights and ≈ 2.5–3 h of the 9 h
runtime limit — no DINO/RadImageNet chain and no LB-probed per-target weights. Gold-58 has a macro-AUC standard
error of ~0.02, so it guards against regressions rather than ranking close candidates.

### Phase 2: our own models (2026-10)

Own ConvNeXt-nano 336 px, 18 epochs, 5-fold models on the public pre-decoded corpus. "Solo" means our model alone,
no public stack.

| Submission | What | Gold-58 | Public LB | Private LB |
|---|---|---:|---:|---:|
| `56654316` | R1: public 0.943 + our 3-fold ConvNeXt-nano, rank blend w = 0.35 | 0.905 | 0.938 | 0.924 |
| `56753338` | R1 leg alone (3-fold) | 0.905 | 0.917 | 0.907 |
| `56787175` | R3 solo (`targets_r2`: report labels + OOF mix) | 0.898 | 0.920 | 0.906 |
| `56787692` | **public 0.943 unchanged + R3 at w = 0.10** | — | **0.942** | **0.926** |
| `56799230` | R4 solo (heavy-OOF `targets_r4`) | 0.891 | 0.917 | 0.903 |
| `56803647` | R5t solo (report labels only, no OOF) | 0.905 | 0.922 | 0.912 |
| `56814752` | R5t solo + 2 full-data models (all 4,349 studies) | — | 0.924 | 0.913 |

Training on report labels alone (R5t) beats mixing in out-of-fold pseudo-labels (R3, R4) on gold-58 and on both
leaderboards, and two extra full-data models add +0.001–0.002.

### Phase 3: endgame (2026-10-10 → 10-15)

Since 2026-10-07 every public notebook at ≥ 0.950 uses `nartaa/rsna-knee-publication-swa-weights-20261007`, one
CoAtNet trained with **OAI external knee MRIs**. The hosts did not rule on OAI, so submissions were split into an
**OAI track** (public OAI-trained weights) and a **clean track** (competition data + previously used public weights).
We never downloaded OAI data ourselves.

| Submission | Track | What | Gold-58 | Public LB | Private LB |
|---|---|---|---:|---:|---:|
| `57051397` | OAI | A2: goodpjw 0.950 (nartaa + reader) + nartaa 224-crop checkpoint at 0.25 | 0.9275 | 0.950 | 0.932 |
| `57051394` | OAI | A3: A2 + our R5t leg at 0.10 | 0.9280 | 0.949 | 0.932 |
| `57051391` | clean | C2: community stack + goodpjw reader 0.30 + **our ftC** 0.20 | — | 0.945 | 0.930 |
| `57110710` | OAI | A5: goodpjw 0.954 notebook unchanged (nartaa + heliosli S6 + reader) | — | 0.954 | 0.937 |
| `57185288` | OAI | A6: varshithakolupuri 0.956 notebook unchanged | — | 0.956 | 0.941 |
| `57185425` | OAI | A7: goodpjw v12 unchanged | — | 0.956 | 0.941 |
| `57186035` | clean | **C3: C2 with our ftC5o replacing ftC** (clean-track best) | 0.9242 | **0.946** | **0.931** |
| `57197655` | OAI | A10: heliosli 0.9573 notebook | — | 0.956 | 0.941 |
| `57197656` | OAI | A12: A10 + nartaa s6fall as a third equal leg | 0.9404 | 0.957 | 0.942 |

- **ftC / ftC5o** (ours): public clean Raptor v10 / v5 CoAtNet fine-tuned on nartaa's 96-slice native-384 input
  (`rsna_knee.mil.c96`) with our report labels (teach4, then teach4 + Claude-Opus-4.5 mix), 320 crop,
  anatomical-mirror aug/TTA. Gold-58 0.921 / 0.924, the best clean single models here.
- Rejected, each against a rule fixed before the result: fine-tuning nartaa on teach4, 64-slice maxspan input,
  multi-scale TTA, flip TTA, averaging sibling fine-tunes, and our R5t leg (−0.001 on the LB in both tracks).

### Phase 4: final week (2026-10-15 → 10-22)

The only lever that had worked was a strong (gold ≥ 0.93), decorrelated (ρ ≤ 0.88) extra leg. Six candidates were
tried; most failed their pre-registered gates (OrthoFoundation ViT-L gold 0.896, V-JEPA 2.1 gold 0.80, DINOv2
ViT-S gold 0.714). **S6T** — heliosli's S6 recipe retrained on our teach4opus labels with gold-58 excluded, trained
on Kaggle GPUs and TPUs — reached gold 0.9315 but correlated 0.945 with s6fall, so it went in by sharing the S6 slot
rather than as a new leg.

| Submission | What | Gold-58 | Public LB | Private LB |
|---|---|---:|---:|---:|
| `57269763` | A14: A12 with the compact leg R7 → heliosli R6 plain | — | 0.957 | **0.943** |
| `57461357` | A16: heliosli public 0.9586 notebook unchanged | 0.9387 | 0.958 | 0.942 |
| `57401136` | A17: A16 + nartaa s6fall as a third equal leg | 0.9395 | 0.958 | **0.943** |
| **`57490753`** | **A20: A17 + our S6T sharing the S6 slot with s6fall — final scored pick** | **0.9396** | **0.958** | **0.943** |
| `57507780` | A22: runtime-safe A21 (cohort + s6fall + S6T) | 0.9428 | 0.958 | **0.943** |
| `57507775` | A23: A22 with R6 plain | 0.9439 | 0.957 | 0.941 |
| `57525938` | A24: A22 with S6-slot weight 1/2 | 0.9411 | 0.958 | 0.942 |
| `57526860` | A26: A22 with S6-slot weight 2/5 | 0.9424 | 0.957 | 0.941 |

A19, A21 and A25 exceeded the 9 h runtime limit on the hidden test and have no score.

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
                            MIL pipeline (inference, corpus trainer, own k-fold models, submission)
tests/                      unit tests for src/rsna_knee (run locally without Kaggle data or weights)
scripts/llm_label_gold.py   LLM report-labeling pipeline
scripts/kaggle_submit.py    offline Kaggle submission entry point (rsna_knee.mil.submit)
scripts/kaggle_mil_train.py Kaggle launcher for training plans (rsna_knee.mil.train / plan)
data/                       gold labels, LLM-generated labels, usage logs
docs/report.md              the 2026-09 study: why the 0.94 stack works, experiments, recommended path
docs/experiment_ledger.md   every hypothesis, experiment, result and decision rule, in order
docs/phase2_notes.md        phase 2 study: label work, own k-fold models, A/B results
docs/handoff_2026-10-04.md  phase 2 state, running kernels, how to submit the solo and blend kernels
docs/phase3_notes.md        phase 3: OAI landscape, two-track blends, d96 fine-tunes, pre-registered rules, LB results
docs/phase3_report.md       phase 3 summary
docs/phase4_notes.md        phase 4: extra-leg search (S6T, R6, V1, VJ, V2) and the A16–A26 submissions
docs/requirements.md        competition task description
reports/final-selection.md  evidence table behind the final two picks
experiments/phase2/         Kaggle kernel launchers and analysis scripts for phases 2 and 3
experiments/phase4/         Kaggle kernels and analysis scripts for phase 4
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
pytest
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
