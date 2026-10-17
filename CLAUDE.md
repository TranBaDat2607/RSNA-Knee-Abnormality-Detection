# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository. The repo is driven with the
**`kaggle-grandmaster` plugin** (project-scoped, see "Operating mode"); the long history that used to live
in this file is archived in `docs/archive/CLAUDE_md_2026-10-15_pre-rewrite.md` and in `docs/`.

## What this repo is

Kaggle's **RSNA Knee Abnormality Detection** (slug `rsna-knee-abnormality-detection`, a **Code Competition**:
offline notebooks, internet off, 9 h limit, hidden test of ~1,322 studies). Predict twelve findings (ACL, MCL,
medial/lateral meniscus, medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion, fracture) from knee
MRI; scored as macro ROC AUC over the twelve targets. Only 58 training studies ("gold-58") carry expert labels;
every study has a free-text report, which this repo turns into training targets with an LLM labeler.

**Deadline: 2026-10-22 (23:59 UTC). 5 submissions/day, 2 final picks.**

### Where things stand (public LB, read 2026-10-15)

| Role | Submission | Public LB | Notes |
|---|---|---|---|
| OAI-track final pick | A12 `57197656` | **0.957** | A10 + nartaa s6fall as a third equal leg; rank ~211 / 6,043 |
| Clean-track hedge | C3 `57186035` | 0.946 | no gated external data; ~7.5 h of the 9 h limit |
| Own models alone | R5t 0.922 > R3 0.920 > R4 0.917 | | retired from the OAI track (cost ≈ −0.001 each time) |

Two tracks, by the user's decision on 2026-10-10: **OAI** may use any public weights (including ones trained
on OAI/fastMRI/SKM-TEA; the hosts have not ruled); **clean** uses only competition data and public weights
already used before. **Never download OAI data ourselves.** Full account: `docs/phase3_report.md`,
`docs/phase3_notes.md`.

## Operating mode: automation with the kaggle-grandmaster plugin

Plugin: `kaggle-grandmaster@claude-code-kaggle-grandmaster` (declared in `.claude/settings.json`). It provides
the `kg-*` commands, 9 agents, and hooks (session brief, skill router, submission/GPU-push guard, run logger).

**Authority (user, 2026-10-15): Claude may run the whole loop autonomously, without asking per action** —
push Kaggle kernels (GPU and CPU), poll/collect their outputs, download datasets, run
`kaggle competitions submit`, and commit/push to git (branch + PR by default; push to `main` when the user
says so). Do not stop to ask for approval on these.

**Guardrails that still apply in full-auto:**
- Stay within the daily submission budget (5/day) and the cached GPU quota (`kaggle quota`; the plugin hooks
  may still ask when it is exhausted). Quota resets weekly (next: 2026-10-17 00:00 UTC) and does not roll over —
  use it, but only on already-validated or pre-registered work, never untested exploration.
- **Pre-register** the decision rule (what score/gold-58 result means keep/discard) in the docs *before* any
  score is read. Choose weight forms before looking at gold-58.
- **Audit every external notebook/dataset source before pushing**: internet off, only `--no-index` wheel
  installs, no network calls, no secrets. Then a dev run on the 3-row public `test.csv` (schema, NaN, all
  stages complete). Submit only a file that passes `python -m kgkit validate`.
- Final-two selection is ticked on the Kaggle website — only the user can do that. Prepare the evidence table
  (`/kg-final`) and say which two.
- Never print/commit credentials (`~/.kaggle/access_token`, `.env`). Do not publish teammates' private
  notebooks or datasets in this public repo (the external `tranbadat/rsna-knee-abnormality` notebook is
  deliberately **not** in git). Keep our own kernels/datasets private.
- Don't delete kernels/datasets/branches or force-push without being asked.
- Keep the solution independent: public ideas and public notebooks are fine; no merging with other teams.

**Start of every session:** `python -m kgkit status` (or `/kg-status`) — accepted baseline, best CV/LB, GPU
quota, deadline countdown, backlog top, submissions left today. `KGKIT_HOME` is set by the SessionStart hook;
otherwise `PYTHONPATH=~/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/<ver>`.

### The loop

```
kg-ideas → backlog (kgkit backlog) → pre-register rule → build/audit kernel → dev run on 3-row test
→ kaggle kernels push (GPU: kgkit gpu build/push/wait --collect) → collect outputs → ledger add / lb / decide
→ submit only if the pre-registered rule says so → record LB in the ledger → commit result note
```

| Task | Command |
|---|---|
| Where do we stand | `/kg-status`, `python -m kgkit status` |
| Rank next ideas | `/kg-ideas` (error analysis, public notebooks, prior art, gold-58 correlations) |
| One experiment | `/kg-experiment`; `python -m kgkit ledger add <name> <cv> --parent <id> --notes ...` |
| Compare vs accepted baseline | `python -m kgkit ledger compare <new> --truth data/train.csv:<col> --folds ...` (needs OOFs; most of our decisions are gold-58/LB-based, see below) |
| Train on Kaggle GPUs | `/kg-gpu`; `python -m kgkit gpu quota\|plan\|build\|push\|wait\|collect` |
| Blend | `/kg-ensemble`; `python -m kgkit blend ...` |
| Validate / submit | `python -m kgkit validate sub.csv data/sample_submission.csv`; `kaggle competitions submit rsna-knee-abnormality-detection -k <user>/<kernel> -v <ver> -f submission.csv -m "<ledger id> | <what>"` |
| CV↔LB disagreement | `/kg-debug` |
| Final picks | `/kg-final` |

### State files (workspace root; local, not the source of truth for results)

- `.kaggle-gm/competition.json` — metric, code-competition limits, deadline, 5 subs/day, 2 final picks.
- `.kaggle-gm/ledger.jsonl` — experiments with CV (here: **gold-58**, n=58, SE ≈ 0.02), public LB, parent
  baseline, decision. **Accepted baseline = the last KEEP, not the best CV** (currently A12 `0008_a12`).
  Re-mark with `kgkit ledger decide <id> baseline` after logging older entries.
- `.kaggle-gm/submissions.jsonl` (imported from Kaggle; 21 submissions), `backlog.json`, `gpu_runs.jsonl`,
  `gpu_quota.json`. `.kaggle-gm/` and `.claude/` are not committed.
- Narrative results/decisions live in git: `docs/phase3_report.md` (latest), `docs/phase3_notes.md`,
  `docs/phase2_notes.md`, `docs/experiment_ledger.md`, `experiments/phase2/README.md` (kernel → result).
  **After every experiment: commit its code, launcher and a result note.**

### Statistics that steer decisions (do not relearn these)

- **Gold-58 cannot resolve ±0.003** (SE ≈ 0.02); on 10-10 two gold gains (+0.0027, +0.0031) gave 0.000 and
  −0.001 on the LB. Use it as a regression guard and for leg-correlation estimates, never to tune weights.
- Public LB is shown to 3 decimals and is noisy at ~0.001; ~940 teams sit at the 0.956 plateau.
- The lever that worked is **a strong (gold ≥ 0.93), decorrelated (ρ ≤ 0.88 vs the existing legs) extra leg**.
  Weight tweaks, TTA, flip views, more seeds/checkpoints of the same family gave nothing measurable.
- Own models: train on report labels (`teach4.csv`, `teach4opus`), not OOF pseudo-labels; ConvNeXt-nano 336 px,
  18–20 epochs; they correlate ≥ 0.87 with the public stacks and no longer help the OAI track.

## Repo layout

```
eda/                    EDA + baseline imaging notebook (narrative; 0.813 baseline)
src/rsna_knee/          module-per-concern imaging package; mil/ = CoAtNet 2.5-D attention-MIL + own k-fold
                         models (cache, kfold, fleet, ours, submit, train/plan)
tests/                  unit tests for src/rsna_knee (DICOM-free; run locally)
scripts/llm_label_gold.py      LLM report labeler
scripts/kaggle_submit.py       offline submission entry (rsna_knee.mil.submit) — E8 / clean-submit
scripts/kaggle_mil_train.py    launcher for rsna_knee.mil.train/plan
scripts/kaggle_kfold_train.py  launcher for own k-fold models (kaggle_kfold_xla.py = TPU, unused)
experiments/phase2/kernels/    every Kaggle kernel folder (kernel-metadata.json + script/notebook); the
                         README there maps kernel → result. analysis/ = local analysis scripts
data/                   gold + LLM label tables, usage log; train.csv/test.csv/DICOMs are gitignored
docs/                   report.md, experiment_ledger.md, phase2_notes.md, phase3_*.md, handoff_*.md,
                         archive/ (previous CLAUDE.md)
environment.yml         conda env `rsna-knee` (CPU-only; ALL GPU work runs on Kaggle)
.claude/settings.json   project plugin + permissions;  .kaggle-gm/  workspace state (local)
```

`data/train.csv` and `data/test.csv` (a 3-row dry-run stub) are downloaded locally; the DICOM series
(`train_series/`, `test_series/`) are large — fetch only if a local step truly needs them.

## Commands (package)

```bash
conda activate rsna-knee                 # or: pip install -e ".[dev,viz,mil]"
pytest                                   # DICOM-free tests; pytest tests/test_x.py::test_y for one
rsna-knee --epochs 6 --folds 2           # notebook-pipeline run (needs data root / Kaggle)
python scripts/llm_label_gold.py --limit 6   # labeler smoke test (needs .env with OPENAI_API_KEY/MODEL)
```

`dicom/sampling.py` and `model/backbone.py` (real pixel decode / DINOv2 weights) are only exercised on Kaggle.
There is no test suite for `scripts/llm_label_gold.py` or the notebook.

## Architecture and validation-leak guards

- `eda/…baseline.ipynb` is the narrative original; `src/rsna_knee/` is the actively developed package
  (DICOM header → laterality → slot picking → physical-scale decode cache → gold+LLM target merge →
  DINOv2 + `SlotHead` → 4-fold CV → rank-mean submission; `pipeline.py` orchestrates, `config.py` holds every
  tunable). `src/rsna_knee/README.md` has the module map. The rule-based extractor in the notebook is
  historical (0.814) and not in the training-target path.
- `src/rsna_knee/mil/` is shipped to Kaggle as the private dataset `tranbadat/rsna-knee-code` (a zip of
  `src/rsna_knee` plus label tables `teach4*.csv`, `targets_r*.csv`, which are **not** in git). Script kernels
  upload one file, so launchers walk `/kaggle/input` for `rsna_knee/__init__.py`. Re-version the dataset
  after changing `src/` before pushing a kernel that depends on it.
- **Two guards to preserve in any change:** folds are grouped by **report-text hash** (identical reports and
  therefore identical derived labels stay in one fold), and each fold's gold holdout excludes studies that
  fold trained on. `kgkit folds` must use the same grouping.
- Labels: gold-58 use real `train.csv` labels at `gold_weight` 3.0; everything else uses graded 0–1 LLM or
  report-derived targets (`data/llm_labels_full.csv` from `gpt-5.4-mini`, 0.869 gold agreement AUC; best
  clean teacher mean `teach4opus` 0.9014).

## Competition operations (hard-won)

- **Auth:** Kaggle CLI 2.x needs the new-style token in `~/.kaggle/access_token` (username `tranbadat`).
  The legacy `kaggle.json` key cannot push kernels or use competition endpoints.
- **Submitting a code competition:** `kaggle competitions submit <slug> -k <user>/<kernel> -v <version>
  -f <output filename the kernel wrote> -m <msg>` — `-f` is the kernel's output name (`submission.csv`), not a
  local path; a raw file upload fails with a bodyless 400. The kernel **version must reference only
  versioned datasets** (`FAILED_PRECONDITION: non-versioned datasets` shows only in the response body).
- **Grading time:** a submission re-runs the notebook on the hidden test. Budget **2.5 h (E8) to ~7 h (blends)**
  of the 9 h limit; the CLI shows no progress while PENDING. Dev runs on the 3-row test take minutes and do
  **not** predict hidden-test runtime — measure per-stage time from the kernel log before adding any leg to A12.
- **Quota/GPUs:** 30 h/week GPU (the quota API once misreported 6 h), max 2 concurrent GPU sessions, 1 TPU
  session (TPU queue was too long; unused). Use `machine_shape: "NvidiaTeslaT4"` (2×T4); P100 is unsupported
  by the image's PyTorch (sm_60). T4 has no bf16 → fp16 autocast + GradScaler. ~4 vCPUs and ~29 GB host RAM:
  decode-bound inference; a host-OOM only shows as `terminated with signal SIGKILL` — run stages in
  subprocesses, open memmaps lazily, ≤ 2 non-persistent DataLoader workers; DDP + timm gradient checkpointing
  needs `static_graph=True`.
- **Monitoring:** `kaggle kernels status <k>`; fetch only logs/receipts with
  `kaggle kernels output <k> -p <dir> --file-pattern '(\.log$|\.json$)'`.
- **Old kernel versions can't be pulled** (`kaggle kernels pull <k>/<ver>` → 403). Latest only; save every
  submitted kernel into `experiments/phase2/kernels/` at push time.
- A12 is time-guarded and falls back to A10's file if the s6fall stage fails/runs out of time/covers < 90 % of
  studies; keep that property for anything added.
- Windows only: `kaggle datasets create|version -p <abs path>` crashes — `cd` to the parent and pass a
  relative folder.

## Key findings by phase (details in docs/)

- **Phase 1 (labels, to 2026-08):** LLM labeler `gpt-5.4-mini` over all 4,407 reports (2.4M tokens, shared
  2.5M/day pool across the mini/nano family; resumable; `reasoning_effort: "none"`). Baseline notebook 0.813;
  the external multi-person ensemble notebook (not in git) 0.910/0.909 — blend reshuffling among the same three
  arms hit its ceiling. EfficientNet-B3 second arm: weak/mixed, not pursued.
- **CoAtNet MIL program (to 2026-09-12):** the public 0.94 stack rests on the CoAtNet-RMLP-2 @384 2.5-D MIL
  family; pipeline/label diversity adds, more checkpoints of the same pipeline and backbone swaps do not.
  E8 clean baseline 0.939 (Raptor v5/v10/v8 + residual-gated CoAtNet 0.4). Canonical-orientation windows were
  rejected (E5b). Details: `docs/report.md`, `docs/experiment_ledger.md`.
- **Phase 2 (own k-fold models, to 10-10):** R3/R4/R5t as above; blends with the public 0.943 notebook peaked at
  0.942. Pre-registered blend weight rule: solo < 0.925 → 0.10; 0.925–0.934 → 0.20; ≥ 0.935 → 0.35.
  Details: `docs/phase2_notes.md`, `docs/handoff_2026-10-04.md`.
- **Phase 3 (endgame, 10-10 → 10-15):** forked public stacks after auditing them (OAI track), nartaa's
  OAI CoAtNet + heliosli S6/R7 legs, then A12's third equal leg. Clean track: community stack + goodpjw reader
  + our fine-tuned ftC5o (C3). Rejected: our R5t as a leg, ftO, multi-scale TTA, flip TTA, S6 in the compact
  slot, 64-slice maxspan. Details: `docs/phase3_report.md`.

## Open items (update here when they change)

- Another strong, decorrelated leg for A12: phase 4 tried three and all failed their gates (S6T 0.9315 but ρ s6fall 0.945;
  R6 swap = A14, public 0.957 = A12; V1 OrthoFoundation ViT-L ρ ≤ 0.89 but gold 0.896). Only a new public release that
  passes an audit is still worth testing (`experiments/phase4/analysis/watch_public.py 0.957` daily). Details:
  `docs/phase4_notes.md`.
- Clean track: ep0 fine-tune checkpoints (gold 0.925–0.927 vs ep1) are a cheap candidate.
- Final picks: A12 + C3 unless the hosts rule on OAI (then re-decide with `/kg-final`).
- Use the GPU quota that expires 2026-10-17 00:00 UTC only on pre-registered work.
