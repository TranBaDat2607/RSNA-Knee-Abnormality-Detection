# Session summary: 2026-10-14 (phase 3, branch `feat/phase3-oai956`)

**Goal:** raise the public LB above the previous best (0.954, A5 `57110710`).
**Result:** best public LB is now **0.956** (A6 `57185288`). The clean track went from 0.945 to **0.946** (C3 `57186035`).
Two more variants were still grading when this was written (A10, A12; see the end).

Full details, pre-registered rules and numbers are in `docs/phase3_notes.md` (sections from "2026-10-14").

## 1. What changed on the public leaderboard

- heliosli released **R7** (`heliosli/rsna-knee-compact-r7cnxfsss`). It is a ConvNeXt-small with 4 folds, knee-joint anchors
  and fastMRI auxiliary training.
- A 50/50 rank blend of nartaa's CoAtNet (OAI-trained) and R7 scored public 0.956. About 25 public notebooks reached 0.956.
- The top public notebook scored 0.957. It adds a private leg (knee-centred crops), so a fork of it only gives 0.956.
- Late on 10-14, heliosli's "0.9573" notebook appeared (the same recipe, with the 224 checkpoint run without the mirror view).
- nartaa released **s6fall**: heliosli's S6 architecture retrained on all data, with no OAI and with gold-58 excluded. It
  scores 0.950 public alone.
- All of these legs use gated external data (OAI, fastMRI or the SKM-TEA anatomy model), so they belong to the OAI track only.

## 2. Submissions this session

| Submission | Track | What | Public LB | Decision |
|---|---|---|---|---|
| A6 `57185288` | OAI | varshithakolupuri's 0.956 notebook unchanged: nartaa 384 + nartaa 224 (mirror) at 0.35 of the Raptor slot, then R7 50/50 | **0.956** | **OAI final pick** (replaces A5 0.954) |
| A7 `57185425` | OAI | goodpjw v12: A6 + heliosli S6 at 0.3 of the compact slot + reader at 1/10 | 0.956 | tie, so A6 stays (rule: strictly higher) |
| C3 `57186035` | CLEAN | clean C2 with our ftC5o (teach4opus labels, gold 0.924) in place of ftC at 0.20 | **0.946** | **clean final pick** (replaces C2 0.945) |
| A10 `57197655` | OAI | heliosli 0.9573 notebook; R7 read from the versioned dataset (v1 was refused: non-versioned dataset) | pending | replaces A6 only if > 0.956 |
| A12 `57197656` | OAI | A10 + nartaa s6fall as a third equal leg (2/3 A10 + 1/3 rank s6fall), time-guarded | pending | replaces the OAI pick only if strictly higher |

Every variant was audited before it was pushed (internet off, only `--no-index` installs, no network calls). Each one had a
dev run on the 3-row test (schema, NaN and stage-completion checks) and a decision rule recorded before its score was read.

## 3. Gold-58 diagnostics (offline, no submission spent)

All three kernels run on the 58 gold-labelled training studies through a stand-in competition root.

| Leg / blend | gold-58 macro | Note |
|---|---|---|
| nartaa 384 | 0.9229 | selected on gold, so optimistic |
| nartaa 224 | 0.9150 | |
| goodpjw reader | 0.9153 | |
| R7 | 0.9312 | gold never trained on |
| R7 + flip TTA | 0.9317 | Spearman 0.998 with plain R7 |
| heliosli S6 | 0.9267 | |
| **nartaa s6fall** | **0.9314** | gold excluded; Spearman 0.87 with R7 and with nartaa |
| A6/A10 blend | 0.9365 | |
| A11 (s6fall at 0.3 of the compact slot) | 0.9362 | fails its rule, not built |
| **A12 (equal three-way)** | **0.9406** | +0.0042, CI [−0.002, +0.0105], 9/12 findings up. The weight form was chosen after the gold read, so only the LB can validate it |

Ideas tested and not submitted:

- **A8, R7 flip TTA:** +0.0001 in the blend, with about +1.5 h of runtime on the hidden test.
- **Our ftC5o as an OAI-blend leg:** no gain. It correlates 0.93 with nartaa, and our legs have cost about 0.001 on the LB
  twice before.
- **An R7 weight probe:** gold is flat for R7 weights from 0.5 to 0.7.

## 4. Kernels and code added

`experiments/phase2/kernels/`:

| Folder | Purpose |
|---|---|
| `oai-956` | A6 |
| `oai-956s6` | A7 |
| `clean-c3` | C3 |
| `oai-957` | A10 |
| `oai-957s6f` | A12 (`build.py`) |
| `oai-956flip` | A8, staged and not pushed |
| `oai-gold58-r7` | gold diagnostic: R7 plain / flip and S6 |
| `oai-gold58-s6fall` | gold diagnostic: s6fall, run as nartaa's unchanged script through its own `K1_KAGGLE_ROOT` hook |

Analysis script: `experiments/phase2/analysis/gold_compact.py`.

## 5. Where things stand and what is next

- **Final-pick candidates (joint decision with the user):** OAI A6 `57185288` (0.956), or A10/A12 if either scores strictly
  higher; clean hedge C3 `57186035` (0.946).
- **Pending:** read the A10/A12 scores (`kaggle competitions submissions rsna-knee-abnormality-detection`) and record them in
  `docs/phase3_notes.md`.
- **If A12 does not beat 0.956:** the remaining lever is a new, strong, decorrelated leg (several GPU days). About 18 h of GPU
  quota were left until the 10-17 reset. The deadline is 2026-10-22.
- **OAI ruling:** still none from the hosts (forum thread 749145). Keep both tracks.
