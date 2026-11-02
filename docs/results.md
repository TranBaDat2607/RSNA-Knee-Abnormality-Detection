# Full results

Every scored phase, in order. The final standing and the summary are in the [README](../README.md); the raw list of
submissions is [`reports/submissions_final.csv`](../reports/submissions_final.csv).

All scores are from the Kaggle submissions list, read 2026-10-23 after the competition closed.
Gold-58 is the macro AUC on the 58 expert-labeled training studies, held out from every model scored on it.

### Report labels → DINOv2 baseline (original contribution)

| Label source | Public LB | Private LB |
|---|---:|---:|
| Original (rule-based report extractor, Roman Rozen baseline, version forked here) | 0.809 | — |
| **This repo (`gpt-5.4-mini`-labeled targets)**, submission `55429575` | **0.813** | **0.804** |

The same run's out-of-fold macro AUC on the 58 held-out annotated studies is 0.7675
(`EDA_BASELINE_RESULTS.md` §3). The baseline notebook has since been updated upstream and now scores 0.894.

### Earlier ensemble notebook (2026-08)

| Submission | What | Public LB | Private LB |
|---|---|---:|---:|
| `55503088` | my earlier ensemble notebook (DINOv2 TTA + DINOv3 + RadImageNet, `challenger` blend) | 0.910 | 0.889 |
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
