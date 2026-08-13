# EDA & Baseline Results — RSNA Knee Abnormality Detection

Summary of `eda/rsna-knee-data-structure-eda-baseline.ipynb`, read against the task
defined in `docs/requirements.md` (predict 12 per-study binary findings from knee MRI,
scored as macro-averaged ROC AUC across labels).

## 1. Dataset facts surfaced by the EDA

- **4,407 training studies**, each with a free-text `Report` and up to 12 binary labels —
  but only **~58 studies (~1.3%) carry a full set of annotated labels**. Everything else
  has to be supervised some other way, which is why the notebook builds a report-label
  extractor before touching images.
- **24,371 training series** across those studies → **mean 5.24 of 6 possible "slots"
  present per study** (min 3, max 6), where a slot = one anatomical plane × fat-sat/fluid
  combination.
- Sequence type × fat-suppression breakdown (series counts):

  | Weighting | Non-fat-sat | Fat-sat |
  |---|---:|---:|
  | PD  | 2,724 | 10,922 |
  | T1  | 5,299 | 84 |
  | T2  | 2,188 | 2,926 |
  | GRE | 226 | 0 |
  | Unknown | 0 | 2 |

  PD fat-sat is by far the most common series type; T1 is almost always non-fat-sat, as
  expected physically.
- **Laterality (L/R) tag is present on only 50% of studies.** Where both the DICOM tag and
  the patient-position sign are available, they agree 91.4% of the time (n=2,050
  comparable studies) — good enough to trust a position-based fallback, which raises
  laterality coverage from 50% → **98%**.
- Reports span **9 languages** (EN, ES, FR, NL, DE, TR, HR/SR/BS, EL, BG/RU), detected via
  Unicode script + stopword-margin voting rather than substring heuristics.
- `test.csv` has **no `Report` column** — a structural fact that rules out any
  image+text fusion model at inference and forces the report to be used only for
  (a) generating training targets and (b) confidence-weighting studies.

## 2. Report-label extractor (turns text into training targets)

A multilingual, clause-level rule extractor classifies each clause as
assertion / negation / hedge, matches anatomy + pathology stems per target, and grades
severity — producing a graded proxy label plus a per-study confidence weight (used later
as a training sample weight).

**Validation against the ~58 gold-annotated studies** (Hanley–McNeil 95% CI, mean AUC
0.814 ± 0.125):

| Target | Agreement AUC | Silence rate |
|---|---:|---:|
| MCL | 0.940 | 30.2% |
| ACL | 0.930 | 28.8% |
| Baker's | 0.916 | 55.3% |
| Medial Meniscus | 0.895 | 29.2% |
| Lateral Meniscus | 0.865 | 32.2% |
| Medial OA | 0.783 | 75.5% |
| Fracture | 0.778 | 80.0% |
| Contusion | 0.778 | 63.4% |
| PF OA | 0.773 | 39.1% |
| Lateral OA | 0.763 | 78.3% |
| Effusion | 0.719 | 15.7% |
| **Synovitis** | **0.628** | **88.2%** |

Synovitis is both the hardest label to extract reliably *and* the one the extractor says
the least about (88% of reports yield no mention at all) — it's flagged in the notebook
as the top target for lexicon improvement.

## 3. Baseline model

**Architecture**
- DINOv2-small backbone, partially unfrozen (last 6 of 12 transformer blocks + final
  norm trainable, 10.7M trainable params), two-speed optimizer (backbone LR 8e-6, head LR
  1e-3) — a frozen-encoder baseline was tried first and saturated, motivating the
  fine-tune.
- Each study reduced to up to 6 "slots" (3 planes × weighting axis), sampled at a
  **fixed physical scale** (160mm crop → 224px, 0.71mm/pixel) rather than fixed-pixel
  resize, so anatomy occupies a consistent number of pixels regardless of acquisition.
- Left/right knees normalized to one convention (flip for coronal/axial, slice-order
  reversal for sagittal) so the 4 medial/lateral-paired labels aren't asked to learn an
  axis they can't observe.
- A small per-diagnosis attention head (`SlotHead`) pools the 6 slot embeddings with a
  masked softmax — one query vector per target — rather than a plain mean, since each
  finding is best seen on particular sequences/planes.
- Targets are the report-derived graded labels from §2, weighted by extractor confidence.
- Training: affine + intensity augmentation only (**no vertical or horizontal flip** —
  flipping would swap femur/tibia surfaces or undo the laterality normalization), weight
  EMA (decay 0.997), small pairwise ranking loss term (weight 0.05) added to the
  cross-entropy loss.

**Validation setup**
- 4-fold CV, folds assigned by a hash of report text (keeps studies sharing an identical
  report — and therefore an identical derived label — inside one fold, closing a leakage
  path).
- Two references tracked per epoch: AUC against derived (report-based) labels on the
  fold's holdout, and AUC against the annotated studies held out of that fold. Epoch
  selection uses `min(derived, annotated)` so an improvement on one reference alone
  doesn't win.

**Results**

| Fold | Epoch-selection score (worse-of-two) |
|---|---:|
| 0 | 0.7706 |
| 1 | 0.7665 |
| 2 | 0.7364 |
| 3 | 0.7335 |

**Out-of-fold macro AUC on the 58 held-out annotated studies (the test-like metric): 0.7675**

Per-target OOF AUC:

| Target | OOF AUC |
|---|---:|
| Medial OA | 0.918 |
| Effusion | 0.908 |
| Lateral OA | 0.855 |
| Baker's | 0.808 |
| Contusion | 0.760 |
| Fracture | 0.751 |
| Lateral Meniscus | 0.743 |
| PF OA | 0.725 |
| MCL | 0.721 |
| Synovitis | 0.700 |
| ACL | 0.695 |
| **Medial Meniscus** | **0.626** |

Predictions from the 4 fold models are combined by **rank mean** (not averaged
probabilities), since the competition metric only rewards correct ordering.

Full run (data caching + 4-fold training) completed in ~100 minutes in the notebook's own
timer — comfortably inside the 9-hour code-competition budget.

## 4. Reading the two result tables together

- **Medial OA and Effusion are the strongest targets** (>0.9 OOF AUC) — both have decent
  positive counts and comparatively unambiguous imaging signal.
- **Medial Meniscus and ACL are the weakest imaging results (0.626, 0.695)** despite the
  *extractor* agreeing very well with annotations for both (0.895, 0.930). This points to
  the image model, not the label quality, as the current bottleneck for those two targets.
- **Synovitis inverts this pattern**: worst label-extraction agreement (0.628) and highest
  silence rate (88%), yet a middling imaging AUC (0.700) — the few confidently-labeled
  cases are apparently learnable even though most reports say nothing usable about it.

## 5. Where the notebook flags future work

- **Silence rate is the priority work queue** — Synovitis (88%), Fracture (80%), and the
  three OA compartments (75–78%) are the targets most starved of extracted signal; a
  frozen encoder plateaued regardless of resolution/coverage/aggregation changes, so the
  binding constraint was representation, not sampling — addressed here by partially
  fine-tuning DINOv2.
- **No calibration, thresholding, or class rebalancing** — correctly out of scope, since
  AUC only depends on ranking and the competition explicitly warns prevalence may shift
  between train/public/private sets.
- **Next likely wins**: broaden the report lexicon for the high-silence targets, add a
  second/larger backbone per fold to decorrelate ensemble errors (rank-mean combination
  already supports this), and explore the resolution vs. slice-coverage trade-off, which
  is currently fixed but noted as untested.
