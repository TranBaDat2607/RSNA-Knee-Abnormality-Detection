# Team-merge outreach (draft — to be posted by the account owner)

Merge deadline: **2026-10-15**. Candidates seen in the competition forum on 2026-10-02 (topic ids for
`kaggle`/kagglesdk lookups):

| Poster | Topic | What they offer |
|---|---|---|
| KalyanG17 | 745091 "Looking for teammates: 0.942 solo, own 5-fold models + a GPU to share" | CoAtNet@384 (0.935), ConvNeXt-T (0.930), RadImageNet R50 (0.924) on own LLM labels, 5-fold OOF, RTX 5070 |
| Md Kishor Morol | 744230 (bf16 PSA) | independent DINOv2 ViT-S arm, 6 series × 16 slices @336, 5 folds, shares no weights with the public stack; wants a 0.945+ team |
| Prateek Grover | 743374 (fast submissions) | CoAtNet inference < 10 min; wants teammates strong on labels or with a resilient ≥ 0.940 single model |
| (topic) | 744511 "Looking for teammates—open to merging or collaborating" | — |

What we bring (facts to quote):
- Public LB **0.942** (and 0.939 from a clean, no-LB-tuned CoAtNet pipeline).
- Our own 5-fold ConvNeXt-nano 336 px models (R3/R4), trained from scratch on our own soft labels
  (4 leak-checked public label tables + our LLM labels + OOF pseudo-labels): solo LB 0.917–0.920, OOF macro
  0.880 vs report labels, full OOF for all 4,349 training studies — usable for weight fitting on a merged team.
- A tested offline submission framework: shared one-pass DICOM decode for many arms, fixed-weight rank fusion,
  automatic fallback if any arm fails; 30 GPU-h/week of Kaggle quota.
- A leak audit of ~20 public label tables (9 contain the gold labels verbatim).

## Draft reply (for topic 745091, adapt for the others)

> Hi @KalyanG17 — interested in merging. We're at 0.942 public. Our own models are 5-fold ConvNeXt-nano
> @336 trained on our own soft labels (OOF for every training study, solo LB 0.920), plus a CoAtNet pipeline
> that reproduces the public Raptor family exactly from DICOM and a submission harness that decodes each study
> once for all arms (whole stack well inside the 9 h). We've also leak-checked ~20 public label tables against
> the 58 gold studies (9 contain the gold labels verbatim), which matters if you validate on gold.
> Your CoAtNet/ConvNeXt/RadImageNet on your own LLM labels sound like exactly the kind of independent arms our
> stack lacks. Happy to start by comparing OOF predictions on a shared set of studies to measure correlation
> before committing. Reach me at: <your Discord / email>.
