# 320th place (solo, bronze): report labels, audited public stacks, and what I missed

Thanks to RSNA, the annotators and Kaggle, and to the authors whose public work this solution relies on: heliosli, nartaa,
goodpjw2008, varshithakolupuri, dreaddevelopment and mattiaangeli.

**Result:** private 0.943, public 0.958, rank 320 of 6,299, solo.
**Code, all experiments and pre-registered rules:** https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection
**Final notebook:** https://www.kaggle.com/code/tranbadat/rsna-knee-oai-a20
**S6T weights:** https://www.kaggle.com/datasets/tranbadat/rsna-knee-s6t-weights

Most of this score comes from public notebooks and weights, so this is mainly a write-up about process and about what did
and didn't move the score.

## The final submission

A rank blend with three parts:

- the public 0.9586 recipe by heliosli (nartaa's OAI-trained CoAtNet, heliosli's R6 ConvNeXt, goodpjw2008's reader)
- nartaa's s6fall weights, as a third equal leg
- my S6T, sharing the S6 slot: heliosli's S6 recipe retrained on my report labels, with gold-58 excluded

| | Public | Private |
|---|---:|---:|
| Baseline notebook + my LLM report labels | 0.813 | 0.804 |
| My own ConvNeXt-nano models alone | 0.924 | 0.913 |
| Best blend without OAI-trained weights | 0.946 | 0.931 |
| Public 0.9586 notebook, unchanged | 0.958 | 0.942 |
| **Final: + s6fall + S6T** | **0.958** | **0.943** |

## What I'd share as useful

**Gold-58 can't rank close candidates.** Across my 12 strongest blends, its rank correlation with private was 0.03, against
0.70 for the public leaderboard. My highest gold-58 submission (0.9439) scored 0.941 private. It works as a regression guard
and for large effects, nothing finer.

**Pre-registering rules saved submissions.** I wrote down the keep/discard rule before reading each result. OrthoFoundation
ViT-L (gold 0.896), V-JEPA 2.1 (0.80) and DINOv2 ViT-S (0.714) were all rejected without spending a submission.

**A strong, decorrelated leg was the only lever that worked.** Weight tweaks, TTA, flip views and more seeds or checkpoints
of the same family gave nothing measurable.

**Public to private was steady for me:** the drop averaged 0.015 (range 0.009–0.021) over 30 scored submissions.

**Budget runtime from the start.** Three of my last ten submissions exceeded 9 h and returned no score.

## What I missed

My labeler read report text only (`gpt-5.4-mini`, gold-58 agreement 0.869; my best label mean reached 0.90). I had noted that
gold labels are image-based, but I kept working on models and blends instead of labels. The top teams here had vision models
read the MRI with the report and reached 0.94–0.95 on gold-58, and that gap explains the result better than anything else in
my pipeline. Their write-ups are worth reading before mine.
