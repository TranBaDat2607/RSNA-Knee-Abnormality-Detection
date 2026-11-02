# Post-mortem: RSNA Knee Abnormality Detection 2026

Written 2026-11-02, from the final leaderboard read that day, all scored submissions
(`reports/submissions_final.csv`) and the top write-ups in the competition forum.

## Result

| | |
|---|---|
| Private leaderboard | **0.943, rank 320 / 6,299**, top 5.1 %, bronze |
| Public leaderboard | 0.958, rank 422 |
| Winning private score | 0.960 |
| Best private score on any of our submissions | 0.943 (A14, A17, A20, A22) |
| Our best private without OAI-trained weights | 0.931 (C3) |
| Our best private with our own models alone | 0.913 (R5t + 2 full-data models) |

Silver at Kaggle's top-5 % rule ends near rank 315, five places above us. 287 teams share 0.943 at three decimals
(ranks 297–583), so most of the distance to silver is hidden in the rounding. Rank moved from 340 / 6,425 on 10-23
to 320 / 6,299 after Kaggle removed teams. The medal follows from that rule; it was not read from the site, because the API returns no medal field.

The score rests mostly on public notebooks and weights, audited and blended. Our own additions at the top of the stack
(S6T in the S6 slot, the runtime-safe forms) moved the public and private scores by at most 0.001.

## What predicted the private leaderboard

Measured on our scored submissions, private and public to 3 decimals.

| Signal | Set | Spearman with private |
|---|---|---|
| Public LB | all 30 scored submissions | 0.98 |
| Public LB | 17 ledger entries with gold-58 | 0.90 |
| Gold-58 | same 17 | 0.67 |
| Public LB | top 12 of those (public ≥ 0.956) | 0.70 |
| Gold-58 | same top 12 | **0.03** |

- The public-to-private drop was steady: mean 0.015, range 0.009–0.021. There was no shake-up in our submissions.
- Gold-58 separates big steps and nothing near the top. Among the late blends it says nothing about private order. That
  confirms the rule used since 10-10: regression guard only, never a tiebreak on close candidates. Our highest gold-58
  (A23, 0.9439) scored 0.941 private, below A20's 0.943 on gold 0.9396.
- The pre-registered A16 pick would have scored 0.942. The score of 0.943 shows the final selection included one of the
  0.943 submissions. The API does not expose which two boxes were ticked.

## What worked

- **Auditing and forking the strongest public stacks.** From 0.942 to 0.958 public in under three weeks, for little GPU.
- **A second track with no OAI-trained weights.** It never scored, but it kept the OAI question from being a single point
  of failure until the end.
- **Pre-registered gates.** Six candidate legs were rejected before they could waste submissions: V1, V-JEPA and DINOv2
  all failed on gold-58, and S6T failed its correlation gate as a new leg.
- **Report labels over OOF pseudo-labels for our own models.** R5t beat R3 and R4 on gold-58 and on both leaderboards.

## What didn't work

- **We never let an LLM read the images.** This is the main gap (next section). Our labeler read report text only, and
  `gpt-5.4-mini` reached 0.869 on gold-58, the best report-label mean 0.9014. In September our notes already recorded that
  gold labels are image-based and reports disagree with them, and we did not act on it.
- **Blending instead of building after 10-10.** Once the OAI stacks arrived, the time went into combining other people's
  legs. Own models were retired at 0.92 public. Six further legs produced no measurable private gain.
- **Runtime misses in the final 48 hours.** A19, A21 and A25 exceeded 9 h on the hidden test, using three of the last ten
  submissions on no score. A20 also finished close to the limit.

## What the winners did that we didn't

Read: 1st (752468), 2nd (752370), 3rd (752510), 5th (752190), 7th (752217), 38th (751924).

- **Labels from images, not only reports.** The top five all had frontier vision models read the MRI
  along with the report and output soft labels. Those reporting a gold-58 number reached 0.936–0.956, against our 0.90. Costs were modest:
  the 2nd-place team used about two weekly allowances of a $200/month subscription for its image pass; 5th used 20–30 % of
  one for its pass.
- **Treat the report as a draft to check.** 7th got 0.947 by asking the model to verify the report against the images;
  report-only reads were 0.89–0.91 and image-only 0.919.
- **Fix label definitions.** Hosts count moderate or large effusion and high-grade MCL, while reports mention milder cases.
  38th tagged cells as below threshold (25 % positive on gold), going from 0.902 to 0.947 on gold-58.
- **Remove site bias.** 38th trained a 29-site × 12-finding bias dropped at test time, its biggest single jump (+0.006).
  7th corrected Synovitis per site using scanner model as a proxy for the hospital.
- **Small ensembles of strong models.** 7th's single model family scored 0.958 private with no external training data. Our
  OAI blend used many more legs for 0.943, so OAI weights were not the deciding factor.
- **Judge labels by more than AUC.** 1st tracked log loss and confident errors on gold-58, because lucky AUC runs hid large
  gaps in both.

Where we matched them: 2.5-D MIL as the family, anatomical slots, report labels over pseudo-labels at first, gold-58 as a
gate rather than a target (38th's rule is close to ours), per-leg time guards with fallback (38th's fail-soft blend).

## Validation lessons

- With 58 labeled studies, no validation number can rank close candidates. Say so up front and decide by rule.
- Public LB was the better signal here because public and private behaved alike. That is specific to this split, so check
  the drop on early submissions before relying on it.
- Gold-58 is still valid for large effects. The winners' label gains (0.90 → 0.95) were far outside its noise, and the
  signal was available to us.

## Process lessons

- Time went to the part of the pipeline that was easiest to measure (models and blends), not the largest error source
  (labels). An error analysis on gold-58 vs labels by finding, repeated in September, would have pointed there.
- Automating the loop with Claude Code kept audits, dev runs, the ledger and kernels moving. It also made more blending
  cheap, which reinforced the wrong priority. Automation doesn't choose the direction.
- Runtime needs a hard budget per stage from the first version of a blend, not a retrofit in the last 48 hours.

## Reusable takeaways

1. When labels are the bottleneck, spend first on better labels: compare label sources against the small gold set before
   training anything else, and try vision-capable LLMs on images, not only text.
2. A small gold set is a gate for large effects. Test its rank correlation with the leaderboard among near-identical
   candidates; here it was 0.03.
3. Before blending public work, ask what its authors could not do. Blends of the same family plateau at their shared
   label source.
4. Check for site or scanner effects as soon as the data is multi-site. A training-only site bias is a cheap test.
5. Give every late-stage stage a hard time cap with a tested fallback, and keep enough margin that a hidden-test run
   cannot use up the last submissions.
