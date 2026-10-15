# Phase 3 report: endgame, 2026-10-10 to 10-15 (branch `feat/phase3-oai956`)

*This report summarises phase 3. The detailed per-experiment notes, pre-registered rules and raw numbers are in
[`phase3_notes.md`](phase3_notes.md). The day-by-day record is in [`session_2026-10-14_summary.md`](session_2026-10-14_summary.md).
Scores and standing were read from the Kaggle API on 2026-10-15.*

## TL;DR

- **Best public LB: 0.957** (A12 `57197656`), up from **0.942** at the start of phase 3. Rank **211 / 6,043**. The top
  5 % cutoff is 0.957 and the top 10 % cutoff is 0.956, so the team sits in the public silver zone. The top of the board
  is 0.967.
- **Clean track (no gated external data): 0.946** (C3 `57186035`), up from 0.942.
- **Final picks under the pre-registered rules:** OAI track **A12 `57197656` (0.957)** and clean hedge **C3 `57186035` (0.946)**.
- **What moved the score:** the external-data legs other teams published (nartaa's OAI CoAtNet, heliosli's S6 and R7,
  nartaa's s6fall). Most of the gain came from forking those stacks after auditing them. The only gain that came from
  our own judgment on the OAI track was A12. A12 adds nartaa's s6fall as a third leg with equal weight and gains +0.001
  over the 0.956 plateau where about 940 teams sit.
- **Our own models:** they were retired from the OAI track because they cost about −0.001 each time they were added
  (A3 and the phase 2 stack). In the clean track, fine-tuned CoAtNet legs trained on better report labels added +0.001
  twice (C2 and C3).
- **Gold-58 does not predict differences of ±0.003.** The two gold-58 gains on 10-10 (A2 +0.0027, A3 +0.0031) gave
  0.000 and −0.001 on the LB. Gold-58 caught one real gain (A12, +0.0042 → +0.001), but the weight form for that gain
  was chosen after the gold-58 numbers were read.

## 1. Starting point

The phase began on 2026-10-10 with best public 0.942 (jiweiliu's 0.943 stack plus our R3 leg). By then the leaderboard
had moved to about 0.95. Every public notebook at 0.950 or above depended on nartaa's CoAtNet, which was trained with
**2,399 OAI knees** as external data. The hosts have not ruled on OAI (forum threads 748276 and 749145). OAI access is
uneven: NIH has barred NDA access from several countries since 2025-04, and the hosts banned KneeCoT for a similar
reason.

**The user's decision (10-10) was to run two tracks.**

- The **OAI track** may use any public weights, including weights trained on OAI, fastMRI or SKM-TEA.
- The **clean track** uses only competition data and the public weights we had already used before.
- We never download OAI ourselves.

## 2. Submissions

| Date (UTC) | Submission | Track | What | gold-58 | Public LB | Decision |
|---|---|---|---|---|---|---|
| 10-10 | `57051397` A2 | OAI | goodpjw 0.950 notebook + nartaa 224 checkpoint at 0.25 | 0.9275 (+0.0027) | 0.950 | OAI pick (tie with parent) |
| 10-10 | `57051394` A3 | OAI | A2 + our R5t leg at 0.10 | 0.9280 (+0.0031) | 0.949 | our leg retired |
| 10-10 | `57051391` C2 | CLEAN | community stack + goodpjw reader 0.30 + our ftC 0.20 | – | 0.945 | clean pick |
| 10-12 | `57110710` A5 | OAI | goodpjw 0.954 notebook (nartaa + heliosli S6 70/30 + reader) | – | 0.954 | OAI pick |
| 10-14 | `57185288` A6 | OAI | public 0.956 notebook (nartaa 384 + 224 mirror, R7 50/50) | 0.9365 | 0.956 | OAI pick |
| 10-14 | `57185425` A7 | OAI | A6 + S6 0.3 in the compact slot + reader 1/10 | (S6 part −0.0014) | 0.956 | tie, so A6 stays |
| 10-14 | `57186035` C3 | CLEAN | C2 with our ftC5o (teach4opus labels) in place of ftC | leg 0.9242 vs 0.9211 | **0.946** | **clean final pick** |
| 10-14 | `57197655` A10 | OAI | heliosli "0.9573" notebook (A6 with the 224 checkpoint run without the mirror view) | 0.9365 | 0.956 | tie, so A6 stays (rule: > 0.956) |
| 10-14 | `57197656` A12 | OAI | A10 + nartaa s6fall as a third equal leg (2/3 A10 + 1/3 rank s6fall), time-guarded | 0.9406 (+0.0042) | **0.957** | **OAI final pick** |

Every submission followed the same protocol. Its decision rule was written before any score was read. Its source was
audited before the push: internet off, only `--no-index` wheel installs, no network calls and no secrets. Each one also
passed a dev run on the 3-row test (schema, NaN and stage-completion checks).

**A12 ran its s6fall stage on the hidden test set.** A12 falls back to A10's file if s6fall fails, runs out of time or
scores fewer than 90 % of studies. Its score differs from A10's (0.957 vs 0.956), so the fallback did not trigger. The
public and private scores come from the same kernel run, so the private score uses the same three-leg blend.

## 3. What worked

1. **Forking public stacks after auditing them.** This gave 0.942 → 0.950 → 0.954 → 0.956, following the public
   plateau as new legs appeared (nartaa OAI, then S6, then R7). Each step was one submission under a fixed rule.
2. **A third leg that is decorrelated and equally strong (A12, +0.001).** s6fall is heliosli's S6 architecture,
   retrained by nartaa on all data with no OAI and with gold-58 excluded. On gold-58 it scores 0.9314, the same as R7
   (0.9312). Its Spearman correlation is 0.87 with R7 and 0.88 with nartaa, about the same as R7 vs nartaa (0.865).
   Adding it at 0.3 of the compact slot (A11) did nothing on gold. Treating it as a third equal leg did help. The
   reasoning for this form is structural: three legs of equal strength and equal pairwise correlation call for equal
   weights. **However, we picked the form after seeing the gold-58 numbers, so only the LB validates it.**
3. **Better report labels for the clean fine-tunes.** `teach4opus` is the mean of teach4 and the Opus 4.5 v1 public
   table. It agrees with gold at 0.9014, against 0.8963 for teach4 alone. It raised every fine-tune's epoch-0 gold
   score from 0.917–0.923 to 0.925–0.927. ftC5o took C2 to C3 (0.945 → 0.946).
4. **The goodpjw reader as a clean leg.** It scores 0.929 alone and has gold-58 0.915. Adding it to the community stack
   at a flat 0.30 gave most of the clean gain from 0.942 to 0.945.

## 4. What did not work (rejected by pre-registered rules or by evidence)

| Idea | Evidence | Outcome |
|---|---|---|
| Our R5t leg in OAI blends (A3) | public −0.001 (and −0.001 earlier on the clean stack) | retired from both tracks |
| ftO: nartaa fine-tuned on teach4 | gold 0.9229 → 0.9099, Spearman with start 0.972 | rejected; teach4 pulls nartaa away from the expert labels |
| ft96b sibling family (v8 and v5 starts) | rank-mean +0.0001 gold, siblings correlate at 0.94–0.95 | not used; would cost 40–60 min of runtime |
| Multi-scale TTA on nartaa (320/352/384) | best mean 0.9226 vs 0.9229, ρ ≥ 0.990 | rejected |
| R7 flip TTA (A8) | +0.0001 in the blend, ρ 0.998 with the plain pass, +1.5 h runtime | not submitted |
| heliosli S6 in the compact slot (A7) | gold −0.0014, public 0.956 tie | A6 kept |
| s6fall at 0.3 of the compact slot (A11) | gold 0.9362 vs 0.9365 | failed its rule, not built |
| Our ftC5o as an OAI-blend leg | gold +0.0008 at 0.20, CI includes 0, ρ 0.93 with nartaa | not submitted |
| R7 weight probe | gold flat for R7 weights from 0.5 to 0.7 | not submitted |
| 64-slice maxspan input (fold 0) | 0.8727 / 0.8683 vs 0.8753 | rejected |

## 5. How well gold-58 predicted the LB

| Change | gold-58 Δ | Public LB Δ (3 decimals) |
|---|---|---|
| A2 vs parent (+ nartaa 224) | +0.0027 | 0.000 |
| A3 vs parent (+ our R5t) | +0.0031 | −0.001 |
| A7 vs A6 (+ S6, + reader) | −0.0014 (S6 part only) | 0.000 |
| A10 vs A6 (224 without mirror) | 0.0000 | 0.000 |
| C3 vs C2 (ftC5o leg) | +0.0031 (leg level) | +0.001 |
| A12 vs A10 (+ s6fall equal leg) | +0.0042 | +0.001 |

On 58 studies the standard error is about 0.02, so gold-58 cannot resolve changes of ±0.003, and this table shows it.
A2 and A3 looked positive on gold but did not move or went down on the LB. Gold-58 remains useful as a regression guard
and as a source of correlation estimates between legs, which is what found s6fall. It should not be used to tune
weights. The public LB has its own limits. Scores are shown to 3 decimals, so 0.956 vs 0.957 could be anything from
about 0.0001 to 0.0019, and a public split of a few hundred studies is noisy at that scale. **A12's lead over A6/A10 is
probably real, because it adds a strong leg with low correlation, but it is not proven for the private LB.**

## 6. Final picks and risks

| Pick | Submission | Public | Risk |
|---|---|---|---|
| OAI track | **A12 `57197656`** | 0.957 | (1) If the hosts rule OAI, fastMRI or SKM-TEA out, it is disqualified; every leg in it depends on one of them. (2) The equal three-way weight was chosen after the gold read. A6/A10 at 0.956 is the less-tuned alternative. |
| Clean hedge | **C3 `57186035`** | 0.946 | It is far below the OAI pick. It only protects the result if the hosts rule against external data. It runs about 7.5 h of the 9 h limit. |

Under the rules, A12 replaces A6 because 0.957 is strictly greater than 0.956. This is still a joint decision with the
user. The real choice is about hedging: two OAI submissions (A12 + A6) would maximise the expected score if OAI is
allowed. A12 + C3 protects against an OAI ban. Without a host ruling, **A12 + C3** stays the recommendation.

## 7. Standing (public LB, 2026-10-15 14:22 UTC)

- Rank 211 / 6,043 at 0.957. Ties at 0.957 cover ranks 138–347 and are ordered by submission time.
- 137 teams are strictly above 0.957, 70 are at 0.960 or above, and the top score is 0.967.
- The public LB has the medal cutoffs at about 0.957 (silver, top 5 % ≈ rank 302) and 0.956 (bronze, top 10 %).
  Shake-up between public and private can easily move a team across a 0.001 boundary.

## 8. What is left (deadline 2026-10-22, 7 days)

- **The lever that A12 confirmed is another strong, decorrelated leg.** Weight tweaks and TTA on the existing legs have
  given nothing measurable since 10-12. Our own models are too correlated with nartaa (ρ ≈ 0.93) to add value.
  A useful new leg would need to be about as strong as R7 on gold (≥ 0.93) with ρ ≤ 0.88 against both R7 and nartaa.
  Two ways to get one:
  - **Watch public releases.** This has been the cheapest source so far: each new leg moved the plateau within a day.
  - **Train one ourselves** with a recipe different from ours, for example an S6/R7-style knee-centred ConvNeXt on clean
    data. This takes several GPU days. About 18 h of quota remain until the 10-17 reset, then 30 h.
- **Runtime headroom.** Before adding anything to A12, measure its hidden-test runtime from the kernel log. A12 is
  time-guarded, but a new leg must fit inside what s6fall leaves.
- **Clean track.** Using ep0 fine-tune checkpoints (gold 0.925–0.927, ahead of ep1) is a cheap candidate. The expected
  gain is about 0.001.
- **Keep the protocol:** a rule fixed before each read, at most about 3 submissions a day, and gold-58 used only as a
  regression guard.
