# Final selection (`/kg-final`, 2026-10-21 15:30 UTC)

The deadline is **2026-10-22 23:59 UTC** (2026-10-23 06:59 at UTC+7, the clock on this machine). The host confirmed it in
thread 751277. Only the user can tick the two picks: Submissions tab → "Use for Final Score". If nothing is ticked,
Kaggle picks the two best public scores. Here those are A16 and A17, two near-identical OAI submissions with no clean hedge.

## Evidence

| Submission | Public | Gold-58 | Track | Risk elements |
|---|---|---|---|---|
| **A16 `57461357`** | **0.958** | 0.9387 | OAI | heliosli's public 0.9586 notebook unchanged; its per-finding Raptor and reader weights were probed on the public LB (goodpjw `PROBE_CLASS`) |
| A17 `57401136` | 0.958 | 0.9395 | OAI | 2/3 A16 + 1/3 s6fall; the LB-probed weights carry 2/3 of the blend |
| A12 `57197656` | 0.957 | **0.9404** | OAI | A10 (heliosli 0.9573, Raptor half + R7 50/50) + s6fall; no per-finding LB-probed weights, no reader |
| **C3 `57186035`** | 0.946 | — | clean | no OAI/fastMRI/SKM-TEA weights; ~7.5 h of the 9 h limit |

Pairwise on gold-58 (`experiments/phase4/analysis/final_evidence.py`, mean Spearman over the 12 findings, paired bootstrap 95 % CI):

| Pair | ρ | Gold gap | CI |
|---|---|---|---|
| A12 − A16 | 0.980 | +0.0018 | [−0.0056, +0.0093] |
| A12 − A17 | 0.993 | +0.0009 | [−0.0038, +0.0064] |
| A17 − A16 | 0.989 | +0.0008 | [−0.0047, +0.0062] |

**Noise.** Neither source separates the three OAI candidates. The public split is a few hundred studies, and 0.957 vs 0.958
at 3 decimals could be a gap of 0.0001 to 0.0019. Gold-58 has 9–12 positives per finding, and every CI above spans zero.
The two sources disagree on order (public: A16 = A17 > A12; gold-58: A12 > A17 > A16). On 10-19 gold-58 also showed that
the LB-probed weights cost −0.0007 against uniform 0.5 (CI [−0.0054, +0.0038]), which is weak evidence, but forum thread
751021 makes the same point about heliosli's hand-tuned weights.

**Robustness.** All four candidates completed on the hidden test with a score. A code-competition run scores the public
and private rows in one pass, so no second run is needed. Each file passed `kgkit validate` at its dev run. A12 (≤ 4.77 h
measured) and A16 (≈ 5.3 h estimated) have runtime margin; C3 used about 7.5 h.

**Host ruling.** As of 10-21 15:00 UTC the hosts have not answered threads 750899 (OAI-fine-tuned public checkpoints)
or 750889 (fastMRI / SKM-TEA). Every OAI-track candidate depends on such weights. C3 does not.

## Recommendation

1. **Pick 1: A16 `57461357` (0.958).** This is the best public score. The 10-19 pre-registered rule (tie with A17 → A16)
   makes it the OAI pick.
2. **Pick 2: C3 `57186035` (0.946).** This is the user's two-track hedge from 10-10. It still holds because the
   external-weights question is unresolved. It protects against a ruling that removes OAI-weight submissions, a case in
   which every OAI pick scores nothing.

**Alternative pick 2, if the user accepts the OAI risk: A12 `57197656` (0.957).** It is the most different OAI-track
submission (ρ 0.980 with A16, against 0.989 for A17). It has the best gold-58, and it holds none of the LB-probed per-finding
weights. It protects against those weights losing their public edge on the private 70 %. A17 is not a good second pick:
two thirds of it is A16.

**Update 2026-10-22 03:30 UTC.** A20 (A17 + our S6T) scored 0.958, a tie, so A16 stays pick 1. A19 and A21 exceeded the
9 h limit on the hidden test (no score). Their runtime-safe forms A22 (= A21, every extra capped) and A23 (A22 with R6 plain)
are grading, with results expected about 12:00–13:00 UTC. If either is strictly above 0.958, it replaces A16 as pick 1
(A22 on a tie between them), and this file will say so.

Phase 4 tested six more legs or swaps (S6T, R6 swap, V1, VJ, VJf, V2), and all of them failed their pre-registered gates
(`docs/phase4_notes.md`). No public notebook scores above 0.958 as of 10-21 14:40 UTC. A background watch checks every
90 min until 22:00 UTC on 10-22. If a stronger public notebook passes an audit and its submission beats 0.958, it replaces
A16 as pick 1, and this file is updated.

**Update 2026-10-22 12:20 UTC.** A22 scored 0.958 (tie) and A23 0.957; both finished inside 9 h. Neither is strictly
above 0.958, so the recommendation is unchanged: **A16 `57461357` + C3 `57186035`** (or A16 + A12 if the user accepts the
OAI risk).
