# Project notes for Claude

## LLM report-labeling pipeline (full corpus DONE)

`scripts/llm_label_gold.py` uses an LLM (`OPENAI_MODEL` in `.env`) to generate
graded per-finding labels (0-1 for each of the 12 targets in `TARGETS`) from
the free-text `Report` column, as an alternative to the rule-based extractor
described in `EDA_BASELINE_RESULTS.md`.

**Current model: `gpt-5.4-mini`** (switched from `gpt-5.6-sol`, see "Model
history" below for why).

### Results on the 58 gold studies so far

| Model | Mean agreement AUC | Tokens for 58 studies | Est. tokens for full 4,407 corpus |
|---|---:|---:|---:|
| Rule-based extractor (baseline) | 0.814 | -- | -- |
| `gpt-5.6-sol` | 0.869 | not instrumented (heavy hidden-reasoning waste) | ~2.77M+ (needed multiple days) |
| **`gpt-5.4-mini`** | **0.869** | **34,783 (~600/study)** | **~2.64M -- fits in ~1 run** |

Same mean AUC, same per-target ranking (Synovitis weakest at ~0.68, ACL
strongest at ~0.97-0.99), but `gpt-5.4-mini` is ~80x cheaper per study and
doesn't have the hidden-reasoning-token problem `gpt-5.6-sol` had. **Use
`gpt-5.4-mini` for the full-corpus run.**

Output files:
- `data/gold_annotated.csv` -- the 58 fully-annotated studies + their real labels.
- `data/llm_labels.csv` -- `gpt-5.6-sol` result on the 58 gold studies (0.869 AUC).
- `data/llm_labels_gpt54mini.csv` -- `gpt-5.4-mini` result on the 58 gold studies (0.869 AUC).
- `data/llm_labels_full_gpt56sol_STALE_726of4407.csv` -- **stale, do not use.**
  Partial full-corpus run (726/4407) made with `gpt-5.6-sol` before switching
  models. Renamed rather than deleted so the work isn't lost, but it should
  NOT be merged with a `gpt-5.4-mini` full-corpus run -- different model,
  would mix label distributions.

**Full corpus: DONE.** `data/llm_labels_full.csv` has all 4,407 studies
labeled with `gpt-5.4-mini` (graded 0-1 scores for the 12 `TARGETS`, same
column layout as `gold_annotated.csv` but without the ground-truth columns).
Took 2 runs: first stopped cleanly at the 2.4M-token budget with 4,374/4,407
done, a top-up run finished the remaining 33. Total: **2,415,243 tokens**.

This is the actual deliverable -- the 58-sample runs were quality checks
against ground truth only, this full run is what should feed the imaging
model's training targets next.

**Bug fixed along the way:** `load_done()` originally returned pandas Series
for previously-labeled rows (from `df.iterrows()`), which broke when mixed
with the plain dicts used for newly-labeled rows in the same
`pd.DataFrame(rows)` call (`AttributeError: 'dict' object has no attribute
'dtype'`). Only surfaced on the first resume with a large prior batch (4,374
rows) -- earlier resumes in testing were empty or tiny. Fixed by converting
to `.to_dict()` in `load_done()`. No data was lost when it crashed; the one
in-flight batch just got recomputed on retry.

**Next step (not started):** hook `data/llm_labels_full.csv` up as training
targets for the imaging model (the DINOv2 pipeline in
`eda/rsna-knee-data-structure-eda-baseline.ipynb`), replacing or
supplementing the rule-based extractor's derived labels there.

### Daily token budget

**The 2,500,000 token/day limit is shared across a whole family of models**,
not just one -- confirmed via the OpenAI account limits page: it covers
`gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.1-codex-mini`, `gpt-5-mini`,
`gpt-5-nano`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o1-mini`, `o3-mini`, `o4-mini`,
and `codex-mini-latest`. `gpt-5.6-sol` (the original model) is NOT in this
list, which may be why it behaved so differently (heavier reasoning, no
temperature support) -- it's likely a different tier/pool entirely. If any
other tool or process is also calling one of the listed mini/nano models
against the same account on the same day, it eats into this same 2.5M pool.

`scripts/llm_label_gold.py` enforces a **2,400,000-token stop threshold**
per run (`--token-budget`, default `DEFAULT_TOKEN_BUDGET` in the script) --
it checks projected usage before each batch (using a rolling average of
recent real batches) and stops cleanly once the next batch would cross that
line, rather than erroring out mid-run.

**The script is resumable by design**: it skips any `StudyInstanceUID`
already present in `--out-csv`, so re-running the exact same command on a
later day (once the quota resets) picks up where it left off. No manual
bookkeeping needed between runs -- just re-run the same command.

Each run appends one line to `data/llm_label_usage.log` (JSON lines) with
tokens used, studies labelled, and studies remaining.

**Sizing with `gpt-5.4-mini`:** measured at ~600 tokens/study on the 58 gold
studies. Full corpus (4,407 studies) -> ~2.64M tokens estimated total, i.e.
it should mostly fit in a single ~2.4M-token run, with a small top-up run
likely needed to finish the last few hundred studies. Much better than the
`gpt-5.6-sol` estimate (~2.77M+, needing 2+ days) since gpt-5.4-mini doesn't
burn tokens on hidden reasoning for this task.

### Model history / known quirks

**`gpt-5.6-sol`** (original model, now abandoned for this task): a
reasoning-family model with non-standard Chat Completions behavior,
discovered by trial and error:
- Rejects `temperature` entirely (any value).
- Needs `max_completion_tokens`, not `max_tokens`.
- Defaulted to spending 60-100% of the completion budget on hidden reasoning
  tokens for this simple classification task, occasionally consuming the
  *entire* budget and returning empty content. Fixed with
  `"reasoning_effort": "none"`, cutting tokens ~3x with no visible quality
  loss -- but even with that fix it was still far more expensive per study
  than gpt-5.4-mini turned out to be, and not part of the free 2.5M/day pool.

**`gpt-5.4-mini`** (current model): worked cleanly out of the box with the
same request body (including `reasoning_effort: "none"` and
`max_completion_tokens`) -- no adaptation needed, no errors, no truncation.
Matched gpt-5.6-sol's AUC at a fraction of the token cost.

`request_body()` in the script still sends `reasoning_effort: "none"`
unconditionally -- if a future model rejects that parameter,
`_adapt_body_for_error()` already has a fallback to drop it and retry (same
pattern used for temperature/max_tokens).

### Batch API path exists but is separate

`--submit-batch` / `--check-batch` use OpenAI's async Batch API (~50%
cheaper, up to 24h turnaround) instead of the real-time path. **The token
budget guard above only applies to the real-time path** -- not confirmed
whether Batch API jobs draw from the same daily token pool as the real-time
family list above. Given gpt-5.4-mini's real-time cost is now cheap enough
that the full corpus likely fits in ~1 real-time run anyway, Batch API is
probably not needed for this corpus size, but worth reconsidering if the
per-study cost estimate turns out higher in practice.
