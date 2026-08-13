"""Label the 58 gold-annotated studies with an LLM, for comparison against the
rule-based extractor's 0.814 mean agreement AUC (see EDA_BASELINE_RESULTS.md).

Token-usage reduction applied here (see notes inline):
  1. Prompt caching (OpenAI automatic): the system prompt is a fixed block, sent
     identically and first on every call, padded with few-shot examples past the
     1024-token floor so the API actually caches it.
  2. Batched requests: several reports go into one call, so the (now larger,
     cached) system prompt is amortised over many studies instead of paid in
     full per study.
  3. Positional array output (Structured Outputs), not a named JSON object per
     study: no repeated key strings in the response, and a strict schema avoids
     malformed-JSON retries (which otherwise waste a full round trip of tokens).
  4. Light whitespace normalisation on report text before sending.
  5. A tight max_tokens cap as a runaway-output guard.
  6. Optional OpenAI Batch API path (--submit-batch / --check-batch): official
     ~50% discount on both input and output tokens, for when this scales to the
     full 4,407-report corpus. Turnaround can take up to 24h, so submit and
     check are separate steps.

Usage:
    python scripts/llm_label_gold.py                    # real-time, all 58, resumable
    python scripts/llm_label_gold.py --limit 6           # quick smoke test
    python scripts/llm_label_gold.py --batch-size 10     # fewer, bigger calls

    python scripts/llm_label_gold.py --submit-batch      # Batch API: submit and exit
    python scripts/llm_label_gold.py --check-batch        # Batch API: poll / collect
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# override=True: a stale OPENAI_API_KEY in the parent shell/session must not
# shadow what's actually in .env, which is the source of truth for this project.
load_dotenv(ROOT / ".env", override=True)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
N_TARGETS = len(TARGETS)

DEFINITIONS = {
    "ACL": "anterior cruciate ligament injury",
    "MCL": "medial collateral ligament injury",
    "Medial Meniscus": "medial meniscus tear",
    "Lateral Meniscus": "lateral meniscus tear",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment",
    "PF OA": "patellofemoral osteoarthritis",
    "Effusion": "joint effusion / excess fluid",
    "Synovitis": "inflammation of the joint lining",
    "Baker's": "Baker's cyst",
    "Contusion": "bone contusion / bone bruise",
    "Fracture": "fracture",
}

# (report text, expected 12 scores in TARGETS order). Kept short and synthetic,
# adapted from the rule-based extractor's own self-test cases (see the EDA
# notebook, §2 "Reading a report in nine languages"). Their purpose is dual:
# they calibrate assertion/negation/hedge behaviour across languages, AND they
# pad the system prompt past the 1024-token floor OpenAI needs to cache it.
FEW_SHOT = [
    (
        "Complete tear of the anterior cruciate ligament. Menisci intact. No effusion.",
        [0.95, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.5],
    ),
    (
        "Fractures :\nAucune.",  # French: heading + "None." -- a negation, not an assertion
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05],
    ),
    (
        "Ligamentos cruzados y colaterales dentro de limites normales.",  # Spanish, both cruciate+collateral groups cleared at once
        [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    (
        "Eklem icerisinde yaygin sivi artisi mevcuttur.",  # Turkish: widespread fluid increase in the joint -> large effusion
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5],
    ),
    (
        "Tricompartmental osteoarthritis.",  # global OA statement covers all three compartments
        [0.5, 0.5, 0.5, 0.5, 0.85, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    (
        "Ρηξη του εσω μηνισκου.",  # Greek: tear of the medial meniscus
        [0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]


def _build_system_prompt() -> str:
    findings = "\n".join(f"{i}. {t}: {d}" for i, (t, d) in enumerate(DEFINITIONS.items()))
    examples = []
    for text, scores in FEW_SHOT:
        examples.append(
            f'Report: "{text}"\n-> s = {scores}'
        )
    examples_block = "\n\n".join(examples)

    return f"""You are an expert musculoskeletal radiologist assistant.

You will be given several free-text knee MRI radiology reports at once, each
labelled "### Report N". Reports may be written in any language (English,
Spanish, French, Dutch, German, Turkish, Croatian/Serbian/Bosnian, Greek,
Bulgarian/Russian, or others).

For each report, score these twelve findings, ALWAYS in exactly this order:

{findings}

Scoring rubric, applied independently per finding:
- 0.0-0.2: the report explicitly states the structure is normal/intact, or the
  finding is explicitly negated ("no fracture", "sin derrame", "geen effusie").
- 0.3-0.5: the report does not mention the finding at all, or mentions it only
  as uncertain/hedged ("possible", "cannot exclude", "probable", "suspicious").
- 0.6-1.0: the report explicitly asserts the finding is present. Use higher
  values for more confident or more severe language (e.g. "large", "complete
  tear", "marked", "extensive") and lower values within this range for milder
  language ("trace", "small", "mild").
A statement covering the whole joint or multiple structures at once (e.g.
"tricompartmental osteoarthritis", "ligaments collateraux et croises normaux")
applies to every finding it covers.

Worked examples (s is the 12 scores, in the exact order given above):

{examples_block}

For each report you receive, output one object with:
- "i": the report's index (the N from "### Report N"), as an integer.
- "s": an array of exactly {N_TARGETS} numbers between 0 and 1, in the exact
  order given above.

Return a JSON object of the form {{"results": [{{"i": 0, "s": [...]}}, ...]}}
with one entry per report you were given, in any order. No other text."""


SYSTEM_PROMPT = _build_system_prompt()

RESPONSE_SCHEMA = {
    "name": "knee_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer", "description": "report index"},
                        "s": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": f"exactly {N_TARGETS} scores in the fixed target order",
                        },
                    },
                    "required": ["i", "s"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}

_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_report(text) -> str:
    """Collapse redundant whitespace without touching wording -- free token savings."""
    if not isinstance(text, str):
        return "(empty report)"
    text = text.strip()
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text or "(empty report)"


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def build_user_message(reports: list[str]) -> str:
    parts = [f"### Report {i}\n{r}" for i, r in enumerate(reports)]
    return "\n\n".join(parts)


def max_tokens_for(batch_size: int) -> int:
    return batch_size * 70 + 200


def parse_results(content: str, batch_size: int) -> dict[int, list[float]]:
    if not content:
        raise ValueError("model returned empty content (likely truncated or a transient hiccup)")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"  raw content that failed to parse: {content[:500]!r}", file=sys.stderr)
        raise
    out = {}
    for item in data.get("results", []):
        i = int(item["i"])
        s = list(item.get("s", []))
        s = [float(x) for x in s][:N_TARGETS]
        if len(s) < N_TARGETS:
            s = s + [0.5] * (N_TARGETS - len(s))
        s = [min(1.0, max(0.0, x)) for x in s]
        out[i] = s
    for i in range(batch_size):
        if i not in out:
            print(f"  warning: model omitted report {i}, filling with 0.5s", file=sys.stderr)
            out[i] = [0.5] * N_TARGETS
    return out


def request_body(model: str, reports: list[str]) -> dict:
    # temperature=0 is rejected by this model (a reasoning-family model, given it
    # also required max_completion_tokens); omitted rather than adapted-and-retried
    # per call, which would otherwise waste one extra round trip on every batch.
    #
    # reasoning_effort="none": measured directly against this model -- default
    # reasoning effort spent 60-100% of the completion budget on hidden reasoning
    # tokens for this simple per-clause classification task (and occasionally
    # consumed the whole budget, truncating the answer to nothing). "none" cut
    # total tokens per call by roughly 3x with no visible drop in label quality
    # on a spot check against the gold labels.
    return {
        "model": model,
        "max_completion_tokens": max_tokens_for(len(reports)),
        "reasoning_effort": "none",
        "response_format": {"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(reports)},
        ],
    }


# --------------------------------------------------------------------------- realtime mode

def build_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAI_API_KEY is empty. Fill it in .env before running.")
    return OpenAI(api_key=api_key)


def _adapt_body_for_error(body: dict, exc: Exception) -> bool:
    """Some newer models reject 'temperature' or want a different token-limit
    param name. Patch the body in place and return True if we should retry
    immediately without burning a full backoff cycle."""
    msg = str(exc)
    if "'temperature'" in msg and "unsupported" in msg.lower():
        body.pop("temperature", None)
        print("  dropping unsupported 'temperature' param, retrying", file=sys.stderr)
        return True
    if "'max_tokens'" in msg and "max_completion_tokens" in msg:
        body["max_completion_tokens"] = body.pop("max_tokens")
        print("  renaming max_tokens -> max_completion_tokens, retrying", file=sys.stderr)
        return True
    if "'reasoning_effort'" in msg and "unsupported" in msg.lower():
        body.pop("reasoning_effort", None)
        print("  dropping unsupported 'reasoning_effort' param, retrying", file=sys.stderr)
        return True
    return False


def call_batch(client, model: str, reports: list[str],
               retries: int = 3) -> tuple[dict[int, list[float]], int]:
    """Returns (parsed results, total_tokens actually billed for this call)."""
    body = request_body(model, reports)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**body)
            used = getattr(resp.usage, "total_tokens", 0) or 0
            return parse_results(resp.choices[0].message.content, len(reports)), used
        except TypeError:
            # SDK/model combo that doesn't like json_schema -- fall back to json_object,
            # same wrapper shape, still parsed the same way.
            body["response_format"] = {"type": "json_object"}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _adapt_body_for_error(body, exc):
                continue  # retry immediately, don't count against backoff sleep
            print(f"  attempt {attempt + 1}/{retries} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up after {retries} attempts") from last_err


# OpenAI account limit is 2,500,000 tokens/day for this project. Stop calling
# once projected usage would cross this threshold, leaving a 100,000-token
# safety margin rather than cutting it exactly at the account limit.
DEFAULT_DAILY_TOKEN_LIMIT = 2_500_000
DEFAULT_TOKEN_BUDGET = 2_400_000

USAGE_LOG_PATH = ROOT / "data" / "llm_label_usage.log"


class TokenBudget:
    """Tracks cumulative usage for one run and decides when to stop.

    Stops *before* a batch that would likely cross the limit, using a rolling
    average of the last few real batches once we have data, and a conservative
    static estimate before that (so the very first batch of a run is also
    covered rather than only checked in hindsight).
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.history: list[int] = []

    def estimate_next(self, fallback: int) -> int:
        if self.history:
            recent = self.history[-5:]
            return int(sum(recent) / len(recent))
        return fallback

    def would_exceed(self, upcoming_estimate: int) -> bool:
        return self.used + upcoming_estimate > self.limit

    def record(self, total_tokens: int):
        self.used += total_tokens
        self.history.append(total_tokens)

    def pct(self) -> float:
        return 100.0 * self.used / self.limit if self.limit else 0.0


def fallback_batch_estimate(reports: list[str], batch_size: int) -> int:
    """Conservative pre-flight estimate, used only until real usage data exists."""
    approx_report_tokens = sum(len(r.split()) for r in reports) * 1.4
    approx_system_tokens = len(SYSTEM_PROMPT.split()) / 0.75
    return int(approx_system_tokens + approx_report_tokens + max_tokens_for(batch_size))


def load_done(out_path: Path) -> dict:
    if not out_path.exists():
        return {}
    prev = pd.read_csv(out_path)
    # .to_dict(): rows must be plain dicts, not pandas Series, or a later
    # pd.DataFrame(rows) call that mixes these with freshly-appended dicts
    # breaks pandas' array construction (AttributeError: 'dict' object has
    # no attribute 'dtype') -- hit for real on the first large resume.
    return {row["StudyInstanceUID"]: row.to_dict() for _, row in prev.iterrows()}


def realtime_run(df: pd.DataFrame, model: str, batch_size: int, out_path: Path,
                  token_budget: int = DEFAULT_TOKEN_BUDGET):
    done = load_done(out_path)
    if done:
        print(f"resuming: {len(done)} studies already labelled")

    todo = [r for r in df.itertuples(index=False) if r.StudyInstanceUID not in done]
    rows = list(done.values())
    client = build_client()
    budget = TokenBudget(token_budget)
    n_chunks = (len(todo) + batch_size - 1) // batch_size
    stopped_early = False

    for chunk_i, chunk in enumerate(chunked(todo, batch_size), 1):
        reports = [clean_report(r.Report) for r in chunk]
        estimate = budget.estimate_next(fallback_batch_estimate(reports, len(chunk)))
        if budget.would_exceed(estimate):
            print(f"stopping: next batch (~{estimate:,} tokens) would push usage past the "
                  f"{token_budget:,}-token budget (used {budget.used:,} so far)")
            stopped_early = True
            break

        print(f"[batch {chunk_i}/{n_chunks}] {len(chunk)} studies")
        results, used = call_batch(client, model, reports)
        budget.record(used)
        print(f"  tokens: {used:,} this batch | {budget.used:,}/{token_budget:,} "
              f"({budget.pct():.1f}%) running total")
        for local_i, r in enumerate(chunk):
            scores = results[local_i]
            rows.append({"StudyInstanceUID": r.StudyInstanceUID,
                         **dict(zip(TARGETS, scores))})
        pd.DataFrame(rows).to_csv(out_path, index=False)  # save after every batch

    remaining = len(df) - len(rows)
    with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "out_csv": str(out_path),
            "tokens_used_this_run": budget.used,
            "studies_labelled_this_run": len(rows) - len(done),
            "total_labelled": len(rows),
            "total_remaining": remaining,
            "stopped_early_due_to_budget": stopped_early,
        }) + "\n")

    if stopped_early:
        print(f"stopped for today: {len(rows)} labelled, {remaining} studies remaining. "
              f"Re-run the same command tomorrow (or once the daily quota resets) -- "
              f"it resumes automatically from {out_path}.")
    else:
        print(f"done: {len(rows)} studies -> {out_path} "
              f"(used {budget.used:,} tokens this run)")


# --------------------------------------------------------------------------- Batch API mode

BATCH_STATE_PATH = ROOT / "data" / "llm_batch_job.json"
BATCH_INPUT_PATH = ROOT / "data" / "llm_batch_input.jsonl"


def submit_batch(df: pd.DataFrame, model: str, batch_size: int, out_path: Path):
    done = load_done(out_path)
    todo = [r for r in df.itertuples(index=False) if r.StudyInstanceUID not in done]
    if not todo:
        print("nothing to submit, all studies already labelled")
        return

    groups = list(chunked(todo, batch_size))
    id_map = {}
    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for gi, group in enumerate(groups):
            custom_id = f"group-{gi}"
            id_map[custom_id] = [r.StudyInstanceUID for r in group]
            reports = [clean_report(r.Report) for r in group]
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body(model, reports),
            }
            f.write(json.dumps(line) + "\n")

    client = build_client()
    uploaded = client.files.create(file=open(BATCH_INPUT_PATH, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    BATCH_STATE_PATH.write_text(json.dumps({
        "batch_id": batch.id,
        "model": model,
        "batch_size": batch_size,
        "id_map": id_map,
        "out_path": str(out_path),
    }, indent=2), encoding="utf-8")
    print(f"submitted batch {batch.id} ({len(groups)} groups, {len(todo)} studies)")
    print(f"status: {batch.status}. Check back later with --check-batch "
          f"(batch jobs can take up to 24h, usually much less).")


def check_batch():
    if not BATCH_STATE_PATH.exists():
        sys.exit("no pending batch job found (data/llm_batch_job.json missing). "
                 "Run --submit-batch first.")
    state = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))
    client = build_client()
    batch = client.batches.retrieve(state["batch_id"])
    print(f"batch {batch.id}: status = {batch.status} "
          f"({batch.request_counts.completed}/{batch.request_counts.total} done)")

    if batch.status != "completed":
        print("not finished yet -- run --check-batch again later.")
        return

    out_path = Path(state["out_path"])
    rows = list(load_done(out_path).values())
    id_map = state["id_map"]

    content = client.files.content(batch.output_file_id).text
    for line in content.splitlines():
        rec = json.loads(line)
        custom_id = rec["custom_id"]
        study_ids = id_map[custom_id]
        if rec.get("error"):
            print(f"  {custom_id} failed: {rec['error']}", file=sys.stderr)
            continue
        body = rec["response"]["body"]
        text = body["choices"][0]["message"]["content"]
        results = parse_results(text, len(study_ids))
        for local_i, sid in enumerate(study_ids):
            rows.append({"StudyInstanceUID": sid, **dict(zip(TARGETS, results[local_i]))})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"collected {len(rows)} studies -> {out_path}")
    BATCH_STATE_PATH.unlink()
    BATCH_INPUT_PATH.unlink(missing_ok=True)


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="only label the first N studies (smoke test)")
    ap.add_argument("--batch-size", type=int, default=6,
                     help="reports per API call (real-time) or per batch group (Batch API)")
    ap.add_argument("--in-csv", default=str(ROOT / "data" / "gold_annotated.csv"))
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "llm_labels.csv"))
    ap.add_argument("--submit-batch", action="store_true",
                     help="submit via the OpenAI Batch API (~50%% cheaper, async) and exit")
    ap.add_argument("--check-batch", action="store_true",
                     help="poll a previously submitted batch job and collect results if done")
    ap.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                     help=f"stop calling once usage would cross this many tokens in one run "
                          f"(default {DEFAULT_TOKEN_BUDGET:,}, account daily limit is "
                          f"{DEFAULT_DAILY_TOKEN_LIMIT:,}). Real-time mode only.")
    args = ap.parse_args()

    if args.check_batch:
        check_batch()
        return

    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not model:
        sys.exit("OPENAI_MODEL is empty. Fill it in .env before running.")
    print(f"model = {model!r}, batch_size = {args.batch_size}")

    df = pd.read_csv(args.in_csv)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out_csv)
    if args.submit_batch:
        submit_batch(df, model, args.batch_size, out_path)
    else:
        realtime_run(df, model, args.batch_size, out_path, args.token_budget)


if __name__ == "__main__":
    main()
