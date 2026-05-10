# ============================================================
# run_poe_ai_semantic_scoring_v1_1_parallel.py
# Purpose:
#   Parallel Poe AI semantic scoring for SEC filing evidence snippets.
#
# Key features:
#   - Multi-threaded Poe API calls
#   - Resume support
#   - Per-row autosave
#   - Retry logic
#   - Error logging
#   - Safe for long overnight runs
# ============================================================

import os
import re
import csv
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------

POE_BASE_URL = "https://api.poe.com/v1"

DEFAULT_MODEL = os.environ.get("POE_MODEL", "GPT-4o-mini")
POE_API_KEY = os.environ.get("POE_API_KEY", "").strip()

CALL_TIMEOUT_SEC = int(os.environ.get("POE_TIMEOUT", "90"))
MAX_RETRIES = int(os.environ.get("POE_MAX_RETRIES", "4"))
SLEEP_ON_RETRY_BASE = float(os.environ.get("POE_RETRY_SLEEP", "3.0"))


# ------------------------------------------------------------
# 2. Prompt
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior banking regulator and fintech strategy expert.
You specialize in evaluating the depth, credibility, and strategic significance of digital transformation disclosures in bank regulatory filings.

Your task is to objectively evaluate whether the provided filing evidence indicates substantive digital transformation.
You must follow the scoring rubric strictly.
You must avoid speculation beyond the provided evidence.
You must distinguish general technology/cybersecurity risk disclosure from substantive digital transformation strategy.
Return valid JSON only.
""".strip()


USER_PROMPT_TEMPLATE = """
You are given digital-transformation-related evidence snippets extracted from one SEC 10-K or 10-Q filing.

Company/Filing metadata:
- CIK: {cik}
- Form: {form}
- Filing date: {filing_date}
- Fiscal year: {fiscal_year}
- File name: {file_name}

Please score the filing on the following dimensions.

Dimension 1: Investment Type, 1-5
1 = routine maintenance or generic IT operations only
2 = compliance-driven or risk-control technology spending
3 = system optimization or process improvement
4 = process transformation affecting customer service, operations, lending, risk, or delivery channels
5 = strategic innovation or business-model-level digital transformation

Dimension 2: Strategic Importance, 1-5
1 = incidental or boilerplate mention only
2 = limited operational importance
3 = meaningful but not central to strategy
4 = clearly linked to strategic priorities, efficiency, growth, customer experience, or competitiveness
5 = explicitly framed as a central strategic pillar or long-term transformation priority

Dimension 3: Implementation Stage, 1-5
1 = concept/general discussion only
2 = planning or early preparation
3 = pilot/partial implementation
4 = deployed initiative or active rollout
5 = broad operationalized implementation across major business functions

Dimension 4: Time Horizon, 1-5
1 = no clear time horizon
2 = short-term cost/risk control
3 = medium-term efficiency or service improvement
4 = medium-to-long-term growth, modernization, or competitiveness
5 = long-term strategic transformation

Auxiliary Dimension: Digital Relevance, 1-5
1 = evidence is mostly unrelated to digital transformation
2 = mostly generic cybersecurity/technology risk
3 = some digital capability or digital service content
4 = clear digital transformation content
5 = highly substantive and central digital transformation evidence

Confidence, 0-100
Assess how confident you are based only on the evidence provided.

Important rules:
- Do not reward generic risk-factor boilerplate too much.
- Cybersecurity disclosure alone should usually score low on Investment Type and Strategic Importance unless connected to broader digital strategy.
- Mentions of digital banking, mobile banking, analytics, cloud, automation, AI, platform modernization, digital lending, or customer experience may justify higher scores only if the evidence shows actual strategy, investment, implementation, or business relevance.
- If no evidence is provided, assign all four main scores as 1, Digital Relevance as 1, and Confidence as 90.

Return JSON only, with exactly this structure:
{{
  "InvestmentType": 1,
  "StrategicImportance": 1,
  "ImplementationStage": 1,
  "TimeHorizon": 1,
  "DigitalRelevance": 1,
  "Confidence": 90,
  "Explanation": "brief explanation grounded in the evidence"
}}

Evidence snippets:
\"\"\"
{evidence_text}
\"\"\"
""".strip()


# ------------------------------------------------------------
# 3. Thread-local Poe client
# ------------------------------------------------------------

_thread_local = threading.local()


def get_client():
    if not POE_API_KEY:
        raise ValueError(
            "POE_API_KEY is empty. Set it first, e.g. Windows CMD: set POE_API_KEY=your_key_here"
        )

    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=POE_API_KEY,
            base_url=POE_BASE_URL,
            timeout=CALL_TIMEOUT_SEC,
        )

    return _thread_local.client


# ------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------

def extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("Empty response")

    text = text.strip()

    # Remove fenced JSON block if any
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:500]}")

    return json.loads(m.group(0))


def to_number(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def normalize_score_dict(d: dict) -> dict:
    out = {}

    fields = [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
        "Confidence",
    ]

    for f in fields:
        out[f] = to_number(d.get(f), None)

    out["Explanation"] = str(d.get("Explanation", "")).strip()

    # Clamp 1-5 dimensions
    for f in [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
    ]:
        if out[f] is None:
            out[f] = None
        else:
            out[f] = max(1, min(5, out[f]))

    # Clamp confidence
    if out["Confidence"] is not None:
        out["Confidence"] = max(0, min(100, out["Confidence"]))

    return out


def compute_derived_scores(result: dict) -> dict:
    main_scores = [
        result.get("InvestmentType"),
        result.get("StrategicImportance"),
        result.get("ImplementationStage"),
        result.get("TimeHorizon"),
    ]

    if all(x is not None for x in main_scores):
        semantic_score = sum(main_scores) / 4.0
        semantic_index_0_100 = (semantic_score - 1.0) / 4.0 * 100.0
    else:
        semantic_score = None
        semantic_index_0_100 = None

    return {
        "AI_semantic_score": semantic_score,
        "AI_semantic_index_0_100": semantic_index_0_100,
    }


def score_one_row(row: dict, model: str) -> dict:
    client = get_client()

    filing_id = str(row.get("filing_id", "")).strip()
    evidence = str(row.get("evidence_text", "") or "").strip()

    prompt = USER_PROMPT_TEMPLATE.format(
        cik=row.get("cik", ""),
        form=row.get("form", ""),
        filing_date=row.get("filing_date", ""),
        fiscal_year=row.get("fiscal_year", ""),
        file_name=row.get("file_name", ""),
        evidence_text=evidence if evidence else "[NO DIGITAL EVIDENCE FOUND]",
    )

    last_err = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            content = resp.choices[0].message.content
            parsed = extract_json(content)
            scores = normalize_score_dict(parsed)
            derived = compute_derived_scores(scores)

            out = dict(row)
            out.update(scores)
            out.update(derived)
            out.update({
                "raw_response": content,
                "api_status": "ok",
                "api_error": "",
                "poe_model": model,
                "scored_at": pd.Timestamp.now().isoformat(),
                "attempts_used": attempt,
            })

            return out

        except Exception as e:
            last_err = repr(e)
            sleep_seconds = SLEEP_ON_RETRY_BASE * attempt
            time.sleep(sleep_seconds)

    # If all attempts fail
    out = dict(row)
    out.update({
        "InvestmentType": None,
        "StrategicImportance": None,
        "ImplementationStage": None,
        "TimeHorizon": None,
        "DigitalRelevance": None,
        "Confidence": None,
        "Explanation": "",
        "AI_semantic_score": None,
        "AI_semantic_index_0_100": None,
        "raw_response": "",
        "api_status": "error",
        "api_error": last_err,
        "poe_model": model,
        "scored_at": pd.Timestamp.now().isoformat(),
        "attempts_used": MAX_RETRIES,
    })

    return out


def load_done_ids(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()

    try:
        old = pd.read_csv(output_csv, dtype=str, usecols=["filing_id"]).fillna("")
        return set(old["filing_id"].astype(str))
    except Exception:
        return set()


def append_row_csv(path: Path, row: dict, fieldnames: list, lock: threading.Lock):
    with lock:
        file_exists = path.exists()

        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


def write_jsonl_log(path: Path, payload: dict, lock: threading.Lock):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only_with_evidence", action="store_true",
                        help="If set, only send rows with evidence_char_len > 0 to Poe. Not recommended for final main run.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    error_log_path = output_path.with_suffix(".errors.jsonl")

    print("=" * 70)
    print("Poe AI semantic scoring - parallel version")
    print("=" * 70)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Error log:", error_log_path)
    print("Model:", args.model)
    print("Workers:", args.workers)
    print("Resume:", args.resume)
    print("Limit:", args.limit)
    print("=" * 70)

    df = pd.read_csv(input_path, dtype=str).fillna("")

    if args.only_with_evidence:
        df["evidence_char_len_num"] = pd.to_numeric(df.get("evidence_char_len", 0), errors="coerce").fillna(0)
        df = df[df["evidence_char_len_num"] > 0].copy()

    if args.limit:
        df = df.head(args.limit).copy()

    done_ids = set()
    if args.resume:
        done_ids = load_done_ids(output_path)
        print(f"Already completed rows found: {len(done_ids)}")

    df["filing_id"] = df["filing_id"].astype(str)
    todo = df[~df["filing_id"].isin(done_ids)].copy()

    print("Rows in input after filters:", len(df))
    print("Rows to score now:", len(todo))

    if len(todo) == 0:
        print("Nothing to do.")
        return

    # Output columns: input columns + scoring columns
    scoring_cols = [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
        "Confidence",
        "Explanation",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "raw_response",
        "api_status",
        "api_error",
        "poe_model",
        "scored_at",
        "attempts_used",
    ]

    fieldnames = list(df.columns)
    for c in scoring_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    csv_lock = threading.Lock()
    log_lock = threading.Lock()

    ok_count = 0
    error_count = 0

    rows = todo.to_dict("records")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(score_one_row, row, args.model): row.get("filing_id", "")
            for row in rows
        }

        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Poe scoring"):
            filing_id = future_to_id[future]

            try:
                result = future.result()
            except Exception as e:
                # This should be rare because score_one_row catches API errors.
                result = {
                    "filing_id": filing_id,
                    "api_status": "fatal_error",
                    "api_error": repr(e),
                    "scored_at": pd.Timestamp.now().isoformat(),
                    "poe_model": args.model,
                }

            append_row_csv(output_path, result, fieldnames, csv_lock)

            if result.get("api_status") == "ok":
                ok_count += 1
            else:
                error_count += 1
                write_jsonl_log(
                    error_log_path,
                    {
                        "filing_id": filing_id,
                        "api_status": result.get("api_status"),
                        "api_error": result.get("api_error"),
                        "scored_at": result.get("scored_at"),
                    },
                    log_lock,
                )

    print("\nDone.")
    print("New ok rows:", ok_count)
    print("New error rows:", error_count)
    print("Saved:", output_path)

    # Simple final audit
    try:
        final = pd.read_csv(output_path, dtype=str).fillna("")
        print("\nCurrent output audit:")
        print("Rows:", len(final))
        print(final["api_status"].value_counts(dropna=False).to_string())
    except Exception as e:
        print("Could not read final output for audit:", repr(e))


if __name__ == "__main__":
    main()