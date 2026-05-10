# ============================================================
# run_poe_full_vs_evidence_comparison_v1.py
# Purpose:
#   Score the same filings twice using Poe:
#   1) evidence-snippet input
#   2) broad full-document input
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


POE_BASE_URL = "https://api.poe.com/v1"
DEFAULT_MODEL = os.environ.get("POE_MODEL", "GPT-4o-mini")
POE_API_KEY = os.environ.get("POE_API_KEY", "").strip()

CALL_TIMEOUT_SEC = int(os.environ.get("POE_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("POE_MAX_RETRIES", "4"))
SLEEP_ON_RETRY_BASE = float(os.environ.get("POE_RETRY_SLEEP", "3.0"))

_thread_local = threading.local()


SYSTEM_PROMPT = """
You are a senior banking regulator and fintech strategy expert.
You specialize in evaluating the depth, credibility, and strategic significance of digital transformation disclosures in bank regulatory filings.

Your task is to objectively evaluate whether the provided filing text indicates substantive digital transformation.
You must follow the scoring rubric strictly.
You must avoid speculation beyond the provided text.
You must distinguish general technology/cybersecurity risk disclosure from substantive digital transformation strategy.
Return valid JSON only.
""".strip()


PROMPT_TEMPLATE = """
You are evaluating one SEC 10-K or 10-Q filing.

Input mode:
{input_mode}

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
1 = text is mostly unrelated to digital transformation
2 = mostly generic cybersecurity/technology risk
3 = some digital capability or digital service content
4 = clear digital transformation content
5 = highly substantive and central digital transformation evidence

Confidence, 0-100
Assess how confident you are based only on the provided text.

Important rules:
- Do not reward generic risk-factor boilerplate too much.
- Cybersecurity disclosure alone should usually score low on Investment Type and Strategic Importance unless connected to broader digital strategy.
- For full-document input, do not infer digital transformation merely from generally positive tone or strong financial performance.
- For evidence-snippet input, focus only on the provided retrieved passages.
- Mentions of digital banking, mobile banking, analytics, cloud, automation, AI, platform modernization, digital lending, or customer experience may justify higher scores only if the text shows actual strategy, investment, implementation, or business relevance.
- If no relevant evidence is provided, assign all four main scores as 1, Digital Relevance as 1, and Confidence as 90.

Return JSON only, with exactly this structure:
{{
  "InvestmentType": 1,
  "StrategicImportance": 1,
  "ImplementationStage": 1,
  "TimeHorizon": 1,
  "DigitalRelevance": 1,
  "Confidence": 90,
  "Explanation": "brief explanation grounded in the provided text"
}}

Text:
\"\"\"
{text}
\"\"\"
""".strip()


def get_client():
    if not POE_API_KEY:
        raise ValueError("POE_API_KEY is empty. Set it first using: set POE_API_KEY=your_key_here")

    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=POE_API_KEY,
            base_url=POE_BASE_URL,
            timeout=CALL_TIMEOUT_SEC,
        )

    return _thread_local.client


def extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("Empty response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found: {text[:500]}")

    return json.loads(m.group(0))


def to_number(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def normalize_score_dict(d: dict) -> dict:
    out = {}
    for f in [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
        "Confidence",
    ]:
        out[f] = to_number(d.get(f), None)

    out["Explanation"] = str(d.get("Explanation", "")).strip()

    for f in [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
    ]:
        if out[f] is not None:
            out[f] = max(1, min(5, out[f]))

    if out["Confidence"] is not None:
        out["Confidence"] = max(0, min(100, out["Confidence"]))

    return out


def compute_score(scores: dict) -> dict:
    vals = [
        scores.get("InvestmentType"),
        scores.get("StrategicImportance"),
        scores.get("ImplementationStage"),
        scores.get("TimeHorizon"),
    ]

    if all(v is not None for v in vals):
        s = sum(vals) / 4.0
        idx = (s - 1.0) / 4.0 * 100.0
    else:
        s = None
        idx = None

    return {
        "AI_semantic_score": s,
        "AI_semantic_index_0_100": idx,
    }


def score_text(row: dict, input_mode: str, text_col: str, model: str) -> dict:
    client = get_client()

    text = str(row.get(text_col, "") or "").strip()
    if not text:
        text = "[NO RELEVANT TEXT PROVIDED]"

    prompt = PROMPT_TEMPLATE.format(
        input_mode=input_mode,
        cik=row.get("cik", ""),
        form=row.get("form", ""),
        filing_date=row.get("filing_date", ""),
        fiscal_year=row.get("fiscal_year", ""),
        file_name=row.get("file_name", ""),
        text=text,
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
            derived = compute_score(scores)

            out = {
                "filing_id": row.get("filing_id", ""),
                "file_name": row.get("file_name", ""),
                "cik": row.get("cik", ""),
                "form": row.get("form", ""),
                "filing_date": row.get("filing_date", ""),
                "fiscal_year": row.get("fiscal_year", ""),
                "input_mode": input_mode,
                "text_col": text_col,
                "text_char_len": len(text),
                **scores,
                **derived,
                "raw_response": content,
                "api_status": "ok",
                "api_error": "",
                "poe_model": model,
                "attempts_used": attempt,
                "scored_at": pd.Timestamp.now().isoformat(),
            }

            return out

        except Exception as e:
            last_err = repr(e)
            time.sleep(SLEEP_ON_RETRY_BASE * attempt)

    return {
        "filing_id": row.get("filing_id", ""),
        "file_name": row.get("file_name", ""),
        "cik": row.get("cik", ""),
        "form": row.get("form", ""),
        "filing_date": row.get("filing_date", ""),
        "fiscal_year": row.get("fiscal_year", ""),
        "input_mode": input_mode,
        "text_col": text_col,
        "text_char_len": len(text),
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
        "attempts_used": MAX_RETRIES,
        "scored_at": pd.Timestamp.now().isoformat(),
    }


def load_done_keys(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()

    try:
        old = pd.read_csv(output_csv, dtype=str).fillna("")
        return set(old["filing_id"].astype(str) + "||" + old["input_mode"].astype(str))
    except Exception:
        return set()


def append_row_csv(path: Path, row: dict, fieldnames: list, lock: threading.Lock):
    with lock:
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="poe_full_vs_evidence_scores_v1.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")
    output_path = Path(args.output_csv)

    done = set()
    if args.resume:
        done = load_done_keys(output_path)
        print("Existing completed scoring tasks:", len(done))

    tasks = []

    for _, r in df.iterrows():
        row = r.to_dict()

        tasks.append({
            "row": row,
            "input_mode": "evidence_snippets",
            "text_col": "evidence_text",
        })

        tasks.append({
            "row": row,
            "input_mode": "broad_full_document",
            "text_col": "full_document_text",
        })

    filtered_tasks = []
    for t in tasks:
        key = str(t["row"].get("filing_id", "")) + "||" + t["input_mode"]
        if key not in done:
            filtered_tasks.append(t)

    print("Rows:", len(df))
    print("Total scoring tasks:", len(tasks))
    print("Tasks to run:", len(filtered_tasks))
    print("Model:", args.model)
    print("Workers:", args.workers)

    fieldnames = [
        "filing_id", "file_name", "cik", "form", "filing_date", "fiscal_year",
        "input_mode", "text_col", "text_char_len",
        "InvestmentType", "StrategicImportance", "ImplementationStage",
        "TimeHorizon", "DigitalRelevance", "Confidence", "Explanation",
        "AI_semantic_score", "AI_semantic_index_0_100",
        "raw_response", "api_status", "api_error", "poe_model",
        "attempts_used", "scored_at",
    ]

    lock = threading.Lock()

    ok = 0
    err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(score_text, t["row"], t["input_mode"], t["text_col"], args.model)
            for t in filtered_tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Full vs evidence scoring"):
            result = fut.result()
            append_row_csv(output_path, result, fieldnames, lock)

            if result.get("api_status") == "ok":
                ok += 1
            else:
                err += 1

    print("Done.")
    print("OK:", ok)
    print("Errors:", err)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()