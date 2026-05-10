# ============================================================
# run_poe_full_document_from_txt_v3_parallel.py
# Purpose:
#   Full-document AI semantic scoring from HTML-parsed visible .txt files.
#
# Important:
#   - Reads the entire .txt file.
#   - Does NOT truncate.
#   - Does NOT use max_chars.
#   - Saves one row after each completed filing.
#   - Supports resume.
#   - Designed for overnight full-run processing.
# ============================================================

import os
import re
import csv
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ------------------------------------------------------------
# Poe / OpenAI-compatible settings
# ------------------------------------------------------------

POE_BASE_URL = "https://api.poe.com/v1"

DEFAULT_MODEL = os.environ.get("POE_MODEL", "GPT-4o-mini")
POE_API_KEY = os.environ.get("POE_API_KEY", "").strip()

CALL_TIMEOUT_SEC = int(os.environ.get("POE_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("POE_MAX_RETRIES", "4"))
SLEEP_ON_RETRY_BASE = float(os.environ.get("POE_RETRY_SLEEP", "8.0"))

_thread_local = threading.local()


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior banking regulator and fintech strategy expert.
You specialize in evaluating the depth, credibility, and strategic significance of digital transformation disclosures in bank regulatory filings.

You will read a cleaned visible-text version of one SEC 10-K or 10-Q filing.
Your task is to evaluate whether the filing contains substantive digital transformation disclosure.

You must distinguish substantive digital transformation from:
- generic cybersecurity risk disclosure,
- generic information technology risk,
- routine IT operations,
- regulatory boilerplate,
- general positive business tone,
- ordinary electronic banking mentions without strategy or implementation detail.

Return valid JSON only.
""".strip()


USER_PROMPT_TEMPLATE = """
You are given the full cleaned visible text of one SEC 10-K or 10-Q filing.

Company/Filing metadata:
- CIK: {cik}
- Form: {form}
- Filing date: {filing_date}
- Fiscal year: {fiscal_year}
- File name: {file_name}

Please read the filing text and score the filing on the following dimensions.

Dimension 1: Investment Type, 1-5
1 = routine maintenance, generic IT operations, or generic technology risk only
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
1 = concept/general discussion only, or no substantive implementation evidence
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
1 = filing contains little or no substantive digital transformation disclosure
2 = mostly generic cybersecurity/technology risk
3 = some digital capability, digital service, or technology-enabled process content
4 = clear digital transformation content
5 = highly substantive and central digital transformation evidence

Confidence, 0-100
Assess how confident you are based only on the provided filing text.

Important scoring rules:
- Do not reward generic cybersecurity risk disclosure too much.
- Do not infer digital transformation merely from good financial performance or positive business tone.
- Ordinary website, mobile app, online banking, or electronic banking availability should not receive high scores unless tied to strategy, investment, implementation, customer transformation, or operating model change.
- Higher scores require evidence of strategy, investment, implementation, digital services, automation, analytics, platform modernization, AI, cloud, customer digital experience, or technology-enabled transformation.
- If the filing only discusses cybersecurity / information systems risk, scores should usually be low.
- If no substantive digital transformation evidence appears, assign the four main scores as 1, Digital Relevance as 1, and Confidence as 90.

Return JSON only, with exactly this structure:
{{
  "InvestmentType": 1,
  "StrategicImportance": 1,
  "ImplementationStage": 1,
  "TimeHorizon": 1,
  "DigitalRelevance": 1,
  "Confidence": 90,
  "Explanation": "brief explanation grounded in the filing text"
}}

Full filing text:
\"\"\"
{full_document_text}
\"\"\"
""".strip()


# ------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------

def infer_fiscal_year_from_filename(stem: str, filing_year=None):
    candidates = []

    for m in re.finditer(r"\d{8}", stem):
        s = m.group(0)

        # YYYYMMDD
        if s[:4].startswith(("19", "20")):
            try:
                candidates.append(datetime.strptime(s, "%Y%m%d"))
            except Exception:
                pass

        # MMDDYYYY
        if s[4:].startswith(("19", "20")):
            try:
                candidates.append(datetime.strptime(s, "%m%d%Y"))
            except Exception:
                pass

    if candidates:
        candidates = sorted(candidates)

        if filing_year is not None:
            try:
                fy = int(filing_year)
                close = [dt for dt in candidates if dt.year <= fy]
                if close:
                    return close[-1].year
            except Exception:
                pass

        return candidates[-1].year

    return filing_year


def parse_filename(path: Path) -> dict:
    name = path.name
    stem = path.stem
    parts = stem.split("_")

    cik = parts[0] if len(parts) >= 1 else ""
    form = parts[1] if len(parts) >= 2 else ""
    filing_date = parts[2] if len(parts) >= 3 else ""

    filing_year = None
    if re.match(r"\d{4}-\d{2}-\d{2}", filing_date):
        filing_year = int(filing_date[:4])

    fiscal_year = infer_fiscal_year_from_filename(stem, filing_year=filing_year)

    return {
        "file_name": name,
        "filing_id": stem,
        "cik": str(cik).lstrip("0") if cik else "",
        "form": form,
        "filing_date": filing_date,
        "filing_year": filing_year,
        "fiscal_year": fiscal_year,
    }


# ------------------------------------------------------------
# Poe helpers
# ------------------------------------------------------------

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


def extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("Empty model response")

    text = text.strip()

    # Remove fenced JSON if present
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
        if out[f] is not None:
            out[f] = max(1, min(5, out[f]))

    # Clamp confidence
    if out["Confidence"] is not None:
        out["Confidence"] = max(0, min(100, out["Confidence"]))

    return out


def compute_derived_scores(scores: dict) -> dict:
    main_scores = [
        scores.get("InvestmentType"),
        scores.get("StrategicImportance"),
        scores.get("ImplementationStage"),
        scores.get("TimeHorizon"),
    ]

    if all(x is not None for x in main_scores):
        semantic_score = sum(main_scores) / 4.0
        semantic_index_0_100 = (semantic_score - 1.0) / 4.0 * 100.0
    else:
        semantic_score = None
        semantic_index_0_100 = None

    return {
        "AI_full_document_score": semantic_score,
        "AI_full_document_index_0_100": semantic_index_0_100,
        "AI_semantic_score": semantic_score,
        "AI_semantic_index_0_100": semantic_index_0_100,
    }


def clean_text_for_prompt(text: str) -> str:
    """
    No truncation. Only light cleanup to remove nulls and excessive blank lines.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def score_one_file(path_str: str, model: str) -> dict:
    path = Path(path_str)
    meta = parse_filename(path)

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        full_text = clean_text_for_prompt(raw_text)

        full_document_char_len = len(full_text)
        full_document_word_count_rough = len(re.findall(r"\b\w+\b", full_text))

        prompt = USER_PROMPT_TEMPLATE.format(
            cik=meta.get("cik", ""),
            form=meta.get("form", ""),
            filing_date=meta.get("filing_date", ""),
            fiscal_year=meta.get("fiscal_year", ""),
            file_name=meta.get("file_name", ""),
            full_document_text=full_text if full_text else "[NO FILING TEXT PROVIDED]",
        )

    except Exception as e:
        out = dict(meta)
        out.update({
            "full_document_char_len": None,
            "full_document_word_count_rough": None,
            "text_char_len_sent_to_poe": None,
            "InvestmentType": None,
            "StrategicImportance": None,
            "ImplementationStage": None,
            "TimeHorizon": None,
            "DigitalRelevance": None,
            "Confidence": None,
            "Explanation": "",
            "AI_full_document_score": None,
            "AI_full_document_index_0_100": None,
            "AI_semantic_score": None,
            "AI_semantic_index_0_100": None,
            "raw_response": "",
            "api_status": "read_error",
            "api_error": repr(e),
            "poe_model": model,
            "scored_at": pd.Timestamp.now().isoformat(),
            "attempts_used": 0,
        })
        return out

    client = get_client()
    last_err = ""
    attempts_used = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used += 1

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

            out = dict(meta)
            out.update(scores)
            out.update(derived)
            out.update({
                "full_document_char_len": full_document_char_len,
                "full_document_word_count_rough": full_document_word_count_rough,
                "text_char_len_sent_to_poe": len(full_text),
                "raw_response": content,
                "api_status": "ok",
                "api_error": "",
                "poe_model": model,
                "scored_at": pd.Timestamp.now().isoformat(),
                "attempts_used": attempts_used,
            })

            return out

        except Exception as e:
            last_err = repr(e)
            time.sleep(SLEEP_ON_RETRY_BASE * attempt)

    out = dict(meta)
    out.update({
        "full_document_char_len": full_document_char_len,
        "full_document_word_count_rough": full_document_word_count_rough,
        "text_char_len_sent_to_poe": len(full_text),
        "InvestmentType": None,
        "StrategicImportance": None,
        "ImplementationStage": None,
        "TimeHorizon": None,
        "DigitalRelevance": None,
        "Confidence": None,
        "Explanation": "",
        "AI_full_document_score": None,
        "AI_full_document_index_0_100": None,
        "AI_semantic_score": None,
        "AI_semantic_index_0_100": None,
        "raw_response": "",
        "api_status": "error",
        "api_error": last_err,
        "poe_model": model,
        "scored_at": pd.Timestamp.now().isoformat(),
        "attempts_used": attempts_used,
    })

    return out


# ------------------------------------------------------------
# File output helpers
# ------------------------------------------------------------

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
        exists = path.exists()

        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

            if not exists:
                writer.writeheader()

            writer.writerow(row)


def write_jsonl_log(path: Path, payload: dict, lock: threading.Lock):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing HTML-parsed visible .txt files")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_csv)
    error_log_path = output_path.with_suffix(".errors.jsonl")

    files = sorted(input_dir.glob("*.txt"))

    if args.limit:
        files = files[:args.limit]

    done_ids = set()
    if args.resume:
        done_ids = load_done_ids(output_path)

    todo_files = []
    for p in files:
        filing_id = p.stem
        if filing_id not in done_ids:
            todo_files.append(p)

    print("=" * 80)
    print("Full-document Poe scoring from .txt files")
    print("=" * 80)
    print("Input dir:", input_dir)
    print("Output CSV:", output_path)
    print("Error log:", error_log_path)
    print("Model:", args.model)
    print("Workers:", args.workers)
    print("Total txt files found:", len(files))
    print("Already completed:", len(done_ids))
    print("Rows to score now:", len(todo_files))
    print("No max_chars. No truncation.")
    print("=" * 80)

    if len(todo_files) == 0:
        print("Nothing to do.")
        return

    fieldnames = [
        "file_name",
        "filing_id",
        "cik",
        "form",
        "filing_date",
        "filing_year",
        "fiscal_year",
        "full_document_char_len",
        "full_document_word_count_rough",
        "text_char_len_sent_to_poe",
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
        "Confidence",
        "Explanation",
        "AI_full_document_score",
        "AI_full_document_index_0_100",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "raw_response",
        "api_status",
        "api_error",
        "poe_model",
        "scored_at",
        "attempts_used",
    ]

    csv_lock = threading.Lock()
    log_lock = threading.Lock()

    ok_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(score_one_file, str(p), args.model): str(p)
            for p in todo_files
        }

        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Full-document Poe scoring"):
            path_str = future_to_path[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "file_name": Path(path_str).name,
                    "filing_id": Path(path_str).stem,
                    "api_status": "fatal_error",
                    "api_error": repr(e),
                    "poe_model": args.model,
                    "scored_at": pd.Timestamp.now().isoformat(),
                }

            append_row_csv(output_path, result, fieldnames, csv_lock)

            if result.get("api_status") == "ok":
                ok_count += 1
            else:
                error_count += 1
                write_jsonl_log(
                    error_log_path,
                    {
                        "file_name": result.get("file_name"),
                        "filing_id": result.get("filing_id"),
                        "api_status": result.get("api_status"),
                        "api_error": result.get("api_error"),
                        "full_document_char_len": result.get("full_document_char_len"),
                        "scored_at": result.get("scored_at"),
                    },
                    log_lock,
                )

    print("\nDone.")
    print("New ok rows:", ok_count)
    print("New error rows:", error_count)
    print("Saved:", output_path)

    try:
        final = pd.read_csv(output_path, dtype=str).fillna("")
        print("\nCurrent output audit:")
        print("Rows:", len(final))
        print(final["api_status"].value_counts(dropna=False).to_string())
    except Exception as e:
        print("Could not read output audit:", repr(e))


if __name__ == "__main__":
    main()