# audit_problematic_keyword_false_positives_fast.py

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

clean_index_file = folder / "sec_clean_text_index_2018_2024.csv"

problem_terms = [
    "api",
    "rpa",
    "mfa",
    "automated",
    "automation",
    "cloud",
    "core system",
    "system conversion",
    "technology platform",
    "data analytics",
    "digital platform",
]

# --------------------------------------------------
# Regex helpers
# --------------------------------------------------

def strict_pattern(term):
    """
    Strict word-boundary matching.
    For multi-word terms, allow flexible whitespace.
    Example:
      "digital platform" -> (?<![A-Za-z0-9])digital\s+platform(?![A-Za-z0-9])
    """
    parts = [re.escape(p) for p in term.split()]
    phrase = r"\s+".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){phrase}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def loose_pattern(term):
    """
    Original loose substring-style matching.
    This intentionally mimics problematic old matching.
    """
    return re.compile(re.escape(term), flags=re.IGNORECASE)


# Compile inside each worker once
LOOSE_PATTERNS = {t: loose_pattern(t) for t in problem_terms}
STRICT_PATTERNS = {t: strict_pattern(t) for t in problem_terms}


def process_one_file(row):
    path = row.get("clean_text_path")

    rec = {
        "CIK": row.get("CIK"),
        "form": row.get("form"),
        "filing_date": row.get("filing_date"),
        "accession_number": row.get("accession_number"),
        "clean_text_path": path,
        "processing_status": "not_processed",
    }

    if pd.isna(path) or not Path(path).exists():
        rec["processing_status"] = "missing_file"
        for term in problem_terms:
            rec[f"{term}_loose"] = 0
            rec[f"{term}_strict"] = 0
            rec[f"{term}_excess"] = 0
        return rec

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")

        for term in problem_terms:
            loose = len(LOOSE_PATTERNS[term].findall(text))
            strict = len(STRICT_PATTERNS[term].findall(text))

            rec[f"{term}_loose"] = loose
            rec[f"{term}_strict"] = strict
            rec[f"{term}_excess"] = loose - strict

        rec["processing_status"] = "processed"
        return rec

    except Exception as e:
        rec["processing_status"] = f"error_{type(e).__name__}: {e}"
        for term in problem_terms:
            rec[f"{term}_loose"] = 0
            rec[f"{term}_strict"] = 0
            rec[f"{term}_excess"] = 0
        return rec


def main():
    clean_index = pd.read_csv(clean_index_file, low_memory=False)

    print("Clean index shape:", clean_index.shape)

    clean_index = clean_index.dropna(subset=["clean_text_path"]).copy()
    print("Rows with clean_text_path:", clean_index.shape)

    rows = clean_index.to_dict("records")

    # More aggressive local processing.
    # If your computer becomes slow, reduce to 8.
    MAX_WORKERS = 12

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_file, row) for row in rows]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Auditing problematic terms with {MAX_WORKERS} processes"
        ):
            results.append(fut.result())

    out = pd.DataFrame(results)

    print("\nProcessing status:")
    print(out["processing_status"].value_counts(dropna=False))

    summary_rows = []

    for term in problem_terms:
        loose_total = out[f"{term}_loose"].sum()
        strict_total = out[f"{term}_strict"].sum()
        excess_total = out[f"{term}_excess"].sum()

        summary_rows.append({
            "term": term,
            "loose_total": loose_total,
            "strict_total": strict_total,
            "excess_total": excess_total,
            "excess_share": excess_total / loose_total if loose_total > 0 else None,
            "files_with_loose": int((out[f"{term}_loose"] > 0).sum()),
            "files_with_strict": int((out[f"{term}_strict"] > 0).sum()),
            "mean_loose_per_file": out[f"{term}_loose"].mean(),
            "mean_strict_per_file": out[f"{term}_strict"].mean(),
            "max_loose_in_file": out[f"{term}_loose"].max(),
            "max_strict_in_file": out[f"{term}_strict"].max(),
        })

    summary = pd.DataFrame(summary_rows).sort_values("excess_total", ascending=False)

    print("\nFalse positive summary:")
    print(summary)

    out_file = folder / "audit_problematic_keyword_false_positives_file_level.csv"
    summary_file = folder / "audit_problematic_keyword_false_positives_summary.csv"

    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(out_file)
    print(summary_file)


if __name__ == "__main__":
    freeze_support()
    main()