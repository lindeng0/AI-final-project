# extract_keyword_level_hits_v2_parallel.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# ==================================================
# 0. Paths
# ==================================================

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

matches_file = folder / "sec_filing_matches_2018_2024.csv"
clean_index_file = folder / "sec_clean_text_index_2018_2024.csv"

out_hit_level = folder / "sec_keyword_level_hits_v2_2018_2024.csv"
out_filing_summary = folder / "sec_filing_keyword_summary_v2_2018_2024.csv"
out_keyword_total = folder / "sec_keyword_summary_total_v2_2018_2024.csv"
out_keyword_year = folder / "sec_keyword_summary_by_year_v2_2018_2024.csv"
out_windows = folder / "sec_keyword_windows_v2_2018_2024.csv"

MAX_WORKERS = 12

# ==================================================
# 1. Final v2 dictionaries
# Aligned with build_revised_traditional_digital_scores_v2.py
# ==================================================

deployment_terms = [
    "digital banking",
    "mobile banking",
    "online banking",
    "internet banking",
    "electronic banking",
    "mobile app",
    "mobile application",
    "digital channel",
    "digital channels",
    "online account opening",
    "digital account opening",
    "digital onboarding",
    "remote deposit",
    "remote deposit capture",
    "digital lending",
    "online lending",
    "digital loan",
    "online loan",
    "digital mortgage",
    "automated underwriting",
    "digital payments",
    "electronic payments",
    "contactless payment",
    "digital wallet",
    "p2p payments",
    "person-to-person payments",
    "zelle",
    "bill pay",
    "self-service",
    "customer portal",
    "virtual assistant",
    "chatbot",
    "omnichannel",
]

capability_terms = [
    "digital transformation",
    "digital strategy",
    "digital initiatives",
    "digital capabilities",
    "digital platform",
    "digital platforms",
    "cloud computing",
    "cloud migration",
    "cloud-based",
    "data lake",
    "data warehouse",
    "platform modernization",
    "core conversion",
    "system conversion",
    "technology platform",
    "automation",
    "robotic process automation",
    "legacy system replacement",
    "system integration",
    "artificial intelligence",
    "machine learning",
    "advanced analytics",
    "data analytics",
    "business intelligence",
]

risk_terms = [
    "cybersecurity",
    "cyber security",
    "information security",
    "technology risk",
    "cyber risk",
    "data breach",
    "security breach",
    "system failure",
    "technology failure",
    "business interruption",
    "digital fraud",
    "online fraud",
    "encryption",
    "multi-factor authentication",
    "zero trust",
]

WORD_PAT = re.compile(r"\b\w+\b")

WINDOW_CHARS = 1500
MAX_WINDOWS_PER_FILING = 10


# ==================================================
# 2. Regex helpers
# ==================================================

def phrase_pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in term.split()]
    phrase = r"\s+".join(words)
    return re.compile(
        rf"(?<![A-Za-z0-9]){phrase}(?![A-Za-z0-9])",
        flags=re.IGNORECASE
    )


def make_keyword_records():
    records = []

    for term in deployment_terms:
        records.append({"category": "deployment", "keyword": term})

    for term in capability_terms:
        records.append({"category": "capability", "keyword": term})

    for term in risk_terms:
        records.append({"category": "risk", "keyword": term})

    df = pd.DataFrame(records).drop_duplicates(subset=["category", "keyword"])
    return df.to_dict("records")


def make_combined_pattern(keyword_records):
    terms = [r["keyword"] for r in keyword_records]

    parts = []
    for term in sorted(set(terms), key=len, reverse=True):
        words = [re.escape(w) for w in term.split()]
        phrase = r"\s+".join(words)
        parts.append(rf"(?<![A-Za-z0-9]){phrase}(?![A-Za-z0-9])")

    return re.compile("|".join(parts), flags=re.IGNORECASE)


# ==================================================
# 3. Worker function
# ==================================================

def process_one_file(file_row):
    """
    Process one unique filing text file.

    Returns:
    {
        "hit_rows": [...],
        "window_row": {...}
    }
    """

    keyword_records = make_keyword_records()

    compiled_keyword_records = []
    for r in keyword_records:
        compiled_keyword_records.append({
            "category": r["category"],
            "keyword": r["keyword"],
            "pattern": phrase_pattern(r["keyword"]),
        })

    combined_pattern = make_combined_pattern(keyword_records)

    cik = str(file_row.get("CIK", "")).strip()
    accession_number = str(file_row.get("accession_number", "")).strip()
    path = file_row.get("clean_text_path")

    base_window_row = {
        "CIK": cik,
        "accession_number": accession_number,
        "clean_text_path": path,
        "n_keyword_hits_all_terms": 0,
        "n_windows": 0,
        "matched_keywords_unique": "",
        "keyword_windows": "",
        "has_digital_keywords": 0,
        "read_status": "not_processed",
    }

    if pd.isna(path) or not Path(path).exists():
        base_window_row["read_status"] = "missing_file"
        return {
            "hit_rows": [],
            "window_row": base_window_row,
        }

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        base_window_row["read_status"] = f"error_{type(e).__name__}: {e}"
        return {
            "hit_rows": [],
            "window_row": base_window_row,
        }

    total_words = len(WORD_PAT.findall(text))

    hit_rows = []

    # ------------------------------
    # A. Keyword-level counts
    # ------------------------------
    for r in compiled_keyword_records:
        category = r["category"]
        keyword = r["keyword"]
        pat = r["pattern"]

        matches_for_keyword = list(pat.finditer(text))
        count = len(matches_for_keyword)

        if count > 0:
            hit_rows.append({
                "CIK": cik,
                "accession_number": accession_number,
                "clean_text_path": path,
                "category": category,
                "keyword": keyword,
                "keyword_count": count,
                "total_words": total_words,
                "keyword_density_per_1000_words": (
                    count / total_words * 1000 if total_words > 0 else np.nan
                ),
            })

    # ------------------------------
    # B. Context windows for audit
    # ------------------------------
    all_hits = list(combined_pattern.finditer(text))

    windows = []
    seen_starts = []
    matched_terms_for_window = []

    for m in all_hits:
        matched_term = m.group(0).lower()
        matched_terms_for_window.append(matched_term)

        start = max(0, m.start() - WINDOW_CHARS)
        end = min(len(text), m.end() + WINDOW_CHARS)

        # Avoid near-duplicate overlapping windows
        if any(abs(start - s) < 700 for s in seen_starts):
            continue

        window = text[start:end].strip()
        window = re.sub(r"\s+", " ", window)

        windows.append(window)
        seen_starts.append(start)

        if len(windows) >= MAX_WINDOWS_PER_FILING:
            break

    window_row = {
        **base_window_row,
        "n_keyword_hits_all_terms": len(all_hits),
        "n_windows": len(windows),
        "matched_keywords_unique": "; ".join(sorted(set(matched_terms_for_window))),
        "keyword_windows": "\n\n--- WINDOW BREAK ---\n\n".join(windows),
        "has_digital_keywords": int(len(all_hits) > 0),
        "read_status": "processed",
    }

    return {
        "hit_rows": hit_rows,
        "window_row": window_row,
    }


# ==================================================
# 4. Main
# ==================================================

def main():
    print("Loading input files...")

    matches = pd.read_csv(matches_file, low_memory=False)
    clean_index = pd.read_csv(clean_index_file, low_memory=False)

    print("Matches shape:", matches.shape)
    print("Clean index shape:", clean_index.shape)

    # Standardize keys
    matches["CIK"] = (
        pd.to_numeric(matches["CIK"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
    )

    clean_index["CIK"] = (
        pd.to_numeric(clean_index["CIK"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
    )

    matches["accession_number"] = matches["accession_number"].astype(str).str.strip()
    clean_index["accession_number"] = clean_index["accession_number"].astype(str).str.strip()

    df = matches.merge(
        clean_index[["CIK", "accession_number", "clean_text_path", "text_length", "status"]],
        on=["CIK", "accession_number"],
        how="left"
    )

    print("\nMatched filing rows:", df.shape)
    print("Rows with clean text path:", df["clean_text_path"].notna().sum())

    unique_files = (
        df.dropna(subset=["clean_text_path"])
        .drop_duplicates(subset=["CIK", "accession_number", "clean_text_path"])
        [["CIK", "accession_number", "clean_text_path"]]
        .reset_index(drop=True)
    )

    print("Unique clean files:", unique_files.shape)
    print(f"Using {MAX_WORKERS} workers...")

    tasks = unique_files.to_dict("records")

    all_hit_rows = []
    all_window_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_file, task) for task in tasks]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Counting keyword-level hits in parallel"
        ):
            result = fut.result()
            all_hit_rows.extend(result["hit_rows"])
            all_window_rows.append(result["window_row"])

    hit_level = pd.DataFrame(all_hit_rows)
    windows = pd.DataFrame(all_window_rows)

    print("\nHit-level shape:", hit_level.shape)
    print("Windows shape:", windows.shape)

    # ==================================================
    # 5. Merge metadata
    # ==================================================

    meta_cols = [
        "GVKEY", "tic", "CIK", "YEAR", "fqtr", "datadate",
        "expected_form", "form", "filing_date", "report_date",
        "accession_number", "primary_document"
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    filing_meta = (
        df[meta_cols]
        .drop_duplicates(subset=["CIK", "accession_number"])
        .copy()
    )

    filing_meta["CIK"] = filing_meta["CIK"].astype(str).str.strip()
    filing_meta["accession_number"] = filing_meta["accession_number"].astype(str).str.strip()

    if not hit_level.empty:
        hit_level["CIK"] = hit_level["CIK"].astype(str).str.strip()
        hit_level["accession_number"] = hit_level["accession_number"].astype(str).str.strip()

        hit_level = filing_meta.merge(
            hit_level,
            on=["CIK", "accession_number"],
            how="right"
        )

    hit_level.to_csv(out_hit_level, index=False, encoding="utf-8-sig")

    if not windows.empty:
        windows["CIK"] = windows["CIK"].astype(str).str.strip()
        windows["accession_number"] = windows["accession_number"].astype(str).str.strip()

        windows = filing_meta.merge(
            windows,
            on=["CIK", "accession_number"],
            how="right"
        )

    windows.to_csv(out_windows, index=False, encoding="utf-8-sig")

    # ==================================================
    # 6. Filing-level summary
    # ==================================================

    if not hit_level.empty:
        filing_summary = (
            hit_level
            .groupby([
                "GVKEY", "tic", "CIK", "YEAR", "fqtr", "datadate",
                "expected_form", "form", "filing_date", "report_date",
                "accession_number", "primary_document"
            ], dropna=False)
            .agg(
                total_keyword_hits=("keyword_count", "sum"),
                n_unique_keywords=("keyword", "nunique"),
                deployment_hits=(
                    "keyword_count",
                    lambda x: x[hit_level.loc[x.index, "category"].eq("deployment")].sum()
                ),
                capability_hits=(
                    "keyword_count",
                    lambda x: x[hit_level.loc[x.index, "category"].eq("capability")].sum()
                ),
                risk_hits=(
                    "keyword_count",
                    lambda x: x[hit_level.loc[x.index, "category"].eq("risk")].sum()
                ),
                total_words=("total_words", "max"),
            )
            .reset_index()
        )

        filing_summary["total_keyword_density_per_1000_words"] = (
            filing_summary["total_keyword_hits"] / filing_summary["total_words"] * 1000
        )

        filing_summary["deployment_density_per_1000_words"] = (
            filing_summary["deployment_hits"] / filing_summary["total_words"] * 1000
        )

        filing_summary["capability_density_per_1000_words"] = (
            filing_summary["capability_hits"] / filing_summary["total_words"] * 1000
        )

        filing_summary["risk_density_per_1000_words"] = (
            filing_summary["risk_hits"] / filing_summary["total_words"] * 1000
        )

    else:
        filing_summary = pd.DataFrame()

    filing_summary.to_csv(out_filing_summary, index=False, encoding="utf-8-sig")

    # ==================================================
    # 7. Keyword summaries
    # ==================================================

    if not hit_level.empty:
        keyword_total = (
            hit_level
            .groupby(["category", "keyword"], dropna=False)
            .agg(
                total_count=("keyword_count", "sum"),
                n_filings_mentioned=("accession_number", "nunique"),
                mean_count_when_mentioned=("keyword_count", "mean"),
                median_count_when_mentioned=("keyword_count", "median"),
                mean_density_when_mentioned=("keyword_density_per_1000_words", "mean"),
            )
            .reset_index()
            .sort_values(["total_count", "n_filings_mentioned"], ascending=False)
        )

        keyword_year = (
            hit_level
            .groupby(["YEAR", "category", "keyword"], dropna=False)
            .agg(
                total_count=("keyword_count", "sum"),
                n_filings_mentioned=("accession_number", "nunique"),
                mean_density_when_mentioned=("keyword_density_per_1000_words", "mean"),
            )
            .reset_index()
            .sort_values(["YEAR", "total_count"], ascending=[True, False])
        )

    else:
        keyword_total = pd.DataFrame()
        keyword_year = pd.DataFrame()

    keyword_total.to_csv(out_keyword_total, index=False, encoding="utf-8-sig")
    keyword_year.to_csv(out_keyword_year, index=False, encoding="utf-8-sig")

    # ==================================================
    # 8. Print preview
    # ==================================================

    print("\nSaved files:")
    print(out_hit_level)
    print(out_filing_summary)
    print(out_keyword_total)
    print(out_keyword_year)
    print(out_windows)

    print("\nTop 30 keywords by total count:")
    if not keyword_total.empty:
        print(keyword_total.head(30))
    else:
        print("No keyword hits found.")

    print("\nKeyword totals by category:")
    if not keyword_total.empty:
        print(
            keyword_total.groupby("category")["total_count"]
            .sum()
            .sort_values(ascending=False)
        )

    print("\nYearly keyword totals by category:")
    if not keyword_year.empty:
        print(
            keyword_year.groupby(["YEAR", "category"])["total_count"]
            .sum()
            .reset_index()
            .pivot(index="YEAR", columns="category", values="total_count")
            .fillna(0)
        )

    print("\nDone.")


if __name__ == "__main__":
    freeze_support()
    main()