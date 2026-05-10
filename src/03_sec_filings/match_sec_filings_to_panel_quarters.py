# match_sec_filings_to_panel_quarters_2018_2024.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

targets = pd.read_csv(folder / "sec_filing_targets_2018_2024.csv", low_memory=False)
index = pd.read_csv(folder / "sec_10k_10q_submissions_index_2018_2024.csv", low_memory=False)

targets["CIK"] = pd.to_numeric(targets["CIK"], errors="coerce").astype("Int64")
targets["YEAR"] = pd.to_numeric(targets["YEAR"], errors="coerce").astype("Int64")
targets["fqtr"] = pd.to_numeric(targets["fqtr"], errors="coerce").astype("Int64")
targets["datadate"] = pd.to_datetime(targets["datadate"], errors="coerce")

index["CIK"] = pd.to_numeric(index["CIK"], errors="coerce").astype("Int64")
index["filing_date"] = pd.to_datetime(index["filing_date"], errors="coerce")
index["report_date"] = pd.to_datetime(index["report_date"], errors="coerce")
index["report_year"] = index["report_date"].dt.year.astype("Int64")
index["report_quarter"] = index["report_date"].dt.quarter.astype("Int64")

targets["expected_form"] = targets["fqtr"].map({
    1: "10-Q",
    2: "10-Q",
    3: "10-Q",
    4: "10-K",
})

# Main exact match
matched = targets.merge(
    index,
    left_on=["CIK", "expected_form", "YEAR", "fqtr"],
    right_on=["CIK", "form", "report_year", "report_quarter"],
    how="left",
    suffixes=("", "_sec")
)

unmatched = matched[matched["accession_number"].isna()].copy()
print("Initial matched rows:", matched["accession_number"].notna().sum())
print("Initial unmatched rows:", len(unmatched))

# Fallback nearest report_date within +/- 1 year
fallback_rows = []

if len(unmatched) > 0:
    for _, row in unmatched.iterrows():
        cik = row["CIK"]
        form = row["expected_form"]
        year = row["YEAR"]
        datadate = row["datadate"]

        candidates = index[
            (index["CIK"] == cik)
            & (index["form"] == form)
            & (index["report_year"].between(year - 1, year + 1))
        ].copy()

        if candidates.empty or pd.isna(datadate):
            fallback_rows.append(None)
            continue

        candidates["date_distance"] = (candidates["report_date"] - datadate).abs()
        best = candidates.sort_values(["date_distance", "filing_date"]).head(1)

        fallback_rows.append(best.iloc[0].to_dict() if len(best) == 1 else None)

matched_final = matched.copy()

if len(unmatched) > 0:
    unmatched_idx = unmatched.index.tolist()

    for idx, best in zip(unmatched_idx, fallback_rows):
        if best is None:
            continue
        for col, val in best.items():
            matched_final.loc[idx, col] = val

def make_sec_doc_url(row):
    if pd.isna(row.get("CIK")) or pd.isna(row.get("accession_number")) or pd.isna(row.get("primary_document")):
        return None

    cik_int = int(row["CIK"])
    accession_no_dash = str(row["accession_number"]).replace("-", "")
    primary_doc = str(row["primary_document"])
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dash}/{primary_doc}"

matched_final["sec_document_url"] = matched_final.apply(make_sec_doc_url, axis=1)

out = folder / "sec_filing_matches_2018_2024.csv"
matched_final.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:", out)
print("Final shape:", matched_final.shape)
print("Final matched rows:", matched_final["accession_number"].notna().sum())
print("Final unmatched rows:", matched_final["accession_number"].isna().sum())

print("\nCoverage by expected form:")
print(
    matched_final.groupby("expected_form")["accession_number"]
    .apply(lambda x: x.notna().mean())
)

print("\nCoverage by year-quarter:")
print(
    matched_final.groupby(["YEAR", "fqtr"])["accession_number"]
    .apply(lambda x: x.notna().mean())
)