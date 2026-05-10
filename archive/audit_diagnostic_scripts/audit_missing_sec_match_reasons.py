# audit_missing_sec_match_reasons.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

missing = pd.read_csv(folder / "audit_missing_clean_text_coverage.csv", low_memory=False)
index = pd.read_csv(folder / "sec_10k_10q_submissions_index_2018_2024.csv", low_memory=False)
targets = pd.read_csv(folder / "sec_filing_targets_2018_2024.csv", low_memory=False)

for df in [missing, index, targets]:
    if "CIK" in df.columns:
        df["CIK"] = pd.to_numeric(df["CIK"], errors="coerce").astype("Int64").astype(str)
        df["CIK"] = df["CIK"].str.replace("<NA>", "", regex=False).str.strip()

index["form"] = index["form"].astype(str).str.strip()
index["filing_date"] = pd.to_datetime(index["filing_date"], errors="coerce")
index["report_date"] = pd.to_datetime(index["report_date"], errors="coerce")
index["report_year"] = index["report_date"].dt.year
index["report_quarter"] = index["report_date"].dt.quarter

missing["YEAR"] = pd.to_numeric(missing["YEAR"], errors="coerce").astype("Int64")
missing["fqtr"] = pd.to_numeric(missing["fqtr"], errors="coerce").astype("Int64")

rows = []

for _, r in missing.iterrows():
    cik = r["CIK"]
    year = int(r["YEAR"])
    fqtr = int(r["fqtr"])
    expected_form = r["expected_form"]

    idx_cik = index[index["CIK"] == cik].copy()
    idx_form = idx_cik[idx_cik["form"] == expected_form].copy()
    idx_year = idx_form[idx_form["report_year"].between(year - 1, year + 1, inclusive="both")].copy()
    idx_exact = idx_form[
        (idx_form["report_year"] == year)
        & (idx_form["report_quarter"] == fqtr)
    ].copy()

    rows.append({
        "GVKEY": r.get("GVKEY"),
        "tic": r.get("tic"),
        "CIK": cik,
        "YEAR": year,
        "fqtr": fqtr,
        "expected_form": expected_form,
        "index_has_cik": len(idx_cik) > 0,
        "index_n_cik_filings": len(idx_cik),
        "index_n_expected_form": len(idx_form),
        "index_n_near_year": len(idx_year),
        "index_n_exact_year_qtr": len(idx_exact),
        "available_forms": "; ".join(sorted(idx_cik["form"].dropna().unique()))[:300],
        "available_report_dates": "; ".join(
            idx_form["report_date"].dropna().dt.strftime("%Y-%m-%d").head(20).tolist()
        ),
    })

out = pd.DataFrame(rows)
out_file = folder / "audit_missing_sec_match_reasons.csv"
out.to_csv(out_file, index=False, encoding="utf-8-sig")

print("Saved:", out_file)

print("\nSummary:")
print(out[[
    "index_has_cik",
    "index_n_expected_form",
    "index_n_near_year",
    "index_n_exact_year_qtr"
]].describe())

print("\nMissing where SEC index has CIK but exact quarter failed:")
print(
    out[
        (out["index_has_cik"] == True)
        & (out["index_n_expected_form"] > 0)
        & (out["index_n_exact_year_qtr"] == 0)
    ].head(50)
)