# export_sec_filing_targets_2018_2024.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

stock_file = folder / "6020_21.xlsx"

stock = pd.read_excel(stock_file)

print("Stock shape:", stock.shape)
print("Columns:", list(stock.columns))

# Standardize
stock["GVKEY"] = stock["GVKEY"].astype("string").str.strip()
stock["tic"] = stock["tic"].astype("string").str.strip().str.upper()
stock["fyearq"] = pd.to_numeric(stock["fyearq"], errors="coerce").astype("Int64")
stock["fqtr"] = pd.to_numeric(stock["fqtr"], errors="coerce").astype("Int64")
stock["datadate"] = pd.to_datetime(stock["datadate"], errors="coerce")

# CIK may not exist in original stock file.
# If not, merge CIK from the CCM link file.
if "CIK" not in stock.columns and "cik" not in stock.columns:
    ccm_file = folder / "external_data" / "ccm_gvkey_permco_link_clean.csv"
    ccm = pd.read_csv(ccm_file, low_memory=False)
    ccm.columns = [c.strip().upper() for c in ccm.columns]

    ccm["GVKEY"] = ccm["GVKEY"].astype("string").str.strip()
    ccm["CIK"] = pd.to_numeric(ccm["CIK"], errors="coerce").astype("Int64")

    ccm_cik = (
        ccm.dropna(subset=["CIK"])
        .sort_values(["GVKEY", "LINKPRIM"])
        .drop_duplicates(subset=["GVKEY"])
        [["GVKEY", "CIK"]]
    )

    stock = stock.merge(ccm_cik, on="GVKEY", how="left")

cik_col = "CIK" if "CIK" in stock.columns else "cik"
stock[cik_col] = pd.to_numeric(stock[cik_col], errors="coerce").astype("Int64")

# Keep 2018-2024
targets = stock[
    stock["fyearq"].between(2018, 2024)
    & stock[cik_col].notna()
    & stock["fqtr"].between(1, 4)
].copy()

targets = targets.rename(columns={
    cik_col: "CIK",
    "fyearq": "YEAR"
})

targets = targets[["GVKEY", "tic", "CIK", "YEAR", "fqtr", "datadate"]].drop_duplicates()

# Expected forms:
# Q1-Q3 -> 10-Q
# Q4 -> 10-K
targets["expected_form"] = targets["fqtr"].map({
    1: "10-Q",
    2: "10-Q",
    3: "10-Q",
    4: "10-K",
})

out = folder / "sec_filing_targets_2018_2024.csv"
targets.to_csv(out, index=False, encoding="utf-8-sig")

print("\nTargets shape:", targets.shape)
print("Unique GVKEY:", targets["GVKEY"].nunique())
print("Unique CIK:", targets["CIK"].nunique())
print("Saved:", out)

print("\nExpected form counts:")
print(targets["expected_form"].value_counts(dropna=False))

print("\nYear-quarter counts:")
print(targets.groupby(["YEAR", "fqtr"])["GVKEY"].nunique())