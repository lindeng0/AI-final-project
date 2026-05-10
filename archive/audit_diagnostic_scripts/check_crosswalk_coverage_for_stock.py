# check_crosswalk_coverage_for_stock.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

stock_file = folder / "6020_21.xlsx"
crosswalk_file = external / "gvkey_permco_rssd_link.csv"

stock = pd.read_excel(stock_file)
cw = pd.read_csv(crosswalk_file, low_memory=False)

stock["GVKEY"] = stock["GVKEY"].astype("string").str.strip()
cw["GVKEY"] = cw["GVKEY"].astype("string").str.strip()

firm_list = (
    stock.groupby(["GVKEY", "tic"], dropna=False)
    .agg(
        first_fyearq=("fyearq", "min"),
        last_fyearq=("fyearq", "max"),
        n_quarters=("datadate", "count"),
        sic=("sic", "first"),
    )
    .reset_index()
)

cw_unique = (
    cw.dropna(subset=["RSSDID"])
    .drop_duplicates(subset=["GVKEY", "RSSDID"])
    [["GVKEY", "PERMCO", "RSSDID", "RSSDHCR", "FRB_NAME", "FRB_INST_TYPE"]]
)

merged = firm_list.merge(cw_unique, on="GVKEY", how="left")

print("Unique stock firms:", firm_list["GVKEY"].nunique())
print("Firms with RSSDID:", merged[merged["RSSDID"].notna()]["GVKEY"].nunique())
print("Firms without RSSDID:", merged[merged["RSSDID"].isna()]["GVKEY"].nunique())

print("\nCoverage rate:")
print(merged[merged["RSSDID"].notna()]["GVKEY"].nunique() / firm_list["GVKEY"].nunique())

print("\nRows with multiple RSSDID per GVKEY:")
multi = (
    merged.dropna(subset=["RSSDID"])
    .groupby("GVKEY")["RSSDID"]
    .nunique()
    .reset_index(name="n_rssdid")
)
print(multi[multi["n_rssdid"] > 1].head(30))
print("Count:", (multi["n_rssdid"] > 1).sum())

print("\nUnmatched firms:")
unmatched = merged[merged["RSSDID"].isna()].copy()
print(unmatched[["GVKEY", "tic", "first_fyearq", "last_fyearq", "n_quarters", "sic"]].head(100))

out = folder / "stock_firm_crosswalk_coverage.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved coverage file:")
print(out)