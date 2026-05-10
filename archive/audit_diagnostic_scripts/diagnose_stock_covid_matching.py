# diagnose_stock_covid_matching.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

df = pd.read_parquet(folder / "stock_stats_with_hcr_covid_exposure.parquet")

df["GVKEY"] = df["GVKEY"].astype("string").str.strip()
df["tic"] = df["tic"].astype("string").str.strip().str.upper()
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

# Firm-level diagnostics
firm_diag = (
    df.groupby(["GVKEY", "tic"], dropna=False)
    .agg(
        first_year=("YEAR", "min"),
        last_year=("YEAR", "max"),
        n_rows=("YEAR", "size"),
        has_rssdhcr=("RSSDHCR", lambda x: x.notna().any()),
        has_covid_health=("covid_health_exposure_main_hcr", lambda x: x.notna().any()),
        sic=("sic", "first"),
    )
    .reset_index()
)

firm_diag["match_status"] = "matched_with_covid"
firm_diag.loc[~firm_diag["has_rssdhcr"], "match_status"] = "no_rssdhcr_mapping"
firm_diag.loc[
    firm_diag["has_rssdhcr"] & ~firm_diag["has_covid_health"],
    "match_status"
] = "rssdhcr_but_no_hcr_exposure"

print("Firm-level match status:")
print(firm_diag["match_status"].value_counts())

print("\nShare:")
print(firm_diag["match_status"].value_counts(normalize=True))

# Row-level by year
row_diag = (
    df.groupby("YEAR")
    .agg(
        n_rows=("GVKEY", "size"),
        n_firms=("GVKEY", "nunique"),
        share_has_rssdhcr=("RSSDHCR", lambda x: x.notna().mean()),
        share_has_covid_health=("covid_health_exposure_main_hcr", lambda x: x.notna().mean()),
        share_has_covid_policy=("covid_policy_exposure_main_hcr", lambda x: x.notna().mean()),
    )
    .reset_index()
)

print("\nRow-level yearly diagnostics:")
print(row_diag)

out_firm = folder / "diagnostic_stock_covid_matching_firm_level.csv"
out_year = folder / "diagnostic_stock_covid_matching_by_year.csv"

firm_diag.to_csv(out_firm, index=False, encoding="utf-8-sig")
row_diag.to_csv(out_year, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(" -", out_firm)
print(" -", out_year)

print("\nUnmatched / no RSSDHCR firms:")
print(
    firm_diag[firm_diag["match_status"] == "no_rssdhcr_mapping"]
    .sort_values(["tic", "GVKEY"])
    .head(100)
)