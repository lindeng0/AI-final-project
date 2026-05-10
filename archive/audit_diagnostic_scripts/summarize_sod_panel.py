# summarize_sod_panel.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

panel = pd.read_parquet(folder / "sod_branch_panel_2018_2025.parquet")

# --------------------------------------------------
# 1. Branch-level summary
# --------------------------------------------------

branch_summary = (
    panel.groupby("YEAR")
    .agg(
        n_rows=("UNINUMBR", "size"),
        n_branches=("UNINUMBR", "nunique"),
        n_banks=("CERT", "nunique"),
        total_branch_deposits=("DEPSUMBR", "sum"),
        avg_branch_deposits=("DEPSUMBR", "mean"),
        median_branch_deposits=("DEPSUMBR", "median"),
    )
    .reset_index()
)

# --------------------------------------------------
# 2. Bank-level variables must be deduplicated first
# --------------------------------------------------

bank_year = (
    panel[["YEAR", "CERT", "ASSET", "DEPSUM"]]
    .drop_duplicates(subset=["YEAR", "CERT"])
)

bank_summary = (
    bank_year.groupby("YEAR")
    .agg(
        total_bank_assets=("ASSET", "sum"),
        total_bank_deposits=("DEPSUM", "sum"),
        avg_bank_assets=("ASSET", "mean"),
        median_bank_assets=("ASSET", "median"),
    )
    .reset_index()
)

summary = branch_summary.merge(bank_summary, on="YEAR", how="left")

# Optional formatting for easier reading in console
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nSummary by year:")
print(summary)

out_csv = folder / "sod_summary_by_year.csv"
out_parquet = folder / "sod_summary_by_year.parquet"

summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
summary.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)