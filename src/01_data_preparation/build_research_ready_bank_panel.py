# build_research_ready_bank_panel.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

bank = pd.read_parquet(folder / "sod_bank_panel_2018_2025.parquet")

print("Original bank panel shape:", bank.shape)

clean = bank.copy()

clean = clean[
    (clean["CERT"].notna()) &
    (clean["n_branches"] > 0) &
    (clean["total_branch_deposits"] >= 0)
].copy()

# Bank size categories by assets
clean["bank_size_category"] = pd.cut(
    clean["bank_assets"],
    bins=[-np.inf, 1000000, 10000000, 100000000, np.inf],
    labels=["small", "medium", "large", "very_large"]
)

# Changes
clean = clean.sort_values(["CERT", "YEAR"])

clean["n_branches_lag"] = clean.groupby("CERT")["n_branches"].shift(1)
clean["bank_assets_lag"] = clean.groupby("CERT")["bank_assets"].shift(1)
clean["bank_deposits_lag"] = clean.groupby("CERT")["bank_deposits"].shift(1)

clean["delta_n_branches"] = clean["n_branches"] - clean["n_branches_lag"]

clean["asset_growth"] = np.where(
    clean["bank_assets_lag"] > 0,
    (clean["bank_assets"] - clean["bank_assets_lag"]) / clean["bank_assets_lag"],
    np.nan
)

clean["bank_deposit_growth"] = np.where(
    clean["bank_deposits_lag"] > 0,
    (clean["bank_deposits"] - clean["bank_deposits_lag"]) / clean["bank_deposits_lag"],
    np.nan
)

out_csv = folder / "research_ready_bank_panel_2018_2025.csv"
out_parquet = folder / "research_ready_bank_panel_2018_2025.parquet"

clean.to_csv(out_csv, index=False, encoding="utf-8-sig")
clean.to_parquet(out_parquet, index=False)

print("Clean bank panel shape:", clean.shape)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nYear counts:")
print(clean.groupby("YEAR")["CERT"].nunique())

print("\nBank size category counts:")
print(clean["bank_size_category"].value_counts(dropna=False))