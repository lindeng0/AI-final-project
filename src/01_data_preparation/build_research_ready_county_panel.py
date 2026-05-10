# build_research_ready_county_panel.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

market = pd.read_parquet(folder / "sod_county_market_panel_2018_2025.parquet")

print("Original market panel shape:", market.shape)

# Keep only meaningful county-year banking markets
clean = market.copy()

clean = clean[
    (clean["market_deposits"] > 0) &
    (clean["HHI"].notna()) &
    (clean["HHI"] >= 0) &
    (clean["HHI"] <= 1) &
    (clean["n_banks"] > 0) &
    (clean["n_branches"] > 0)
].copy()

# Add readable concentration categories
clean["concentration_category"] = pd.cut(
    clean["HHI"],
    bins=[-np.inf, 0.15, 0.25, np.inf],
    labels=["low_concentration", "moderate_concentration", "high_concentration"]
)

# Add year-over-year changes within county
clean = clean.sort_values(["STCNTYBR", "YEAR"])

clean["market_deposits_lag"] = clean.groupby("STCNTYBR")["market_deposits"].shift(1)
clean["HHI_lag"] = clean.groupby("STCNTYBR")["HHI"].shift(1)
clean["n_banks_lag"] = clean.groupby("STCNTYBR")["n_banks"].shift(1)
clean["n_branches_lag"] = clean.groupby("STCNTYBR")["n_branches"].shift(1)

clean["market_deposit_growth"] = np.where(
    clean["market_deposits_lag"] > 0,
    (clean["market_deposits"] - clean["market_deposits_lag"]) / clean["market_deposits_lag"],
    np.nan
)

clean["delta_HHI"] = clean["HHI"] - clean["HHI_lag"]
clean["delta_n_banks"] = clean["n_banks"] - clean["n_banks_lag"]
clean["delta_n_branches"] = clean["n_branches"] - clean["n_branches_lag"]

out_csv = folder / "research_ready_county_market_panel_2018_2025.csv"
out_parquet = folder / "research_ready_county_market_panel_2018_2025.parquet"

clean.to_csv(out_csv, index=False, encoding="utf-8-sig")
clean.to_parquet(out_parquet, index=False)

print("Clean county market panel shape:", clean.shape)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nYear counts:")
print(clean.groupby("YEAR")["STCNTYBR"].nunique())

print("\nHHI summary:")
print(clean["HHI"].describe())

print("\nConcentration category counts:")
print(clean["concentration_category"].value_counts(dropna=False))