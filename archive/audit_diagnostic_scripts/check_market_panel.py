# check_market_panel.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

market = pd.read_csv(folder / "sod_county_market_panel_2018_2025.csv", low_memory=False)

print("Shape:", market.shape)

print("\nHHI summary:")
print(market["HHI"].describe())

print("\nNumber of county markets by year:")
print(market.groupby("YEAR")["STCNTYBR"].nunique())

print("\nMost concentrated 2025 county markets:")
print(
    market[market["YEAR"] == 2025]
    .sort_values("HHI", ascending=False)
    [["STALPBR", "CNTYNAMB", "n_banks", "n_branches", "market_deposits", "HHI", "max_market_share"]]
    .head(20)
)

print("\nLeast concentrated 2025 county markets:")
print(
    market[market["YEAR"] == 2025]
    .sort_values("HHI", ascending=True)
    [["STALPBR", "CNTYNAMB", "n_banks", "n_branches", "market_deposits", "HHI", "max_market_share"]]
    .head(20)
)