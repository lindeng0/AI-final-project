# check_bank_panel.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

bank = pd.read_csv(folder / "sod_bank_panel_2018_2025.csv", low_memory=False)

print("Shape:", bank.shape)
print("\nYears:")
print(bank["YEAR"].value_counts().sort_index())

print("\nTop 20 banks by 2025 branch count:")
print(
    bank[bank["YEAR"] == 2025]
    .sort_values("n_branches", ascending=False)
    [["CERT", "NAMEFULL", "n_branches", "total_branch_deposits", "bank_assets"]]
    .head(20)
)