# check_sod_columns.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

panel = pd.read_parquet(folder / "sod_branch_panel_2018_2025.parquet")

print("Current branch panel columns:")
print(list(panel.columns))

required = ["CERT", "YEAR", "UNINUMBR", "STCNTYBR", "STALPBR", "STCNTY", "STALP", "CITY", "ZIP"]

print("\nRequired columns check:")
for col in required:
    print(col, col in panel.columns)