# check_nyt_rolling_annual.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

files = [
    "covid_county_nyt_rolling_2020.csv",
    "covid_county_nyt_rolling_2021.csv",
    "covid_county_nyt_rolling_2022.csv",
]

for filename in files:
    path = external / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path, low_memory=False)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Date min:", df["date"].min())
    print("Date max:", df["date"].max())
    print("First 5 rows:")
    print(df.head())