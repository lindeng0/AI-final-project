# check_downloaded_covid_data.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

files = [
    "covid_county_nyt_us_counties.csv",
    "covid_county_nyt_rolling_averages.csv",
    "OxCGRT_compact_subnational_v1.csv",
    "OxCGRT_simplified_subnational_v1.csv",
]

for filename in files:
    path = external / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path, nrows=5, low_memory=False)
    print("Path:", path)
    print("Preview shape:", df.shape)
    print("Columns:")
    print(list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())