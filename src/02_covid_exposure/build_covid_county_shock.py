# build_covid_county_shock.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

input_files = [
    external / "covid_county_nyt_rolling_2020.csv",
    external / "covid_county_nyt_rolling_2021.csv",
    external / "covid_county_nyt_rolling_2022.csv",
]

dfs = []

for path in input_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"Reading {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

covid = pd.concat(dfs, ignore_index=True)

print("\nCombined NYT rolling county COVID shape:", covid.shape)
print("Columns:", list(covid.columns))

# --------------------------------------------------
# 1. Parse date and county FIPS
# --------------------------------------------------

covid["date"] = pd.to_datetime(covid["date"], errors="coerce")
covid["YEAR"] = covid["date"].dt.year

# geoid example: USA-53061
covid["county_fips"] = (
    covid["geoid"]
    .astype("string")
    .str.replace("USA-", "", regex=False)
    .str.zfill(5)
)

covid = covid[
    covid["county_fips"].notna()
    & (covid["county_fips"].str.len() == 5)
    & covid["YEAR"].between(2020, 2022)
].copy()

# Numeric columns
num_cols = [
    "cases",
    "cases_avg",
    "cases_avg_per_100k",
    "deaths",
    "deaths_avg",
    "deaths_avg_per_100k",
]

for col in num_cols:
    covid[col] = pd.to_numeric(covid[col], errors="coerce")

# --------------------------------------------------
# 2. Annual county-level health shock
# --------------------------------------------------
# Main shock: average daily rolling deaths per 100k within a year.
# Alternative shocks: peak deaths per 100k, average cases per 100k, peak cases per 100k.

annual = (
    covid.groupby(["county_fips", "YEAR"], dropna=False)
    .agg(
        county=("county", "first"),
        state=("state", "first"),
        avg_cases_per_100k=("cases_avg_per_100k", "mean"),
        max_cases_per_100k=("cases_avg_per_100k", "max"),
        avg_deaths_per_100k=("deaths_avg_per_100k", "mean"),
        max_deaths_per_100k=("deaths_avg_per_100k", "max"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_days=("date", "nunique"),
    )
    .reset_index()
)

# --------------------------------------------------
# 3. Z-score within year
# --------------------------------------------------

def zscore_by_year(df, col):
    return df.groupby("YEAR")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan
    )

for col in [
    "avg_cases_per_100k",
    "max_cases_per_100k",
    "avg_deaths_per_100k",
    "max_deaths_per_100k",
]:
    annual[f"z_{col}"] = zscore_by_year(annual, col)

# Main health shock
annual["covid_health_shock"] = annual["z_avg_deaths_per_100k"]

# Alternative health shocks
annual["covid_health_shock_peak"] = annual["z_max_deaths_per_100k"]
annual["covid_case_shock"] = annual["z_avg_cases_per_100k"]

# --------------------------------------------------
# 4. Save
# --------------------------------------------------

out_csv = external / "county_covid_health_shock_annual.csv"
out_parquet = external / "county_covid_health_shock_annual.parquet"

annual.to_csv(out_csv, index=False, encoding="utf-8-sig")
annual.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nAnnual county COVID shock shape:", annual.shape)

print("\nYear counts:")
print(annual.groupby("YEAR")["county_fips"].nunique())

print("\nDate coverage by year:")
print(
    annual.groupby("YEAR")
    .agg(
        first_date=("first_date", "min"),
        last_date=("last_date", "max"),
        min_days=("n_days", "min"),
        median_days=("n_days", "median"),
        max_days=("n_days", "max"),
    )
)

print("\nMain shock summary:")
print(
    annual.groupby("YEAR")[
        ["avg_deaths_per_100k", "max_deaths_per_100k", "covid_health_shock"]
    ].describe()
)