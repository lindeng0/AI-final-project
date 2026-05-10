# describe_stock_covid_performance_panel.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

df = pd.read_parquet(folder / "analysis_stock_covid_performance_panel_2020_2022.parquet")

print("Panel shape:", df.shape)
print("Unique firms:", df["GVKEY"].nunique())

print("\nYear-quarter firm counts:")
print(df.groupby(["YEAR", "fqtr"])["GVKEY"].nunique())

vars_main = [
    "covid_health_exposure_main_hcr",
    "covid_case_exposure_main_hcr",
    "covid_policy_exposure_main_hcr",
    "covid_overall_exposure_main_hcr",
    "roa_q",
    "roa_q_w",
    "price_return_q",
    "price_return_q_w",
    "mktcap_growth_q",
    "mktcap_growth_q_w",
    "asset_growth_q",
    "asset_growth_q_w",
    "roa_change_q",
    "roa_change_q_w",
    "log_assets_q",
    "log_mktcap_q",
]

vars_main = [c for c in vars_main if c in df.columns]

print("\nDescriptive statistics:")
print(df[vars_main].describe())

print("\nCorrelation matrix:")
print(df[vars_main].corr())

print("\nMean by year-quarter:")
print(
    df.groupby("year_quarter")[vars_main]
    .mean()
)

out = folder / "descriptive_stock_covid_performance_summary.csv"

summary = df[vars_main].describe().T
summary.to_csv(out, encoding="utf-8-sig")

print("\nSaved descriptive summary:")
print(out)