# build_research_ready_bank_covid_panel_2020_2022.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

analysis = pd.read_parquet(folder / "final_bank_covid_panel_2018_2025.parquet")

print("Original final panel shape:", analysis.shape)

# --------------------------------------------------
# 1. Main COVID health-shock sample: 2020-2022
# --------------------------------------------------

main = analysis[
    analysis["YEAR"].between(2020, 2022)
    & analysis["covid_health_exposure_main"].notna()
].copy()

print("\nMain health-shock sample shape:", main.shape)

print("\nDuplicate YEAR + CERT in main sample:")
print(main.duplicated(subset=["YEAR", "CERT"]).sum())

print("\nYear counts:")
print(main.groupby("YEAR")["CERT"].nunique())

# --------------------------------------------------
# 2. Exposure coverage
# --------------------------------------------------

exposure_cols = [
    "covid_health_exposure_main",
    "covid_case_exposure_main",
    "covid_policy_exposure_main",
    "covid_overall_exposure_main",
    "covid_health_exposure_branch_hq",
    "covid_policy_exposure_branch_hq",
]

print("\nExposure coverage:")
for col in exposure_cols:
    if col in main.columns:
        print("\n", col)
        print(main.groupby("YEAR")[col].apply(lambda x: x.notna().mean()))

print("\nExposure summaries:")
for col in exposure_cols:
    if col in main.columns:
        print("\n", col)
        print(main.groupby("YEAR")[col].describe())

# --------------------------------------------------
# 3. Save
# --------------------------------------------------

out_csv = folder / "research_ready_bank_covid_panel_2020_2022.csv"
out_parquet = folder / "research_ready_bank_covid_panel_2020_2022.parquet"

main.to_csv(out_csv, index=False, encoding="utf-8-sig")
main.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)