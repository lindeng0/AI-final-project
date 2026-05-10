# check_final_bank_covid_panel.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

analysis = pd.read_parquet(folder / "final_bank_covid_panel_2018_2025.parquet")
covid = pd.read_parquet(folder / "bank_covid_exposure_annual_2020_2022.parquet")

print("=" * 80)
print("1. Final panel shape")
print("=" * 80)
print(analysis.shape)

print("\nColumns containing NAME:")
name_like_cols = [c for c in analysis.columns if "NAME" in c.upper()]
print(name_like_cols)

print("\nYear counts:")
print(analysis.groupby("YEAR")["CERT"].nunique())

print("\nDuplicate YEAR + CERT:")
print(analysis.duplicated(subset=["YEAR", "CERT"]).sum())

print("\n" + "=" * 80)
print("2. COVID exposure file check")
print("=" * 80)

print("COVID exposure shape:", covid.shape)
print("Expected rows: 5313 banks in 2019 × 3 years = 15939")

print("\nCOVID exposure year counts:")
print(covid.groupby("YEAR")["CERT"].nunique())

print("\nDuplicate YEAR + CERT in COVID exposure:")
print(covid.duplicated(subset=["YEAR", "CERT"]).sum())

print("\n" + "=" * 80)
print("3. Exposure coverage by year in final panel")
print("=" * 80)

exposure_cols = [
    "covid_health_exposure_main",
    "covid_policy_exposure_main",
    "covid_case_exposure_main",
    "covid_overall_exposure_main",
    "covid_health_exposure_branch_hq",
    "covid_policy_exposure_branch_hq",
]

for col in exposure_cols:
    if col in analysis.columns:
        print("\n", col)
        print(analysis.groupby("YEAR")[col].apply(lambda x: x.notna().mean()))

print("\n" + "=" * 80)
print("4. Exposure summary, 2020-2022")
print("=" * 80)

for col in exposure_cols:
    if col in analysis.columns:
        print("\n", col)
        print(
            analysis[analysis["YEAR"].between(2020, 2022)]
            .groupby("YEAR")[col]
            .describe()
        )

print("\n" + "=" * 80)
print("5. Top exposed banks in 2020 by health shock")
print("=" * 80)

# Automatically find a usable bank name column
name_candidates = [
    "NAMEFULL",
    "NAMEFULL_x",
    "NAMEFULL_y",
    "NAMEFULL_bank",
]

name_col = None
for c in name_candidates:
    if c in analysis.columns:
        name_col = c
        break

base_cols = [
    "CERT",
    "n_branches",
    "covid_health_exposure_main",
    "covid_policy_exposure_main",
    "covid_health_exposure_branch_hq",
]

if name_col is not None:
    display_cols = ["CERT", name_col] + [c for c in base_cols if c != "CERT"]
else:
    print("No NAMEFULL-like column found. Showing CERT only.")
    display_cols = base_cols

display_cols = [c for c in display_cols if c in analysis.columns]

top = (
    analysis[
        (analysis["YEAR"] == 2020)
        & (analysis["covid_health_exposure_main"].notna())
    ]
    .sort_values("covid_health_exposure_main", ascending=False)
    [display_cols]
    .head(20)
)

print(top)

print("\n" + "=" * 80)
print("6. Lowest exposed banks in 2020 by health shock")
print("=" * 80)

low = (
    analysis[
        (analysis["YEAR"] == 2020)
        & (analysis["covid_health_exposure_main"].notna())
    ]
    .sort_values("covid_health_exposure_main", ascending=True)
    [display_cols]
    .head(20)
)

print(low)

print("\n" + "=" * 80)
print("7. Missing exposure counts, 2020-2022")
print("=" * 80)

sample = analysis[analysis["YEAR"].between(2020, 2022)].copy()

for col in exposure_cols:
    if col in sample.columns:
        print(col, sample[col].isna().sum())

print("\nQuality check completed.")