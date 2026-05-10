# build_hcr_covid_exposure.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

branch_file = folder / "sod_branch_panel_2018_2025.parquet"
county_covid_file = external / "county_covid_health_shock_annual.parquet"
state_policy_file = external / "state_covid_policy_shock_annual.parquet"

branch = pd.read_parquet(branch_file)
county_covid = pd.read_parquet(county_covid_file)
state_policy = pd.read_parquet(state_policy_file)

print("Branch panel:", branch.shape)
print("County COVID:", county_covid.shape)
print("State policy:", state_policy.shape)

# --------------------------------------------------
# 1. Required columns
# --------------------------------------------------

required_cols = [
    "YEAR", "CERT", "NAMEFULL", "UNINUMBR",
    "RSSDHCR", "NAMEHCR",
    "STCNTYBR", "STALPBR",
    "DEPSUMBR", "ASSET", "DEPSUM",
]

missing = [c for c in required_cols if c not in branch.columns]
if missing:
    raise ValueError("Missing required SOD columns: " + str(missing))

# --------------------------------------------------
# 2. Standardize keys
# --------------------------------------------------

branch["YEAR"] = pd.to_numeric(branch["YEAR"], errors="coerce").astype("Int64")

for col in ["CERT", "UNINUMBR", "RSSDHCR", "STCNTYBR"]:
    branch[col] = branch[col].astype("string").str.strip()

branch["STCNTYBR"] = (
    pd.to_numeric(branch["STCNTYBR"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

branch["STALPBR"] = branch["STALPBR"].astype("string").str.strip().str.upper()
branch["DEPSUMBR"] = pd.to_numeric(branch["DEPSUMBR"], errors="coerce")
branch["ASSET"] = pd.to_numeric(branch["ASSET"], errors="coerce")
branch["DEPSUM"] = pd.to_numeric(branch["DEPSUM"], errors="coerce")

county_covid["county_fips"] = (
    pd.to_numeric(county_covid["county_fips"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)
county_covid["YEAR"] = pd.to_numeric(county_covid["YEAR"], errors="coerce").astype("Int64")

state_policy["YEAR"] = pd.to_numeric(state_policy["YEAR"], errors="coerce").astype("Int64")
state_policy["STALPBR"] = state_policy["STALPBR"].astype("string").str.strip().str.upper()

# --------------------------------------------------
# 3. Use 2019 branch network as pre-pandemic exposure base
# --------------------------------------------------

base_year = 2019
covid_years = [2020, 2021, 2022]

base = branch[branch["YEAR"] == base_year].copy()

# Keep only rows with holding company RSSD
base = base[base["RSSDHCR"].notna() & (base["RSSDHCR"] != "")].copy()

print("\nBase year:", base_year)
print("Base branches:", base.shape)
print("Number of holding companies:", base["RSSDHCR"].nunique())

# --------------------------------------------------
# 4. Branch weights within holding company
# --------------------------------------------------

base["deposit_weight_raw"] = base["DEPSUMBR"].clip(lower=0)

hcr_deposit_sum = base.groupby("RSSDHCR")["deposit_weight_raw"].transform("sum")

base["branch_deposit_weight"] = np.where(
    hcr_deposit_sum > 0,
    base["deposit_weight_raw"] / hcr_deposit_sum,
    np.nan,
)

base["branch_equal_weight"] = (
    1 / base.groupby("RSSDHCR")["UNINUMBR"].transform("count")
)

base["branch_weight_main"] = (
    base["branch_deposit_weight"].fillna(base["branch_equal_weight"])
)

weight_check = (
    base.groupby("RSSDHCR")["branch_weight_main"]
    .sum()
    .reset_index(name="weight_sum")
)

print("\nWeight sum check:")
print(weight_check["weight_sum"].describe())

# --------------------------------------------------
# 5. Expand base branch network to COVID years
# --------------------------------------------------

base_expanded = pd.concat(
    [base.assign(YEAR=y) for y in covid_years],
    ignore_index=True
)

# --------------------------------------------------
# 6. Merge county health shock
# --------------------------------------------------

health_cols = [
    "county_fips", "YEAR",
    "avg_cases_per_100k", "max_cases_per_100k",
    "avg_deaths_per_100k", "max_deaths_per_100k",
    "z_avg_cases_per_100k", "z_max_cases_per_100k",
    "z_avg_deaths_per_100k", "z_max_deaths_per_100k",
    "covid_health_shock", "covid_health_shock_peak", "covid_case_shock",
]

base_expanded = base_expanded.merge(
    county_covid[health_cols],
    left_on=["STCNTYBR", "YEAR"],
    right_on=["county_fips", "YEAR"],
    how="left",
)

# --------------------------------------------------
# 7. Merge state policy shock
# --------------------------------------------------

policy_cols = [c for c in state_policy.columns if c not in ["STALPBR", "YEAR"]]

base_expanded = base_expanded.merge(
    state_policy[["STALPBR", "YEAR"] + policy_cols],
    on=["STALPBR", "YEAR"],
    how="left",
)

# --------------------------------------------------
# 8. Weighted average function
# --------------------------------------------------

def weighted_avg(g, value_col, weight_col):
    valid = g[value_col].notna() & g[weight_col].notna()
    if valid.sum() == 0:
        return np.nan
    v = g.loc[valid, value_col]
    w = g.loc[valid, weight_col]
    if w.sum() <= 0:
        return np.nan
    return np.average(v, weights=w)

# --------------------------------------------------
# 9. Holding-company-level exposure
# --------------------------------------------------

shock_cols = [
    "avg_cases_per_100k",
    "max_cases_per_100k",
    "avg_deaths_per_100k",
    "max_deaths_per_100k",
    "covid_health_shock",
    "covid_health_shock_peak",
    "covid_case_shock",
    "covid_policy_shock",
    "covid_policy_shock_containment",
    "covid_policy_shock_government_response",
]

rows = []

for (rssdhcr, year), g in base_expanded.groupby(["RSSDHCR", "YEAR"], dropna=False):
    row = {
        "RSSDHCR": rssdhcr,
        "YEAR": year,
        "NAMEHCR": g["NAMEHCR"].dropna().iloc[0] if g["NAMEHCR"].dropna().size > 0 else np.nan,
        "n_base_branches_hcr": g["UNINUMBR"].nunique(),
        "n_base_bank_subsidiaries": g["CERT"].nunique(),
        "base_total_branch_deposits_hcr": g["DEPSUMBR"].sum(),
    }

    for col in shock_cols:
        if col in g.columns:
            row[f"hcr_exp_{col}_deposit_w"] = weighted_avg(g, col, "branch_weight_main")
            row[f"hcr_exp_{col}_equal_w"] = weighted_avg(g, col, "branch_equal_weight")

    rows.append(row)

hcr_exposure = pd.DataFrame(rows)

# Main names
hcr_exposure["covid_health_exposure_main_hcr"] = (
    hcr_exposure["hcr_exp_covid_health_shock_deposit_w"]
)

hcr_exposure["covid_case_exposure_main_hcr"] = (
    hcr_exposure["hcr_exp_covid_case_shock_deposit_w"]
)

hcr_exposure["covid_policy_exposure_main_hcr"] = (
    hcr_exposure["hcr_exp_covid_policy_shock_deposit_w"]
)

hcr_exposure["covid_overall_exposure_main_hcr"] = (
    hcr_exposure["covid_health_exposure_main_hcr"]
    + hcr_exposure["covid_policy_exposure_main_hcr"]
)

# --------------------------------------------------
# 10. Save
# --------------------------------------------------

out_csv = folder / "hcr_covid_exposure_annual_2020_2022.csv"
out_parquet = folder / "hcr_covid_exposure_annual_2020_2022.parquet"

hcr_exposure.to_csv(out_csv, index=False, encoding="utf-8-sig")
hcr_exposure.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nHCR exposure shape:", hcr_exposure.shape)
print("\nHCR exposure counts by year:")
print(hcr_exposure.groupby("YEAR")["RSSDHCR"].nunique())

print("\nCoverage:")
for col in [
    "covid_health_exposure_main_hcr",
    "covid_case_exposure_main_hcr",
    "covid_policy_exposure_main_hcr",
    "covid_overall_exposure_main_hcr",
]:
    print("\n", col)
    print(hcr_exposure.groupby("YEAR")[col].apply(lambda x: x.notna().mean()))