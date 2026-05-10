# build_bank_covid_exposure.py

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
# 1. Required columns check
# --------------------------------------------------

required_branch_cols = [
    "CERT", "YEAR", "UNINUMBR", "NAMEFULL",
    "STCNTYBR", "STALPBR",
    "STCNTY", "STALP",
    "DEPSUMBR", "ASSET", "DEPSUM"
]

missing_cols = [c for c in required_branch_cols if c not in branch.columns]

if missing_cols:
    raise ValueError(
        "Missing columns in branch panel: " + str(missing_cols)
    )

# --------------------------------------------------
# 2. Standardize keys
# --------------------------------------------------

branch["CERT"] = branch["CERT"].astype("string").str.strip()
branch["YEAR"] = pd.to_numeric(branch["YEAR"], errors="coerce").astype("Int64")

branch["STCNTYBR"] = (
    pd.to_numeric(branch["STCNTYBR"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

branch["STCNTY"] = (
    pd.to_numeric(branch["STCNTY"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

branch["STALPBR"] = branch["STALPBR"].astype("string").str.strip().str.upper()
branch["STALP"] = branch["STALP"].astype("string").str.strip().str.upper()

branch["DEPSUMBR"] = pd.to_numeric(branch["DEPSUMBR"], errors="coerce")
branch["ASSET"] = pd.to_numeric(branch["ASSET"], errors="coerce")
branch["DEPSUM"] = pd.to_numeric(branch["DEPSUM"], errors="coerce")

county_covid["county_fips"] = (
    pd.to_numeric(county_covid["county_fips"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

county_covid["YEAR"] = pd.to_numeric(
    county_covid["YEAR"], errors="coerce"
).astype("Int64")

state_policy["YEAR"] = pd.to_numeric(
    state_policy["YEAR"], errors="coerce"
).astype("Int64")

state_policy["STALPBR"] = (
    state_policy["STALPBR"].astype("string").str.strip().str.upper()
)

# --------------------------------------------------
# 3. Use 2019 branch network as exposure base
# --------------------------------------------------

base_year = 2019
covid_years = [2020, 2021, 2022]

base = branch[branch["YEAR"] == base_year].copy()

print("\nBase branch network year:", base_year)
print("Base shape:", base.shape)
print("Number of banks in base year:", base["CERT"].nunique())

# --------------------------------------------------
# 4. Branch weights
# --------------------------------------------------

base["deposit_weight_raw"] = base["DEPSUMBR"].clip(lower=0)

bank_deposit_sum = base.groupby("CERT")["deposit_weight_raw"].transform("sum")

base["branch_deposit_weight"] = np.where(
    bank_deposit_sum > 0,
    base["deposit_weight_raw"] / bank_deposit_sum,
    np.nan
)

base["branch_equal_weight"] = (
    1 / base.groupby("CERT")["UNINUMBR"].transform("count")
)

base["branch_weight_main"] = (
    base["branch_deposit_weight"].fillna(base["branch_equal_weight"])
)

weight_check = (
    base.groupby("CERT")["branch_weight_main"]
    .sum()
    .reset_index(name="weight_sum")
)

print("\nWeight sum check:")
print(weight_check["weight_sum"].describe())

# --------------------------------------------------
# 5. Expand 2019 branch network to COVID years
# --------------------------------------------------

base_expanded = pd.concat(
    [base.assign(YEAR=y) for y in covid_years],
    ignore_index=True
)

# --------------------------------------------------
# 6. Merge branch county health shock
# --------------------------------------------------

health_cols = [
    "county_fips", "YEAR",
    "avg_cases_per_100k", "max_cases_per_100k",
    "avg_deaths_per_100k", "max_deaths_per_100k",
    "z_avg_cases_per_100k", "z_max_cases_per_100k",
    "z_avg_deaths_per_100k", "z_max_deaths_per_100k",
    "covid_health_shock", "covid_health_shock_peak", "covid_case_shock"
]

base_expanded = base_expanded.merge(
    county_covid[health_cols],
    left_on=["STCNTYBR", "YEAR"],
    right_on=["county_fips", "YEAR"],
    how="left"
)

# --------------------------------------------------
# 7. Merge branch state policy shock
# --------------------------------------------------

policy_cols = [c for c in state_policy.columns if c not in ["STALPBR", "YEAR"]]

base_expanded = base_expanded.merge(
    state_policy[["STALPBR", "YEAR"] + policy_cols],
    on=["STALPBR", "YEAR"],
    how="left"
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
# 9. Branch-network exposure
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

exposure_rows = []

for (cert, year), g in base_expanded.groupby(["CERT", "YEAR"], dropna=False):
    row = {
        "CERT": cert,
        "YEAR": year,
        "n_base_branches": g["UNINUMBR"].nunique(),
        "base_total_branch_deposits": g["DEPSUMBR"].sum(),
    }

    for col in shock_cols:
        if col in g.columns:
            row[f"branch_exp_{col}_deposit_w"] = weighted_avg(
                g, col, "branch_weight_main"
            )
            row[f"branch_exp_{col}_equal_w"] = weighted_avg(
                g, col, "branch_equal_weight"
            )

    exposure_rows.append(row)

branch_exposure = pd.DataFrame(exposure_rows)

print("\nBranch exposure shape:", branch_exposure.shape)

# --------------------------------------------------
# 10. Headquarters exposure
# --------------------------------------------------

hq = (
    base.sort_values(["CERT", "DEPSUMBR"], ascending=[True, False])
    .groupby("CERT", dropna=False)
    .agg(
        NAMEFULL=("NAMEFULL", "first"),
        hq_county_fips=("STCNTY", "first"),
        hq_state=("STALP", "first"),
        bank_assets_2019=("ASSET", "max"),
        bank_deposits_2019=("DEPSUM", "max"),
    )
    .reset_index()
)

hq_expanded = pd.concat(
    [hq.assign(YEAR=y) for y in covid_years],
    ignore_index=True
)

# HQ health shock
hq_health = county_covid[health_cols].rename(
    columns={
        "county_fips": "hq_county_fips",
        "avg_cases_per_100k": "hq_avg_cases_per_100k",
        "max_cases_per_100k": "hq_max_cases_per_100k",
        "avg_deaths_per_100k": "hq_avg_deaths_per_100k",
        "max_deaths_per_100k": "hq_max_deaths_per_100k",
        "z_avg_cases_per_100k": "hq_z_avg_cases_per_100k",
        "z_max_cases_per_100k": "hq_z_max_cases_per_100k",
        "z_avg_deaths_per_100k": "hq_z_avg_deaths_per_100k",
        "z_max_deaths_per_100k": "hq_z_max_deaths_per_100k",
        "covid_health_shock": "hq_covid_health_shock",
        "covid_health_shock_peak": "hq_covid_health_shock_peak",
        "covid_case_shock": "hq_covid_case_shock",
    }
)

hq_expanded = hq_expanded.merge(
    hq_health,
    on=["hq_county_fips", "YEAR"],
    how="left"
)

# HQ policy shock
state_policy_hq = state_policy.rename(columns={"STALPBR": "hq_state"}).copy()

rename_policy = {}

for col in state_policy_hq.columns:
    if col not in ["hq_state", "YEAR"]:
        rename_policy[col] = f"hq_{col}"

state_policy_hq = state_policy_hq.rename(columns=rename_policy)

hq_expanded = hq_expanded.merge(
    state_policy_hq,
    on=["hq_state", "YEAR"],
    how="left"
)

# --------------------------------------------------
# 11. Combine branch and HQ exposure
# --------------------------------------------------

bank_covid_exposure = branch_exposure.merge(
    hq_expanded,
    on=["CERT", "YEAR"],
    how="left"
)

# Main variables
bank_covid_exposure["covid_health_exposure_main"] = (
    bank_covid_exposure["branch_exp_covid_health_shock_deposit_w"]
)

bank_covid_exposure["covid_policy_exposure_main"] = (
    bank_covid_exposure["branch_exp_covid_policy_shock_deposit_w"]
)

bank_covid_exposure["covid_case_exposure_main"] = (
    bank_covid_exposure["branch_exp_covid_case_shock_deposit_w"]
)

bank_covid_exposure["covid_overall_exposure_main"] = (
    bank_covid_exposure["covid_health_exposure_main"]
    + bank_covid_exposure["covid_policy_exposure_main"]
)

# Branch + HQ combined version
alpha_branch = 0.8
alpha_hq = 0.2

bank_covid_exposure["covid_health_exposure_branch_hq"] = (
    alpha_branch * bank_covid_exposure["branch_exp_covid_health_shock_deposit_w"]
    + alpha_hq * bank_covid_exposure["hq_covid_health_shock"]
)

bank_covid_exposure["covid_policy_exposure_branch_hq"] = (
    alpha_branch * bank_covid_exposure["branch_exp_covid_policy_shock_deposit_w"]
    + alpha_hq * bank_covid_exposure["hq_covid_policy_shock"]
)

# --------------------------------------------------
# 12. Save
# --------------------------------------------------

out_csv = folder / "bank_covid_exposure_annual_2020_2022.csv"
out_parquet = folder / "bank_covid_exposure_annual_2020_2022.parquet"

bank_covid_exposure.to_csv(out_csv, index=False, encoding="utf-8-sig")
bank_covid_exposure.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nBank COVID exposure shape:", bank_covid_exposure.shape)

print("\nExposure coverage by year:")

for col in [
    "covid_health_exposure_main",
    "covid_policy_exposure_main",
    "covid_case_exposure_main",
    "covid_overall_exposure_main",
    "covid_health_exposure_branch_hq",
    "covid_policy_exposure_branch_hq",
]:
    print("\n", col)
    print(bank_covid_exposure.groupby("YEAR")[col].apply(lambda x: x.notna().mean()))

print("\nPreview:")
print(bank_covid_exposure.head())