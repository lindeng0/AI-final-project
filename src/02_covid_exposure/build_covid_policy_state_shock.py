# build_covid_policy_state_shock.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

input_file = external / "OxCGRT_compact_subnational_v1.csv"

policy = pd.read_csv(input_file, low_memory=False)

print("Raw OxCGRT shape:", policy.shape)
print("Columns:", list(policy.columns))

# --------------------------------------------------
# 1. Keep U.S. records
# --------------------------------------------------

policy = policy[policy["CountryCode"] == "USA"].copy()

print("\nUSA-only shape:", policy.shape)
print("\nJurisdiction counts:")
print(policy["Jurisdiction"].value_counts(dropna=False).head(20))

# --------------------------------------------------
# 2. Parse date
# --------------------------------------------------

policy["Date"] = pd.to_datetime(
    policy["Date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

policy["YEAR"] = policy["Date"].dt.year

# --------------------------------------------------
# 3. Extract state code
# --------------------------------------------------

policy["RegionCode"] = policy["RegionCode"].astype("string")

policy = policy[
    policy["RegionCode"].notna()
    & policy["RegionCode"].str.startswith("US_")
].copy()

policy["STALPBR"] = policy["RegionCode"].str[-2:].str.upper()

policy = policy[policy["YEAR"].between(2020, 2022)].copy()

# --------------------------------------------------
# 4. Policy index variables
# --------------------------------------------------

policy_cols = [
    "StringencyIndex_Average",
    "GovernmentResponseIndex_Average",
    "ContainmentHealthIndex_Average",
    "EconomicSupportIndex",
]

for col in policy_cols:
    policy[col] = pd.to_numeric(policy[col], errors="coerce")

# --------------------------------------------------
# 5. Annual state-level policy shock
# --------------------------------------------------

annual_policy = (
    policy.groupby(["STALPBR", "YEAR"], dropna=False)
    .agg(
        policy_stringency=("StringencyIndex_Average", "mean"),
        policy_government_response=("GovernmentResponseIndex_Average", "mean"),
        policy_containment_health=("ContainmentHealthIndex_Average", "mean"),
        policy_economic_support=("EconomicSupportIndex", "mean"),
        n_policy_days=("Date", "nunique"),
    )
    .reset_index()
)

# --------------------------------------------------
# 6. Z-score within year
# --------------------------------------------------

def zscore_by_year(df, col):
    return df.groupby("YEAR")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan
    )

annual_policy["z_policy_stringency"] = zscore_by_year(
    annual_policy, "policy_stringency"
)

annual_policy["z_policy_government_response"] = zscore_by_year(
    annual_policy, "policy_government_response"
)

annual_policy["z_policy_containment_health"] = zscore_by_year(
    annual_policy, "policy_containment_health"
)

annual_policy["z_policy_economic_support"] = zscore_by_year(
    annual_policy, "policy_economic_support"
)

# Main policy shock
annual_policy["covid_policy_shock"] = annual_policy["z_policy_stringency"]

# Alternative policy shocks
annual_policy["covid_policy_shock_containment"] = (
    annual_policy["z_policy_containment_health"]
)

annual_policy["covid_policy_shock_government_response"] = (
    annual_policy["z_policy_government_response"]
)

# --------------------------------------------------
# 7. Save
# --------------------------------------------------

out_csv = external / "state_covid_policy_shock_annual.csv"
out_parquet = external / "state_covid_policy_shock_annual.parquet"

annual_policy.to_csv(out_csv, index=False, encoding="utf-8-sig")
annual_policy.to_parquet(out_parquet, index=False)

print("\nSaved:")
print(" -", out_csv)
print(" -", out_parquet)

print("\nAnnual state policy shock shape:", annual_policy.shape)

print("\nState counts by year:")
print(annual_policy.groupby("YEAR")["STALPBR"].nunique())

print("\nPolicy shock summary:")
print(
    annual_policy.groupby("YEAR")[
        ["policy_stringency", "policy_containment_health", "covid_policy_shock"]
    ].describe()
)