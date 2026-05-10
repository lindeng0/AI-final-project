# summarize_each_bank_stock_covid.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

# Prefer parquet because it preserves data types better
input_file = folder / "analysis_stock_covid_performance_panel_2020_2022.parquet"

if input_file.exists():
    df = pd.read_parquet(input_file)
else:
    input_file = folder / "analysis_stock_covid_performance_panel_2020_2022.csv"
    df = pd.read_csv(input_file, low_memory=False)

print("Input file:", input_file)
print("Input shape:", df.shape)

# --------------------------------------------------
# 1. Standardize key variables
# --------------------------------------------------

df["GVKEY"] = df["GVKEY"].astype("string").str.strip()
df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
df["fqtr"] = pd.to_numeric(df["fqtr"], errors="coerce").astype("Int64")

# There may be both tic and TIC columns
if "tic" in df.columns:
    df["ticker"] = df["tic"].astype("string").str.strip().str.upper()
elif "TIC" in df.columns:
    df["ticker"] = df["TIC"].astype("string").str.strip().str.upper()
else:
    df["ticker"] = pd.NA

# There may be CONM / NAMEHCR / FRB_NAME
if "CONM" in df.columns:
    df["company_name"] = df["CONM"]
elif "NAMEHCR" in df.columns:
    df["company_name"] = df["NAMEHCR"]
elif "FRB_NAME" in df.columns:
    df["company_name"] = df["FRB_NAME"]
else:
    df["company_name"] = pd.NA

# Standardize RSSDHCR if present
if "RSSDHCR" in df.columns:
    df["RSSDHCR"] = df["RSSDHCR"].astype("string").str.strip()

# Numeric variables
numeric_cols = [
    "atq", "niq", "prccq", "cshoq",
    "mktcap_q", "roa_q",
    "price_return_q", "price_return_q_w",
    "mktcap_growth_q", "mktcap_growth_q_w",
    "asset_growth_q", "asset_growth_q_w",
    "niq_growth_q", "niq_growth_q_w",
    "roa_change_q", "roa_change_q_w",
    "log_assets_q", "log_mktcap_q",
    "covid_health_exposure_main_hcr",
    "covid_case_exposure_main_hcr",
    "covid_policy_exposure_main_hcr",
    "covid_overall_exposure_main_hcr",
    "n_base_branches_hcr",
    "n_base_bank_subsidiaries",
    "base_total_branch_deposits_hcr",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------
# 2. Create bank-quarter detail table
# --------------------------------------------------

detail_cols = [
    "GVKEY", "ticker", "company_name", "RSSDHCR",
    "datadate", "YEAR", "fqtr",
    "covid_health_exposure_main_hcr",
    "covid_case_exposure_main_hcr",
    "covid_policy_exposure_main_hcr",
    "covid_overall_exposure_main_hcr",
    "roa_q", "roa_q_w",
    "price_return_q", "price_return_q_w",
    "mktcap_growth_q", "mktcap_growth_q_w",
    "asset_growth_q", "asset_growth_q_w",
    "roa_change_q", "roa_change_q_w",
    "atq", "niq", "prccq", "cshoq", "mktcap_q",
    "log_assets_q", "log_mktcap_q",
    "n_base_branches_hcr",
    "n_base_bank_subsidiaries",
    "base_total_branch_deposits_hcr",
    "active_both_links",
]

detail_cols = [c for c in detail_cols if c in df.columns]

bank_quarter_detail = (
    df[detail_cols]
    .sort_values(["GVKEY", "datadate"])
    .copy()
)

# --------------------------------------------------
# 3. Create bank-level overview table
# --------------------------------------------------

def first_nonmissing(x):
    x = x.dropna()
    return x.iloc[0] if len(x) > 0 else np.nan

agg_dict = {
    "ticker": first_nonmissing,
    "company_name": first_nonmissing,
    "RSSDHCR": first_nonmissing,
    "datadate": ["min", "max", "count"],
    "YEAR": ["min", "max", "nunique"],
    "fqtr": "count",
}

optional_aggs = {
    # COVID exposure
    "covid_health_exposure_main_hcr": ["mean", "min", "max"],
    "covid_case_exposure_main_hcr": ["mean"],
    "covid_policy_exposure_main_hcr": ["mean"],
    "covid_overall_exposure_main_hcr": ["mean"],

    # Performance
    "roa_q": ["mean", "min", "max"],
    "roa_q_w": ["mean"],
    "price_return_q": ["mean", "min", "max"],
    "price_return_q_w": ["mean"],
    "mktcap_growth_q": ["mean"],
    "mktcap_growth_q_w": ["mean"],
    "asset_growth_q": ["mean"],
    "asset_growth_q_w": ["mean"],
    "roa_change_q": ["mean"],
    "roa_change_q_w": ["mean"],

    # Scale
    "atq": ["mean", "max"],
    "mktcap_q": ["mean", "max"],
    "log_assets_q": ["mean"],
    "log_mktcap_q": ["mean"],

    # 2019 branch network base
    "n_base_branches_hcr": ["mean", "max"],
    "n_base_bank_subsidiaries": ["mean", "max"],
    "base_total_branch_deposits_hcr": ["mean", "max"],

    # Link quality
    "active_both_links": "mean",
}

for col, funcs in optional_aggs.items():
    if col in df.columns:
        agg_dict[col] = funcs

bank_overview = df.groupby("GVKEY", dropna=False).agg(agg_dict)

# Flatten MultiIndex columns
bank_overview.columns = [
    "_".join([str(x) for x in col if x != ""]).strip("_")
    for col in bank_overview.columns.to_flat_index()
]

bank_overview = bank_overview.reset_index()

# Rename common columns
rename_map = {
    "ticker_first_nonmissing": "ticker",
    "company_name_first_nonmissing": "company_name",
    "RSSDHCR_first_nonmissing": "RSSDHCR",
    "datadate_min": "first_datadate",
    "datadate_max": "last_datadate",
    "datadate_count": "n_quarters",
    "YEAR_min": "first_year",
    "YEAR_max": "last_year",
    "YEAR_nunique": "n_years",
    "fqtr_count": "n_firm_quarters",
    "active_both_links_mean": "share_active_both_links",
}

bank_overview = bank_overview.rename(columns=rename_map)

# Easier variable names
rename_more = {
    "covid_health_exposure_main_hcr_mean": "avg_covid_health_exposure",
    "covid_health_exposure_main_hcr_min": "min_covid_health_exposure",
    "covid_health_exposure_main_hcr_max": "max_covid_health_exposure",
    "covid_case_exposure_main_hcr_mean": "avg_covid_case_exposure",
    "covid_policy_exposure_main_hcr_mean": "avg_covid_policy_exposure",
    "covid_overall_exposure_main_hcr_mean": "avg_covid_overall_exposure",
    "roa_q_mean": "avg_roa",
    "roa_q_min": "min_roa",
    "roa_q_max": "max_roa",
    "roa_q_w_mean": "avg_roa_w",
    "price_return_q_mean": "avg_price_return",
    "price_return_q_min": "min_price_return",
    "price_return_q_max": "max_price_return",
    "price_return_q_w_mean": "avg_price_return_w",
    "mktcap_growth_q_mean": "avg_mktcap_growth",
    "mktcap_growth_q_w_mean": "avg_mktcap_growth_w",
    "asset_growth_q_mean": "avg_asset_growth",
    "asset_growth_q_w_mean": "avg_asset_growth_w",
    "roa_change_q_mean": "avg_roa_change",
    "roa_change_q_w_mean": "avg_roa_change_w",
    "atq_mean": "avg_assets",
    "atq_max": "max_assets",
    "mktcap_q_mean": "avg_mktcap",
    "mktcap_q_max": "max_mktcap",
    "log_assets_q_mean": "avg_log_assets",
    "log_mktcap_q_mean": "avg_log_mktcap",
    "n_base_branches_hcr_mean": "base_branches_2019",
    "n_base_branches_hcr_max": "base_branches_2019_max",
    "n_base_bank_subsidiaries_mean": "base_bank_subsidiaries_2019",
    "base_total_branch_deposits_hcr_mean": "base_branch_deposits_2019",
}

bank_overview = bank_overview.rename(columns=rename_more)

# --------------------------------------------------
# 4. Add rankings
# --------------------------------------------------

if "avg_covid_health_exposure" in bank_overview.columns:
    bank_overview["rank_covid_health_high"] = (
        bank_overview["avg_covid_health_exposure"]
        .rank(ascending=False, method="min")
    )

if "avg_roa_w" in bank_overview.columns:
    bank_overview["rank_roa_high"] = (
        bank_overview["avg_roa_w"]
        .rank(ascending=False, method="min")
    )

if "avg_price_return_w" in bank_overview.columns:
    bank_overview["rank_price_return_high"] = (
        bank_overview["avg_price_return_w"]
        .rank(ascending=False, method="min")
    )

# Sort for readability
sort_cols = []
if "avg_covid_health_exposure" in bank_overview.columns:
    sort_cols.append("avg_covid_health_exposure")

if sort_cols:
    bank_overview = bank_overview.sort_values(sort_cols, ascending=False)
else:
    bank_overview = bank_overview.sort_values("GVKEY")

# --------------------------------------------------
# 5. Print summary
# --------------------------------------------------

print("\nBank-quarter detail shape:", bank_quarter_detail.shape)
print("Bank overview shape:", bank_overview.shape)

print("\nUnique banks:", bank_overview["GVKEY"].nunique())

print("\nTop 20 banks by COVID health exposure:")
display_cols = [
    "GVKEY", "ticker", "company_name", "RSSDHCR",
    "n_quarters",
    "avg_covid_health_exposure",
    "avg_covid_policy_exposure",
    "avg_roa_w",
    "avg_price_return_w",
    "avg_asset_growth_w",
    "avg_log_assets",
    "base_branches_2019",
    "share_active_both_links",
]

display_cols = [c for c in display_cols if c in bank_overview.columns]
print(bank_overview[display_cols].head(20))

print("\nLowest 20 banks by COVID health exposure:")
print(bank_overview[display_cols].tail(20))

# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------

out_overview_csv = folder / "bank_overview_stock_covid_performance_2020_2022.csv"
out_detail_csv = folder / "bank_quarter_detail_stock_covid_performance_2020_2022.csv"
out_excel = folder / "bank_overview_stock_covid_performance_2020_2022.xlsx"

bank_overview.to_csv(out_overview_csv, index=False, encoding="utf-8-sig")
bank_quarter_detail.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    bank_overview.to_excel(writer, sheet_name="bank_overview", index=False)
    bank_quarter_detail.to_excel(writer, sheet_name="quarter_detail", index=False)

    # Additional useful sheets
    if "avg_covid_health_exposure" in bank_overview.columns:
        bank_overview.head(50).to_excel(writer, sheet_name="top_covid_exposure", index=False)
        bank_overview.tail(50).to_excel(writer, sheet_name="low_covid_exposure", index=False)

    if "avg_roa_w" in bank_overview.columns:
        bank_overview.sort_values("avg_roa_w", ascending=False).head(50).to_excel(
            writer, sheet_name="top_roa", index=False
        )

    if "avg_price_return_w" in bank_overview.columns:
        bank_overview.sort_values("avg_price_return_w", ascending=False).head(50).to_excel(
            writer, sheet_name="top_price_return", index=False
        )

print("\nSaved:")
print(" -", out_overview_csv)
print(" -", out_detail_csv)
print(" -", out_excel)