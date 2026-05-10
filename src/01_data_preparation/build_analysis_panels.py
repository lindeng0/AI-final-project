# build_analysis_panels.py

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 0. Paths
# --------------------------------------------------

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
input_file = folder / "sod_branch_panel_2018_2025.parquet"

print("Reading branch panel...")
panel = pd.read_parquet(input_file)

print("Raw branch panel shape:", panel.shape)
print("Years:", sorted(panel["YEAR"].dropna().unique()))

# --------------------------------------------------
# 1. Basic type cleaning
# --------------------------------------------------

id_cols = ["CERT", "UNINUMBR", "BRNUM", "RSSDID", "RSSDHCR", "STCNTYBR", "MSABR"]

for col in id_cols:
    if col in panel.columns:
        panel[col] = panel[col].astype("string").str.strip()

panel["YEAR"] = pd.to_numeric(panel["YEAR"], errors="coerce").astype("Int64")

numeric_cols = [
    "ASSET", "DEPDOM", "DEPSUM", "DEPSUMBR",
    "SIMS_LATITUDE", "SIMS_LONGITUDE",
    "METROBR", "MICROBR"
]

for col in numeric_cols:
    if col in panel.columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")


def safe_log(series):
    """
    Return natural log only for positive values.
    Non-positive or missing values become NaN.
    This avoids RuntimeWarning: divide by zero encountered in log.
    """
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = series > 0
    out.loc[mask] = np.log(series.loc[mask])
    return out


# --------------------------------------------------
# 2. Bank-year panel
# --------------------------------------------------

print("\nBuilding bank-year panel...")

# Use YEAR + CERT as the true bank-year key.
# Do not group by NAMEFULL, because the same CERT may occasionally have multiple names.
bank_panel = (
    panel.groupby(["YEAR", "CERT"], dropna=False)
    .agg(
        NAMEFULL=("NAMEFULL", lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) > 0 else x.dropna().iloc[0] if x.dropna().size > 0 else np.nan),
        n_branches=("UNINUMBR", "nunique"),
        total_branch_deposits=("DEPSUMBR", "sum"),
        bank_assets=("ASSET", "max"),
        bank_deposits=("DEPSUM", "max"),
        n_states=("STALPBR", "nunique"),
        n_counties=("STCNTYBR", "nunique"),
        n_msas=("MSABR", "nunique"),
    )
    .reset_index()
)

bank_panel = bank_panel.sort_values(["CERT", "YEAR"])

bank_panel["branch_lag"] = bank_panel.groupby("CERT")["n_branches"].shift(1)
bank_panel["branch_growth"] = np.where(
    bank_panel["branch_lag"] > 0,
    (bank_panel["n_branches"] - bank_panel["branch_lag"]) / bank_panel["branch_lag"],
    np.nan
)

bank_panel["deposit_lag"] = bank_panel.groupby("CERT")["total_branch_deposits"].shift(1)
bank_panel["deposit_growth"] = np.where(
    bank_panel["deposit_lag"] > 0,
    (bank_panel["total_branch_deposits"] - bank_panel["deposit_lag"]) / bank_panel["deposit_lag"],
    np.nan
)

bank_panel["log_n_branches"] = np.log1p(bank_panel["n_branches"])
bank_panel["log_total_branch_deposits"] = np.log1p(bank_panel["total_branch_deposits"])
bank_panel["log_bank_assets"] = safe_log(bank_panel["bank_assets"])
bank_panel["log_bank_deposits"] = safe_log(bank_panel["bank_deposits"])

bank_csv = folder / "sod_bank_panel_2018_2025.csv"
bank_parquet = folder / "sod_bank_panel_2018_2025.parquet"

bank_panel.to_csv(bank_csv, index=False, encoding="utf-8-sig")
bank_panel.to_parquet(bank_parquet, index=False)

print("Saved bank panel:")
print(" -", bank_csv)
print(" -", bank_parquet)
print("Bank panel shape:", bank_panel.shape)


# --------------------------------------------------
# 3. County-year market panel
# --------------------------------------------------

print("\nBuilding county-year market panel...")

market = (
    panel.groupby(["YEAR", "STCNTYBR", "CNTYNAMB", "STALPBR"], dropna=False)
    .agg(
        market_deposits=("DEPSUMBR", "sum"),
        n_branches=("UNINUMBR", "nunique"),
        n_banks=("CERT", "nunique"),
    )
    .reset_index()
)

bank_market = (
    panel.groupby(["YEAR", "STCNTYBR", "CERT"], dropna=False)
    .agg(
        bank_market_deposits=("DEPSUMBR", "sum")
    )
    .reset_index()
)

bank_market = bank_market.merge(
    market[["YEAR", "STCNTYBR", "market_deposits"]],
    on=["YEAR", "STCNTYBR"],
    how="left"
)

bank_market["market_share"] = np.nan
valid_market = bank_market["market_deposits"] > 0
bank_market.loc[valid_market, "market_share"] = (
    bank_market.loc[valid_market, "bank_market_deposits"] /
    bank_market.loc[valid_market, "market_deposits"]
)

hhi = (
    bank_market.groupby(["YEAR", "STCNTYBR"], dropna=False)
    .agg(
        HHI=(
            "market_share",
            lambda x: np.sum(x.dropna() ** 2) if x.notna().any() else np.nan
        ),
        max_market_share=("market_share", "max")
    )
    .reset_index()
)

market = market.merge(hhi, on=["YEAR", "STCNTYBR"], how="left")

market["log_market_deposits"] = safe_log(market["market_deposits"])
market["log_n_branches"] = np.log1p(market["n_branches"])
market["log_n_banks"] = np.log1p(market["n_banks"])

market["competition_index"] = np.where(
    market["HHI"].notna(),
    1 - market["HHI"],
    np.nan
)

market_csv = folder / "sod_county_market_panel_2018_2025.csv"
market_parquet = folder / "sod_county_market_panel_2018_2025.parquet"

market.to_csv(market_csv, index=False, encoding="utf-8-sig")
market.to_parquet(market_parquet, index=False)

print("Saved county market panel:")
print(" -", market_csv)
print(" -", market_parquet)
print("County market panel shape:", market.shape)


# --------------------------------------------------
# 4. Branch-year change panel
# --------------------------------------------------

print("\nBuilding branch-year change panel...")

branch = panel.sort_values(["UNINUMBR", "YEAR"]).copy()

branch["deposit_lag"] = branch.groupby("UNINUMBR")["DEPSUMBR"].shift(1)

branch["deposit_growth"] = np.where(
    branch["deposit_lag"] > 0,
    (branch["DEPSUMBR"] - branch["deposit_lag"]) / branch["deposit_lag"],
    np.nan
)

branch["log_branch_deposits"] = safe_log(branch["DEPSUMBR"])

# Need fillna(False), otherwise first observation for each branch becomes NA.
branch["observed_prev_year"] = (
    branch.groupby("UNINUMBR")["YEAR"].diff().eq(1).fillna(False)
)

branch["observed_next_year"] = (
    branch.groupby("UNINUMBR")["YEAR"].diff(-1).eq(-1).fillna(False)
)

branch["entry_in_sample"] = ~branch["observed_prev_year"]
branch["exit_in_sample"] = ~branch["observed_next_year"]

# Left-censoring and right-censoring adjustment:
# first sample year cannot be treated as true entry;
# last sample year cannot be treated as true exit.
first_year = branch["YEAR"].min()
last_year = branch["YEAR"].max()

branch.loc[branch["YEAR"] == first_year, "entry_in_sample"] = False
branch.loc[branch["YEAR"] == last_year, "exit_in_sample"] = False

branch_csv = folder / "sod_branch_change_panel_2018_2025.csv"
branch_parquet = folder / "sod_branch_change_panel_2018_2025.parquet"

branch.to_csv(branch_csv, index=False, encoding="utf-8-sig")
branch.to_parquet(branch_parquet, index=False)

print("Saved branch change panel:")
print(" -", branch_csv)
print(" -", branch_parquet)
print("Branch change panel shape:", branch.shape)

print("\nDone.")