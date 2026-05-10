# ============================================================
# build_final_2018_2024_panel_v2.py
# Purpose:
#   Build final 2018–2024 analysis panel:
#   - Traditional + AI full-document index, 2018–2024
#   - Stock/accounting performance, 2018–2024
#   - COVID exposure only where available, mainly 2020–2022
#
# Key principle:
#   Do NOT restrict sample to 2020–2022.
#   Keep COVID variables missing outside 2020–2022.
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COVID_COL_PATTERNS = [
    "covid",
    "hcr_exp",
    "cases_per_100k",
    "deaths_per_100k",
    "policy_shock",
    "health_shock",
    "case_shock",
]


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_gvkey(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = x.replace(".0", "")
    return x


def clean_cik(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = x.replace(".0", "")
    x = x.lstrip("0")
    return x


def clean_tic(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def compound_return(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan

    # If return appears in percent, e.g. 5 instead of 0.05, convert.
    if x.abs().median() > 2:
        x = x / 100.0

    return float((1.0 + x).prod() - 1.0)


def first_nonmissing(x):
    x = x.replace("", np.nan)
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return x.iloc[0]


def last_nonmissing(x):
    x = x.replace("", np.nan)
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return x.iloc[-1]


def identify_columns(df):
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}

    def get(name_list):
        for n in name_list:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    return {
        "gvkey": get(["GVKEY", "gvkey"]),
        "cik": get(["CIK", "cik"]),
        "tic": get(["TIC", "tic", "ticker"]),
        "year": get(["YEAR", "year", "fyear", "fyearq"]),
        "datadate": get(["datadate", "DATADATE"]),
    }


def prepare_stock_quarterly(stock_csv):
    stock = pd.read_csv(stock_csv, dtype=str).fillna("")

    ids = identify_columns(stock)

    if ids["gvkey"] is None:
        raise ValueError("Stock/performance file must contain GVKEY.")

    stock["GVKEY_clean"] = stock[ids["gvkey"]].apply(clean_gvkey)

    if ids["cik"]:
        stock["CIK_clean"] = stock[ids["cik"]].apply(clean_cik)
    else:
        stock["CIK_clean"] = ""

    if ids["tic"]:
        stock["tic_clean"] = stock[ids["tic"]].apply(clean_tic)
    else:
        stock["tic_clean"] = ""

    # Year handling
    if ids["year"]:
        stock["YEAR_key"] = safe_num(stock[ids["year"]]).astype("Int64")
    elif ids["datadate"]:
        stock["YEAR_key"] = pd.to_datetime(stock[ids["datadate"]], errors="coerce").dt.year.astype("Int64")
    else:
        raise ValueError("Stock/performance file must contain YEAR or datadate.")

    if ids["datadate"]:
        stock["datadate_dt"] = pd.to_datetime(stock[ids["datadate"]], errors="coerce")
    else:
        stock["datadate_dt"] = pd.NaT

    # Convert numeric candidate columns
    for c in stock.columns:
        cl = str(c).lower()
        if any(k in cl for k in [
            "roa", "roe", "return", "growth", "atq", "niq", "log_assets",
            "asset", "covid", "hcr_exp", "cases", "deaths", "policy",
            "shock", "deposit"
        ]):
            stock[c] = safe_num(stock[c])

    return stock


def annualize_stock_performance(stock):
    stock = stock.copy()
    stock = stock[(stock["YEAR_key"] >= 2018) & (stock["YEAR_key"] <= 2024)].copy()

    # Sort within firm-year
    if "datadate_dt" in stock.columns:
        stock = stock.sort_values(["GVKEY_clean", "YEAR_key", "datadate_dt"])
    else:
        stock = stock.sort_values(["GVKEY_clean", "YEAR_key"])

    group_cols = ["GVKEY_clean", "YEAR_key"]

    # Candidate performance and control columns
    mean_cols = [
        "roa_q", "roa_q_w",
        "ROA", "roa",
        "roe_q", "roe_q_w",
        "asset_growth_q", "asset_growth_q_w",
        "niq_growth_q", "niq_growth_q_w",
        "roa_change_q", "roa_change_q_w",
        "log_assets_q",
        "atq", "niq",
        "base_total_branch_deposits_hcr",
    ]

    return_cols = [
        "price_return_q",
        "price_return_q_w",
        "stock_return_q",
        "stock_return_q_w",
        "ret_q",
        "ret_q_w",
    ]

    # COVID variables: keep as annual mean if available, but later force missing outside 2020–2022
    covid_cols = []
    for c in stock.columns:
        cl = str(c).lower()
        if any(p in cl for p in COVID_COL_PATTERNS):
            if c not in group_cols and c not in ["datadate_dt"]:
                covid_cols.append(c)

    rows = []

    for (gvkey, year), g in stock.groupby(group_cols, dropna=False):
        row = {
            "GVKEY_clean": gvkey,
            "YEAR_key": int(year),
            "n_quarters_stock": len(g),
            "stock_first_datadate": first_nonmissing(g["datadate_dt"].astype(str)) if "datadate_dt" in g.columns else "",
            "stock_last_datadate": last_nonmissing(g["datadate_dt"].astype(str)) if "datadate_dt" in g.columns else "",
            "stock_tic": first_nonmissing(g["tic_clean"]) if "tic_clean" in g.columns else "",
            "stock_cik": first_nonmissing(g["CIK_clean"]) if "CIK_clean" in g.columns else "",
        }

        # Annual average for accounting/performance/control variables
        for c in mean_cols:
            if c in g.columns:
                row[f"annual_mean_{c}"] = safe_num(g[c]).mean(skipna=True)
                row[f"annual_last_{c}"] = safe_num(g[c]).dropna().iloc[-1] if safe_num(g[c]).notna().sum() > 0 else np.nan

        # Compound annual stock return if return columns exist
        for c in return_cols:
            if c in g.columns:
                row[f"annual_compound_{c}"] = compound_return(g[c])
                row[f"annual_mean_{c}"] = safe_num(g[c]).mean(skipna=True)

        # COVID exposures: annual mean, but only meaningful where data exist
        for c in covid_cols:
            if c in g.columns:
                row[f"annual_mean_{c}"] = safe_num(g[c]).mean(skipna=True)

        rows.append(row)

    annual = pd.DataFrame(rows)

    # Explicit COVID availability flags
    annual["COVID_data_period_2020_2022"] = annual["YEAR_key"].isin([2020, 2021, 2022]).astype(int)
    annual["Pre_COVID_2018_2019"] = annual["YEAR_key"].isin([2018, 2019]).astype(int)
    annual["Post_COVID_2023_2024"] = annual["YEAR_key"].isin([2023, 2024]).astype(int)
    annual["Post_2020"] = (annual["YEAR_key"] >= 2020).astype(int)

    # Force COVID variables to missing outside 2020–2022
    # This follows user's requested design.
    covid_annual_cols = [
        c for c in annual.columns
        if any(p in c.lower() for p in COVID_COL_PATTERNS)
    ]

    outside_covid_period = ~annual["YEAR_key"].isin([2020, 2021, 2022])
    for c in covid_annual_cols:
        annual.loc[outside_covid_period, c] = np.nan

    return annual


def merge_base_with_annual_performance(base_csv, annual_perf, output_csv, audit_csv):
    base = pd.read_csv(base_csv, dtype=str).fillna("")

    # Standardize base keys
    if "GVKEY" not in base.columns:
        raise ValueError("Base file must contain GVKEY.")

    base["GVKEY_clean"] = base["GVKEY"].apply(clean_gvkey)

    if "YEAR_num" in base.columns:
        base["YEAR_key"] = safe_num(base["YEAR_num"]).astype("Int64")
    elif "YEAR_final" in base.columns:
        base["YEAR_key"] = safe_num(base["YEAR_final"]).astype("Int64")
    elif "YEAR" in base.columns:
        base["YEAR_key"] = safe_num(base["YEAR"]).astype("Int64")
    else:
        raise ValueError("Base file must contain YEAR / YEAR_num / YEAR_final.")

    base = base[(base["YEAR_key"] >= 2018) & (base["YEAR_key"] <= 2024)].copy()

    merged = base.merge(
        annual_perf,
        on=["GVKEY_clean", "YEAR_key"],
        how="left",
        indicator="performance_merge_status",
    )

    # Core flags
    merged["has_performance_data"] = merged["performance_merge_status"].eq("both")
    merged["has_ai"] = pd.to_numeric(
        merged.get("AI_full_document_index_0_100_v3", np.nan),
        errors="coerce"
    ).notna()
    merged["has_traditional"] = pd.to_numeric(
        merged.get("Traditional_index_0_100", np.nan),
        errors="coerce"
    ).notna()

    # Prefer clean names for common outcomes if available
    rename_candidates = {
        "annual_mean_roa_q_w": "ROA_annual_mean_w",
        "annual_mean_roa_q": "ROA_annual_mean",
        "annual_last_roa_q_w": "ROA_annual_last_w",
        "annual_last_roa_q": "ROA_annual_last",
        "annual_compound_price_return_q_w": "StockReturn_annual_compound_w",
        "annual_compound_price_return_q": "StockReturn_annual_compound",
        "annual_mean_price_return_q_w": "StockReturn_annual_mean_w",
        "annual_mean_price_return_q": "StockReturn_annual_mean",
        "annual_mean_asset_growth_q_w": "AssetGrowth_annual_mean_w",
        "annual_mean_asset_growth_q": "AssetGrowth_annual_mean",
        "annual_mean_roa_change_q_w": "ROAChange_annual_mean_w",
        "annual_mean_roa_change_q": "ROAChange_annual_mean",
        "annual_mean_log_assets_q": "LogAssets_annual_mean",
        "annual_last_atq": "Assets_atq_annual_last",
    }

    for old, new in rename_candidates.items():
        if old in merged.columns and new not in merged.columns:
            merged[new] = merged[old]

    # Z-score core variables
    z_cols = [
        "Traditional_index_0_100",
        "AI_full_document_index_0_100_v3",
        "ROA_annual_mean_w",
        "ROA_annual_mean",
        "StockReturn_annual_compound_w",
        "StockReturn_annual_compound",
        "AssetGrowth_annual_mean_w",
        "ROAChange_annual_mean_w",
        "LogAssets_annual_mean",
    ]

    for c in z_cols:
        if c in merged.columns:
            x = pd.to_numeric(merged[c], errors="coerce")
            std = x.std(skipna=True)
            mean = x.mean(skipna=True)
            if pd.notna(std) and std > 0:
                merged[f"z_{c}"] = (x - mean) / std

    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Audit
    audit = {
        "base_rows_2018_2024": len(base),
        "annual_perf_rows_2018_2024": len(annual_perf),
        "merged_rows": len(merged),
        "matched_performance_rows": int(merged["has_performance_data"].sum()),
        "missing_performance_rows": int((~merged["has_performance_data"]).sum()),
        "performance_coverage_rate": float(merged["has_performance_data"].mean()),
        "has_ai_rows": int(merged["has_ai"].sum()),
        "has_traditional_rows": int(merged["has_traditional"].sum()),
    }

    for c in [
        "ROA_annual_mean_w",
        "ROA_annual_mean",
        "StockReturn_annual_compound_w",
        "StockReturn_annual_compound",
        "AssetGrowth_annual_mean_w",
        "ROAChange_annual_mean_w",
        "LogAssets_annual_mean",
    ]:
        if c in merged.columns:
            audit[f"n_nonmissing_{c}"] = int(pd.to_numeric(merged[c], errors="coerce").notna().sum())

    pd.DataFrame([audit]).to_csv(audit_csv, index=False, encoding="utf-8-sig")

    by_year = (
        merged.groupby("YEAR_key")
        .agg(
            n=("YEAR_key", "size"),
            n_perf=("has_performance_data", "sum"),
            n_ai=("has_ai", "sum"),
            n_traditional=("has_traditional", "sum"),
            ai_mean=("AI_full_document_index_0_100_v3", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            traditional_mean=("Traditional_index_0_100", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
    )

    # Add outcome yearly means if present
    for c in ["ROA_annual_mean_w", "StockReturn_annual_compound_w", "AssetGrowth_annual_mean_w"]:
        if c in merged.columns:
            tmp = merged.groupby("YEAR_key")[c].mean().reset_index(name=f"{c}_mean")
            by_year = by_year.merge(tmp, on="YEAR_key", how="left")

    by_year_path = Path(audit_csv).with_name(Path(audit_csv).stem + "_by_year.csv")
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")

    print("\nSaved:", output_csv)
    print("Saved audit:", audit_csv)
    print("Saved by-year audit:", by_year_path)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nBy year:")
    print(by_year.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", default="analysis_ready_traditional_ai_v1.csv")
    parser.add_argument("--stock_csv", default="stock_stats_with_hcr_covid_exposure_time_aware.csv")
    parser.add_argument("--annual_perf_csv", default="annual_stock_performance_covid_2018_2024_v2.csv")
    parser.add_argument("--output_csv", default="analysis_ready_final_2018_2024_v2.csv")
    parser.add_argument("--audit_csv", default="analysis_ready_final_2018_2024_v2_audit.csv")
    args = parser.parse_args()

    print("=" * 80)
    print("Build final 2018–2024 Traditional + AI + Performance + COVID panel")
    print("=" * 80)
    print("Base:", args.base_csv)
    print("Stock/performance/COVID source:", args.stock_csv)

    stock = prepare_stock_quarterly(args.stock_csv)
    annual = annualize_stock_performance(stock)

    annual.to_csv(args.annual_perf_csv, index=False, encoding="utf-8-sig")
    print("\nSaved annual performance panel:", args.annual_perf_csv)
    print("Annual performance shape:", annual.shape)

    merge_base_with_annual_performance(
        base_csv=args.base_csv,
        annual_perf=annual,
        output_csv=args.output_csv,
        audit_csv=args.audit_csv,
    )


if __name__ == "__main__":
    main()