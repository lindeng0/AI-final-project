# ============================================================
# merge_stock_return_into_final_panel_v3.py
# Purpose:
#   Merge annual stock return from 6020_21.xlsx into final 2018–2024 panel.
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_gvkey(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace(".0", "")


def zscore(s):
    x = safe_num(s)
    sd = x.std(skipna=True)
    mu = x.mean(skipna=True)
    if pd.notna(sd) and sd > 0:
        return (x - mu) / sd
    return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_panel_csv", default="analysis_ready_final_2018_2024_v2.csv")
    parser.add_argument("--stock_return_csv", default="annual_stock_return_2018_2024_from_6020_v1.csv")
    parser.add_argument("--output_csv", default="analysis_ready_final_2018_2024_v3_with_stock_return.csv")
    parser.add_argument("--audit_csv", default="analysis_ready_final_2018_2024_v3_with_stock_return_audit.csv")
    args = parser.parse_args()

    panel = pd.read_csv(args.final_panel_csv, dtype=str).fillna("")
    ret = pd.read_csv(args.stock_return_csv, dtype=str).fillna("")

    panel["GVKEY_clean"] = panel["GVKEY"].apply(clean_gvkey)
    panel["YEAR_key"] = safe_num(panel["YEAR_key"]).astype("Int64")

    ret["GVKEY_clean"] = ret["GVKEY_clean"].apply(clean_gvkey)
    ret["YEAR_key"] = safe_num(ret["YEAR_key"]).astype("Int64")

    keep_cols = [
        "GVKEY_clean",
        "YEAR_key",
        "tic_stock_6020",
        "n_quarters_6020",
        "first_datadate_6020",
        "last_datadate_6020",
        "annual_compound_price_return_6020",
        "annual_compound_price_return_6020_w",
        "annual_mean_price_return_6020",
        "annual_mean_price_return_6020_w",
        "annual_last_prccq_6020",
        "annual_last_mktcap_6020",
        "annual_mean_log_mktcap_6020",
        "annual_mean_roa_from_6020",
        "annual_mean_roa_from_6020_w",
    ]

    keep_cols = [c for c in keep_cols if c in ret.columns]
    ret = ret[keep_cols].drop_duplicates(["GVKEY_clean", "YEAR_key"], keep="first")

    merged = panel.merge(
        ret,
        on=["GVKEY_clean", "YEAR_key"],
        how="left",
        indicator="stock_return_merge_status",
    )

    # Convenient names
    merged["StockReturn_annual_compound"] = safe_num(merged["annual_compound_price_return_6020"])
    merged["StockReturn_annual_compound_w"] = safe_num(merged["annual_compound_price_return_6020_w"])
    merged["StockReturn_annual_mean_w"] = safe_num(merged["annual_mean_price_return_6020_w"])
    merged["LogMktCap_annual_mean"] = safe_num(merged["annual_mean_log_mktcap_6020"])

    # z-scores
    for c in [
        "StockReturn_annual_compound",
        "StockReturn_annual_compound_w",
        "StockReturn_annual_mean_w",
        "LogMktCap_annual_mean",
    ]:
        if c in merged.columns:
            merged[f"z_{c}"] = zscore(merged[c])

    merged.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "panel_rows": len(panel),
        "stock_return_rows": len(ret),
        "merged_rows": len(merged),
        "matched_stock_return_rows": int(merged["stock_return_merge_status"].eq("both").sum()),
        "missing_stock_return_rows": int(merged["stock_return_merge_status"].eq("left_only").sum()),
        "stock_return_coverage_rate": float(merged["stock_return_merge_status"].eq("both").mean()),
        "n_nonmissing_StockReturn_annual_compound_w": int(merged["StockReturn_annual_compound_w"].notna().sum()),
        "mean_StockReturn_annual_compound_w": float(merged["StockReturn_annual_compound_w"].mean(skipna=True)),
        "median_StockReturn_annual_compound_w": float(merged["StockReturn_annual_compound_w"].median(skipna=True)),
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    by_year = (
        merged.groupby("YEAR_key")
        .agg(
            n=("YEAR_key", "size"),
            n_stock_return=("StockReturn_annual_compound_w", lambda x: x.notna().sum()),
            mean_stock_return=("StockReturn_annual_compound_w", "mean"),
            median_stock_return=("StockReturn_annual_compound_w", "median"),
            mean_ai=("AI_full_document_index_0_100_v3", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            mean_traditional=("Traditional_index_0_100", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
    )

    by_year_path = Path(args.audit_csv).with_name(
        Path(args.audit_csv).stem + "_by_year.csv"
    )
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Saved audit:", args.audit_csv)
    print("Saved by-year audit:", by_year_path)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nBy year:")
    print(by_year.to_string(index=False))


if __name__ == "__main__":
    main()