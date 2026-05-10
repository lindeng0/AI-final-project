# ============================================================
# build_stock_return_2018_2024_from_6020_v1.py
# Purpose:
#   Construct 2018–2024 annual stock return from Compustat quarterly price data.
#
# Input:
#   6020_21.xlsx
#
# Key variables:
#   GVKEY, datadate, fyearq, fqtr, tic, prccq, cshoq, atq, niq
#
# Output:
#   annual_stock_return_2018_2024_from_6020_v1.csv
#   annual_stock_return_2018_2024_from_6020_v1_audit.csv
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
    x = str(x).strip().replace(".0", "")
    return x


def clean_tic(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def winsorize(s, lower=0.01, upper=0.99):
    x = pd.to_numeric(s, errors="coerce")
    lo = x.quantile(lower)
    hi = x.quantile(upper)
    return x.clip(lo, hi)


def compound_return(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", default="6020_21.xlsx")
    parser.add_argument("--output_csv", default="annual_stock_return_2018_2024_from_6020_v1.csv")
    parser.add_argument("--audit_csv", default="annual_stock_return_2018_2024_from_6020_v1_audit.csv")
    args = parser.parse_args()

    print("=" * 80)
    print("Build annual stock return from 6020_21.xlsx")
    print("=" * 80)
    print("Input:", args.input_xlsx)

    df = pd.read_excel(args.input_xlsx, dtype=str)

    # Standardize columns
    required = ["GVKEY", "datadate", "fyearq", "fqtr", "tic", "prccq", "cshoq", "atq", "niq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["GVKEY_clean"] = df["GVKEY"].apply(clean_gvkey)
    df["tic_clean"] = df["tic"].apply(clean_tic)

    df["datadate_dt"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["calendar_year"] = df["datadate_dt"].dt.year.astype("Int64")

    df["fyearq_num"] = safe_num(df["fyearq"]).astype("Int64")
    df["fqtr_num"] = safe_num(df["fqtr"]).astype("Int64")

    for c in ["prccq", "cshoq", "atq", "niq"]:
        df[c] = safe_num(df[c])

    # Keep plausible rows
    df = df[
        df["GVKEY_clean"].ne("")
        & df["datadate_dt"].notna()
        & df["prccq"].notna()
        & (df["prccq"] > 0)
    ].copy()

    # Sort by firm-date
    df = df.sort_values(["GVKEY_clean", "datadate_dt"])

    # Quarterly price return using lag price
    df["lag_prccq"] = df.groupby("GVKEY_clean")["prccq"].shift(1)
    df["price_return_q"] = df["prccq"] / df["lag_prccq"] - 1.0

    # Remove impossible extreme returns before winsorization
    df.loc[df["price_return_q"] < -0.95, "price_return_q"] = np.nan
    df.loc[df["price_return_q"] > 10, "price_return_q"] = np.nan

    df["price_return_q_w"] = winsorize(df["price_return_q"], 0.01, 0.99)

    # Market cap
    df["mktcap_q"] = df["prccq"] * df["cshoq"]
    df["log_mktcap_q"] = np.log(df["mktcap_q"].where(df["mktcap_q"] > 0))

    # ROA in this raw file, for consistency check
    df["roa_q_from_6020"] = df["niq"] / df["atq"]
    df["roa_q_from_6020_w"] = winsorize(df["roa_q_from_6020"], 0.01, 0.99)

    # We use calendar year to align with stock performance year.
    df_2018_2024 = df[
        (df["calendar_year"] >= 2018)
        & (df["calendar_year"] <= 2024)
    ].copy()

    rows = []

    for (gvkey, year), g in df_2018_2024.groupby(["GVKEY_clean", "calendar_year"], dropna=False):
        g = g.sort_values("datadate_dt")

        row = {
            "GVKEY_clean": gvkey,
            "YEAR_key": int(year),
            "tic_stock_6020": g["tic_clean"].dropna().iloc[-1] if g["tic_clean"].notna().sum() else "",
            "n_quarters_6020": len(g),
            "first_datadate_6020": g["datadate_dt"].min(),
            "last_datadate_6020": g["datadate_dt"].max(),

            "annual_compound_price_return_6020": compound_return(g["price_return_q"]),
            "annual_compound_price_return_6020_w": compound_return(g["price_return_q_w"]),

            "annual_mean_price_return_6020": g["price_return_q"].mean(skipna=True),
            "annual_mean_price_return_6020_w": g["price_return_q_w"].mean(skipna=True),

            "annual_last_prccq_6020": g["prccq"].dropna().iloc[-1] if g["prccq"].notna().sum() else np.nan,
            "annual_last_mktcap_6020": g["mktcap_q"].dropna().iloc[-1] if g["mktcap_q"].notna().sum() else np.nan,
            "annual_mean_log_mktcap_6020": g["log_mktcap_q"].mean(skipna=True),

            "annual_mean_roa_from_6020": g["roa_q_from_6020"].mean(skipna=True),
            "annual_mean_roa_from_6020_w": g["roa_q_from_6020_w"].mean(skipna=True),
        }

        rows.append(row)

    annual = pd.DataFrame(rows)

    annual.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "raw_rows": len(df),
        "rows_2018_2024": len(df_2018_2024),
        "annual_firm_year_rows": len(annual),
        "n_unique_gvkey": annual["GVKEY_clean"].nunique(),
        "n_nonmissing_annual_compound_price_return_6020": int(annual["annual_compound_price_return_6020"].notna().sum()),
        "n_nonmissing_annual_compound_price_return_6020_w": int(annual["annual_compound_price_return_6020_w"].notna().sum()),
        "mean_annual_compound_price_return_6020_w": float(annual["annual_compound_price_return_6020_w"].mean(skipna=True)),
        "median_annual_compound_price_return_6020_w": float(annual["annual_compound_price_return_6020_w"].median(skipna=True)),
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    by_year = (
        annual.groupby("YEAR_key")
        .agg(
            n=("GVKEY_clean", "size"),
            n_return=("annual_compound_price_return_6020_w", lambda x: x.notna().sum()),
            mean_return=("annual_compound_price_return_6020_w", "mean"),
            median_return=("annual_compound_price_return_6020_w", "median"),
            mean_roa=("annual_mean_roa_from_6020_w", "mean"),
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