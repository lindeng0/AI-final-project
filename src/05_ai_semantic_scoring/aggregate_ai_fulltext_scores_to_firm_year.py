# ============================================================
# aggregate_ai_full_document_scores_v3_fixed.py
# Purpose:
#   Aggregate full-document Poe AI scores to firm-year level.
#
# Fixes:
#   - Robust fiscal year inference
#   - Ignores impossible years like 1910 / 2003 caused by filename strings
#   - Maps early-year 10-K filings to previous fiscal year
#   - Uses 10-K if successfully scored; otherwise falls back to scored 10-Q
#   - Default final analysis window: 2018–2024
# ============================================================

import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


MAIN_SCORE_COLS = [
    "InvestmentType",
    "StrategicImportance",
    "ImplementationStage",
    "TimeHorizon",
]


def safe_float(s):
    return pd.to_numeric(s, errors="coerce")


def infer_fiscal_year_fixed(file_name: str, form: str, filing_date: str):
    """
    Robust fiscal-year inference.

    Rules:
    1. Use valid YYYYMMDD / MMDDYYYY dates in filename only if year is plausible.
    2. Ignore impossible years like 1910 / 2003 created by strings such as 12311910k.
    3. For 10-K filed in Jan–Jun, fiscal year is usually filing_year - 1.
    4. For 10-Q, fallback to filing_year.
    """
    stem = str(file_name).replace(".txt", "")
    form = str(form).upper()

    filing_year = np.nan
    filing_month = np.nan

    if re.match(r"\d{4}-\d{2}-\d{2}", str(filing_date)):
        filing_year = int(str(filing_date)[:4])
        filing_month = int(str(filing_date)[5:7])

    candidates = []

    for m in re.finditer(r"\d{8}", stem):
        s = m.group(0)

        # YYYYMMDD
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            if 2017 <= dt.year <= 2025:
                if pd.isna(filing_year) or (filing_year - 2 <= dt.year <= filing_year):
                    candidates.append(dt)
        except Exception:
            pass

        # MMDDYYYY
        try:
            dt = datetime.strptime(s, "%m%d%Y")
            if 2017 <= dt.year <= 2025:
                if pd.isna(filing_year) or (filing_year - 2 <= dt.year <= filing_year):
                    candidates.append(dt)
        except Exception:
            pass

    if candidates:
        return sorted(candidates)[-1].year

    if pd.isna(filing_year):
        return np.nan

    # Fallback logic
    if form == "10-K":
        # Most bank 10-K filings in Feb–Mar 2025 refer to FY2024.
        if filing_month <= 6:
            return filing_year - 1
        return filing_year

    # 10-Q generally belongs to same calendar filing year,
    # except when filename has a valid period-end date, already handled above.
    return filing_year


def weighted_avg(g, col, weight_col):
    x = safe_float(g[col])
    w = safe_float(g[weight_col])

    mask = x.notna() & w.notna() & (w > 0)

    if mask.sum() == 0:
        return np.nan

    return float(np.average(x[mask], weights=w[mask]))


def aggregate_group(g):
    """
    Annual-first with fallback:
    - Prefer successfully scored 10-K.
    - If no scored 10-K exists, use all successfully scored filings.
    - If nothing scored successfully, keep missing.
    """
    g = g.copy()

    g_ok = g[g["api_ok"]].copy()

    has_10k_any = g["form_upper"].eq("10-K").any()
    has_10k_ok = g_ok["form_upper"].eq("10-K").any() if len(g_ok) else False

    if has_10k_ok:
        used = g_ok[g_ok["form_upper"].eq("10-K")].copy()
        aggregation_rule = "10K_ok_only"
    elif len(g_ok) > 0:
        used = g_ok.copy()
        aggregation_rule = "fallback_all_ok_no_10K_ok"
    else:
        used = g.iloc[0:0].copy()
        aggregation_rule = "all_failed"

    cik = g["cik"].iloc[0]
    fiscal_year = int(g["fiscal_year_fixed"].iloc[0])

    row = {
        "cik": cik,
        "fiscal_year": fiscal_year,
        "aggregation_rule": aggregation_rule,

        "n_filings_scored_all": len(g),
        "n_filings_used": len(used),
        "n_api_ok_all": int(g["api_ok"].sum()),
        "n_api_error_all": int((~g["api_ok"]).sum()),

        "n_10k_all": int(g["form_upper"].eq("10-K").sum()),
        "n_10k_ok": int((g["form_upper"].eq("10-K") & g["api_ok"]).sum()),
        "n_10q_all": int(g["form_upper"].eq("10-Q").sum()),
        "n_10q_ok": int((g["form_upper"].eq("10-Q") & g["api_ok"]).sum()),

        "has_10k_any": int(has_10k_any),
        "has_10k_ok": int(has_10k_ok),
    }

    if len(used) == 0:
        row.update({
            "mean_digital_relevance_used": np.nan,
            "mean_confidence_used": np.nan,
            "mean_full_document_char_len_used": np.nan,
            "mean_text_char_len_sent_to_poe_used": np.nan,

            "AI_full_investment_type": np.nan,
            "AI_full_strategic_importance": np.nan,
            "AI_full_implementation_stage": np.nan,
            "AI_full_time_horizon": np.nan,
            "AI_full_digital_relevance": np.nan,
            "AI_full_confidence": np.nan,
            "AI_full_document_score_v3": np.nan,
            "AI_full_document_index_0_100_v3": np.nan,
        })
        return row

    row["mean_digital_relevance_used"] = float(used["DigitalRelevance"].mean(skipna=True))
    row["mean_confidence_used"] = float(used["Confidence"].mean(skipna=True))

    if "full_document_char_len" in used.columns:
        row["mean_full_document_char_len_used"] = float(used["full_document_char_len"].mean(skipna=True))
    else:
        row["mean_full_document_char_len_used"] = np.nan

    if "text_char_len_sent_to_poe" in used.columns:
        row["mean_text_char_len_sent_to_poe_used"] = float(used["text_char_len_sent_to_poe"].mean(skipna=True))
    else:
        row["mean_text_char_len_sent_to_poe_used"] = np.nan

    row["AI_full_investment_type"] = weighted_avg(used, "InvestmentType", "final_weight")
    row["AI_full_strategic_importance"] = weighted_avg(used, "StrategicImportance", "final_weight")
    row["AI_full_implementation_stage"] = weighted_avg(used, "ImplementationStage", "final_weight")
    row["AI_full_time_horizon"] = weighted_avg(used, "TimeHorizon", "final_weight")
    row["AI_full_digital_relevance"] = weighted_avg(used, "DigitalRelevance", "final_weight")
    row["AI_full_confidence"] = weighted_avg(used, "Confidence", "final_weight")

    main_vals = [
        row["AI_full_investment_type"],
        row["AI_full_strategic_importance"],
        row["AI_full_implementation_stage"],
        row["AI_full_time_horizon"],
    ]

    if all(pd.notna(x) for x in main_vals):
        row["AI_full_document_score_v3"] = float(np.mean(main_vals))
        row["AI_full_document_index_0_100_v3"] = float(
            (row["AI_full_document_score_v3"] - 1.0) / 4.0 * 100.0
        )
    else:
        row["AI_full_document_score_v3"] = np.nan
        row["AI_full_document_index_0_100_v3"] = np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="poe_ai_full_document_scores_v3_fulltext.csv")
    parser.add_argument("--output_csv", default="ai_full_document_firm_year_v3_fixed.csv")
    parser.add_argument("--audit_csv", default="ai_full_document_firm_year_audit_v3_fixed.csv")
    parser.add_argument("--year_min", type=int, default=2018)
    parser.add_argument("--year_max", type=int, default=2024)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")

    numeric_cols = MAIN_SCORE_COLS + [
        "DigitalRelevance",
        "Confidence",
        "AI_full_document_score",
        "AI_full_document_index_0_100",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "filing_year",
        "fiscal_year",
        "full_document_char_len",
        "full_document_word_count_rough",
        "text_char_len_sent_to_poe",
        "attempts_used",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_float(df[c])

    df["api_ok"] = df["api_status"].astype(str).str.lower().eq("ok")
    df["form_upper"] = df["form"].astype(str).str.upper()

    df["fiscal_year_fixed"] = df.apply(
        lambda r: infer_fiscal_year_fixed(
            r.get("file_name", ""),
            r.get("form", ""),
            r.get("filing_date", ""),
        ),
        axis=1,
    )

    df["fiscal_year_fixed"] = safe_float(df["fiscal_year_fixed"]).astype("Int64")

    # Clean CIK
    df["cik"] = (
        df["cik"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
    )
    df.loc[df["cik"].eq(""), "cik"] = np.nan

    # Keep only requested fiscal-year window
    valid = (
        df["cik"].notna()
        & df["fiscal_year_fixed"].notna()
        & (df["fiscal_year_fixed"] >= args.year_min)
        & (df["fiscal_year_fixed"] <= args.year_max)
    )

    df_valid = df[valid].copy()

    # Weighting
    df_valid["form_weight"] = np.where(df_valid["form_upper"].eq("10-K"), 1.0, 0.35)

    df_valid["confidence_weight"] = safe_float(df_valid["Confidence"]) / 100.0
    df_valid["confidence_weight"] = df_valid["confidence_weight"].clip(lower=0, upper=1)

    df_valid["relevance_weight"] = safe_float(df_valid["DigitalRelevance"]) / 5.0
    df_valid["relevance_weight"] = df_valid["relevance_weight"].clip(lower=0.2, upper=1)

    df_valid["final_weight"] = (
        df_valid["form_weight"]
        * df_valid["confidence_weight"]
        * df_valid["relevance_weight"]
    )

    df_valid.loc[~df_valid["api_ok"], "final_weight"] = 0

    rows = []

    for _, g in df_valid.groupby(["cik", "fiscal_year_fixed"], dropna=False):
        rows.append(aggregate_group(g))

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["cik", "fiscal_year"])

    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "n_filing_rows_original": len(df),
        "n_filing_rows_in_year_window": len(df_valid),
        "n_api_ok_in_year_window": int(df_valid["api_ok"].sum()),
        "n_api_error_in_year_window": int((~df_valid["api_ok"]).sum()),

        "n_firm_year_rows": len(out),
        "n_missing_firm_year_ai_score": int(out["AI_full_document_score_v3"].isna().sum()) if len(out) else 0,

        "n_10K_ok_only_firm_years": int(out["aggregation_rule"].eq("10K_ok_only").sum()) if len(out) else 0,
        "n_fallback_all_ok_no_10K_ok_firm_years": int(out["aggregation_rule"].eq("fallback_all_ok_no_10K_ok").sum()) if len(out) else 0,
        "n_all_failed_firm_years": int(out["aggregation_rule"].eq("all_failed").sum()) if len(out) else 0,

        "mean_AI_full_document_index_0_100_v3": float(out["AI_full_document_index_0_100_v3"].mean(skipna=True)) if len(out) else np.nan,
        "median_AI_full_document_index_0_100_v3": float(out["AI_full_document_index_0_100_v3"].median(skipna=True)) if len(out) else np.nan,
        "year_min": args.year_min,
        "year_max": args.year_max,
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Shape:", out.shape)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    if len(out) > 0:
        print("\nAggregation rule counts:")
        print(out["aggregation_rule"].value_counts(dropna=False).to_string())

        print("\nBy fiscal year:")
        print(
            out.groupby("fiscal_year")["AI_full_document_index_0_100_v3"]
            .agg(["count", "mean", "median"])
            .to_string()
        )

        print("\nPreview:")
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()