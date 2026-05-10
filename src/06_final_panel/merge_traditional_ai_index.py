# ============================================================
# merge_traditional_ai_index_v1.py
# Purpose:
#   Merge traditional keyword-based index with full-document AI semantic index.
#
# Inputs:
#   analysis_ready_traditional_index_v3_1.csv
#   ai_full_document_firm_year_v3_fixed.csv
#
# Output:
#   analysis_ready_traditional_ai_v1.csv
#   analysis_ready_traditional_ai_v1_audit.csv
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_cik(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = x.replace(".0", "")
    x = x.lstrip("0")
    return x


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traditional_csv",
        default="analysis_ready_traditional_index_v3_1.csv",
    )
    parser.add_argument(
        "--ai_csv",
        default="ai_full_document_firm_year_v3_fixed.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="analysis_ready_traditional_ai_v1.csv",
    )
    parser.add_argument(
        "--audit_csv",
        default="analysis_ready_traditional_ai_v1_audit.csv",
    )
    args = parser.parse_args()

    traditional = pd.read_csv(args.traditional_csv, dtype=str).fillna("")
    ai = pd.read_csv(args.ai_csv, dtype=str).fillna("")

    print("=" * 70)
    print("Merge Traditional + AI Full-Document Index")
    print("=" * 70)
    print("Traditional input:", args.traditional_csv)
    print("AI input:", args.ai_csv)

    # --------------------------------------------------------
    # 1. Standardize keys
    # --------------------------------------------------------

    traditional["CIK_clean"] = traditional["CIK"].apply(clean_cik)

    # Traditional file uses YEAR
    traditional["YEAR_num"] = safe_num(traditional["YEAR"]).astype("Int64")

    ai["CIK_clean"] = ai["cik"].apply(clean_cik)
    ai["YEAR_num"] = safe_num(ai["fiscal_year"]).astype("Int64")

    # --------------------------------------------------------
    # 2. Select useful AI columns
    # --------------------------------------------------------

    ai_keep = [
        "CIK_clean",
        "YEAR_num",
        "aggregation_rule",
        "n_filings_scored_all",
        "n_filings_used",
        "n_api_ok_all",
        "n_api_error_all",
        "n_10k_all",
        "n_10k_ok",
        "n_10q_all",
        "n_10q_ok",
        "has_10k_any",
        "has_10k_ok",
        "mean_digital_relevance_used",
        "mean_confidence_used",
        "mean_full_document_char_len_used",
        "mean_text_char_len_sent_to_poe_used",
        "AI_full_investment_type",
        "AI_full_strategic_importance",
        "AI_full_implementation_stage",
        "AI_full_time_horizon",
        "AI_full_digital_relevance",
        "AI_full_confidence",
        "AI_full_document_score_v3",
        "AI_full_document_index_0_100_v3",
    ]

    ai_keep = [c for c in ai_keep if c in ai.columns]
    ai_small = ai[ai_keep].copy()

    # Deduplicate AI if needed
    dup_ai = ai_small.duplicated(["CIK_clean", "YEAR_num"], keep=False).sum()
    if dup_ai > 0:
        print(f"Warning: AI file has duplicated CIK-year rows: {dup_ai}")
        ai_small = ai_small.drop_duplicates(["CIK_clean", "YEAR_num"], keep="first")

    # --------------------------------------------------------
    # 3. Merge
    # --------------------------------------------------------

    merged = traditional.merge(
        ai_small,
        on=["CIK_clean", "YEAR_num"],
        how="left",
        indicator="ai_merge_status",
    )

    # Rename clean year key to simple final name
    merged["YEAR_final"] = merged["YEAR_num"]

    # Numeric conversions for key variables
    numeric_cols = [
        "Traditional_score",
        "Traditional_index_0_100",
        "Traditional_score_risk_adjusted",
        "Traditional_index_risk_adjusted_0_100",
        "Traditional_score_tfidf",
        "Digital_risk_disclosure_score",
        "AI_full_document_score_v3",
        "AI_full_document_index_0_100_v3",
        "AI_full_investment_type",
        "AI_full_strategic_importance",
        "AI_full_implementation_stage",
        "AI_full_time_horizon",
        "AI_full_digital_relevance",
        "AI_full_confidence",
        "mean_confidence_used",
        "mean_digital_relevance_used",
        "n_filings",
        "total_words_v3",
    ]

    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = safe_num(merged[c])

    # --------------------------------------------------------
    # 4. Audit variables
    # --------------------------------------------------------

    merged["has_traditional_index"] = merged["Traditional_index_0_100"].notna()
    merged["has_ai_full_document_index"] = merged["AI_full_document_index_0_100_v3"].notna()

    merged["both_traditional_and_ai"] = (
        merged["has_traditional_index"] & merged["has_ai_full_document_index"]
    )

    # Difference / gap variables
    merged["AI_minus_Traditional_index"] = (
        merged["AI_full_document_index_0_100_v3"] - merged["Traditional_index_0_100"]
    )

    # Optional z-scores within merged sample
    for c in ["Traditional_index_0_100", "AI_full_document_index_0_100_v3"]:
        if c in merged.columns:
            mean = merged[c].mean(skipna=True)
            std = merged[c].std(skipna=True)
            if pd.notna(std) and std > 0:
                merged[f"z_{c}"] = (merged[c] - mean) / std
            else:
                merged[f"z_{c}"] = np.nan

    merged["AI_minus_Traditional_z"] = (
        merged.get("z_AI_full_document_index_0_100_v3", np.nan)
        - merged.get("z_Traditional_index_0_100", np.nan)
    )

    # --------------------------------------------------------
    # 5. Save final merged data
    # --------------------------------------------------------

    merged.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 6. Audit table
    # --------------------------------------------------------

    audit = {}

    audit["traditional_rows"] = len(traditional)
    audit["ai_rows"] = len(ai)
    audit["merged_rows"] = len(merged)

    audit["traditional_unique_cik_year"] = traditional[["CIK_clean", "YEAR_num"]].drop_duplicates().shape[0]
    audit["ai_unique_cik_year"] = ai[["CIK_clean", "YEAR_num"]].drop_duplicates().shape[0]

    audit["matched_ai_rows"] = int(merged["has_ai_full_document_index"].sum())
    audit["missing_ai_rows"] = int((~merged["has_ai_full_document_index"]).sum())
    audit["ai_coverage_rate"] = float(merged["has_ai_full_document_index"].mean())

    audit["both_traditional_and_ai_rows"] = int(merged["both_traditional_and_ai"].sum())

    if "Traditional_index_0_100" in merged.columns:
        audit["mean_traditional_index"] = float(merged["Traditional_index_0_100"].mean(skipna=True))
        audit["median_traditional_index"] = float(merged["Traditional_index_0_100"].median(skipna=True))

    if "AI_full_document_index_0_100_v3" in merged.columns:
        audit["mean_ai_full_document_index"] = float(merged["AI_full_document_index_0_100_v3"].mean(skipna=True))
        audit["median_ai_full_document_index"] = float(merged["AI_full_document_index_0_100_v3"].median(skipna=True))

    valid_corr = merged[
        ["Traditional_index_0_100", "AI_full_document_index_0_100_v3"]
    ].dropna()

    if len(valid_corr) >= 3:
        audit["corr_traditional_ai"] = float(
            valid_corr["Traditional_index_0_100"].corr(
                valid_corr["AI_full_document_index_0_100_v3"]
            )
        )
    else:
        audit["corr_traditional_ai"] = np.nan

    # By-year audit
    by_year = (
        merged.groupby("YEAR_num")
        .agg(
            n_rows=("YEAR_num", "size"),
            n_ai_matched=("has_ai_full_document_index", "sum"),
            ai_coverage_rate=("has_ai_full_document_index", "mean"),
            traditional_mean=("Traditional_index_0_100", "mean"),
            ai_mean=("AI_full_document_index_0_100_v3", "mean"),
            traditional_median=("Traditional_index_0_100", "median"),
            ai_median=("AI_full_document_index_0_100_v3", "median"),
        )
        .reset_index()
    )

    audit_df = pd.DataFrame([audit])
    audit_df.to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    by_year_path = Path(args.audit_csv).with_name(
        Path(args.audit_csv).stem + "_by_year.csv"
    )
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Shape:", merged.shape)

    print("\nSaved audit:", args.audit_csv)
    print("Saved by-year audit:", by_year_path)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nBy year:")
    print(by_year.to_string(index=False))


if __name__ == "__main__":
    main()