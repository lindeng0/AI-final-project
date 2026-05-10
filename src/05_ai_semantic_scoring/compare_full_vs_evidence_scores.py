# ============================================================
# compare_full_vs_evidence_scores_v1.py
# Purpose:
#   Compare Poe scores from evidence-snippet input vs broad full-document input.
# ============================================================

import argparse
import pandas as pd
import numpy as np


SCORE_COLS = [
    "InvestmentType",
    "StrategicImportance",
    "ImplementationStage",
    "TimeHorizon",
    "DigitalRelevance",
    "Confidence",
    "AI_semantic_score",
    "AI_semantic_index_0_100",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="poe_full_vs_evidence_scores_v1.csv")
    parser.add_argument("--output_comparison_csv", default="full_vs_evidence_comparison_v1.csv")
    parser.add_argument("--output_audit_csv", default="full_vs_evidence_audit_v1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")

    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ok = df[df["api_status"].eq("ok")].copy()

    wide = ok.pivot_table(
        index=["filing_id", "file_name", "cik", "form", "filing_date", "fiscal_year"],
        columns="input_mode",
        values=SCORE_COLS,
        aggfunc="first",
    )

    wide.columns = [f"{col}_{mode}" for col, mode in wide.columns]
    wide = wide.reset_index()

    # Differences: full - evidence
    pairs = [
        "InvestmentType",
        "StrategicImportance",
        "ImplementationStage",
        "TimeHorizon",
        "DigitalRelevance",
        "Confidence",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
    ]

    for p in pairs:
        e = f"{p}_evidence_snippets"
        f = f"{p}_broad_full_document"
        if e in wide.columns and f in wide.columns:
            wide[f"diff_{p}_full_minus_evidence"] = wide[f] - wide[e]
            wide[f"absdiff_{p}"] = wide[f"diff_{p}_full_minus_evidence"].abs()

    wide.to_csv(args.output_comparison_csv, index=False, encoding="utf-8-sig")

    audit = {}

    audit["n_total_scoring_rows"] = len(df)
    audit["n_ok_scoring_rows"] = len(ok)
    audit["n_unique_filings_compared"] = len(wide)

    main_diff = "diff_AI_semantic_index_0_100_full_minus_evidence"
    abs_main_diff = "absdiff_AI_semantic_index_0_100"

    if main_diff in wide.columns:
        audit["mean_full_index"] = wide["AI_semantic_index_0_100_broad_full_document"].mean()
        audit["mean_evidence_index"] = wide["AI_semantic_index_0_100_evidence_snippets"].mean()
        audit["mean_difference_full_minus_evidence"] = wide[main_diff].mean()
        audit["median_difference_full_minus_evidence"] = wide[main_diff].median()
        audit["mean_absolute_difference"] = wide[abs_main_diff].mean()
        audit["median_absolute_difference"] = wide[abs_main_diff].median()

        audit["corr_full_evidence_index"] = wide[
            ["AI_semantic_index_0_100_broad_full_document", "AI_semantic_index_0_100_evidence_snippets"]
        ].corr().iloc[0, 1]

        audit["pct_absdiff_le_10"] = (wide[abs_main_diff] <= 10).mean()
        audit["pct_absdiff_le_20"] = (wide[abs_main_diff] <= 20).mean()
        audit["pct_full_higher_than_evidence"] = (wide[main_diff] > 0).mean()
        audit["pct_full_lower_than_evidence"] = (wide[main_diff] < 0).mean()

    audit_df = pd.DataFrame([audit])
    audit_df.to_csv(args.output_audit_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.output_comparison_csv)
    print("Saved:", args.output_audit_csv)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nLargest differences:")
    if abs_main_diff in wide.columns:
        cols = [
            "file_name",
            "form",
            "filing_date",
            "fiscal_year",
            "AI_semantic_index_0_100_evidence_snippets",
            "AI_semantic_index_0_100_broad_full_document",
            main_diff,
            abs_main_diff,
        ]
        print(
            wide.sort_values(abs_main_diff, ascending=False)
                .head(15)[cols]
                .to_string(index=False)
        )


if __name__ == "__main__":
    main()