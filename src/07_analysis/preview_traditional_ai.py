# ============================================================
# preview_traditional_ai_v1.py
# Purpose:
#   Create preview figures and summary tables for traditional vs AI index.
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def save_hist(df, col, out_path, title, xlabel):
    x = df[col].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of firm-years")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_year_trend(df, out_path):
    d = df.copy()

    year_col = "YEAR_num" if "YEAR_num" in d.columns else "YEAR_final"

    trend = (
        d.groupby(year_col)
        .agg(
            Traditional_mean=("Traditional_index_0_100", "mean"),
            AI_mean=("AI_full_document_index_0_100_v3", "mean"),
            Traditional_median=("Traditional_index_0_100", "median"),
            AI_median=("AI_full_document_index_0_100_v3", "median"),
            N=("AI_full_document_index_0_100_v3", "count"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(trend[year_col], trend["Traditional_mean"], marker="o", label="Traditional index, mean")
    plt.plot(trend[year_col], trend["AI_mean"], marker="o", label="AI full-document index, mean")
    plt.title("Traditional vs AI Digital Transformation Index by Year")
    plt.xlabel("Fiscal year")
    plt.ylabel("Index, 0–100")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return trend


def save_scatter(df, out_path):
    d = df[["Traditional_index_0_100", "AI_full_document_index_0_100_v3"]].dropna()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        d["Traditional_index_0_100"],
        d["AI_full_document_index_0_100_v3"],
        alpha=0.45,
        s=16,
    )
    plt.title("Traditional Index vs AI Full-Document Index")
    plt.xlabel("Traditional keyword-based index, 0–100")
    plt.ylabel("AI full-document semantic index, 0–100")

    # 45-degree reference line
    lim_min = 0
    lim_max = 100
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_gap_hist(df, out_path):
    x = df["AI_minus_Traditional_index"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=30)
    plt.axvline(0, linestyle="--")
    plt.title("Gap Between AI and Traditional Index")
    plt.xlabel("AI index − Traditional index")
    plt.ylabel("Number of firm-years")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_summary_tables(df, out_dir):
    cols = [
        "Traditional_index_0_100",
        "AI_full_document_index_0_100_v3",
        "AI_full_investment_type",
        "AI_full_strategic_importance",
        "AI_full_implementation_stage",
        "AI_full_time_horizon",
        "AI_full_digital_relevance",
        "AI_full_confidence",
        "AI_minus_Traditional_index",
    ]

    cols = [c for c in cols if c in df.columns]

    summary = df[cols].describe().T
    summary.to_csv(out_dir / "summary_statistics_traditional_ai_v1.csv", encoding="utf-8-sig")

    corr_cols = [
        "Traditional_index_0_100",
        "Traditional_score_tfidf",
        "Traditional_index_risk_adjusted_0_100",
        "Digital_risk_disclosure_score",
        "AI_full_document_index_0_100_v3",
        "AI_full_investment_type",
        "AI_full_strategic_importance",
        "AI_full_implementation_stage",
        "AI_full_time_horizon",
    ]

    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    corr.to_csv(out_dir / "correlation_matrix_traditional_ai_v1.csv", encoding="utf-8-sig")

    # Top gap cases
    gap_cols = [
        "GVKEY",
        "CIK",
        "tic",
        "YEAR",
        "Traditional_index_0_100",
        "AI_full_document_index_0_100_v3",
        "AI_minus_Traditional_index",
        "aggregation_rule",
        "n_filings",
        "n_filings_used",
        "mean_confidence_used",
    ]

    gap_cols = [c for c in gap_cols if c in df.columns]

    high_ai_gap = (
        df.dropna(subset=["AI_minus_Traditional_index"])
        .sort_values("AI_minus_Traditional_index", ascending=False)
        .head(30)
    )

    high_trad_gap = (
        df.dropna(subset=["AI_minus_Traditional_index"])
        .sort_values("AI_minus_Traditional_index", ascending=True)
        .head(30)
    )

    high_ai_gap[gap_cols].to_csv(out_dir / "top30_ai_higher_than_traditional.csv", index=False, encoding="utf-8-sig")
    high_trad_gap[gap_cols].to_csv(out_dir / "top30_traditional_higher_than_ai.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="analysis_ready_traditional_ai_v1.csv")
    parser.add_argument("--output_dir", default="preview_traditional_ai_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    numeric_cols = [
        "Traditional_index_0_100",
        "AI_full_document_index_0_100_v3",
        "AI_minus_Traditional_index",
        "AI_full_investment_type",
        "AI_full_strategic_importance",
        "AI_full_implementation_stage",
        "AI_full_time_horizon",
        "AI_full_digital_relevance",
        "AI_full_confidence",
        "YEAR_num",
        "YEAR_final",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # Keep rows with both indices for most preview figures
    both = df[
        df["Traditional_index_0_100"].notna()
        & df["AI_full_document_index_0_100_v3"].notna()
    ].copy()

    print("Rows total:", len(df))
    print("Rows with both indices:", len(both))

    # Figures
    save_hist(
        both,
        "Traditional_index_0_100",
        out_dir / "fig1_traditional_index_distribution.png",
        "Distribution of Traditional Keyword-Based Index",
        "Traditional index, 0–100",
    )

    save_hist(
        both,
        "AI_full_document_index_0_100_v3",
        out_dir / "fig2_ai_full_document_index_distribution.png",
        "Distribution of AI Full-Document Semantic Index",
        "AI full-document index, 0–100",
    )

    trend = save_year_trend(
        both,
        out_dir / "fig3_traditional_vs_ai_year_trend.png",
    )
    trend.to_csv(out_dir / "year_trend_traditional_ai_v1.csv", index=False, encoding="utf-8-sig")

    save_scatter(
        both,
        out_dir / "fig4_traditional_vs_ai_scatter.png",
    )

    save_gap_hist(
        both,
        out_dir / "fig5_ai_minus_traditional_gap_distribution.png",
    )

    make_summary_tables(both, out_dir)

    print("\nSaved preview outputs to:", out_dir)


if __name__ == "__main__":
    main()