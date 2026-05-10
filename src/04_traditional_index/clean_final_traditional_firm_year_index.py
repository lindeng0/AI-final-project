# clean_final_traditional_firm_year_index_v3_1.py

import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 0. Paths
# ==================================================

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

in_file = folder / "final_traditional_firm_year_digital_index_v3.csv"

out_clean = folder / "final_traditional_firm_year_digital_index_v3_1_clean.csv"
out_analysis = folder / "analysis_ready_traditional_index_v3_1.csv"
out_missing = folder / "audit_no_text_firm_years_v3_1.csv"
out_audit = folder / "audit_traditional_index_v3_1_clean.xlsx"

# ==================================================
# 1. Helper functions
# ==================================================

def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def winsorize_by_sample(s, sample_mask, lower=0.01, upper=0.99):
    """
    Winsorize only using valid analysis sample.
    Non-sample rows remain NaN.
    """
    s = to_num(s)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    valid = sample_mask & s.notna()
    if valid.sum() == 0:
        return out

    lo = s.loc[valid].quantile(lower)
    hi = s.loc[valid].quantile(upper)

    out.loc[valid] = s.loc[valid].clip(lower=lo, upper=hi)
    return out


def z_by_year_valid(s, year, sample_mask):
    """
    Year-by-year z-score using only valid analysis rows.
    Non-sample rows remain NaN.
    """
    s = to_num(s)
    year = to_num(year)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    valid = sample_mask & s.notna() & year.notna()

    for y in sorted(year.loc[valid].unique()):
        idx = valid & (year == y)
        x = s.loc[idx]

        if x.notna().sum() > 2 and x.std() > 0:
            out.loc[idx] = (x - x.mean()) / x.std()
        else:
            out.loc[idx] = np.nan

    return out


def percentile_0_100_valid(s, sample_mask):
    """
    Percentile rank on valid analysis rows only.
    Non-sample rows remain NaN.
    """
    s = to_num(s)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    valid = sample_mask & s.notna()
    if valid.sum() > 0:
        out.loc[valid] = s.loc[valid].rank(pct=True) * 100

    return out


def safe_corr(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].corr()


# ==================================================
# 2. Load current v3
# ==================================================

df = pd.read_csv(in_file, low_memory=False)

print("Loaded:", in_file)
print("Shape:", df.shape)

# Basic numeric conversion
numeric_cols = [
    "YEAR",
    "n_filings",
    "total_words_v3",
    "deployment_count_v3",
    "capability_count_v3",
    "risk_count_v3",
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
    "deployment_tfidf_v3",
    "capability_tfidf_v3",
    "risk_tfidf_v3",
    "n_unique_keywords_total_v3",
    "n_unique_deployment_keywords_v3",
    "n_unique_capability_keywords_v3",
    "n_unique_risk_keywords_v3",
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = to_num(df[c])

# ==================================================
# 3. Define valid analysis sample
# ==================================================
# Important:
# n_filings = 0 means no usable SEC text was matched.
# This is missing coverage, not zero digitalization.

df["has_text_v3_1"] = (
    (df["n_filings"].fillna(0) > 0)
    & (df["total_words_v3"].notna())
    & (df["total_words_v3"] > 0)
)

df["main_analysis_sample_v3_1"] = df["has_text_v3_1"].astype(int)

# Filing coverage flags
df["full_four_filing_year_v3_1"] = (df["n_filings"] >= 4).astype(int)
df["partial_filing_year_v3_1"] = (
    (df["n_filings"] > 0) & (df["n_filings"] < 4)
).astype(int)
df["no_text_firm_year_v3_1"] = (~df["has_text_v3_1"]).astype(int)

print("\nSample coverage:")
print(df[["has_text_v3_1", "full_four_filing_year_v3_1", "partial_filing_year_v3_1", "no_text_firm_year_v3_1"]].sum())

# Save no-text firm-years for audit
no_text = df.loc[~df["has_text_v3_1"]].copy()
no_text.to_csv(out_missing, index=False, encoding="utf-8-sig")

print("\nSaved no-text firm-year audit:")
print(out_missing)
print("No-text firm-years:", no_text.shape[0])

# ==================================================
# 4. Force non-analysis rows to missing for score inputs
# ==================================================

sample_mask = df["has_text_v3_1"]

raw_input_cols = [
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
    "deployment_tfidf_v3",
    "capability_tfidf_v3",
    "risk_tfidf_v3",
]

for c in raw_input_cols:
    if c in df.columns:
        df.loc[~sample_mask, c] = np.nan

# ==================================================
# 5. Winsorize valid sample only
# ==================================================

winsor_inputs = [
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
    "deployment_tfidf_v3",
    "capability_tfidf_v3",
    "risk_tfidf_v3",
]

for c in winsor_inputs:
    if c in df.columns:
        df[f"{c}_winsor_v3_1"] = winsorize_by_sample(
            df[c],
            sample_mask,
            lower=0.01,
            upper=0.99
        )

# ==================================================
# 6. Recalculate year-standardized z-scores
# ==================================================

df["z_deployment_density_v3_1"] = z_by_year_valid(
    df["deployment_density_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

df["z_capability_density_v3_1"] = z_by_year_valid(
    df["capability_density_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

df["z_risk_density_v3_1"] = z_by_year_valid(
    df["risk_density_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

df["z_deployment_tfidf_v3_1"] = z_by_year_valid(
    df["deployment_tfidf_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

df["z_capability_tfidf_v3_1"] = z_by_year_valid(
    df["capability_tfidf_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

df["z_risk_tfidf_v3_1"] = z_by_year_valid(
    df["risk_tfidf_v3_winsor_v3_1"],
    df["YEAR"],
    sample_mask
)

# ==================================================
# 7. Rebuild traditional scores
# ==================================================
# Main interpretation:
# - deployment: customer-facing digital channel disclosure
# - capability: digital capability / infrastructure disclosure
# - risk: digital-related risk disclosure, not positive adoption
#
# Therefore:
# - core score excludes risk
# - risk-adjusted score penalizes risk
# - risk disclosure score is kept separately

df["traditional_core_score_v3_1"] = (
    df["z_deployment_density_v3_1"]
    + 0.25 * df["z_capability_density_v3_1"]
)

df["traditional_risk_adjusted_score_v3_1"] = (
    df["z_deployment_density_v3_1"]
    + 0.25 * df["z_capability_density_v3_1"]
    - 0.25 * df["z_risk_density_v3_1"]
)

df["traditional_deployment_only_score_v3_1"] = (
    df["z_deployment_density_v3_1"]
)

df["digital_risk_disclosure_score_v3_1"] = (
    df["z_risk_density_v3_1"]
)

df["traditional_tfidf_core_score_v3_1"] = (
    df["z_deployment_tfidf_v3_1"]
    + 0.25 * df["z_capability_tfidf_v3_1"]
)

df["traditional_tfidf_risk_adjusted_score_v3_1"] = (
    df["z_deployment_tfidf_v3_1"]
    + 0.25 * df["z_capability_tfidf_v3_1"]
    - 0.25 * df["z_risk_tfidf_v3_1"]
)

# 0-100 percentile versions, valid sample only
df["traditional_core_index_0_100_v3_1"] = percentile_0_100_valid(
    df["traditional_core_score_v3_1"],
    sample_mask
)

df["traditional_risk_adjusted_index_0_100_v3_1"] = percentile_0_100_valid(
    df["traditional_risk_adjusted_score_v3_1"],
    sample_mask
)

df["traditional_deployment_only_index_0_100_v3_1"] = percentile_0_100_valid(
    df["traditional_deployment_only_score_v3_1"],
    sample_mask
)

df["traditional_tfidf_core_index_0_100_v3_1"] = percentile_0_100_valid(
    df["traditional_tfidf_core_score_v3_1"],
    sample_mask
)

df["traditional_tfidf_risk_adjusted_index_0_100_v3_1"] = percentile_0_100_valid(
    df["traditional_tfidf_risk_adjusted_score_v3_1"],
    sample_mask
)

# ==================================================
# 8. Final recommended variables
# ==================================================

df["Traditional_score"] = df["traditional_core_score_v3_1"]
df["Traditional_index_0_100"] = df["traditional_core_index_0_100_v3_1"]

df["Traditional_score_risk_adjusted"] = df["traditional_risk_adjusted_score_v3_1"]
df["Traditional_index_risk_adjusted_0_100"] = df["traditional_risk_adjusted_index_0_100_v3_1"]

df["Traditional_score_tfidf"] = df["traditional_tfidf_core_score_v3_1"]
df["Digital_risk_disclosure_score"] = df["digital_risk_disclosure_score_v3_1"]

# ==================================================
# 9. Save clean full file and analysis-ready subset
# ==================================================

df = df.sort_values(["GVKEY", "YEAR"]).reset_index(drop=True)

df.to_csv(out_clean, index=False, encoding="utf-8-sig")

analysis = df.loc[df["main_analysis_sample_v3_1"] == 1].copy()
analysis.to_csv(out_analysis, index=False, encoding="utf-8-sig")

print("\nSaved clean full file:")
print(out_clean)
print("Shape:", df.shape)

print("\nSaved analysis-ready file:")
print(out_analysis)
print("Shape:", analysis.shape)

# ==================================================
# 10. Audit tables
# ==================================================

score_cols = [
    "Traditional_score",
    "Traditional_index_0_100",
    "Traditional_score_risk_adjusted",
    "Traditional_index_risk_adjusted_0_100",
    "Traditional_score_tfidf",
    "Digital_risk_disclosure_score",
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
    "deployment_tfidf_v3",
    "capability_tfidf_v3",
    "risk_tfidf_v3",
    "n_filings",
    "total_words_v3",
    "n_unique_keywords_total_v3",
]

score_cols = [c for c in score_cols if c in df.columns]

summary_stats = analysis[score_cols].describe().T

year_summary = (
    df.groupby("YEAR", dropna=False)
    .agg(
        n_firm_years=("GVKEY", "nunique"),
        n_analysis_sample=("main_analysis_sample_v3_1", "sum"),
        n_no_text=("no_text_firm_year_v3_1", "sum"),
        mean_n_filings=("n_filings", "mean"),
        mean_total_words=("total_words_v3", "mean"),
        mean_Traditional_score=("Traditional_score", "mean"),
        mean_Traditional_index_0_100=("Traditional_index_0_100", "mean"),
        mean_risk_disclosure_score=("Digital_risk_disclosure_score", "mean"),
    )
    .reset_index()
)

coverage_summary = pd.DataFrame({
    "metric": [
        "total_firm_year_rows",
        "analysis_sample_rows",
        "no_text_firm_year_rows",
        "unique_gvkey_total",
        "unique_gvkey_analysis_sample",
        "duplicate_firm_year_rows",
    ],
    "value": [
        df.shape[0],
        analysis.shape[0],
        no_text.shape[0],
        df["GVKEY"].nunique(),
        analysis["GVKEY"].nunique(),
        df.duplicated(subset=["GVKEY", "YEAR"]).sum(),
    ]
})

top_core = analysis.sort_values("Traditional_score", ascending=False).head(30)
bottom_core = analysis.sort_values("Traditional_score", ascending=True).head(30)
top_risk_adjusted = analysis.sort_values("Traditional_score_risk_adjusted", ascending=False).head(30)
top_tfidf = analysis.sort_values("Traditional_score_tfidf", ascending=False).head(30)
top_risk = analysis.sort_values("Digital_risk_disclosure_score", ascending=False).head(30)

corr_cols = [
    "Traditional_score",
    "Traditional_score_risk_adjusted",
    "Traditional_score_tfidf",
    "Digital_risk_disclosure_score",
    "traditional_deployment_only_score_v3_1",
]

corr_table = safe_corr(analysis, corr_cols)

with pd.ExcelWriter(out_audit, engine="openpyxl") as writer:
    coverage_summary.to_excel(writer, sheet_name="coverage_summary", index=False)
    summary_stats.to_excel(writer, sheet_name="summary_stats")
    year_summary.to_excel(writer, sheet_name="year_summary", index=False)
    corr_table.to_excel(writer, sheet_name="score_correlations")
    top_core.to_excel(writer, sheet_name="top_traditional_score", index=False)
    bottom_core.to_excel(writer, sheet_name="bottom_traditional_score", index=False)
    top_risk_adjusted.to_excel(writer, sheet_name="top_risk_adjusted", index=False)
    top_tfidf.to_excel(writer, sheet_name="top_tfidf", index=False)
    top_risk.to_excel(writer, sheet_name="top_risk_disclosure", index=False)
    no_text.to_excel(writer, sheet_name="no_text_firm_years", index=False)

print("\nSaved audit workbook:")
print(out_audit)

# ==================================================
# 11. Print preview
# ==================================================

print("\n==============================")
print("v3.1 traditional index complete")
print("==============================")

print("\nCoverage summary:")
print(coverage_summary)

print("\nYear summary:")
print(year_summary)

print("\nScore correlations:")
print(corr_table)

preview_cols = [
    "GVKEY", "tic", "YEAR", "n_filings",
    "deployment_count_v3", "capability_count_v3", "risk_count_v3",
    "deployment_density_v3", "capability_density_v3", "risk_density_v3",
    "Traditional_score",
    "Traditional_index_0_100",
    "Traditional_score_risk_adjusted",
    "Traditional_score_tfidf",
    "Digital_risk_disclosure_score",
]

preview_cols = [c for c in preview_cols if c in analysis.columns]

print("\nTop 20 Traditional_score:")
print(top_core[preview_cols].head(20))

print("\nTop 20 Digital_risk_disclosure_score:")
print(top_risk[preview_cols].head(20))

print("\nRecommended files:")
print("1.", out_clean)
print("2.", out_analysis)
print("3.", out_audit)

print("\nRecommended main variable:")
print("Traditional_score = traditional_core_score_v3_1")
print("Traditional_index_0_100 = traditional_core_index_0_100_v3_1")

print("\nDone.")