# build_final_traditional_firm_year_index_v3.py

import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 0. Paths
# ==================================================

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

score_file = folder / "sec_revised_traditional_digital_scores_v2_2018_2024.csv"
hit_file = folder / "sec_keyword_level_hits_v2_2018_2024.csv"
filing_keyword_summary_file = folder / "sec_filing_keyword_summary_v2_2018_2024.csv"

out_firm_year = folder / "final_traditional_firm_year_digital_index_v3.csv"
out_filing_level = folder / "final_traditional_filing_level_dedup_v3.csv"
out_audit_excel = folder / "audit_final_traditional_index_v3.xlsx"
out_duplicate_audit = folder / "audit_duplicate_accessions_v3.csv"

# ==================================================
# 1. Helper functions
# ==================================================

def clean_id_series(s):
    """
    Standardize IDs read from CSV.
    """
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def winsorize_series(s, lower=0.01, upper=0.99):
    """
    Winsorize a numeric pandas Series.
    """
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def z_by_year(series, year):
    """
    Year-by-year z-score.
    """
    series = pd.to_numeric(series, errors="coerce")
    return series.groupby(year).transform(
        lambda x: (x - x.mean()) / x.std()
        if x.notna().sum() > 2 and x.std() > 0 else np.nan
    )


def percentile_0_100(series):
    """
    Convert a score into 0-100 percentile scale.
    Higher = more digital disclosure.
    """
    series = pd.to_numeric(series, errors="coerce")
    return series.rank(pct=True) * 100


def safe_sum(x):
    return pd.to_numeric(x, errors="coerce").sum(min_count=1)


def join_unique(x):
    vals = []
    for v in x.dropna().astype(str):
        v = v.strip()
        if v and v.lower() not in ["nan", "none", "<na>"]:
            vals.append(v)
    return "; ".join(sorted(set(vals)))


# ==================================================
# 2. Load v2 score file
# ==================================================

print("Loading v2 traditional score file...")
df = pd.read_csv(score_file, low_memory=False)

print("Raw score file shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

required_cols = [
    "GVKEY", "YEAR", "CIK", "accession_number",
    "deployment_count_v2", "capability_count_v2", "risk_count_v2",
    "deployment_density_v2", "capability_density_v2", "risk_density_v2",
    "total_words_v2"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in score file: {missing}")

# Standardize key columns
df["GVKEY"] = clean_id_series(df["GVKEY"])
df["CIK"] = clean_id_series(df["CIK"])
df["accession_number"] = df["accession_number"].astype(str).str.strip()

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

# Keep usable rows
df = df.dropna(subset=["YEAR"])
df["YEAR"] = df["YEAR"].astype(int)

# ==================================================
# 3. Audit duplicate accession mappings
# ==================================================

dup_audit = (
    df.groupby(["CIK", "accession_number"], dropna=False)
    .agg(
        n_rows=("accession_number", "size"),
        n_gvkey=("GVKEY", "nunique"),
        n_year=("YEAR", "nunique"),
        gvkeys=("GVKEY", join_unique),
        years=("YEAR", lambda x: "; ".join(map(str, sorted(set(x.dropna()))))),
        tics=("tic", join_unique) if "tic" in df.columns else ("GVKEY", join_unique),
        forms=("expected_form", join_unique) if "expected_form" in df.columns else ("GVKEY", join_unique),
    )
    .reset_index()
    .sort_values(["n_rows", "n_gvkey", "n_year"], ascending=False)
)

dup_audit.to_csv(out_duplicate_audit, index=False, encoding="utf-8-sig")

print("\nDuplicate accession audit saved:")
print(out_duplicate_audit)

print("\nTop duplicate accession mappings:")
print(dup_audit.head(20))

# ==================================================
# 4. Deduplicate to filing-level
# ==================================================
# We deduplicate by CIK + accession_number because the same SEC filing
# may appear multiple times in the filing-quarter match table.

sort_cols = []
for c in ["GVKEY", "YEAR", "fqtr", "expected_form", "filing_date"]:
    if c in df.columns:
        sort_cols.append(c)

if sort_cols:
    df = df.sort_values(sort_cols)

filing = (
    df.drop_duplicates(subset=["CIK", "accession_number"], keep="first")
    .copy()
)

print("\nFiling-level deduplicated shape:", filing.shape)

# Ensure numeric columns
numeric_cols = [
    "deployment_count_v2", "capability_count_v2", "risk_count_v2",
    "deployment_density_v2", "capability_density_v2", "risk_density_v2",
    "total_words_v2",
    "traditional_digital_deployment_score_v2"
]

for c in numeric_cols:
    if c in filing.columns:
        filing[c] = pd.to_numeric(filing[c], errors="coerce")

filing.to_csv(out_filing_level, index=False, encoding="utf-8-sig")

print("Saved filing-level deduplicated file:")
print(out_filing_level)

# ==================================================
# 5. Aggregate to firm-year level
# ==================================================

group_keys = ["GVKEY", "YEAR"]

agg_dict = {
    "CIK": ("CIK", "first"),
    "tic": ("tic", "first") if "tic" in filing.columns else ("GVKEY", "first"),
    "n_filings": ("accession_number", "nunique"),
    "accessions": ("accession_number", join_unique),
    "total_words_v3": ("total_words_v2", safe_sum),
    "deployment_count_v3": ("deployment_count_v2", safe_sum),
    "capability_count_v3": ("capability_count_v2", safe_sum),
    "risk_count_v3": ("risk_count_v2", safe_sum),
    "mean_filing_v2_score": ("traditional_digital_deployment_score_v2", "mean")
        if "traditional_digital_deployment_score_v2" in filing.columns
        else ("deployment_density_v2", "mean"),
}

if "expected_form" in filing.columns:
    agg_dict["forms"] = ("expected_form", join_unique)

if "form" in filing.columns:
    agg_dict["raw_forms"] = ("form", join_unique)

firm_year = (
    filing.groupby(group_keys, dropna=False)
    .agg(**agg_dict)
    .reset_index()
)

# Recompute firm-year density from summed counts / summed words
firm_year["deployment_density_v3"] = (
    firm_year["deployment_count_v3"] / firm_year["total_words_v3"] * 1000
)

firm_year["capability_density_v3"] = (
    firm_year["capability_count_v3"] / firm_year["total_words_v3"] * 1000
)

firm_year["risk_density_v3"] = (
    firm_year["risk_count_v3"] / firm_year["total_words_v3"] * 1000
)

# ==================================================
# 6. Add keyword breadth and TF-IDF from keyword-level hit file
# ==================================================

print("\nLoading keyword-level hit file...")
if hit_file.exists():
    hit = pd.read_csv(hit_file, low_memory=False)
    print("Hit-level file shape:", hit.shape)

    # Standardize keys
    hit["GVKEY"] = clean_id_series(hit["GVKEY"])
    hit["YEAR"] = pd.to_numeric(hit["YEAR"], errors="coerce").astype("Int64")
    hit = hit.dropna(subset=["YEAR"])
    hit["YEAR"] = hit["YEAR"].astype(int)

    hit["keyword_count"] = pd.to_numeric(hit["keyword_count"], errors="coerce").fillna(0)
    hit["category"] = hit["category"].astype(str).str.strip()
    hit["keyword"] = hit["keyword"].astype(str).str.strip()

    # ------------------------------
    # 6A. Breadth: number of unique keywords by firm-year
    # ------------------------------
    breadth_total = (
        hit.groupby(group_keys, dropna=False)
        .agg(
            n_unique_keywords_total_v3=("keyword", "nunique"),
            n_keyword_records_v3=("keyword", "size"),
            total_keyword_hits_check_v3=("keyword_count", "sum"),
        )
        .reset_index()
    )

    breadth_cat = (
        hit.groupby(group_keys + ["category"], dropna=False)
        .agg(
            n_unique_keywords=("keyword", "nunique"),
            total_keyword_hits=("keyword_count", "sum")
        )
        .reset_index()
    )

    breadth_pivot_keywords = (
        breadth_cat.pivot_table(
            index=group_keys,
            columns="category",
            values="n_unique_keywords",
            fill_value=0,
            aggfunc="sum"
        )
        .reset_index()
    )

    breadth_pivot_keywords = breadth_pivot_keywords.rename(columns={
        "deployment": "n_unique_deployment_keywords_v3",
        "capability": "n_unique_capability_keywords_v3",
        "risk": "n_unique_risk_keywords_v3",
    })

    breadth_pivot_hits = (
        breadth_cat.pivot_table(
            index=group_keys,
            columns="category",
            values="total_keyword_hits",
            fill_value=0,
            aggfunc="sum"
        )
        .reset_index()
    )

    breadth_pivot_hits = breadth_pivot_hits.rename(columns={
        "deployment": "deployment_hits_check_v3",
        "capability": "capability_hits_check_v3",
        "risk": "risk_hits_check_v3",
    })

    firm_year = firm_year.merge(breadth_total, on=group_keys, how="left")
    firm_year = firm_year.merge(breadth_pivot_keywords, on=group_keys, how="left")
    firm_year = firm_year.merge(breadth_pivot_hits, on=group_keys, how="left")

    # Fill missing breadth values
    breadth_cols = [
        c for c in firm_year.columns
        if c.startswith("n_unique_") or c.endswith("_hits_check_v3")
    ]
    for c in breadth_cols:
        firm_year[c] = firm_year[c].fillna(0)

    # ------------------------------
    # 6B. TF-IDF by firm-year keyword
    # ------------------------------
    # TF = keyword count per 1,000 words at firm-year level
    # IDF = log((N + 1) / (df + 1)) + 1
    # where df = number of firm-years mentioning the keyword

    # Firm-year total words for merging
    fy_words = firm_year[group_keys + ["total_words_v3"]].copy()

    fy_kw = (
        hit.groupby(group_keys + ["category", "keyword"], dropna=False)
        .agg(keyword_count_fy=("keyword_count", "sum"))
        .reset_index()
    )

    fy_kw = fy_kw.merge(fy_words, on=group_keys, how="left")

    fy_kw["tf_per_1000_words"] = (
        fy_kw["keyword_count_fy"] / fy_kw["total_words_v3"] * 1000
    )

    N = firm_year[group_keys].drop_duplicates().shape[0]

    keyword_df = (
        fy_kw.groupby("keyword", dropna=False)
        .agg(df_firm_year=("GVKEY", "nunique"))
        .reset_index()
    )

    keyword_df["idf"] = np.log((N + 1) / (keyword_df["df_firm_year"] + 1)) + 1

    fy_kw = fy_kw.merge(keyword_df[["keyword", "idf"]], on="keyword", how="left")
    fy_kw["tfidf"] = fy_kw["tf_per_1000_words"] * fy_kw["idf"]

    tfidf_cat = (
        fy_kw.groupby(group_keys + ["category"], dropna=False)
        .agg(tfidf_sum=("tfidf", "sum"))
        .reset_index()
    )

    tfidf_pivot = (
        tfidf_cat.pivot_table(
            index=group_keys,
            columns="category",
            values="tfidf_sum",
            fill_value=0,
            aggfunc="sum"
        )
        .reset_index()
    )

    tfidf_pivot = tfidf_pivot.rename(columns={
        "deployment": "deployment_tfidf_v3",
        "capability": "capability_tfidf_v3",
        "risk": "risk_tfidf_v3",
    })

    firm_year = firm_year.merge(tfidf_pivot, on=group_keys, how="left")

    for c in ["deployment_tfidf_v3", "capability_tfidf_v3", "risk_tfidf_v3"]:
        if c not in firm_year.columns:
            firm_year[c] = 0
        firm_year[c] = firm_year[c].fillna(0)

else:
    print("Warning: keyword-level hit file not found. Skipping keyword breadth and TF-IDF.")
    firm_year["n_unique_keywords_total_v3"] = np.nan
    firm_year["deployment_tfidf_v3"] = np.nan
    firm_year["capability_tfidf_v3"] = np.nan
    firm_year["risk_tfidf_v3"] = np.nan

# ==================================================
# 7. Winsorize and construct final traditional scores
# ==================================================

density_cols = [
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
]

tfidf_cols = [
    "deployment_tfidf_v3",
    "capability_tfidf_v3",
    "risk_tfidf_v3",
]

for c in density_cols + tfidf_cols:
    if c in firm_year.columns:
        firm_year[f"{c}_winsor"] = winsorize_series(firm_year[c])

# Year-standardized density scores
firm_year["z_deployment_density_v3"] = z_by_year(
    firm_year["deployment_density_v3_winsor"],
    firm_year["YEAR"]
)

firm_year["z_capability_density_v3"] = z_by_year(
    firm_year["capability_density_v3_winsor"],
    firm_year["YEAR"]
)

firm_year["z_risk_density_v3"] = z_by_year(
    firm_year["risk_density_v3_winsor"],
    firm_year["YEAR"]
)

# Year-standardized TF-IDF scores
firm_year["z_deployment_tfidf_v3"] = z_by_year(
    firm_year["deployment_tfidf_v3_winsor"],
    firm_year["YEAR"]
)

firm_year["z_capability_tfidf_v3"] = z_by_year(
    firm_year["capability_tfidf_v3_winsor"],
    firm_year["YEAR"]
)

firm_year["z_risk_tfidf_v3"] = z_by_year(
    firm_year["risk_tfidf_v3_winsor"],
    firm_year["YEAR"]
)

# --------------------------------------------------
# Main traditional index
# --------------------------------------------------
# Interpretation:
# deployment = customer-facing digital deployment
# capability = infrastructure / strategic digital capability
# risk = digital-related risk disclosure, not adoption
#
# Main score excludes risk as a positive adoption signal.
# Risk-adjusted score penalizes risk disclosure.
# --------------------------------------------------

firm_year["traditional_core_score_v3"] = (
    firm_year["z_deployment_density_v3"]
    + 0.25 * firm_year["z_capability_density_v3"]
)

firm_year["traditional_risk_adjusted_score_v3"] = (
    firm_year["z_deployment_density_v3"]
    + 0.25 * firm_year["z_capability_density_v3"]
    - 0.25 * firm_year["z_risk_density_v3"]
)

firm_year["traditional_deployment_only_score_v3"] = (
    firm_year["z_deployment_density_v3"]
)

firm_year["digital_risk_disclosure_score_v3"] = (
    firm_year["z_risk_density_v3"]
)

# TF-IDF robustness version
firm_year["traditional_tfidf_core_score_v3"] = (
    firm_year["z_deployment_tfidf_v3"]
    + 0.25 * firm_year["z_capability_tfidf_v3"]
)

firm_year["traditional_tfidf_risk_adjusted_score_v3"] = (
    firm_year["z_deployment_tfidf_v3"]
    + 0.25 * firm_year["z_capability_tfidf_v3"]
    - 0.25 * firm_year["z_risk_tfidf_v3"]
)

# 0-100 percentile versions
firm_year["traditional_core_index_0_100_v3"] = percentile_0_100(
    firm_year["traditional_core_score_v3"]
)

firm_year["traditional_risk_adjusted_index_0_100_v3"] = percentile_0_100(
    firm_year["traditional_risk_adjusted_score_v3"]
)

firm_year["traditional_deployment_only_index_0_100_v3"] = percentile_0_100(
    firm_year["traditional_deployment_only_score_v3"]
)

firm_year["traditional_tfidf_core_index_0_100_v3"] = percentile_0_100(
    firm_year["traditional_tfidf_core_score_v3"]
)

firm_year["traditional_tfidf_risk_adjusted_index_0_100_v3"] = percentile_0_100(
    firm_year["traditional_tfidf_risk_adjusted_score_v3"]
)

# ==================================================
# 8. Sort and save final firm-year dataset
# ==================================================

firm_year = firm_year.sort_values(["GVKEY", "YEAR"]).reset_index(drop=True)

firm_year.to_csv(out_firm_year, index=False, encoding="utf-8-sig")

print("\nSaved final firm-year traditional digital index:")
print(out_firm_year)
print("Shape:", firm_year.shape)

# ==================================================
# 9. Audit tables
# ==================================================

summary_cols = [
    "n_filings",
    "total_words_v3",
    "deployment_count_v3",
    "capability_count_v3",
    "risk_count_v3",
    "deployment_density_v3",
    "capability_density_v3",
    "risk_density_v3",
    "traditional_core_score_v3",
    "traditional_risk_adjusted_score_v3",
    "traditional_core_index_0_100_v3",
    "traditional_risk_adjusted_index_0_100_v3",
    "traditional_tfidf_core_score_v3",
    "traditional_tfidf_risk_adjusted_score_v3",
]

summary_cols = [c for c in summary_cols if c in firm_year.columns]

summary_stats = firm_year[summary_cols].describe().T

year_summary = (
    firm_year.groupby("YEAR", dropna=False)
    .agg(
        n_firm_years=("GVKEY", "nunique"),
        mean_deployment_density=("deployment_density_v3", "mean"),
        mean_capability_density=("capability_density_v3", "mean"),
        mean_risk_density=("risk_density_v3", "mean"),
        mean_core_score=("traditional_core_score_v3", "mean"),
        mean_risk_adjusted_score=("traditional_risk_adjusted_score_v3", "mean"),
        mean_core_index_0_100=("traditional_core_index_0_100_v3", "mean"),
    )
    .reset_index()
)

top_core = (
    firm_year.sort_values("traditional_core_score_v3", ascending=False)
    .head(30)
)

top_risk_adjusted = (
    firm_year.sort_values("traditional_risk_adjusted_score_v3", ascending=False)
    .head(30)
)

top_risk = (
    firm_year.sort_values("digital_risk_disclosure_score_v3", ascending=False)
    .head(30)
)

bottom_core = (
    firm_year.sort_values("traditional_core_score_v3", ascending=True)
    .head(30)
)

# Correlation among main indices
corr_cols = [
    "traditional_core_score_v3",
    "traditional_risk_adjusted_score_v3",
    "traditional_deployment_only_score_v3",
    "traditional_tfidf_core_score_v3",
    "traditional_tfidf_risk_adjusted_score_v3",
    "digital_risk_disclosure_score_v3",
]

corr_cols = [c for c in corr_cols if c in firm_year.columns]
corr_table = firm_year[corr_cols].corr()

with pd.ExcelWriter(out_audit_excel, engine="openpyxl") as writer:
    summary_stats.to_excel(writer, sheet_name="summary_stats")
    year_summary.to_excel(writer, sheet_name="year_summary", index=False)
    corr_table.to_excel(writer, sheet_name="score_correlations")
    top_core.to_excel(writer, sheet_name="top_core_score", index=False)
    top_risk_adjusted.to_excel(writer, sheet_name="top_risk_adjusted", index=False)
    top_risk.to_excel(writer, sheet_name="top_risk_disclosure", index=False)
    bottom_core.to_excel(writer, sheet_name="bottom_core_score", index=False)
    dup_audit.head(500).to_excel(writer, sheet_name="duplicate_accessions", index=False)

print("\nSaved audit workbook:")
print(out_audit_excel)

# ==================================================
# 10. Print preview
# ==================================================

print("\n==============================")
print("Final traditional index preview")
print("==============================")

print("\nMain output:")
print(out_firm_year)

print("\nFirm-year shape:")
print(firm_year.shape)

print("\nSummary statistics:")
print(summary_stats)

print("\nYear summary:")
print(year_summary)

print("\nTop 20 traditional core score:")
preview_cols = [
    "GVKEY", "tic", "YEAR", "n_filings",
    "deployment_count_v3", "capability_count_v3", "risk_count_v3",
    "deployment_density_v3", "capability_density_v3", "risk_density_v3",
    "traditional_core_score_v3",
    "traditional_risk_adjusted_score_v3",
    "traditional_core_index_0_100_v3",
]

preview_cols = [c for c in preview_cols if c in firm_year.columns]
print(top_core[preview_cols].head(20))

print("\nTop 20 digital risk disclosure:")
print(top_risk[preview_cols].head(20))

print("\nRecommended main variables:")
print("1. traditional_core_score_v3")
print("2. traditional_core_index_0_100_v3")
print("3. traditional_risk_adjusted_score_v3")
print("4. traditional_tfidf_core_score_v3")
print("5. digital_risk_disclosure_score_v3")

print("\nDone.")