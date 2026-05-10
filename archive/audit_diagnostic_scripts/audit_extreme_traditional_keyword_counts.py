# audit_extreme_traditional_keyword_counts.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
df = pd.read_csv(folder / "sec_digital_windows_traditional_scores_2018_2024.csv", low_memory=False)

count_cols = [
    "deployment_keyword_count",
    "capability_keyword_count",
    "ai_data_keyword_count",
    "risk_keyword_count",
    "traditional_deployment_score",
]

print("Summary:")
print(df[count_cols].describe())

base_cols = [
    "GVKEY", "tic", "CIK", "YEAR", "fqtr", "expected_form", "form",
    "filing_date", "report_date", "accession_number",
    "deployment_keyword_count", "capability_keyword_count",
    "ai_data_keyword_count",
    "risk_keyword_count",
    "traditional_deployment_score",
    "matched_keywords",
]

base_cols = [c for c in base_cols if c in df.columns]

for col in ["deployment_keyword_count", "capability_keyword_count", "ai_data_keyword_count", "risk_keyword_count", "traditional_deployment_score"]:
    print("\n" + "=" * 80)
    print(f"Top 30 by {col}")
    print("=" * 80)
    top = df.sort_values(col, ascending=False)[base_cols].head(30)
    print(top)

    out = folder / f"audit_top30_{col}.csv"
    top.to_csv(out, index=False, encoding="utf-8-sig")
    print("Saved:", out)

# Firm-level aggregation
firm = (
    df.groupby(["GVKEY", "tic"], dropna=False)[
        ["deployment_keyword_count", "capability_keyword_count", "ai_data_keyword_count", "risk_keyword_count"]
    ]
    .sum()
    .reset_index()
)

print("\nTop 30 firms by capability count:")
print(firm.sort_values("capability_keyword_count", ascending=False).head(30))

firm.to_csv(folder / "audit_firm_level_keyword_counts.csv", index=False, encoding="utf-8-sig")
print("\nSaved firm-level audit.")