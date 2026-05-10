# audit_revised_v2_top_scores.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

df = pd.read_csv(folder / "sec_revised_traditional_digital_scores_v2_2018_2024.csv", low_memory=False)

cols = [
    "GVKEY", "tic", "CIK", "YEAR", "fqtr", "datadate",
    "expected_form", "form", "filing_date", "report_date",
    "accession_number",
    "deployment_count_v2", "capability_count_v2", "risk_count_v2",
    "deployment_density_v2", "capability_density_v2", "risk_density_v2",
    "traditional_customer_channel_intensity_v2",
    "traditional_capability_intensity_v2",
    "traditional_risk_intensity_v2",
    "traditional_digital_deployment_score_v2",
]
cols = [c for c in cols if c in df.columns]

top_score = df.sort_values("traditional_digital_deployment_score_v2", ascending=False)[cols].head(100)
top_dep = df.sort_values("deployment_count_v2", ascending=False)[cols].head(100)
top_risk = df.sort_values("risk_count_v2", ascending=False)[cols].head(100)

out_xlsx = folder / "audit_revised_v2_top_scores.xlsx"

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    top_score.to_excel(writer, sheet_name="top_composite_score", index=False)
    top_dep.to_excel(writer, sheet_name="top_deployment_count", index=False)
    top_risk.to_excel(writer, sheet_name="top_risk_count", index=False)

print("Saved:", out_xlsx)

print("\nTop composite score:")
print(top_score.head(20))

print("\nTop deployment count:")
print(top_dep.head(20))

print("\nTop risk count:")
print(top_risk.head(20))