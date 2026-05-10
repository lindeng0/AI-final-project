# audit_missing_clean_text_coverage.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

df = pd.read_csv(folder / "sec_digital_windows_traditional_scores_2018_2024.csv", low_memory=False)

print("Shape:", df.shape)

df["has_clean_text"] = df["clean_text_path"].notna() & (df["clean_text_path"].astype(str).str.len() > 0)
df["has_score"] = df["traditional_deployment_score"].notna()

print("\nOverall:")
print("Rows:", len(df))
print("Has clean text:", df["has_clean_text"].sum())
print("Has traditional score:", df["has_score"].sum())
print("Has digital keywords:", df["has_digital_keywords"].sum())

print("\nCoverage among all rows:")
print(df[["has_clean_text", "has_score", "has_digital_keywords"]].mean())

print("\nCoverage among rows with clean text:")
clean = df[df["has_clean_text"]].copy()
print(clean[["has_score", "has_digital_keywords"]].mean())

print("\nMissing clean text by year:")
print(
    df.groupby("YEAR")["has_clean_text"]
    .agg(["mean", "sum", "count"])
)

print("\nMissing clean text by form:")
print(
    df.groupby("expected_form")["has_clean_text"]
    .agg(["mean", "sum", "count"])
)

print("\nRows missing clean text examples:")
cols = ["GVKEY", "tic", "CIK", "YEAR", "fqtr", "expected_form", "form", "filing_date", "report_date", "accession_number"]
cols = [c for c in cols if c in df.columns]
print(df[~df["has_clean_text"]][cols].head(50))

out = folder / "audit_missing_clean_text_coverage.csv"
df[~df["has_clean_text"]][cols].to_csv(out, index=False, encoding="utf-8-sig")
print("\nSaved missing examples:", out)