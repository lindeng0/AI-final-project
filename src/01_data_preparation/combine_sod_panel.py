# combine_sod_panel.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
files = sorted(folder.glob("SOD_CustomDownload_ALL_*.csv"))

print("Files found:")
for f in files:
    print(" -", f.name)

if len(files) == 0:
    raise FileNotFoundError("No SOD_CustomDownload_ALL_*.csv files found.")

# --------------------------------------------------
# Selected variables
# --------------------------------------------------

selected_cols = [
    # identifiers
    "YEAR", "CERT", "UNINUMBR", "BRNUM",

    # bank-level info
    "NAMEFULL", "BKCLASS", "ASSET", "DEPDOM", "DEPSUM",
    "STALP", "STNAME", "STCNTY", "CITY", "ZIP",
    "RSSDID", "CALL", "CHARTER", "CHRTAGNT",

    # branch-level info
    "NAMEBR", "ADDRESBR", "CITYBR", "STALPBR", "STNAMEBR", "ZIPBR",
    "CNTYNAMB", "CNTYNUMB", "STCNTYBR",
    "DEPSUMBR", "BRSERTYP",

    # geography
    "SIMS_LATITUDE", "SIMS_LONGITUDE",
    "MSABR", "MSANAMB", "METROBR", "MICROBR",
    "CSABR", "CSANAMBR",

    # branch dates / status
    "SIMS_ESTABLISHED_DATE", "SIMS_ACQUIRED_DATE", "SIMS_DESCRIPTION",

    # holding company
    "NAMEHCR", "RSSDHCR", "HCTMULT",
]

# --------------------------------------------------
# Read and combine files
# --------------------------------------------------

dfs = []

for f in files:
    print(f"\nReading {f.name}...")

    # Check columns in current file
    header_cols = pd.read_csv(f, nrows=0).columns.tolist()
    available_cols = [c for c in selected_cols if c in header_cols]
    missing_cols = [c for c in selected_cols if c not in header_cols]

    if missing_cols:
        print("Warning: these selected columns are missing from this file:")
        print(missing_cols)

    df = pd.read_csv(
        f,
        usecols=available_cols,
        low_memory=False
    )

    # Ensure YEAR exists and is numeric
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    # Standardize ID columns as strings
    id_cols = [
        "CERT", "UNINUMBR", "BRNUM", "RSSDID", "RSSDHCR",
        "STCNTY", "STCNTYBR", "MSABR"
    ]

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Standardize state codes
    for col in ["STALP", "STALPBR"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()

    # Numeric variables
    numeric_cols = [
        "ASSET", "DEPDOM", "DEPSUM", "DEPSUMBR",
        "SIMS_LATITUDE", "SIMS_LONGITUDE",
        "METROBR", "MICROBR"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs.append(df)

panel = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# Final checks
# --------------------------------------------------

print("\nCombined shape:", panel.shape)
print("Years:", sorted(panel["YEAR"].dropna().unique()))

print("\nChecking duplicated branch-year observations...")
dup = panel.duplicated(subset=["YEAR", "UNINUMBR"], keep=False)
print("Duplicated YEAR-UNINUMBR rows:", dup.sum())

print("\nChecking required columns for COVID exposure:")
required_for_covid = ["CERT", "YEAR", "UNINUMBR", "STCNTYBR", "STALPBR", "STCNTY", "STALP"]
for col in required_for_covid:
    print(col, col in panel.columns)

# --------------------------------------------------
# Save
# --------------------------------------------------

out_parquet = folder / "sod_branch_panel_2018_2025.parquet"
out_csv = folder / "sod_branch_panel_2018_2025.csv"

panel.to_parquet(out_parquet, index=False)
panel.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out_parquet)
print(out_csv)