# audit_sod_files.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

files = sorted(folder.glob("SOD_CustomDownload_ALL_*.csv"))

print("Files found:")
for f in files:
    print(" -", f.name)

print("\nNumber of files:", len(files))

col_sets = {}
row_counts = {}

for f in files:
    year = f.name.split("_")[-3]  # e.g. 2018
    header = pd.read_csv(f, nrows=0)
    cols = list(header.columns)
    col_sets[f.name] = cols

    # count rows
    n = sum(1 for _ in open(f, "r", encoding="utf-8", errors="replace")) - 1
    row_counts[f.name] = n

print("\nRow counts:")
for name, n in row_counts.items():
    print(f"{name}: {n:,}")

# compare columns with first file
base_name = files[0].name
base_cols = col_sets[base_name]

print(f"\nBase file: {base_name}")
print(f"Number of columns: {len(base_cols)}")

all_same = True

for name, cols in col_sets.items():
    if cols != base_cols:
        all_same = False
        print(f"\nColumn mismatch in {name}")
        print("Missing from this file:", sorted(set(base_cols) - set(cols)))
        print("Extra in this file:", sorted(set(cols) - set(base_cols)))

if all_same:
    print("\nAll files have identical columns.")
else:
    print("\nSome files have different columns. Fix this before merging.")