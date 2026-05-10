# check_crsp_frb_link_file.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

# Change this if your downloaded file has a different name
path = external / "crsp_frb_permco_rssd_link.csv"

frb = pd.read_csv(path, low_memory=False)

print("Shape:", frb.shape)
print("Columns:")
print(list(frb.columns))

print("\nFirst 10 rows:")
print(frb.head(10))

permco_candidates = [c for c in frb.columns if "permco" in c.lower()]
rssd_candidates = [c for c in frb.columns if "rssd" in c.lower()]

print("\nPERMCO candidate columns:", permco_candidates)
print("RSSD candidate columns:", rssd_candidates)

if permco_candidates:
    c = permco_candidates[0]
    print("\nUnique PERMCO:", frb[c].nunique())

if rssd_candidates:
    c = rssd_candidates[0]
    print("Unique RSSD:", frb[c].nunique())