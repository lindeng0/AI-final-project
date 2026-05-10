# check_ccm_link_file.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
path = folder / "external_data" / "ccm_gvkey_permco_link.csv"

ccm = pd.read_csv(path, low_memory=False)

print("Shape:", ccm.shape)
print("Columns:")
print(list(ccm.columns))

print("\nFirst 10 rows:")
print(ccm.head(10))

# normalize column names for quick diagnostics
cols = {c.lower(): c for c in ccm.columns}

for needed in ["gvkey", "lpermco", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"]:
    print(needed, needed in cols)

if "gvkey" in cols:
    print("\nUnique GVKEY:", ccm[cols["gvkey"]].nunique())

if "lpermco" in cols:
    print("Unique LPERMCO:", ccm[cols["lpermco"]].nunique())

if "linktype" in cols:
    print("\nlinktype counts:")
    print(ccm[cols["linktype"]].value_counts(dropna=False))

if "linkprim" in cols:
    print("\nlinkprim counts:")
    print(ccm[cols["linkprim"]].value_counts(dropna=False))