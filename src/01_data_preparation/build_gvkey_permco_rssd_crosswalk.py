# build_gvkey_permco_rssd_crosswalk.py

import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

ccm_file = external / "ccm_gvkey_permco_link_clean.csv"
frb_file = external / "crsp_frb_permco_rssd_link.csv"

ccm = pd.read_csv(ccm_file, low_memory=False)
frb = pd.read_csv(frb_file, low_memory=False)

print("CCM shape:", ccm.shape)
print("CCM columns:", list(ccm.columns))

print("\nFRB shape:", frb.shape)
print("FRB columns:", list(frb.columns))

# --------------------------------------------------
# 1. Standardize CCM
# --------------------------------------------------

ccm.columns = [c.strip().upper() for c in ccm.columns]

if "GVKEY" not in ccm.columns or "LPERMCO" not in ccm.columns:
    raise ValueError("CCM file must contain GVKEY and LPERMCO.")

ccm["GVKEY"] = ccm["GVKEY"].astype("string").str.strip()
ccm["PERMCO"] = pd.to_numeric(ccm["LPERMCO"], errors="coerce").astype("Int64")

if "LPERMNO" in ccm.columns:
    ccm["PERMNO"] = pd.to_numeric(ccm["LPERMNO"], errors="coerce").astype("Int64")

if "LINKDT" in ccm.columns:
    ccm["LINKDT"] = pd.to_datetime(ccm["LINKDT"], errors="coerce")

if "LINKENDDT" in ccm.columns:
    ccm["LINKENDDT"] = pd.to_datetime(ccm["LINKENDDT"], errors="coerce")

if "LINKENDDT_FILLED" in ccm.columns:
    ccm["LINKENDDT_FILLED"] = pd.to_datetime(ccm["LINKENDDT_FILLED"], errors="coerce")
else:
    ccm["LINKENDDT_FILLED"] = ccm["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))

# --------------------------------------------------
# 2. Standardize CRSP-FRB
# --------------------------------------------------

frb.columns = [c.strip().lower() for c in frb.columns]

required_frb = ["permco", "entity"]
missing = [c for c in required_frb if c not in frb.columns]
if missing:
    raise ValueError("FRB file missing required columns: " + str(missing))

frb = frb.rename(
    columns={
        "permco": "PERMCO",
        "entity": "RSSDID",
        "name": "FRB_NAME",
        "inst_type": "FRB_INST_TYPE",
        "dt_start": "RSSD_START_DATE",
        "dt_end": "RSSD_END_DATE",
    }
)

frb["PERMCO"] = pd.to_numeric(frb["PERMCO"], errors="coerce").astype("Int64")
frb["RSSDID"] = pd.to_numeric(frb["RSSDID"], errors="coerce").astype("Int64")

frb["RSSD_START_DATE"] = pd.to_datetime(
    frb["RSSD_START_DATE"].astype("string"),
    format="%Y%m%d",
    errors="coerce"
)

frb["RSSD_END_DATE"] = pd.to_datetime(
    frb["RSSD_END_DATE"].astype("string"),
    format="%Y%m%d",
    errors="coerce"
)

frb["RSSD_END_DATE_FILLED"] = frb["RSSD_END_DATE"].fillna(pd.Timestamp("2099-12-31"))

# --------------------------------------------------
# 3. Merge GVKEY → PERMCO → RSSDID
# --------------------------------------------------

crosswalk = ccm.merge(
    frb,
    on="PERMCO",
    how="left",
    validate="many_to_many"
)

# For our HCR exposure file, RSSDID is the candidate RSSDHCR.
# Documentation says CRSP-FRB typically matches PERMCOs to the highest corporate parent.
crosswalk["RSSDHCR"] = crosswalk["RSSDID"]

# --------------------------------------------------
# 4. Optional overlap check between CCM and FRB link windows
# --------------------------------------------------

if "LINKDT" in crosswalk.columns and "RSSD_START_DATE" in crosswalk.columns:
    crosswalk["LINK_OVERLAP"] = (
        crosswalk["LINKDT"].fillna(pd.Timestamp("1900-01-01"))
        <= crosswalk["RSSD_END_DATE_FILLED"].fillna(pd.Timestamp("2099-12-31"))
    ) & (
        crosswalk["RSSD_START_DATE"].fillna(pd.Timestamp("1900-01-01"))
        <= crosswalk["LINKENDDT_FILLED"].fillna(pd.Timestamp("2099-12-31"))
    )
else:
    crosswalk["LINK_OVERLAP"] = True

# --------------------------------------------------
# 5. Diagnostics
# --------------------------------------------------

print("\nCrosswalk shape:", crosswalk.shape)

print("\nUnique counts:")
print("GVKEY:", crosswalk["GVKEY"].nunique())
print("PERMCO:", crosswalk["PERMCO"].nunique())
print("RSSDID:", crosswalk["RSSDID"].nunique())

print("\nRows missing RSSDID:", crosswalk["RSSDID"].isna().sum())
print("Rows with overlapping link windows:", crosswalk["LINK_OVERLAP"].sum())

multi_rssd = (
    crosswalk.dropna(subset=["RSSDID"])
    .groupby("GVKEY")["RSSDID"]
    .nunique()
    .reset_index(name="n_rssdid")
)

print("\nGVKEYs linked to multiple RSSDID:", (multi_rssd["n_rssdid"] > 1).sum())

print("\nFRB institution type counts:")
if "FRB_INST_TYPE" in crosswalk.columns:
    print(crosswalk["FRB_INST_TYPE"].value_counts(dropna=False))

# --------------------------------------------------
# 6. Keep useful columns
# --------------------------------------------------

keep_cols = [
    "GVKEY",
    "CONM",
    "TIC",
    "CUSIP",
    "CIK",
    "SIC",
    "NAICS",
    "PERMCO",
    "PERMNO",
    "LPERMCO",
    "LPERMNO",
    "LINKDT",
    "LINKENDDT",
    "LINKTYPE",
    "LINKPRIM",
    "RSSDID",
    "RSSDHCR",
    "FRB_NAME",
    "FRB_INST_TYPE",
    "RSSD_START_DATE",
    "RSSD_END_DATE",
    "LINK_OVERLAP",
]

keep_cols = [c for c in keep_cols if c in crosswalk.columns]
crosswalk = crosswalk[keep_cols].drop_duplicates()

# Prefer rows that have RSSDID and overlapping date windows
crosswalk = crosswalk.sort_values(
    by=["GVKEY", "PERMCO", "RSSDID", "LINK_OVERLAP"],
    ascending=[True, True, True, False]
)

# --------------------------------------------------
# 7. Save
# --------------------------------------------------

out = external / "gvkey_permco_rssd_link.csv"
crosswalk.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out)

print("\nPreview:")
print(crosswalk.head(20))