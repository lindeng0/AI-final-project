# diagnose_multiple_rssd_mappings.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"

cw = pd.read_csv(external / "gvkey_permco_rssd_link.csv", low_memory=False)

cw["GVKEY"] = cw["GVKEY"].astype("string").str.strip()

multi = (
    cw.dropna(subset=["RSSDID"])
    .groupby("GVKEY")["RSSDID"]
    .nunique()
    .reset_index(name="n_rssdid")
)

multi_gvkeys = multi[multi["n_rssdid"] > 1]["GVKEY"]

multi_detail = cw[cw["GVKEY"].isin(multi_gvkeys)].copy()

sort_cols = [c for c in ["GVKEY", "LINKDT", "LINKENDDT", "RSSD_START_DATE", "RSSD_END_DATE"] if c in multi_detail.columns]
multi_detail = multi_detail.sort_values(sort_cols)

print("GVKEYs with multiple RSSDID:", len(multi_gvkeys))
print("Rows:", multi_detail.shape)

display_cols = [
    "GVKEY", "CONM", "TIC", "PERMCO", "RSSDID", "RSSDHCR",
    "FRB_NAME", "FRB_INST_TYPE",
    "LINKDT", "LINKENDDT", "RSSD_START_DATE", "RSSD_END_DATE", "LINK_OVERLAP"
]

display_cols = [c for c in display_cols if c in multi_detail.columns]

print(multi_detail[display_cols].head(100))

out = folder / "diagnostic_multiple_rssd_mappings.csv"
multi_detail.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out)