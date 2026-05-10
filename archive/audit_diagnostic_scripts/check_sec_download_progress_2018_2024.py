# check_sec_download_progress_2018_2024.py

import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

log_file = folder / "sec_filing_download_log_2018_2024.csv"
raw_dir = folder / "sec_filings_2018_2024" / "raw_html"

log = pd.read_csv(log_file, low_memory=False)

print("Log shape:", log.shape)

print("\nStatus counts:")
print(log["status"].value_counts(dropna=False))

files = list(raw_dir.glob("*"))
print("\nFiles in raw_html:", len(files))

nonempty = [p for p in files if p.stat().st_size > 0]
print("Non-empty files:", len(nonempty))

ok = log[log["status"].isin(["downloaded", "already_exists"])]
print("\nSuccessful rows:", len(ok))

failed = log[~log["status"].isin(["downloaded", "already_exists"])]
print("Failed rows:", len(failed))

if len(failed) > 0:
    print("\nFailed examples:")
    print(failed.head(30))