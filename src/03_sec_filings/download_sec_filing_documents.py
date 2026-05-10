# download_sec_filing_documents_2018_2024_fast8.py

import pandas as pd
import requests
import time
import re
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
matches_file = folder / "sec_filing_matches_2018_2024.csv"

outdir = folder / "sec_filings_2018_2024" / "raw_html"
outdir.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(matches_file, low_memory=False)

# IMPORTANT: replace with your real email
USER_AGENT = "BankDigitalDeploymentResearch/1.0 your_email@example.com"

headers = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

filings = (
    matches.dropna(subset=["sec_document_url", "accession_number"])
    .drop_duplicates(subset=["CIK", "accession_number", "primary_document"])
    .copy()
)

print("Unique filings to download:", len(filings))

def safe_filename(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180]

# Global rate limiter: target 8 requests / second
lock = threading.Lock()
last_request_time = [0.0]
MIN_INTERVAL = 1.0 / 8.0

def wait_for_rate_limit():
    with lock:
        now = time.time()
        elapsed = now - last_request_time[0]
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        last_request_time[0] = time.time()

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(headers)
        thread_local.session = session
    return thread_local.session

def download_one(row):
    cik = int(row["CIK"])
    form = str(row["form"])
    filing_date = str(row["filing_date"])[:10]
    accession = str(row["accession_number"])
    primary_doc = str(row["primary_document"])
    url = row["sec_document_url"]

    fname = safe_filename(f"{cik}_{form}_{filing_date}_{accession}_{primary_doc}")
    outpath = outdir / fname

    if outpath.exists() and outpath.stat().st_size > 0:
        return {
            "CIK": cik,
            "form": form,
            "filing_date": filing_date,
            "accession_number": accession,
            "url": url,
            "local_path": str(outpath),
            "status": "already_exists",
        }

    session = get_session()
    max_retries = 4
    status = None

    for attempt in range(1, max_retries + 1):
        try:
            wait_for_rate_limit()
            r = session.get(url, timeout=60)

            if r.status_code == 200 and len(r.content) > 0:
                outpath.write_bytes(r.content)
                status = "downloaded"
                break

            elif r.status_code in [429, 403]:
                status = f"rate_limited_or_forbidden_{r.status_code}_attempt_{attempt}"
                # Back off more strongly if SEC complains
                time.sleep(30 * attempt)

            else:
                status = f"failed_status_{r.status_code}_attempt_{attempt}"
                time.sleep(2 * attempt)

        except Exception as e:
            status = f"error_{type(e).__name__}_attempt_{attempt}"
            time.sleep(3 * attempt)

    return {
        "CIK": cik,
        "form": form,
        "filing_date": filing_date,
        "accession_number": accession,
        "url": url,
        "local_path": str(outpath),
        "status": status,
    }

rows = filings.to_dict("records")
results = []

# 8 workers is okay because global limiter controls total speed.
MAX_WORKERS = 8

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_one, row) for row in rows]

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading SEC filings at <=8 req/s"):
        try:
            results.append(fut.result())
        except Exception as e:
            results.append({"status": f"future_error_{type(e).__name__}", "error": str(e)})

log = pd.DataFrame(results)
outlog = folder / "sec_filing_download_log_2018_2024.csv"
log.to_csv(outlog, index=False, encoding="utf-8-sig")

print("\nSaved log:", outlog)
print(log["status"].value_counts(dropna=False))

failed = log[~log["status"].isin(["downloaded", "already_exists"])]
print("\nFailed rows:", len(failed))
if len(failed) > 0:
    print(failed.head(20))