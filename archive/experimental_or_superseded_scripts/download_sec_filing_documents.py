# download_sec_filing_documents_8rps.py

import pandas as pd
import requests
import time
import re
import threading
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
matches_file = folder / "sec_filing_matches_2020_2022.csv"

outdir = folder / "sec_filings" / "raw_html"
outdir.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(matches_file, low_memory=False)

# ==================================================
# IMPORTANT: replace this with your real email/contact
# ==================================================
USER_AGENT = "BankDigitalDeploymentResearch/1.0 your_email@example.com"

headers = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Keep unique filings only
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

# --------------------------------------------------
# Global rate limiter: target 8 requests / second
# --------------------------------------------------

RATE_LIMIT_PER_SECOND = 8
MIN_INTERVAL = 1.0 / RATE_LIMIT_PER_SECOND

rate_lock = threading.Lock()
last_request_time = [0.0]

def wait_for_rate_limit():
    """
    Enforce a global request rate across all threads.
    Adds small jitter to avoid burst-like timing.
    """
    with rate_lock:
        now = time.time()
        elapsed = now - last_request_time[0]

        sleep_needed = MIN_INTERVAL - elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)

        # Small jitter: prevents perfectly periodic traffic
        time.sleep(random.uniform(0.005, 0.025))

        last_request_time[0] = time.time()

# --------------------------------------------------
# Thread-local sessions
# --------------------------------------------------

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(headers)
        thread_local.session = session
    return thread_local.session

# --------------------------------------------------
# Download function
# --------------------------------------------------

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
            "primary_document": primary_doc,
            "url": url,
            "local_path": str(outpath),
            "status": "already_exists",
            "bytes": outpath.stat().st_size,
        }

    session = get_session()
    max_retries = 4
    status = None
    nbytes = 0

    for attempt in range(1, max_retries + 1):
        try:
            wait_for_rate_limit()
            r = session.get(url, timeout=60)

            if r.status_code == 200 and len(r.content) > 0:
                outpath.write_bytes(r.content)
                status = "downloaded"
                nbytes = len(r.content)
                break

            elif r.status_code in [429, 403]:
                status = f"rate_limited_or_forbidden_{r.status_code}_attempt_{attempt}"
                wait_time = 20 * attempt
                time.sleep(wait_time)

            elif r.status_code in [500, 502, 503, 504]:
                status = f"server_error_{r.status_code}_attempt_{attempt}"
                wait_time = 5 * attempt
                time.sleep(wait_time)

            else:
                status = f"failed_status_{r.status_code}_attempt_{attempt}"
                time.sleep(2 * attempt)

        except requests.exceptions.Timeout:
            status = f"timeout_attempt_{attempt}"
            time.sleep(5 * attempt)

        except Exception as e:
            status = f"error_{type(e).__name__}_attempt_{attempt}"
            time.sleep(5 * attempt)

    return {
        "CIK": cik,
        "form": form,
        "filing_date": filing_date,
        "accession_number": accession,
        "primary_document": primary_doc,
        "url": url,
        "local_path": str(outpath),
        "status": status,
        "bytes": nbytes,
    }

# --------------------------------------------------
# Run downloads
# --------------------------------------------------

rows = filings.to_dict("records")
results = []

# 6 workers is enough because global limiter controls total request rate.
# More workers help with waiting/latency, but will not exceed 8 req/s.
MAX_WORKERS = 6

checkpoint_log = folder / "sec_filing_download_log_checkpoint.csv"
final_log = folder / "sec_filing_download_log.csv"

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_one, row) for row in rows]

    for i, fut in enumerate(
        tqdm(as_completed(futures), total=len(futures), desc="Downloading SEC filings at <=8 req/s"),
        start=1
    ):
        try:
            result = fut.result()
        except Exception as e:
            result = {
                "CIK": None,
                "form": None,
                "filing_date": None,
                "accession_number": None,
                "primary_document": None,
                "url": None,
                "local_path": None,
                "status": f"future_error_{type(e).__name__}",
                "bytes": 0,
            }

        results.append(result)

        # Save checkpoint every 200 completed futures
        if i % 200 == 0:
            pd.DataFrame(results).to_csv(checkpoint_log, index=False, encoding="utf-8-sig")

log = pd.DataFrame(results)
log.to_csv(final_log, index=False, encoding="utf-8-sig")

print("\nSaved final log:")
print(final_log)

print("\nStatus counts:")
print(log["status"].value_counts(dropna=False))

print("\nSuccessful downloads or already existing:")
print(log["status"].isin(["downloaded", "already_exists"]).sum(), "/", len(log))

print("\nTotal downloaded bytes:")
print(log["bytes"].sum())