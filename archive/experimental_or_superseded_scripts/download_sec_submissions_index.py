# download_sec_submissions_index.py

import pandas as pd
import requests
import time
from pathlib import Path
from tqdm import tqdm

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
outdir = folder / "sec_filings"
outdir.mkdir(exist_ok=True)

targets_file = folder / "sec_filing_targets_2020_2022.csv"
targets = pd.read_csv(targets_file, low_memory=False)

# --------------------------------------------------
# IMPORTANT: replace with your real contact info
# SEC asks for a descriptive User-Agent with contact info.
# --------------------------------------------------

USER_AGENT = "BankDigitalDeploymentResearch/1.0 your_email@example.com"

headers = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

unique_ciks = (
    pd.to_numeric(targets["CIK"], errors="coerce")
    .dropna()
    .astype(int)
    .drop_duplicates()
    .sort_values()
)

rows = []

for cik in tqdm(unique_ciks, desc="Downloading SEC submissions index"):
    cik10 = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"

    try:
        r = requests.get(url, headers=headers, timeout=30)

        if r.status_code != 200:
            print(f"Failed CIK {cik}: status {r.status_code}")
            time.sleep(0.2)
            continue

        data = r.json()

        company_name = data.get("name")
        sec_cik = data.get("cik")

        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_doc_desc = recent.get("primaryDocDescription", [])

        for i in range(len(forms)):
            rows.append({
                "CIK": sec_cik,
                "CIK10": cik10,
                "company_name_sec": company_name,
                "form": forms[i],
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                "report_date": report_dates[i] if i < len(report_dates) else None,
                "accession_number": accession_numbers[i] if i < len(accession_numbers) else None,
                "primary_document": primary_docs[i] if i < len(primary_docs) else None,
                "primary_doc_description": primary_doc_desc[i] if i < len(primary_doc_desc) else None,
            })

        # Polite delay. Keep below SEC rate limits.
        time.sleep(0.12)

    except Exception as e:
        print(f"Error CIK {cik}: {e}")
        time.sleep(0.5)

index = pd.DataFrame(rows)

# Keep relevant forms
index = index[index["form"].isin(["10-K", "10-Q"])].copy()

index["filing_date"] = pd.to_datetime(index["filing_date"], errors="coerce")
index["report_date"] = pd.to_datetime(index["report_date"], errors="coerce")
index["report_year"] = index["report_date"].dt.year
index["report_month"] = index["report_date"].dt.month

out = folder / "sec_10k_10q_submissions_index.csv"
index.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:", out)
print("Index shape:", index.shape)
print("\nForm counts:")
print(index["form"].value_counts(dropna=False))
print("\nFiling date range:")
print(index["filing_date"].min(), index["filing_date"].max())