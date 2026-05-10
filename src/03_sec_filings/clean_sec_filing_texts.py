# clean_sec_filing_texts_2018_2024.py

import pandas as pd
import re
import html
import warnings
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:
    pass

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

log_file = folder / "sec_filing_download_log_2018_2024.csv"

raw_dir = folder / "sec_filings_2018_2024" / "raw_html"
outdir = folder / "sec_filings_2018_2024" / "clean_text"
outdir.mkdir(parents=True, exist_ok=True)

log = pd.read_csv(log_file, low_memory=False)

success = log[log["status"].isin(["downloaded", "already_exists"])].copy()

print("Successful downloaded filings:", len(success))
print("Raw HTML directory:", raw_dir)
print("Clean text output directory:", outdir)

# --------------------------------------------------
# Fast HTML/XML to text
# --------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript).*?</\1>",
    flags=re.IGNORECASE | re.DOTALL
)
WHITESPACE_RE = re.compile(r"\s+")


def decode_bytes(raw):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def fast_html_to_text(path):
    raw = Path(path).read_bytes()
    text = decode_bytes(raw)

    # Remove scripts/styles
    text = SCRIPT_STYLE_RE.sub(" ", text)

    # Replace common structural tags with spaces before removing all tags
    text = re.sub(r"</(div|p|br|tr|td|th|li|table|section|article|header|footer)>", " ", text, flags=re.IGNORECASE)

    # Remove all remaining tags
    text = TAG_RE.sub(" ", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def clean_one(row):
    raw_path = Path(row["local_path"])

    if not raw_path.exists():
        return {
            "CIK": row.get("CIK"),
            "form": row.get("form"),
            "filing_date": row.get("filing_date"),
            "accession_number": row.get("accession_number"),
            "raw_path": str(raw_path),
            "clean_text_path": None,
            "text_length": None,
            "status": "raw_file_missing",
        }

    outname = raw_path.stem + ".txt"
    outpath = outdir / outname

    # Skip existing non-empty clean text
    if outpath.exists() and outpath.stat().st_size > 0:
        try:
            text_length = outpath.stat().st_size
        except Exception:
            text_length = None

        return {
            "CIK": row.get("CIK"),
            "form": row.get("form"),
            "filing_date": row.get("filing_date"),
            "accession_number": row.get("accession_number"),
            "raw_path": str(raw_path),
            "clean_text_path": str(outpath),
            "text_length": text_length,
            "status": "already_cleaned",
        }

    try:
        text = fast_html_to_text(raw_path)
        outpath.write_text(text, encoding="utf-8", errors="ignore")

        return {
            "CIK": row.get("CIK"),
            "form": row.get("form"),
            "filing_date": row.get("filing_date"),
            "accession_number": row.get("accession_number"),
            "raw_path": str(raw_path),
            "clean_text_path": str(outpath),
            "text_length": len(text),
            "status": "cleaned",
        }

    except Exception as e:
        return {
            "CIK": row.get("CIK"),
            "form": row.get("form"),
            "filing_date": row.get("filing_date"),
            "accession_number": row.get("accession_number"),
            "raw_path": str(raw_path),
            "clean_text_path": None,
            "text_length": None,
            "status": f"error_{type(e).__name__}: {e}",
        }


# --------------------------------------------------
# Multi-thread local cleaning
# --------------------------------------------------

rows = success.to_dict("records")
results = []

# Local CPU/file processing; safe to use more workers.
# If your computer becomes slow, reduce to 4.
MAX_WORKERS = 8

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(clean_one, row) for row in rows]

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Cleaning SEC filings fast"):
        results.append(fut.result())

clean_index = pd.DataFrame(results)

out = folder / "sec_clean_text_index_2018_2024.csv"
clean_index.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:", out)

print("\nStatus counts:")
print(clean_index["status"].value_counts(dropna=False))

print("\nText length summary:")
print(clean_index["text_length"].describe())

print("\nVery short files, text_length < 1000:")
short_files = clean_index[clean_index["text_length"].fillna(0) < 1000]
print(len(short_files))
if len(short_files) > 0:
    print(short_files.head(20))