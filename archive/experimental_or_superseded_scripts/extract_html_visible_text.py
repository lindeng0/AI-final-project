# ============================================================
# extract_html_visible_text_v1_1_parallel.py
# Purpose:
#   Convert raw SEC HTML / Inline XBRL filings into improved visible clean text.
#
# Key features:
#   - Parallel processing
#   - Resume support
#   - Per-file audit
#   - Suitable for thousands of SEC filings
# ============================================================

import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_block_from_sec_wrapper(raw: str) -> str:
    m = re.search(r"<TEXT>(.*)</TEXT>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    return raw


def remove_invisible_and_xbrl_nodes(soup: BeautifulSoup):
    for tag in soup.find_all(["script", "style", "noscript", "meta", "link"]):
        tag.decompose()

    for tag in soup.find_all():
        tag_name = tag.name.lower() if tag.name else ""

        if (
            tag_name.startswith("ix:")
            or tag_name.startswith("xbrli:")
            or tag_name.startswith("xbrldi:")
            or tag_name.startswith("link:")
            or tag_name.startswith("xlink:")
            or tag_name in {"ix:hidden", "ix:header", "ix:references", "ix:resources"}
        ):
            tag.decompose()
            continue

        style = tag.get("style", "")
        if isinstance(style, str):
            low_style = style.lower().replace(" ", "")
            if "display:none" in low_style or "visibility:hidden" in low_style:
                tag.decompose()
                continue

        if tag.has_attr("hidden"):
            tag.decompose()
            continue


def normalize_html_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("&#160;", " ")
    text = text.replace("&nbsp;", " ")

    text = text.replace("&#8217;", "'")
    text = text.replace("&#8220;", '"')
    text = text.replace("&#8221;", '"')
    text = text.replace("&#8211;", "-")
    text = text.replace("&#8212;", "-")

    # Remove remaining XBRL/taxonomy-like tokens
    text = re.sub(r"\b[a-zA-Z]{2,20}:[A-Za-z0-9_\-\.]+", " ", text)

    # Remove very long technical IDs
    text = re.sub(r"\b[A-Za-z0-9_\-]{35,}\b", " ", text)

    # Preserve important section boundaries
    section_patterns = [
        r"Item\s+1A\.",
        r"Item\s+1B\.",
        r"Item\s+1\.",
        r"Item\s+7A\.",
        r"Item\s+7\.",
        r"Item\s+8\.",
        r"ITEM\s+1A\.",
        r"ITEM\s+1B\.",
        r"ITEM\s+1\.",
        r"ITEM\s+7A\.",
        r"ITEM\s+7\.",
        r"ITEM\s+8\.",
    ]

    for p in section_patterns:
        text = re.sub(rf"\b({p})", r"\n\n\1", text, flags=re.IGNORECASE)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        # Drop numeric/table-heavy lines
        if len(line) > 20:
            alpha_ratio = sum(ch.isalpha() for ch in line) / max(len(line), 1)
            digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)

            if alpha_ratio < 0.25 and digit_ratio > 0.35:
                continue

        # Drop obvious SEC XML namespace remnants
        low = line.lower()
        if low.startswith(("xmlns:", "contextref=", "unitref=", "decimals=", "schemaref")):
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def html_to_visible_text(raw: str) -> str:
    html = extract_text_block_from_sec_wrapper(raw)

    # lxml is faster and more robust for messy SEC HTML
    soup = BeautifulSoup(html, "lxml")

    remove_invisible_and_xbrl_nodes(soup)

    # Add line breaks around block-ish elements
    for tag in soup.find_all(["div", "p", "tr", "table", "br", "hr", "li", "h1", "h2", "h3"]):
        tag.append("\n")

    text = soup.get_text(separator="\n")
    text = normalize_html_text(text)

    return text


def parse_metadata_from_filename(path: Path) -> dict:
    name = path.name
    stem = path.stem
    parts = stem.split("_")

    cik = parts[0] if len(parts) >= 1 else ""
    form = parts[1] if len(parts) >= 2 else ""
    filing_date = parts[2] if len(parts) >= 3 else ""

    return {
        "file_name": name,
        "output_txt_name": stem + ".txt",
        "cik": cik.lstrip("0"),
        "form": form,
        "filing_date": filing_date,
    }


def process_one_file(args_tuple):
    """
    Worker function. Must be top-level for multiprocessing on Windows.
    """
    file_path_str, output_dir_str, resume = args_tuple

    path = Path(file_path_str)
    output_dir = Path(output_dir_str)

    meta = parse_metadata_from_filename(path)
    out_path = output_dir / meta["output_txt_name"]

    row = dict(meta)
    row["input_path"] = str(path)
    row["output_path"] = str(out_path)

    if resume and out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = out_path.read_text(encoding="utf-8", errors="ignore")
            row.update({
                "status": "skipped_existing",
                "raw_char_len": None,
                "visible_char_len": len(existing),
                "visible_word_count_rough": len(re.findall(r"\b\w+\b", existing)),
                "contains_us_gaap_after_clean": int("us-gaap:" in existing.lower()),
                "contains_xbrli_after_clean": int("xbrli:" in existing.lower()),
                "error": "",
            })
            return row
        except Exception:
            pass

    try:
        raw = read_file(path)
        visible_text = html_to_visible_text(raw)

        out_path.write_text(visible_text, encoding="utf-8")

        row.update({
            "status": "ok",
            "raw_char_len": len(raw),
            "visible_char_len": len(visible_text),
            "visible_word_count_rough": len(re.findall(r"\b\w+\b", visible_text)),
            "contains_us_gaap_after_clean": int("us-gaap:" in visible_text.lower()),
            "contains_xbrli_after_clean": int("xbrli:" in visible_text.lower()),
            "error": "",
        })

    except Exception as e:
        row.update({
            "status": "error",
            "raw_char_len": None,
            "visible_char_len": None,
            "visible_word_count_rough": None,
            "contains_us_gaap_after_clean": None,
            "contains_xbrli_after_clean": None,
            "error": repr(e),
        })

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing raw .htm/.html files")
    parser.add_argument("--output_dir", required=True, help="Folder for improved visible .txt files")
    parser.add_argument("--audit_csv", default="html_visible_text_audit_v1_1.csv")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["process", "thread"],
        default="process",
        help="process is usually faster for HTML parsing; thread is safer if process has issues on Windows.",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.htm")) + list(input_dir.glob("*.html")))

    if args.limit:
        files = files[:args.limit]

    print("=" * 70)
    print("HTML visible text extraction v1.1 parallel")
    print("=" * 70)
    print("Input dir:", input_dir)
    print("Output dir:", output_dir)
    print("Audit CSV:", args.audit_csv)
    print("Files:", len(files))
    print("Workers:", args.workers)
    print("Mode:", args.mode)
    print("Resume:", args.resume)
    print("=" * 70)

    tasks = [(str(p), str(output_dir), args.resume) for p in files]

    rows = []

    Executor = ProcessPoolExecutor if args.mode == "process" else ThreadPoolExecutor

    with Executor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one_file, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting HTML visible text"):
            rows.append(fut.result())

    audit = pd.DataFrame(rows)
    audit = audit.sort_values(["file_name"])

    audit.to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    print("\nSaved visible text folder:", output_dir)
    print("Saved audit:", args.audit_csv)
    print("Files processed:", len(audit))

    if "status" in audit.columns:
        print("\nStatus counts:")
        print(audit["status"].value_counts(dropna=False).to_string())

    ok_like = audit[audit["status"].isin(["ok", "skipped_existing"])].copy()

    if len(ok_like) > 0:
        print("\nMean visible char len:", pd.to_numeric(ok_like["visible_char_len"], errors="coerce").mean())
        print("Median visible char len:", pd.to_numeric(ok_like["visible_char_len"], errors="coerce").median())
        print("Remaining us-gaap count:", pd.to_numeric(ok_like["contains_us_gaap_after_clean"], errors="coerce").sum())
        print("Remaining xbrli count:", pd.to_numeric(ok_like["contains_xbrli_after_clean"], errors="coerce").sum())


if __name__ == "__main__":
    main()