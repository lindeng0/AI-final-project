# ============================================================
# extract_html_visible_text_v1_2_parallel.py
# Purpose:
#   Robust parallel extraction of visible clean text from SEC HTML / Inline XBRL.
#
# Fixes from v1.1:
#   - Avoids NoneType tag errors during BeautifulSoup decompose()
#   - Separates removal passes
#   - More robust hidden / inline-XBRL cleanup
#   - Suitable for thousands of filings
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


def safe_decompose(tag):
    try:
        tag.decompose()
    except Exception:
        pass


def remove_invisible_and_xbrl_nodes(soup: BeautifulSoup):
    """
    Robust removal using separate passes.
    Avoid accessing attributes after decompose().
    """

    # Pass 1: remove obvious non-visible tags
    for tag in list(soup.find_all(["script", "style", "noscript", "meta", "link"])):
        safe_decompose(tag)

    # Pass 2: remove inline XBRL technical containers by tag name
    xbrl_prefixes = (
        "ix:",
        "xbrli:",
        "xbrldi:",
        "link:",
        "xlink:",
        "xsi:",
        "utr:",
        "ixt:",
        "ixt-sec:",
    )

    for tag in list(soup.find_all(True)):
        try:
            tag_name = str(tag.name).lower() if tag.name else ""
        except Exception:
            continue

        if tag_name.startswith(xbrl_prefixes):
            safe_decompose(tag)

    # Pass 3: remove hidden/display-none nodes
    for tag in list(soup.find_all(True)):
        try:
            if tag is None or tag.name is None:
                continue

            if tag.has_attr("hidden"):
                safe_decompose(tag)
                continue

            style = tag.attrs.get("style", "")
            if isinstance(style, str):
                low_style = style.lower().replace(" ", "")
                if "display:none" in low_style or "visibility:hidden" in low_style:
                    safe_decompose(tag)
                    continue

            # Some inline XBRL headers are represented with class/role/id
            cls = " ".join(tag.get("class", [])) if isinstance(tag.get("class", []), list) else str(tag.get("class", ""))
            tag_id = str(tag.get("id", ""))
            role = str(tag.get("role", ""))

            combined = f"{cls} {tag_id} {role}".lower()
            if "hidden" in combined and ("ix" in combined or "xbrl" in combined):
                safe_decompose(tag)

        except Exception:
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

    # Remove XBRL/taxonomy-like tokens that survive parsing
    text = re.sub(r"\b[a-zA-Z]{2,20}:[A-Za-z0-9_\-\.]+", " ", text)

    # Remove very long technical IDs
    text = re.sub(r"\b[A-Za-z0-9_\-]{35,}\b", " ", text)

    # Preserve important section boundaries
    section_patterns = [
        r"Item\s+1A\.",
        r"Item\s+1B\.",
        r"Item\s+1\.",
        r"Item\s+2\.",
        r"Item\s+7A\.",
        r"Item\s+7\.",
        r"Item\s+8\.",
        r"PART\s+I",
        r"PART\s+II",
    ]

    for p in section_patterns:
        text = re.sub(rf"\b({p})", r"\n\n\1", text, flags=re.IGNORECASE)

    lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        low = line.lower()

        # Drop obvious XML / XBRL remnants
        if low.startswith((
            "xmlns:",
            "contextref=",
            "unitref=",
            "decimals=",
            "schemaref",
            "xbrli:",
            "ix:",
            "us-gaap:",
            "dei:",
        )):
            continue

        # Drop lines that are mainly technical ids
        if len(line) > 80:
            alpha_ratio = sum(ch.isalpha() for ch in line) / max(len(line), 1)
            digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)

            if alpha_ratio < 0.25 and digit_ratio > 0.25:
                continue

        # Drop very short noisy lines, but keep item headings
        if len(line) <= 2:
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def html_to_visible_text(raw: str) -> str:
    html = extract_text_block_from_sec_wrapper(raw)

    soup = BeautifulSoup(html, "lxml")
    remove_invisible_and_xbrl_nodes(soup)

    # Add line breaks around block-like tags.
    for tag in list(soup.find_all(["div", "p", "tr", "table", "br", "hr", "li", "h1", "h2", "h3"])):
        try:
            tag.append("\n")
        except Exception:
            pass

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
    file_path_str, output_dir_str, resume = args_tuple

    path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                "contains_ix_hidden_after_clean": int("ix:hidden" in existing.lower()),
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
            "contains_ix_hidden_after_clean": int("ix:hidden" in visible_text.lower()),
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
            "contains_ix_hidden_after_clean": None,
            "error": repr(e),
        })

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_csv", default="html_visible_text_audit_v1_2.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=["process", "thread"], default="process")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.htm")) + list(input_dir.glob("*.html")))

    if args.limit:
        files = files[:args.limit]

    print("=" * 70)
    print("HTML visible text extraction v1.2 parallel")
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
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "file_name": "",
                    "output_txt_name": "",
                    "cik": "",
                    "form": "",
                    "filing_date": "",
                    "input_path": "",
                    "output_path": "",
                    "status": "fatal_error",
                    "raw_char_len": None,
                    "visible_char_len": None,
                    "visible_word_count_rough": None,
                    "contains_us_gaap_after_clean": None,
                    "contains_xbrli_after_clean": None,
                    "contains_ix_hidden_after_clean": None,
                    "error": repr(e),
                })

    audit = pd.DataFrame(rows)

    if "file_name" in audit.columns:
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
        print("Remaining ix:hidden count:", pd.to_numeric(ok_like["contains_ix_hidden_after_clean"], errors="coerce").sum())


if __name__ == "__main__":
    main()