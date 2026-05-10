# ============================================================
# build_ai_semantic_input_v1_2.py
# Purpose:
#   Build Poe input from improved HTML-visible clean text.
#
# Key update:
#   evidence_text = short filing context + digital evidence snippets
# ============================================================

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm


DEPLOYMENT_TERMS = [
    "digital banking", "mobile banking", "online banking", "digital channel",
    "digital channels", "digital onboarding", "remote deposit capture",
    "customer portal", "virtual assistant", "chatbot", "self-service",
    "digital wallet", "contactless payment", "peer-to-peer payment", "p2p",
    "digital lending", "automated underwriting", "robo-advisory",
    "open banking", "omnichannel", "fintech partnership", "embedded finance",
    "electronic banking", "internet banking", "online services",
]

CAPABILITY_TERMS = [
    "artificial intelligence", "machine learning", "deep learning",
    "natural language processing", "generative ai", "predictive analytics",
    "advanced analytics", "data analytics", "data science", "big data",
    "cloud", "cloud computing", "cloud migration", "public cloud",
    "private cloud", "hybrid cloud", "api", "open api", "microservices",
    "data lake", "data warehouse", "data governance", "business intelligence",
    "automation", "robotic process automation", "rpa", "core system",
    "platform modernization", "digital platform", "system integration",
    "legacy system replacement", "cybersecurity", "information security",
    "encryption", "multi-factor authentication", "mfa", "zero trust",
    "technology platform", "information technology", "technology infrastructure",
]

RISK_TERMS = [
    "cybersecurity risk", "cyber risk", "information security risk",
    "data breach", "privacy risk", "operational risk", "technology risk",
    "systems failure", "business continuity", "third-party technology",
    "vendor risk", "fraud detection", "real-time monitoring",
    "unauthorized access", "computer systems", "information systems",
]

STRATEGIC_CUE_TERMS = [
    "strategy", "strategic", "transformation", "modernization", "initiative",
    "investment", "investing", "innovation", "growth", "efficiency",
    "customer experience", "competitive", "long-term", "platform",
    "implementation", "deployment", "launched", "rolled out", "expanded",
    "enhance", "improve", "modernize", "upgrade",
]

ALL_TERMS = sorted(set(DEPLOYMENT_TERMS + CAPABILITY_TERMS + RISK_TERMS))


def infer_fiscal_year_from_filename(stem: str, filing_year=None):
    candidates = []

    for m in re.finditer(r"\d{8}", stem):
        s = m.group(0)

        # YYYYMMDD
        if s[:4].startswith(("19", "20")):
            try:
                candidates.append(datetime.strptime(s, "%Y%m%d"))
            except Exception:
                pass

        # MMDDYYYY
        if s[4:].startswith(("19", "20")):
            try:
                candidates.append(datetime.strptime(s, "%m%d%Y"))
            except Exception:
                pass

    if candidates:
        candidates = sorted(candidates)

        if filing_year is not None:
            try:
                fy = int(filing_year)
                close = [dt for dt in candidates if dt.year <= fy]
                if close:
                    return close[-1].year
            except Exception:
                pass

        return candidates[-1].year

    return filing_year


def parse_filename(path: Path) -> dict:
    name = path.name
    stem = path.stem

    parts = stem.split("_")
    cik = parts[0] if len(parts) >= 1 else ""
    form = parts[1] if len(parts) >= 2 else ""
    filing_date = parts[2] if len(parts) >= 3 else ""

    filing_year = None
    if re.match(r"\d{4}-\d{2}-\d{2}", filing_date):
        filing_year = int(filing_date[:4])

    fiscal_year = infer_fiscal_year_from_filename(stem, filing_year=filing_year)

    return {
        "file_name": name,
        "filing_id": stem,
        "cik": str(cik).lstrip("0") if cik else "",
        "form": form,
        "filing_date": filing_date,
        "filing_year": filing_year,
        "fiscal_year": fiscal_year,
    }


def basic_clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\b[a-zA-Z]{2,20}:[A-Za-z0-9_\-\.]+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9_\-]{35,}\b", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def useful_window_score(window: str) -> float:
    low = window.lower()
    score = 0.0

    for t in DEPLOYMENT_TERMS:
        if t in low:
            score += 3.0

    for t in CAPABILITY_TERMS:
        if t in low:
            score += 2.0

    for t in RISK_TERMS:
        if t in low:
            score += 1.2

    for t in STRATEGIC_CUE_TERMS:
        if t in low:
            score += 1.5

    alpha_ratio = sum(ch.isalpha() for ch in window) / max(len(window), 1)
    digit_ratio = sum(ch.isdigit() for ch in window) / max(len(window), 1)

    if alpha_ratio < 0.45:
        score *= 0.5

    if digit_ratio > 0.25:
        score *= 0.6

    return score


def extract_evidence(text: str, window_chars: int = 1800, top_k: int = 8, max_chars: int = 12000) -> dict:
    clean = basic_clean_text(text)
    low = clean.lower()

    matches = []
    for term in ALL_TERMS:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        for m in pattern.finditer(low):
            matches.append((m.start(), m.end(), term))

    if not matches:
        return {
            "evidence_only_text": "",
            "evidence_terms": "",
            "n_evidence_windows": 0,
            "evidence_char_len": 0,
        }

    windows = []
    for start, end, term in matches:
        left = max(0, start - window_chars // 2)
        right = min(len(clean), end + window_chars // 2)
        snippet = clean[left:right].strip()
        windows.append({
            "start": left,
            "end": right,
            "term": term,
            "text": snippet,
            "score": useful_window_score(snippet),
        })

    windows = sorted(windows, key=lambda x: (x["start"], x["end"]))

    merged = []
    for w in windows:
        if not merged or w["start"] > merged[-1]["end"] + 300:
            merged.append(w.copy())
        else:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["text"] = clean[merged[-1]["start"]:merged[-1]["end"]].strip()
            merged[-1]["term"] += "; " + w["term"]
            merged[-1]["score"] = useful_window_score(merged[-1]["text"])

    merged = sorted(merged, key=lambda x: x["score"], reverse=True)
    selected = merged[:top_k]

    parts = []
    used_terms = set()
    total_len = 0

    for i, w in enumerate(selected, start=1):
        snippet = w["text"]

        if total_len + len(snippet) > max_chars:
            remaining = max_chars - total_len
            if remaining < 500:
                break
            snippet = snippet[:remaining]

        terms_here = [t.strip() for t in w["term"].split(";") if t.strip()]
        used_terms.update(terms_here)

        parts.append(
            f"[Digital Evidence Window {i} | matched terms: {', '.join(sorted(set(terms_here)))[:300]}]\n"
            f"{snippet}"
        )
        total_len += len(snippet)

    evidence_only_text = "\n\n---\n\n".join(parts)

    return {
        "evidence_only_text": evidence_only_text,
        "evidence_terms": "; ".join(sorted(used_terms)),
        "n_evidence_windows": len(parts),
        "evidence_char_len": len(evidence_only_text),
    }


def find_section_window(text: str, patterns, max_chars: int = 2500) -> str:
    low = text.lower()
    for p in patterns:
        m = re.search(p, low, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start())
            end = min(len(text), start + max_chars)
            return text[start:end].strip()
    return ""


def extract_filing_context(text: str, max_total_chars: int = 5000) -> str:
    """
    Extract short broad context from Business / Risk / MD&A.
    This is not the main digital evidence. It only helps interpret strategy and time horizon.
    """
    clean = basic_clean_text(text)

    business = find_section_window(
        clean,
        patterns=[
            r"item\s+1\.\s+business",
            r"\bbusiness\b",
            r"\boverview\b",
        ],
        max_chars=2200,
    )

    risk = find_section_window(
        clean,
        patterns=[
            r"item\s+1a\.\s+risk factors",
            r"\brisk factors\b",
        ],
        max_chars=1800,
    )

    mda = find_section_window(
        clean,
        patterns=[
            r"item\s+7\.\s+management",
            r"management'?s discussion and analysis",
            r"\bresults of operations\b",
        ],
        max_chars=2200,
    )

    parts = []

    if business:
        parts.append("[Brief Business / Strategy Context]\n" + business)

    if mda:
        parts.append("[Brief MD&A Context]\n" + mda)

    if risk:
        parts.append("[Brief Risk Context]\n" + risk)

    context = "\n\n---\n\n".join(parts)
    return context[:max_total_chars].strip()


def combine_context_and_evidence(context: str, evidence: str) -> str:
    if evidence.strip():
        return (
            "Important instruction: The brief filing context below is provided only to help interpret "
            "strategic importance and time horizon. The score should be based primarily on the digital "
            "transformation evidence windows.\n\n"
            "[Brief Filing Context]\n"
            f"{context if context.strip() else '[NO BROAD FILING CONTEXT EXTRACTED]'}\n\n"
            "====================\n\n"
            "[Digital Transformation Evidence Windows]\n"
            f"{evidence}"
        )

    return (
        "Important instruction: No digital-transformation evidence windows were found by the retrieval step. "
        "Use the brief filing context only to confirm whether there is any clear digital transformation evidence. "
        "If none is present, assign low digital transformation scores.\n\n"
        "[Brief Filing Context]\n"
        f"{context if context.strip() else '[NO BROAD FILING CONTEXT EXTRACTED]'}\n\n"
        "====================\n\n"
        "[Digital Transformation Evidence Windows]\n"
        "[NO DIGITAL EVIDENCE FOUND]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing improved visible .txt files")
    parser.add_argument("--output_csv", default="poe_ai_semantic_input_v1_2.csv")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_evidence_chars", type=int, default=12000)
    parser.add_argument("--max_context_chars", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.txt"))

    if args.limit:
        files = files[:args.limit]

    rows = []

    for path in tqdm(files, desc="Building AI semantic input v1.2"):
        meta = parse_filename(path)

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            context = extract_filing_context(text, max_total_chars=args.max_context_chars)
            evidence = extract_evidence(
                text,
                window_chars=1800,
                top_k=args.top_k,
                max_chars=args.max_evidence_chars,
            )

            combined = combine_context_and_evidence(context, evidence["evidence_only_text"])

            row = {
                **meta,
                **evidence,
                "filing_context_text": context,
                "filing_context_char_len": len(context),
                "evidence_text": combined,
                "combined_input_char_len": len(combined),
                "status": "ok",
                "error": "",
            }

        except Exception as e:
            row = {
                **meta,
                "evidence_only_text": "",
                "evidence_terms": "",
                "n_evidence_windows": 0,
                "evidence_char_len": 0,
                "filing_context_text": "",
                "filing_context_char_len": 0,
                "evidence_text": "",
                "combined_input_char_len": 0,
                "status": "error",
                "error": repr(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    df["has_digital_evidence"] = pd.to_numeric(df["evidence_char_len"], errors="coerce").fillna(0).astype(int) > 0
    df["has_context"] = pd.to_numeric(df["filing_context_char_len"], errors="coerce").fillna(0).astype(int) > 0

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "n_files": len(df),
        "n_with_evidence": int(df["has_digital_evidence"].sum()),
        "n_without_evidence": int((~df["has_digital_evidence"]).sum()),
        "n_with_context": int(df["has_context"].sum()),
        "mean_evidence_char_len": float(df["evidence_char_len"].mean()),
        "mean_context_char_len": float(df["filing_context_char_len"].mean()),
        "mean_combined_input_char_len": float(df["combined_input_char_len"].mean()),
    }

    audit_path = Path(args.output_csv).with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nSaved:", args.output_csv)
    print("Shape:", df.shape)
    print("Saved audit:", audit_path)

    for k, v in audit.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()