# ============================================================
# build_full_vs_evidence_input_v1_3_balanced.py
# Purpose:
#   Build a larger balanced full-document vs evidence comparison sample.
#
# Difference from v1.2:
#   - Does not over-restrict to 10-K
#   - Samples both 10-K and 10-Q
#   - Stratifies by evidence length
#   - Supports n=100 / n=200 validation samples
# ============================================================

import re
import argparse
from pathlib import Path

import pandas as pd


def clean_broad_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\b[a-zA-Z]{2,20}:[A-Za-z0-9_\-\.]+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9_\-]{35,}\b", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_broad_full_document_text(text: str, max_chars: int = 60000) -> str:
    """
    Broad full-document input:
    wider than evidence snippets, but capped for Poe stability.
    """
    text = clean_broad_text(text)

    if len(text) <= max_chars:
        return text

    # Preserve beginning: company overview / table of contents / early business context.
    beginning = text[:12000]

    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 80]

    broad_terms = [
        "business", "strategy", "strategic", "management discussion",
        "risk factors", "operations", "technology", "digital",
        "customer", "banking", "lending", "data", "cybersecurity",
        "mobile", "online", "platform", "investment", "initiative",
        "growth", "efficiency", "innovation", "transformation",
        "information technology", "electronic banking", "self-service",
        "digital services", "digital delivery", "automation", "analytics",
        "customer experience", "modernization",
    ]

    scored = []

    for c in chunks:
        low = c.lower()
        alpha_ratio = sum(ch.isalpha() for ch in c) / max(len(c), 1)
        digit_ratio = sum(ch.isdigit() for ch in c) / max(len(c), 1)

        score = alpha_ratio

        if digit_ratio > 0.25:
            score -= 0.4

        for t in broad_terms:
            if t in low:
                score += 0.25

        scored.append((score, c))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    selected = []
    current_len = len(beginning)

    for _, c in scored:
        if current_len >= max_chars:
            break

        if c[:120] in beginning:
            continue

        add_len = len(c) + 2

        if current_len + add_len > max_chars:
            remain = max_chars - current_len
            if remain > 500:
                selected.append(c[:remain])
            break

        selected.append(c)
        current_len += add_len

    out = beginning + "\n\n[Additional broad filing context]\n\n" + "\n\n".join(selected)
    return out[:max_chars]


def assign_evidence_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["evidence_char_len_num"] = pd.to_numeric(df["evidence_char_len"], errors="coerce").fillna(0)

    def bucket(x):
        if x <= 0:
            return "no_evidence"
        elif x < 2500:
            return "low_evidence"
        elif x < 8000:
            return "medium_evidence"
        else:
            return "high_evidence"

    df["evidence_bucket"] = df["evidence_char_len_num"].apply(bucket)
    return df


def balanced_sample(df: pd.DataFrame, n: int, seed: int = 42, tenk_share: float = 0.7) -> pd.DataFrame:
    """
    Balanced sampling by form and evidence bucket.
    """
    df = df.copy()
    df["form_upper"] = df["form"].astype(str).str.upper()
    df = assign_evidence_bucket(df)

    target_10k = int(round(n * tenk_share))
    target_10q = n - target_10k

    pieces = []

    for form_value, target_n in [("10-K", target_10k), ("10-Q", target_10q)]:
        form_df = df[df["form_upper"].eq(form_value)].copy()

        if len(form_df) == 0 or target_n <= 0:
            continue

        buckets = ["no_evidence", "low_evidence", "medium_evidence", "high_evidence"]
        per_bucket = max(1, target_n // len(buckets))

        form_pieces = []

        for i, b in enumerate(buckets):
            sub = form_df[form_df["evidence_bucket"].eq(b)]
            if len(sub) > 0:
                form_pieces.append(
                    sub.sample(
                        min(per_bucket, len(sub)),
                        random_state=seed + i + (0 if form_value == "10-K" else 100),
                    )
                )

        sampled_form = pd.concat(form_pieces, ignore_index=True) if form_pieces else pd.DataFrame()

        # Fill remaining quota within this form
        used = set(sampled_form["filing_id"].astype(str)) if len(sampled_form) > 0 else set()
        remain = form_df[~form_df["filing_id"].astype(str).isin(used)]

        need = target_n - len(sampled_form)
        if need > 0 and len(remain) > 0:
            sampled_form = pd.concat(
                [
                    sampled_form,
                    remain.sample(min(need, len(remain)), random_state=seed + 999),
                ],
                ignore_index=True,
            )

        pieces.append(sampled_form)

    sampled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    # Fill if there are not enough 10-K / 10-Q to hit n
    used = set(sampled["filing_id"].astype(str)) if len(sampled) > 0 else set()
    remain_all = df[~df["filing_id"].astype(str).isin(used)]

    need = n - len(sampled)
    if need > 0 and len(remain_all) > 0:
        sampled = pd.concat(
            [
                sampled,
                remain_all.sample(min(need, len(remain_all)), random_state=seed + 2024),
            ],
            ignore_index=True,
        )

    sampled = sampled.sample(frac=1, random_state=seed + 3000).head(n).reset_index(drop=True)

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12_input_csv", required=True)
    parser.add_argument("--visible_text_dir", required=True)
    parser.add_argument("--output_csv", default="poe_full_vs_evidence_input_balanced_v1_3.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max_full_chars", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tenk_share", type=float, default=0.7)
    args = parser.parse_args()

    df = pd.read_csv(args.v12_input_csv, dtype=str).fillna("")
    df = assign_evidence_bucket(df)

    sample = balanced_sample(
        df,
        n=args.n,
        seed=args.seed,
        tenk_share=args.tenk_share,
    )

    visible_dir = Path(args.visible_text_dir)

    rows = []

    for _, row in sample.iterrows():
        row = row.to_dict()

        file_name = row["file_name"]
        txt_path = visible_dir / file_name

        full_text = ""
        full_status = "missing_file"

        if txt_path.exists():
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="ignore")
                full_text = build_broad_full_document_text(raw, max_chars=args.max_full_chars)
                full_status = "ok"
            except Exception as e:
                full_text = ""
                full_status = f"error: {repr(e)}"

        out = dict(row)
        out["full_document_text"] = full_text
        out["full_document_char_len"] = len(full_text)
        out["full_document_status"] = full_status

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.output_csv)
    print("Shape:", out_df.shape)

    print("\nForm distribution:")
    print(out_df["form"].value_counts(dropna=False).to_string())

    print("\nEvidence bucket distribution:")
    print(out_df["evidence_bucket"].value_counts(dropna=False).to_string())

    print("\nFull document status:")
    print(out_df["full_document_status"].value_counts(dropna=False).to_string())

    print("\nMean evidence_char_len:", pd.to_numeric(out_df["evidence_char_len"], errors="coerce").mean())
    print("Mean combined_input_char_len:", pd.to_numeric(out_df["combined_input_char_len"], errors="coerce").mean())
    print("Mean full_document_char_len:", pd.to_numeric(out_df["full_document_char_len"], errors="coerce").mean())


if __name__ == "__main__":
    main()