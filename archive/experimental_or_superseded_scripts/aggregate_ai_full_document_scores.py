# ============================================================
# aggregate_ai_full_document_scores_v2.py
# Purpose:
#   Aggregate filing-level full-document Poe AI semantic scores
#   to firm-year level.
#
# Difference from evidence aggregator:
#   - Does not require evidence_char_len
#   - Uses full_document_char_len / text_char_len_sent_to_poe
#   - Annual-first rule: if 10-K exists, use 10-K only
# ============================================================

import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


MAIN_SCORE_COLS = [
    "InvestmentType",
    "StrategicImportance",
    "ImplementationStage",
    "TimeHorizon",
]


def safe_float(s):
    return pd.to_numeric(s, errors="coerce")


def infer_fiscal_year_from_filename(file_name: str, filing_year=None):
    stem = str(file_name).replace(".txt", "")
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

        if pd.notna(filing_year):
            try:
                fy = int(float(filing_year))
                close = [dt for dt in candidates if dt.year <= fy]
                if close:
                    return close[-1].year
            except Exception:
                pass

        return candidates[-1].year

    if pd.notna(filing_year):
        try:
            return int(float(filing_year))
        except Exception:
            return np.nan

    return np.nan


def weighted_avg(g, col, weight_col):
    x = safe_float(g[col])
    w = safe_float(g[weight_col])

    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan

    return float(np.average(x[mask], weights=w[mask]))


def aggregate_group(g):
    """
    Annual-first rule:
    - If 10-K exists for the firm-year, use 10-K rows only.
    - Otherwise use available 10-Q rows.
    """
    g = g.copy()

    has_10k = g["form_upper"].eq("10-K").any()

    if has_10k:
        used = g[g["form_upper"].eq("10-K")].copy()
        aggregation_rule = "10K_only"
    else:
        used = g.copy()
        aggregation_rule = "all_available_no_10K"

    row = {
        "cik": used["cik"].iloc[0],
        "fiscal_year": int(used["fiscal_year_fixed"].iloc[0]),
        "aggregation_rule": aggregation_rule,

        "n_filings_scored_all": len(g),
        "n_filings_used": len(used),
        "n_api_ok_all": int(g["api_ok"].sum()),
        "n_api_ok_used": int(used["api_ok"].sum()),

        "n_10k_all": int(g["form_upper"].eq("10-K").sum()),
        "n_10q_all": int(g["form_upper"].eq("10-Q").sum()),

        "mean_digital_relevance_used": float(used["DigitalRelevance"].mean(skipna=True)),
        "mean_confidence_used": float(used["Confidence"].mean(skipna=True)),

        "mean_full_document_char_len_used": float(
            used["full_document_char_len"].mean(skipna=True)
            if "full_document_char_len" in used.columns else np.nan
        ),
        "mean_text_char_len_sent_to_poe_used": float(
            used["text_char_len_sent_to_poe"].mean(skipna=True)
            if "text_char_len_sent_to_poe" in used.columns else np.nan
        ),
        "n_capped_used": int(
            used["was_capped"].fillna(0).astype(float).sum()
            if "was_capped" in used.columns else 0
        ),
    }

    row["AI_full_investment_type"] = weighted_avg(used, "InvestmentType", "final_weight")
    row["AI_full_strategic_importance"] = weighted_avg(used, "StrategicImportance", "final_weight")
    row["AI_full_implementation_stage"] = weighted_avg(used, "ImplementationStage", "final_weight")
    row["AI_full_time_horizon"] = weighted_avg(used, "TimeHorizon", "final_weight")
    row["AI_full_digital_relevance"] = weighted_avg(used, "DigitalRelevance", "final_weight")
    row["AI_full_confidence"] = weighted_avg(used, "Confidence", "final_weight")

    main_vals = [
        row["AI_full_investment_type"],
        row["AI_full_strategic_importance"],
        row["AI_full_implementation_stage"],
        row["AI_full_time_horizon"],
    ]

    if all(pd.notna(x) for x in main_vals):
        row["AI_full_document_score_v2"] = float(np.mean(main_vals))
        row["AI_full_document_index_0_100_v2"] = float(
            (row["AI_full_document_score_v2"] - 1.0) / 4.0 * 100.0
        )
    else:
        row["AI_full_document_score_v2"] = np.nan
        row["AI_full_document_index_0_100_v2"] = np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="poe_ai_full_document_scores_v2.csv")
    parser.add_argument("--output_csv", default="ai_full_document_firm_year_v2.csv")
    parser.add_argument("--audit_csv", default="ai_full_document_firm_year_audit_v2.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")

    numeric_cols = MAIN_SCORE_COLS + [
        "DigitalRelevance",
        "Confidence",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "filing_year",
        "fiscal_year",
        "full_document_char_len",
        "full_document_char_len_before_cap",
        "text_char_len_sent_to_poe",
        "was_capped",
        "attempts_used",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_float(df[c])

    df["api_ok"] = df["api_status"].astype(str).str.lower().eq("ok")
    df["form_upper"] = df["form"].astype(str).str.upper()

    # Fix fiscal year from filename
    df["fiscal_year_fixed"] = df.apply(
        lambda r: infer_fiscal_year_from_filename(
            r.get("file_name", ""),
            filing_year=r.get("filing_year", np.nan),
        ),
        axis=1,
    )

    df["fiscal_year_fixed"] = safe_float(df["fiscal_year_fixed"]).astype("Int64")

    # Clean CIK
    df["cik"] = (
        df["cik"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
    )
    df.loc[df["cik"].eq(""), "cik"] = np.nan

    # Weighting
    # Since annual-first already selects 10-K when available,
    # form_weight is mainly for fallback years without 10-K.
    df["form_weight"] = np.where(df["form_upper"].eq("10-K"), 1.0, 0.35)

    df["confidence_weight"] = safe_float(df["Confidence"]) / 100.0
    df["confidence_weight"] = df["confidence_weight"].clip(lower=0, upper=1)

    df["relevance_weight"] = safe_float(df["DigitalRelevance"]) / 5.0
    df["relevance_weight"] = df["relevance_weight"].clip(lower=0.2, upper=1)

    df["final_weight"] = df["form_weight"] * df["confidence_weight"] * df["relevance_weight"]
    df.loc[~df["api_ok"], "final_weight"] = 0

    valid = df["cik"].notna() & df["fiscal_year_fixed"].notna()
    df_valid = df[valid].copy()

    rows = []

    for _, g in df_valid.groupby(["cik", "fiscal_year_fixed"], dropna=False):
        rows.append(aggregate_group(g))

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["cik", "fiscal_year"])

    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "n_filing_rows": len(df),
        "n_valid_filing_rows": len(df_valid),
        "n_firm_year_rows": len(out),
        "n_api_ok": int(df["api_ok"].sum()),
        "n_api_error": int((~df["api_ok"]).sum()),
        "n_missing_firm_year_ai_score": int(out["AI_full_document_score_v2"].isna().sum()) if len(out) else 0,
        "n_10K_only_firm_years": int(out["aggregation_rule"].eq("10K_only").sum()) if len(out) else 0,
        "n_all_available_no_10K_firm_years": int(out["aggregation_rule"].eq("all_available_no_10K").sum()) if len(out) else 0,
        "mean_AI_full_document_index_0_100_v2": float(out["AI_full_document_index_0_100_v2"].mean(skipna=True)) if len(out) else np.nan,
        "median_AI_full_document_index_0_100_v2": float(out["AI_full_document_index_0_100_v2"].median(skipna=True)) if len(out) else np.nan,
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Shape:", out.shape)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    if len(out) > 0:
        print("\nPreview:")
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()