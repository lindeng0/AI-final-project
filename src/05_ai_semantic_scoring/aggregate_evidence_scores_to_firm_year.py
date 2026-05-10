# ============================================================
# aggregate_ai_semantic_scores_v1_1.py
# Purpose:
#   Aggregate filing-level Poe AI semantic scores to firm-year level.
#   v1.1 fixes fiscal_year inference and uses annual-first aggregation.
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
        "total_evidence_char_len_used": float(used["evidence_char_len"].sum(skipna=True)),
        "mean_digital_relevance_used": float(used["DigitalRelevance"].mean(skipna=True)),
        "mean_confidence_used": float(used["Confidence"].mean(skipna=True)),
    }

    row["AI_investment_type"] = weighted_avg(used, "InvestmentType", "final_weight")
    row["AI_strategic_importance"] = weighted_avg(used, "StrategicImportance", "final_weight")
    row["AI_implementation_stage"] = weighted_avg(used, "ImplementationStage", "final_weight")
    row["AI_time_horizon"] = weighted_avg(used, "TimeHorizon", "final_weight")
    row["AI_digital_relevance"] = weighted_avg(used, "DigitalRelevance", "final_weight")
    row["AI_confidence"] = weighted_avg(used, "Confidence", "final_weight")

    main_vals = [
        row["AI_investment_type"],
        row["AI_strategic_importance"],
        row["AI_implementation_stage"],
        row["AI_time_horizon"],
    ]

    if all(pd.notna(x) for x in main_vals):
        row["AI_semantic_score"] = float(np.mean(main_vals))
        row["AI_semantic_index_0_100"] = float((row["AI_semantic_score"] - 1.0) / 4.0 * 100.0)
    else:
        row["AI_semantic_score"] = np.nan
        row["AI_semantic_index_0_100"] = np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="poe_ai_semantic_scores_v1.csv")
    parser.add_argument("--output_csv", default="ai_semantic_firm_year_v1_1.csv")
    parser.add_argument("--audit_csv", default="ai_semantic_firm_year_audit_v1_1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")

    for c in MAIN_SCORE_COLS + [
        "DigitalRelevance",
        "Confidence",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "filing_year",
        "fiscal_year",
        "evidence_char_len",
        "n_evidence_windows",
    ]:
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

    # Clean cik
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df.loc[df["cik"].eq(""), "cik"] = np.nan

    # Weighting inside selected group
    # Because annual-first already chooses 10-K when available,
    # form_weight is less important, but keep it for fallback years.
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
    out = out.sort_values(["cik", "fiscal_year"])

    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "n_filing_rows": len(df),
        "n_valid_filing_rows": len(df_valid),
        "n_firm_year_rows": len(out),
        "n_api_ok": int(df["api_ok"].sum()),
        "n_api_error": int((~df["api_ok"]).sum()),
        "n_missing_firm_year_ai_score": int(out["AI_semantic_score"].isna().sum()),
        "n_10K_only_firm_years": int(out["aggregation_rule"].eq("10K_only").sum()),
        "n_all_available_no_10K_firm_years": int(out["aggregation_rule"].eq("all_available_no_10K").sum()),
        "mean_AI_semantic_index_0_100": float(out["AI_semantic_index_0_100"].mean(skipna=True)),
        "median_AI_semantic_index_0_100": float(out["AI_semantic_index_0_100"].median(skipna=True)),
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Shape:", out.shape)
    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nPreview:")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()