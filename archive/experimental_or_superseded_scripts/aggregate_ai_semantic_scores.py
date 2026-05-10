# ============================================================
# aggregate_ai_semantic_scores_v1.py
# Purpose:
#   Aggregate filing-level Poe AI semantic scores to firm-year level.
# ============================================================

import argparse
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


def weighted_avg(g, col, weight_col):
    x = safe_float(g[col])
    w = safe_float(g[weight_col])

    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan

    return np.average(x[mask], weights=w[mask])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="poe_ai_semantic_scores_v1.csv")
    parser.add_argument("--output_csv", default="ai_semantic_firm_year_v1.csv")
    parser.add_argument("--audit_csv", default="ai_semantic_firm_year_audit_v1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")

    # Convert needed fields
    for c in MAIN_SCORE_COLS + [
        "DigitalRelevance",
        "Confidence",
        "AI_semantic_score",
        "AI_semantic_index_0_100",
        "fiscal_year",
        "evidence_char_len",
        "n_evidence_windows",
    ]:
        if c in df.columns:
            df[c] = safe_float(df[c])

    # Basic filters
    df["api_ok"] = df["api_status"].astype(str).str.lower().eq("ok")

    # Form weight:
    # 10-K contains annual strategic narrative, 10-Q is useful but shorter/quarterly.
    df["form_upper"] = df["form"].astype(str).str.upper()
    df["form_weight"] = np.where(df["form_upper"].eq("10-K"), 1.0, 0.35)

    # Confidence weight
    df["confidence_weight"] = df["Confidence"] / 100.0
    df["confidence_weight"] = df["confidence_weight"].clip(lower=0, upper=1)

    # Digital relevance weight:
    # Low relevance evidence contributes less to final firm-year signal.
    df["relevance_weight"] = df["DigitalRelevance"] / 5.0
    df["relevance_weight"] = df["relevance_weight"].clip(lower=0.2, upper=1)

    df["final_weight"] = df["form_weight"] * df["confidence_weight"] * df["relevance_weight"]
    df.loc[~df["api_ok"], "final_weight"] = 0

    # Group identifiers
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["fiscal_year"] = safe_float(df["fiscal_year"]).astype("Int64")

    group_cols = ["cik", "fiscal_year"]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        cik, fiscal_year = keys

        row = {
            "cik": cik,
            "fiscal_year": fiscal_year,
            "n_filings_scored": len(g),
            "n_api_ok": int(g["api_ok"].sum()),
            "n_10k": int(g["form_upper"].eq("10-K").sum()),
            "n_10q": int(g["form_upper"].eq("10-Q").sum()),
            "total_evidence_char_len": float(g["evidence_char_len"].sum(skipna=True)),
            "mean_digital_relevance": float(g["DigitalRelevance"].mean(skipna=True)),
            "mean_confidence": float(g["Confidence"].mean(skipna=True)),
            "sum_final_weight": float(g["final_weight"].sum(skipna=True)),
        }

        # Weighted dimensions
        row["AI_investment_type"] = weighted_avg(g, "InvestmentType", "final_weight")
        row["AI_strategic_importance"] = weighted_avg(g, "StrategicImportance", "final_weight")
        row["AI_implementation_stage"] = weighted_avg(g, "ImplementationStage", "final_weight")
        row["AI_time_horizon"] = weighted_avg(g, "TimeHorizon", "final_weight")
        row["AI_digital_relevance"] = weighted_avg(g, "DigitalRelevance", "final_weight")
        row["AI_confidence"] = weighted_avg(g, "Confidence", "final_weight")

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

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(["cik", "fiscal_year"])
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "n_filing_rows": len(df),
        "n_firm_year_rows": len(out),
        "n_api_ok": int(df["api_ok"].sum()),
        "n_api_error": int((~df["api_ok"]).sum()),
        "n_missing_firm_year_ai_score": int(out["AI_semantic_score"].isna().sum()),
        "mean_AI_semantic_index_0_100": float(out["AI_semantic_index_0_100"].mean(skipna=True)),
        "median_AI_semantic_index_0_100": float(out["AI_semantic_index_0_100"].median(skipna=True)),
    }

    pd.DataFrame([audit]).to_csv(args.audit_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.output_csv)
    print("Shape:", out.shape)
    print("\nAudit:")
    for k, v in audit.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()