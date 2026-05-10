# ============================================================
# run_final_preview_regressions_2018_2024_v3.py
# Purpose:
#   Preview regressions using final 2018–2024 panel with:
#   - Traditional index
#   - AI full-document semantic index
#   - ROA
#   - Stock return
#   - COVID exposure, available only in 2020–2022
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def zscore(s):
    x = safe_num(s)
    sd = x.std(skipna=True)
    mu = x.mean(skipna=True)
    if pd.notna(sd) and sd > 0:
        return (x - mu) / sd
    return np.nan


def run_model(df, formula, cluster_col="GVKEY"):
    model = smf.ols(formula, data=df)

    if cluster_col in df.columns:
        try:
            return model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df[cluster_col].astype(str)}
            )
        except Exception:
            return model.fit()

    return model.fit()


def extract_key_results(res, model_name, outcome, terms):
    rows = []
    for t in terms:
        if t in res.params.index:
            rows.append({
                "outcome": outcome,
                "model": model_name,
                "term": t,
                "coef": res.params[t],
                "std_err": res.bse[t],
                "t": res.tvalues[t],
                "p_value": res.pvalues[t],
                "nobs": int(res.nobs),
                "r2": res.rsquared,
                "adj_r2": res.rsquared_adj,
            })
    return rows


def add_numeric_and_z(df, outcomes):
    numeric_cols = [
        "YEAR_key",
        "Traditional_index_0_100",
        "AI_full_document_index_0_100_v3",
        "LogAssets_annual_mean",
        "LogMktCap_annual_mean",
        "COVID_data_period_2020_2022",
        "Pre_COVID_2018_2019",
        "Post_COVID_2023_2024",
        "Post_2020",
        "annual_mean_covid_overall_exposure_main_hcr",
        "annual_mean_covid_health_exposure_main_hcr",
        "annual_mean_covid_case_exposure_main_hcr",
        "annual_mean_covid_policy_exposure_main_hcr",
    ] + outcomes

    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])

    df["z_traditional"] = zscore(df["Traditional_index_0_100"])
    df["z_ai"] = zscore(df["AI_full_document_index_0_100_v3"])

    if "LogAssets_annual_mean" in df.columns:
        df["z_log_assets"] = zscore(df["LogAssets_annual_mean"])

    if "LogMktCap_annual_mean" in df.columns:
        df["z_log_mktcap"] = zscore(df["LogMktCap_annual_mean"])

    covid_vars = [
        "annual_mean_covid_overall_exposure_main_hcr",
        "annual_mean_covid_health_exposure_main_hcr",
        "annual_mean_covid_case_exposure_main_hcr",
        "annual_mean_covid_policy_exposure_main_hcr",
    ]

    for c in covid_vars:
        if c in df.columns:
            df[f"z_{c}"] = zscore(df[c])

    for y in outcomes:
        if y in df.columns:
            df[f"z_{y}"] = zscore(df[y])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="analysis_ready_final_2018_2024_v3_with_stock_return.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="final_preview_regressions_2018_2024_v3"
    )
    parser.add_argument(
        "--outcomes",
        default="ROA_annual_mean,StockReturn_annual_compound_w"
    )
    parser.add_argument("--firm_fe", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    outcomes = [x.strip() for x in args.outcomes.split(",") if x.strip()]
    outcomes = [x for x in outcomes if x in df.columns]

    if not outcomes:
        raise ValueError("No valid outcomes found.")

    df = add_numeric_and_z(df, outcomes)

    fe_firm = " + C(GVKEY)" if args.firm_fe and "GVKEY" in df.columns else ""

    all_rows = []
    summaries = []

    for outcome in outcomes:
        z_outcome = f"z_{outcome}"

        base = df[
            df[z_outcome].notna()
            & df["z_traditional"].notna()
            & df["z_ai"].notna()
            & df["YEAR_key"].notna()
            & df["z_log_assets"].notna()
        ].copy()

        print("\n" + "=" * 80)
        print("Outcome:", outcome)
        print("Usable rows:", len(base))

        formulas = {
            "M1_traditional_yearFE":
                f"{z_outcome} ~ z_traditional + z_log_assets + C(YEAR_key){fe_firm}",

            "M2_ai_yearFE":
                f"{z_outcome} ~ z_ai + z_log_assets + C(YEAR_key){fe_firm}",

            "M3_both_yearFE":
                f"{z_outcome} ~ z_traditional + z_ai + z_log_assets + C(YEAR_key){fe_firm}",

            "M4_both_post2020":
                f"{z_outcome} ~ z_traditional + z_ai + Post_2020 + z_log_assets + C(YEAR_key){fe_firm}",

            "M5_ai_x_postcovid":
                f"{z_outcome} ~ z_traditional + z_ai * Post_COVID_2023_2024 + z_log_assets + C(YEAR_key){fe_firm}",
        }

        key_terms = [
            "z_traditional",
            "z_ai",
            "Post_2020",
            "Post_COVID_2023_2024",
            "z_ai:Post_COVID_2023_2024",
            "z_log_assets",
        ]

        for model_name, formula in formulas.items():
            try:
                res = run_model(base, formula, cluster_col="GVKEY")
                all_rows.extend(extract_key_results(res, model_name, outcome, key_terms))

                summaries.append("\n" + "=" * 100)
                summaries.append(f"Outcome: {outcome}")
                summaries.append(f"Model: {model_name}")
                summaries.append(f"Formula: {formula}")
                summaries.append(str(res.summary()))

                print(f"Done {model_name}: n={int(res.nobs)}, r2={res.rsquared:.3f}")

            except Exception as e:
                print("FAILED:", outcome, model_name, repr(e))

        # COVID exposure subsample, 2020–2022 only
        covid_main = "z_annual_mean_covid_overall_exposure_main_hcr"

        if covid_main in base.columns:
            covid_sample = base[base[covid_main].notna()].copy()

            if len(covid_sample) >= 50:
                covid_formulas = {
                    "C1_covid_exposure":
                        f"{z_outcome} ~ z_traditional + z_ai + {covid_main} + z_log_assets + C(YEAR_key){fe_firm}",

                    "C2_ai_x_covid_exposure":
                        f"{z_outcome} ~ z_traditional + z_ai * {covid_main} + z_log_assets + C(YEAR_key){fe_firm}",
                }

                covid_terms = [
                    "z_traditional",
                    "z_ai",
                    covid_main,
                    f"z_ai:{covid_main}",
                    "z_log_assets",
                ]

                print("COVID exposure subsample rows:", len(covid_sample))

                for model_name, formula in covid_formulas.items():
                    try:
                        res = run_model(covid_sample, formula, cluster_col="GVKEY")
                        all_rows.extend(extract_key_results(res, model_name, outcome, covid_terms))

                        summaries.append("\n" + "=" * 100)
                        summaries.append(f"Outcome: {outcome}")
                        summaries.append(f"Model: {model_name}")
                        summaries.append(f"Formula: {formula}")
                        summaries.append(str(res.summary()))

                        print(f"Done {model_name}: n={int(res.nobs)}, r2={res.rsquared:.3f}")

                    except Exception as e:
                        print("FAILED:", outcome, model_name, repr(e))

    results = pd.DataFrame(all_rows)
    results.to_csv(
        out_dir / "final_preview_regression_key_results_v3.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(out_dir / "final_preview_regression_full_summaries_v3.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))

    print("\nSaved:", out_dir / "final_preview_regression_key_results_v3.csv")
    print("Saved:", out_dir / "final_preview_regression_full_summaries_v3.txt")


if __name__ == "__main__":
    main()