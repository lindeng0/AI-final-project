# build_revised_traditional_digital_scores_v2.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

matches_file = folder / "sec_filing_matches_2018_2024.csv"
clean_index_file = folder / "sec_clean_text_index_2018_2024.csv"

# -----------------------------
# Revised dictionaries
# -----------------------------

deployment_terms = [
    "digital banking",
    "mobile banking",
    "online banking",
    "internet banking",
    "electronic banking",
    "mobile app",
    "mobile application",
    "digital channel",
    "digital channels",
    "online account opening",
    "digital account opening",
    "digital onboarding",
    "remote deposit",
    "remote deposit capture",
    "digital lending",
    "online lending",
    "digital loan",
    "online loan",
    "digital mortgage",
    "automated underwriting",
    "digital payments",
    "electronic payments",
    "contactless payment",
    "digital wallet",
    "p2p payments",
    "person-to-person payments",
    "zelle",
    "bill pay",
    "self-service",
    "customer portal",
    "virtual assistant",
    "chatbot",
    "omnichannel",
]

capability_terms = [
    "digital transformation",
    "digital strategy",
    "digital initiatives",
    "digital capabilities",
    "digital platform",
    "digital platforms",
    "cloud computing",
    "cloud migration",
    "cloud-based",
    "data lake",
    "data warehouse",
    "platform modernization",
    "core conversion",
    "system conversion",
    "technology platform",
    "automation",
    "robotic process automation",
    "legacy system replacement",
    "system integration",
    "artificial intelligence",
    "machine learning",
    "advanced analytics",
    "data analytics",
    "business intelligence",
]

risk_terms = [
    "cybersecurity",
    "cyber security",
    "information security",
    "technology risk",
    "cyber risk",
    "data breach",
    "security breach",
    "system failure",
    "technology failure",
    "business interruption",
    "digital fraud",
    "online fraud",
    "encryption",
    "multi-factor authentication",
    "zero trust",
]

# Ambiguous short terms removed:
# api, rpa, mfa
# cloud alone removed; use cloud computing/cloud migration/cloud-based

def strict_phrase_pattern(terms):
    parts = []
    for term in sorted(set(terms), key=len, reverse=True):
        words = [re.escape(w) for w in term.split()]
        phrase = r"\s+".join(words)
        parts.append(rf"(?<![A-Za-z0-9]){phrase}(?![A-Za-z0-9])")
    return re.compile("|".join(parts), flags=re.IGNORECASE)

deployment_pat = strict_phrase_pattern(deployment_terms)
capability_pat = strict_phrase_pattern(capability_terms)
risk_pat = strict_phrase_pattern(risk_terms)
word_pat = re.compile(r"\b\w+\b")

def process_file(row):
    path = row.get("clean_text_path")
    rec = {
        "CIK": row.get("CIK"),
        "accession_number": row.get("accession_number"),
        "clean_text_path": path,
        "total_words_v2": np.nan,
        "deployment_count_v2": 0,
        "capability_count_v2": 0,
        "risk_count_v2": 0,
        "deployment_density_v2": np.nan,
        "capability_density_v2": np.nan,
        "risk_density_v2": np.nan,
        "status_v2": "not_processed",
    }

    if pd.isna(path) or not Path(path).exists():
        rec["status_v2"] = "missing_file"
        return rec

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        total_words = len(word_pat.findall(text))

        dep = len(deployment_pat.findall(text))
        cap = len(capability_pat.findall(text))
        risk = len(risk_pat.findall(text))

        rec.update({
            "total_words_v2": total_words,
            "deployment_count_v2": dep,
            "capability_count_v2": cap,
            "risk_count_v2": risk,
            "deployment_density_v2": dep / total_words * 1000 if total_words > 0 else np.nan,
            "capability_density_v2": cap / total_words * 1000 if total_words > 0 else np.nan,
            "risk_density_v2": risk / total_words * 1000 if total_words > 0 else np.nan,
            "status_v2": "processed",
        })
        return rec

    except Exception as e:
        rec["status_v2"] = f"error_{type(e).__name__}: {e}"
        return rec

def z_by_year(series, year):
    return series.groupby(year).transform(
        lambda x: (x - x.mean()) / x.std()
        if x.notna().sum() > 2 and x.std() > 0 else np.nan
    )

def main():
    matches = pd.read_csv(matches_file, low_memory=False)
    clean = pd.read_csv(clean_index_file, low_memory=False)

    matches["CIK"] = pd.to_numeric(matches["CIK"], errors="coerce").astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
    clean["CIK"] = pd.to_numeric(clean["CIK"], errors="coerce").astype("Int64").astype(str).str.replace("<NA>", "", regex=False)

    matches["accession_number"] = matches["accession_number"].astype(str).str.strip()
    clean["accession_number"] = clean["accession_number"].astype(str).str.strip()

    df = matches.merge(
        clean[["CIK", "accession_number", "clean_text_path", "text_length"]],
        on=["CIK", "accession_number"],
        how="left"
    )

    unique_files = (
        df.dropna(subset=["clean_text_path"])
        .drop_duplicates(subset=["CIK", "accession_number", "clean_text_path"])
        [["CIK", "accession_number", "clean_text_path"]]
    )

    print("Matched rows:", df.shape)
    print("Unique clean files:", unique_files.shape)

    rows = unique_files.to_dict("records")
    results = []

    MAX_WORKERS = 12

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_file, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Revised traditional v2"):
            results.append(fut.result())

    scores = pd.DataFrame(results)

    scores.to_csv(folder / "revised_traditional_filing_level_scores_v2.csv", index=False, encoding="utf-8-sig")

    for c in ["CIK", "accession_number"]:
        df[c] = df[c].astype(str).str.strip()
        scores[c] = scores[c].astype(str).str.strip()

    out = df.merge(
        scores.drop(columns=["clean_text_path"], errors="ignore"),
        on=["CIK", "accession_number"],
        how="left",
        validate="many_to_one"
    )

    for c in ["deployment_count_v2", "capability_count_v2", "risk_count_v2"]:
        out[c] = out[c].fillna(0)

    out["z_deployment_density_v2"] = z_by_year(out["deployment_density_v2"], out["YEAR"])
    out["z_capability_density_v2"] = z_by_year(out["capability_density_v2"], out["YEAR"])
    out["z_risk_density_v2"] = z_by_year(out["risk_density_v2"], out["YEAR"])

    # Conservative composite:
    # deployment is main; capability helps but with smaller weight; risk does not automatically subtract too much.
    out["traditional_digital_deployment_score_v2"] = (
        out["z_deployment_density_v2"]
        + 0.25 * out["z_capability_density_v2"]
        - 0.25 * out["z_risk_density_v2"]
    )

    out["traditional_customer_channel_intensity_v2"] = out["deployment_density_v2"]
    out["traditional_capability_intensity_v2"] = out["capability_density_v2"]
    out["traditional_risk_intensity_v2"] = out["risk_density_v2"]

    out_file = folder / "sec_revised_traditional_digital_scores_v2_2018_2024.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\nSaved:", out_file)
    print("Shape:", out.shape)

    print("\nSummary v2:")
    print(out[[
        "deployment_count_v2",
        "capability_count_v2",
        "risk_count_v2",
        "traditional_digital_deployment_score_v2"
    ]].describe())

    print("\nCoverage:")
    print(out[["deployment_density_v2", "capability_density_v2", "risk_density_v2"]].notna().mean())

    print("\nTop 20 v2:")
    cols = [
        "GVKEY", "tic", "YEAR", "fqtr", "expected_form",
        "deployment_count_v2", "capability_count_v2", "risk_count_v2",
        "traditional_digital_deployment_score_v2"
    ]
    print(out.sort_values("traditional_digital_deployment_score_v2", ascending=False)[cols].head(20))

if __name__ == "__main__":
    freeze_support()
    main()