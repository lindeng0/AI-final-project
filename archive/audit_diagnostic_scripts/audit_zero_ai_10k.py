import pandas as pd

scores = pd.read_csv("poe_ai_semantic_scores_test100_v1_2.csv")

scores["AI_semantic_index_0_100"] = pd.to_numeric(
    scores["AI_semantic_index_0_100"], errors="coerce"
)
scores["evidence_char_len"] = pd.to_numeric(
    scores["evidence_char_len"], errors="coerce"
)
scores["combined_input_char_len"] = pd.to_numeric(
    scores["combined_input_char_len"], errors="coerce"
)

zero_10k = scores[
    (scores["form"].str.upper() == "10-K")
    & (scores["AI_semantic_index_0_100"] == 0)
].copy()

cols = [
    "file_name",
    "cik",
    "form",
    "filing_date",
    "fiscal_year",
    "has_digital_evidence",
    "evidence_char_len",
    "combined_input_char_len",
    "DigitalRelevance",
    "Confidence",
    "Explanation",
]

zero_10k[cols].to_csv(
    "zero_ai_10k_spotcheck_v1_2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Zero-score 10-K rows:", len(zero_10k))
print(zero_10k[cols].head(30).to_string(index=False))