# AI-Based Digital Transformation Disclosure in U.S. Banks

This repository contains the cleaned code pipeline for a final project that builds and compares two digital transformation disclosure measures for U.S. banks:

1. **Traditional keyword-based index**: a relative 0-100 measure based on digital-transformation-related keyword hits in SEC 10-K/10-Q filings.
2. **AI full-document semantic index**: a Poe/ChatGPT-based semantic score obtained by asking the model to read the full HTML-parsed visible text of each SEC filing and rate the strategic substance of digital transformation disclosure.

The final analysis panel covers **2018-2024** at the **bank firm-year** level. It combines digital disclosure indices, ROA, annual stock return, size controls, market capitalization, and 2020-2022 COVID exposure variables.

## Repository structure

```text
src/
  01_data_preparation/       FDIC SOD, bank panel, crosswalk preparation
  02_covid_exposure/         COVID health/policy/HCR exposure construction
  03_sec_filings/            SEC filing download, cleaning, visible text extraction
  04_traditional_index/      Keyword extraction and traditional index construction
  05_ai_semantic_scoring/    Poe/ChatGPT semantic scoring and aggregation
  06_final_panel/            Final panel and stock return construction
  07_analysis/               Descriptive analysis and regressions
  99_utils/                  Inspection utilities

data/
  final/                     Final analysis-ready panel and small audit files
  interim/                   Small inspection/intermediate files

results/
  regressions/               Main regression output
  regressions_firmfe/        Firm FE robustness output
  preview_traditional_ai/    Preview figures/tables comparing indices

docs/
  METHOD_SUMMARY.md          Main method summary
  RUN_ORDER.md               Suggested pipeline run order
  FILE_MANIFEST.md           Mapping from original messy files to clean repo files
  poe.env.template           Template for Poe API key environment variable
```

## Final dataset

The main final CSV is:

```text
data/final/analysis_ready_final_2018_2024_v3_with_stock_return.csv
```

Key variables:

- `Traditional_index_0_100`
- `AI_full_document_index_0_100_v3`
- `ROA_annual_mean`
- `StockReturn_annual_compound_w`
- `LogAssets_annual_mean`
- `LogMktCap_annual_mean`
- `annual_mean_covid_overall_exposure_main_hcr`

## Main empirical pattern

The AI full-document semantic index is positively associated with annual stock return in pooled year fixed-effect models, while the traditional keyword-based index is not. The relationship is weaker with firm fixed effects, suggesting that the signal is driven more by cross-sectional differences across banks than within-bank year-to-year changes.

COVID exposure models suggest that the positive AI-stock-return relationship is attenuated in higher-COVID-exposure areas during 2020-2022.

## Important notes

- Raw SEC filings, raw FDIC SOD files, and large intermediate text/keyword-window files are intentionally excluded from GitHub.
- Poe API keys must never be committed. Use `docs/poe.env.template`.
- Stock return is an approximation based on Compustat quarterly `prccq`, not CRSP dividend-adjusted total return.
- AI score missing values are technical missingness and should not be filled with zero.
