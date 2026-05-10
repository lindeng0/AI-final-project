# Suggested Run Order

This project was developed iteratively. The cleaned GitHub version keeps only the most important scripts. Raw data are not included in the repo.

## Optional environment setup

```cmd
python -m pip install -r requirements.txt
```

For Poe scoring, set an environment variable before running any Poe script:

```cmd
set POE_API_KEY=your_poe_api_key_here
set POE_MODEL=GPT-4o-mini
```

## Main pipeline overview

### 1. Prepare SEC filings

```cmd
python src/03_sec_filings/download_sec_submissions_index.py
python src/03_sec_filings/export_sec_filing_targets.py
python src/03_sec_filings/download_sec_filing_documents.py
python src/03_sec_filings/extract_html_visible_text.py --resume
```

### 2. Build traditional keyword index

```cmd
python src/04_traditional_index/extract_keyword_level_hits.py
python src/04_traditional_index/build_revised_traditional_scores.py
python src/04_traditional_index/build_final_traditional_firm_year_index.py
python src/04_traditional_index/clean_final_traditional_firm_year_index.py
```

### 3. Build AI full-document semantic index

```cmd
python src/05_ai_semantic_scoring/score_filings_with_poe_fulltext.py --resume
python src/05_ai_semantic_scoring/aggregate_ai_fulltext_scores_to_firm_year.py
```

### 4. Build final panel

```cmd
python src/06_final_panel/merge_traditional_ai_index.py
python src/06_final_panel/build_final_2018_2024_panel.py
python src/06_final_panel/build_stock_return_from_compustat.py
python src/06_final_panel/merge_stock_return_into_final_panel.py
```

### 5. Run analysis

```cmd
python src/07_analysis/preview_traditional_ai.py
python src/07_analysis/run_final_regressions.py
```

## Notes

Exact command-line arguments may need adjustment depending on where raw data are stored locally. See each script's `argparse` defaults and help text.
