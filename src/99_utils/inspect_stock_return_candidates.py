import pandas as pd
from pathlib import Path

FILES = [
    "bank_quarter_detail_stock_covid_performance_2020_2022.csv",
    "analysis_stock_covid_performance_panel_2020_2022.csv",
    "research_ready_stock_covid_panel_2020_2022_time_aware.csv",
    "stock_stats_with_hcr_covid_exposure_time_aware.csv",
    "6020_21.xlsx",
]

KEYWORDS = [
    "return", "ret", "price", "prccq", "prccd", "cshoq",
    "mkvalt", "market", "stock", "datadate", "gvkey", "tic",
    "fyearq", "fqtr", "year"
]

def read_preview(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=20, dtype=str)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, nrows=20, dtype=str)
    else:
        return None

rows = []

for f in FILES:
    path = Path(f)

    if not path.exists():
        rows.append({
            "file_name": f,
            "exists": False,
            "n_cols": None,
            "matched_columns": "",
            "all_columns": "",
        })
        continue

    try:
        df = read_preview(path)
        cols = list(df.columns)

        matched = []
        for c in cols:
            cl = str(c).lower()
            if any(k in cl for k in KEYWORDS):
                matched.append(c)

        rows.append({
            "file_name": f,
            "exists": True,
            "n_cols": len(cols),
            "matched_columns": " | ".join(map(str, matched)),
            "all_columns": " | ".join(map(str, cols)),
        })

    except Exception as e:
        rows.append({
            "file_name": f,
            "exists": True,
            "n_cols": None,
            "matched_columns": "",
            "all_columns": f"ERROR: {repr(e)}",
        })

out = pd.DataFrame(rows)
out.to_csv("stock_return_candidate_inspection_v1.csv", index=False, encoding="utf-8-sig")

print(out[["file_name", "exists", "n_cols", "matched_columns"]].to_string(index=False))
print("\nSaved: stock_return_candidate_inspection_v1.csv")