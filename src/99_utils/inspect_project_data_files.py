# ============================================================
# inspect_project_data_files_v1.py
# Purpose:
#   Inspect CSV / Excel files in the project folder and identify
#   possible performance / COVID / control datasets.
# ============================================================

import argparse
from pathlib import Path

import pandas as pd


KEYWORDS = [
    "gvkey", "cik", "tic", "ticker", "year", "fyear", "datadate",
    "roa", "roe", "ni", "at", "assets", "sale", "revt", "ceq",
    "deposits", "loans", "asset", "income", "return", "performance",
    "covid", "stringency", "cases", "deaths", "policy",
]


def read_preview(path: Path, nrows=5):
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=nrows, dtype=str)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path, nrows=nrows, dtype=str)
        else:
            return None
        return df
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".")
    parser.add_argument("--output_csv", default="project_data_file_inventory_v1.csv")
    args = parser.parse_args()

    folder = Path(args.folder)

    files = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(folder.glob(ext))

    rows = []

    for path in sorted(files):
        df = read_preview(path)

        if df is None:
            rows.append({
                "file_name": path.name,
                "path": str(path),
                "status": "read_error",
                "n_preview_cols": None,
                "columns": "",
                "likely_key_columns": "",
                "likely_relevant_columns": "",
            })
            continue

        cols = list(df.columns)
        cols_lower = [str(c).lower() for c in cols]

        likely_keys = []
        likely_relevant = []

        for c, cl in zip(cols, cols_lower):
            if cl in ["gvkey", "cik", "tic", "ticker", "year", "fyear", "datadate"]:
                likely_keys.append(c)

            if any(k in cl for k in KEYWORDS):
                likely_relevant.append(c)

        rows.append({
            "file_name": path.name,
            "path": str(path),
            "status": "ok",
            "n_preview_cols": len(cols),
            "columns": " | ".join(map(str, cols[:80])),
            "likely_key_columns": " | ".join(map(str, likely_keys)),
            "likely_relevant_columns": " | ".join(map(str, likely_relevant[:80])),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.output_csv)
    print("Files inspected:", len(out))
    print(out[["file_name", "status", "likely_key_columns", "likely_relevant_columns"]].to_string(index=False))


if __name__ == "__main__":
    main()