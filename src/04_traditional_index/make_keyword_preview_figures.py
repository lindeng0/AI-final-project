# make_final_keyword_preview_figures_v2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# 0. Paths
# ==================================================

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")

keyword_total_file = folder / "sec_keyword_summary_total_v2_2018_2024.csv"
keyword_year_file = folder / "sec_keyword_summary_by_year_v2_2018_2024.csv"
filing_summary_file = folder / "sec_filing_keyword_summary_v2_2018_2024.csv"

fig_dir = folder / "final_keyword_preview_figures_v2"
table_dir = folder / "final_keyword_preview_tables_v2"

fig_dir.mkdir(exist_ok=True)
table_dir.mkdir(exist_ok=True)

# ==================================================
# 1. Load data
# ==================================================

keyword_total = pd.read_csv(keyword_total_file)
keyword_year = pd.read_csv(keyword_year_file)
filing_summary = pd.read_csv(filing_summary_file)

print("Loaded files:")
print("keyword_total:", keyword_total.shape)
print("keyword_year:", keyword_year.shape)
print("filing_summary:", filing_summary.shape)

# ==================================================
# 2. Basic cleaning
# ==================================================

keyword_total["total_count"] = pd.to_numeric(keyword_total["total_count"], errors="coerce")
keyword_total["n_filings_mentioned"] = pd.to_numeric(keyword_total["n_filings_mentioned"], errors="coerce")

keyword_year["YEAR"] = pd.to_numeric(keyword_year["YEAR"], errors="coerce")
keyword_year["total_count"] = pd.to_numeric(keyword_year["total_count"], errors="coerce")
keyword_year["n_filings_mentioned"] = pd.to_numeric(keyword_year["n_filings_mentioned"], errors="coerce")

keyword_year = keyword_year.dropna(subset=["YEAR"])
keyword_year["YEAR"] = keyword_year["YEAR"].astype(int)

# ==================================================
# 3. Helper functions
# ==================================================

def save_current_fig(filename):
    out = fig_dir / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def add_bar_labels_horizontal(ax, values, padding_ratio=0.01):
    max_val = max(values) if len(values) > 0 else 0
    padding = max_val * padding_ratio

    for i, v in enumerate(values):
        ax.text(
            v + padding,
            i,
            f"{int(v):,}",
            va="center",
            fontsize=9
        )


# ==================================================
# 4. Figure 1: Top deployment keywords
# Main figure for construct validity
# ==================================================

deployment_top = (
    keyword_total[keyword_total["category"] == "deployment"]
    .sort_values("total_count", ascending=False)
    .head(15)
    .sort_values("total_count", ascending=True)
    .copy()
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(deployment_top["keyword"], deployment_top["total_count"])
ax.set_xlabel("Total keyword count")
ax.set_ylabel("Deployment keyword")
ax.set_title("Top Customer-Facing Digital Deployment Keywords, 2018–2024")

add_bar_labels_horizontal(ax, deployment_top["total_count"].tolist())

save_current_fig("fig1_top_deployment_keywords_clean.png")

deployment_top.sort_values("total_count", ascending=False).to_csv(
    table_dir / "table1_top_deployment_keywords.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 5. Figure 2: Deployment trend by year
# Main figure for time trend
# ==================================================

deployment_year = (
    keyword_year[keyword_year["category"] == "deployment"]
    .groupby("YEAR", dropna=False)["total_count"]
    .sum()
    .reset_index()
    .sort_values("YEAR")
)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    deployment_year["YEAR"],
    deployment_year["total_count"],
    marker="o",
    linewidth=2
)

ax.axvspan(2020, 2021, alpha=0.12, label="COVID period")
ax.set_xlabel("Year")
ax.set_ylabel("Total deployment keyword count")
ax.set_title("Customer-Facing Digital Deployment Keyword Counts, 2018–2024")
ax.grid(alpha=0.3)
ax.legend()

for _, row in deployment_year.iterrows():
    ax.text(
        row["YEAR"],
        row["total_count"],
        f"{int(row['total_count']):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

save_current_fig("fig2_deployment_trend_by_year_clean.png")

deployment_year.to_csv(
    table_dir / "table2_deployment_trend_by_year.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 6. Figure 3: Core banking keyword trends
# Cleaner version with fewer lines
# ==================================================

core_keywords = [
    "mobile banking",
    "online banking",
    "internet banking",
    "electronic banking",
    "digital banking",
]

core_trend = (
    keyword_year[
        (keyword_year["category"] == "deployment")
        & (keyword_year["keyword"].isin(core_keywords))
    ]
    .groupby(["YEAR", "keyword"], dropna=False)["total_count"]
    .sum()
    .reset_index()
)

core_pivot = (
    core_trend
    .pivot(index="YEAR", columns="keyword", values="total_count")
    .fillna(0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 5.8))

for col in core_pivot.columns:
    ax.plot(
        core_pivot.index,
        core_pivot[col],
        marker="o",
        linewidth=2,
        label=col
    )

ax.axvspan(2020, 2021, alpha=0.12, label="COVID period")
ax.set_xlabel("Year")
ax.set_ylabel("Total keyword count")
ax.set_title("Core Digital Banking Keyword Trends, 2018–2024")
ax.grid(alpha=0.3)
ax.legend(title="Keyword", bbox_to_anchor=(1.02, 1), loc="upper left")

save_current_fig("fig3_core_banking_keywords_clean.png")

core_pivot.to_csv(
    table_dir / "table3_core_banking_keyword_trends.csv",
    encoding="utf-8-sig"
)

# ==================================================
# 7. Figure 4: Average annual category count by COVID period
# Fixes the problem that post-COVID has 3 years while other periods have 2 years
# ==================================================

def assign_period(year):
    if year in [2018, 2019]:
        return "Pre-COVID\n2018–2019"
    elif year in [2020, 2021]:
        return "COVID\n2020–2021"
    elif year in [2022, 2023, 2024]:
        return "Post-COVID\n2022–2024"
    else:
        return np.nan

keyword_year["period"] = keyword_year["YEAR"].apply(assign_period)

period_category = (
    keyword_year
    .dropna(subset=["period"])
    .groupby(["period", "category"], dropna=False)
    .agg(
        total_count=("total_count", "sum"),
        n_years=("YEAR", "nunique")
    )
    .reset_index()
)

period_category["average_annual_count"] = (
    period_category["total_count"] / period_category["n_years"]
)

period_order = [
    "Pre-COVID\n2018–2019",
    "COVID\n2020–2021",
    "Post-COVID\n2022–2024",
]

period_category["period"] = pd.Categorical(
    period_category["period"],
    categories=period_order,
    ordered=True
)

period_pivot = (
    period_category
    .pivot(index="period", columns="category", values="average_annual_count")
    .fillna(0)
    .loc[period_order]
)

fig, ax = plt.subplots(figsize=(9, 5.8))

x = np.arange(len(period_pivot.index))
cols = list(period_pivot.columns)
width = 0.25

for i, col in enumerate(cols):
    ax.bar(
        x + (i - (len(cols) - 1) / 2) * width,
        period_pivot[col],
        width=width,
        label=col
    )

ax.set_xticks(x)
ax.set_xticklabels(period_pivot.index)
ax.set_ylabel("Average annual keyword count")
ax.set_title("Average Annual Digital Keyword Counts Across COVID Periods")
ax.legend(title="Category")
ax.grid(axis="y", alpha=0.3)

save_current_fig("fig4_average_annual_category_count_by_period.png")

period_pivot.to_csv(
    table_dir / "table4_average_annual_category_count_by_period.csv",
    encoding="utf-8-sig"
)

# ==================================================
# 8. Figure 5: Risk disclosure spike
# Appendix figure
# ==================================================

risk_year = (
    keyword_year[keyword_year["category"] == "risk"]
    .groupby("YEAR", dropna=False)["total_count"]
    .sum()
    .reset_index()
    .sort_values("YEAR")
)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    risk_year["YEAR"],
    risk_year["total_count"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Year")
ax.set_ylabel("Total risk-related keyword count")
ax.set_title("Digital-Related Risk Disclosure Keywords, 2018–2024")
ax.grid(alpha=0.3)

for _, row in risk_year.iterrows():
    ax.text(
        row["YEAR"],
        row["total_count"],
        f"{int(row['total_count']):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

save_current_fig("fig5_risk_disclosure_spike_appendix.png")

risk_year.to_csv(
    table_dir / "table5_risk_disclosure_trend_by_year.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 9. Figure 6: Capability trend
# Optional supporting figure
# ==================================================

capability_year = (
    keyword_year[keyword_year["category"] == "capability"]
    .groupby("YEAR", dropna=False)["total_count"]
    .sum()
    .reset_index()
    .sort_values("YEAR")
)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    capability_year["YEAR"],
    capability_year["total_count"],
    marker="o",
    linewidth=2
)

ax.axvspan(2020, 2021, alpha=0.12, label="COVID period")
ax.set_xlabel("Year")
ax.set_ylabel("Total capability keyword count")
ax.set_title("Digital Capability Keyword Counts, 2018–2024")
ax.grid(alpha=0.3)
ax.legend()

for _, row in capability_year.iterrows():
    ax.text(
        row["YEAR"],
        row["total_count"],
        f"{int(row['total_count']):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

save_current_fig("fig6_capability_trend_by_year_optional.png")

capability_year.to_csv(
    table_dir / "table6_capability_trend_by_year.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 10. Summary printout
# ==================================================

print("\n==============================")
print("Clean preview figures created")
print("==============================")
print("Figure folder:", fig_dir)
print("Table folder:", table_dir)

print("\nRecommended main figures:")
print("1. fig1_top_deployment_keywords_clean.png")
print("2. fig2_deployment_trend_by_year_clean.png")
print("3. fig3_core_banking_keywords_clean.png")

print("\nRecommended appendix / robustness figures:")
print("4. fig4_average_annual_category_count_by_period.png")
print("5. fig5_risk_disclosure_spike_appendix.png")
print("6. fig6_capability_trend_by_year_optional.png")

print("\nDeployment trend:")
print(deployment_year)

print("\nAverage annual category count by period:")
print(period_pivot)

print("\nRisk trend:")
print(risk_year)

print("\nDone.")