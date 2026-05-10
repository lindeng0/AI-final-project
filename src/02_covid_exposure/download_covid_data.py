# download_covid_data.py

from pathlib import Path
import urllib.request

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"
external.mkdir(exist_ok=True)

files = {
    # Main health shock: NYT county-level cumulative cases/deaths
    "covid_county_nyt_us_counties.csv": (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
    ),

    # Optional: NYT rolling averages, useful later but not necessary now
    "covid_county_nyt_rolling_averages.csv": (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/us-counties.csv"
    ),

    # Policy shock: OxCGRT final dataset, compact subnational version
    "OxCGRT_compact_subnational_v1.csv": (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_subnational_v1.csv"
    ),

    # Optional simpler policy file
    "OxCGRT_simplified_subnational_v1.csv": (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_simplified_subnational_v1.csv"
    ),
}

for filename, url in files.items():
    out = external / filename
    print(f"Downloading {filename}...")
    print(url)

    try:
        urllib.request.urlretrieve(url, out)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"Failed to download {filename}")
        print(e)

print("\nDone.")