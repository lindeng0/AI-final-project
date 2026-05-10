# download_nyt_rolling_annual.py

from pathlib import Path
import urllib.request

folder = Path(r"C:\Users\LinDengdeng\Desktop\ai project")
external = folder / "external_data"
external.mkdir(exist_ok=True)

files = {
    "covid_county_nyt_rolling_2020.csv": (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/us-counties-2020.csv"
    ),
    "covid_county_nyt_rolling_2021.csv": (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/us-counties-2021.csv"
    ),
    "covid_county_nyt_rolling_2022.csv": (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/us-counties-2022.csv"
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
        print(f"Failed: {filename}")
        print(e)

print("\nDone.")