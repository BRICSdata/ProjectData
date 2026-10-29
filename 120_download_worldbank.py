"""
===============================================================================
SCRIPT NAME : 120_download_worldbank.py

PURPOSE
-------
Downloads macroeconomic control variables directly from the official World Bank
World Development Indicators (WDI) REST API and stores them as a reusable local
research dataset.

INPUT
-----
None (downloads directly from the World Bank API)

OUTPUT
------
WorldBank_Controls_2018_2025.csv

INDICATORS
----------
• GDP per capita, PPP (constant international $)
• Total Population
• Trade (% of GDP)

NOTES
-----
- Downloads data for ALL available countries.
- Covers years 2018–2025.
- Does NOT merge with the BRICS panel.
- Produces a reusable research dataset.
===============================================================================
"""

import time
import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

START_YEAR = 2018
END_YEAR = 2025

OUTPUT_FILE = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
    r"\WorldBank_Controls_2018_2025.csv"
)

# =============================================================================
# WORLD BANK INDICATORS
# =============================================================================

INDICATORS = {

    "NY.GDP.PCAP.PP.KD": "GDP_per_Capita_PPP",

    "SP.POP.TOTL": "Population",

    "NE.TRD.GNFS.ZS": "Trade_pct_GDP"

}

BASE_URL = "https://api.worldbank.org/v2/country/all/indicator"


# =============================================================================
# DOWNLOAD FUNCTION
# =============================================================================

def fetch_indicator(indicator_code, variable_name):

    print(f"\nDownloading {variable_name}...")

    url = (
        f"{BASE_URL}/{indicator_code}"
        f"?date={START_YEAR}:{END_YEAR}"
        f"&format=json"
        f"&per_page=25000"
    )

    # Retry mechanism
    for attempt in range(3):

        try:

            response = requests.get(url, timeout=30)

            response.raise_for_status()

            data = response.json()

            break

        except Exception as e:

            print(f"Attempt {attempt+1} failed.")

            if attempt == 2:
                raise RuntimeError(
                    f"Unable to download {variable_name}"
                ) from e

            time.sleep(2)

    # Validate JSON structure

    if not isinstance(data, list):

        raise RuntimeError(
            f"Unexpected response for {variable_name}"
        )

    if len(data) < 2:

        raise RuntimeError(
            f"No data returned for {variable_name}"
        )

    records = []

    for item in data[1]:

        iso3 = item.get("countryiso3code")

        year = item.get("date")

        value = item.get("value")

        if iso3 and iso3 != "" and year:

            records.append({

                "Target_Country": iso3,

                "Year": int(year),

                variable_name: value

            })

    df = pd.DataFrame(records)

    print(f"✓ {len(df):,} observations downloaded")

    return df


# =============================================================================
# MAIN
# =============================================================================

print("=" * 72)
print("SCRIPT 120 : Download World Bank Controls")
print("=" * 72)

dfs = []

for code, name in INDICATORS.items():

    dfs.append(fetch_indicator(code, name))


# =============================================================================
# MERGE INDICATORS TOGETHER
# =============================================================================

print("\nCombining indicators...")

wb = dfs[0]

for df in dfs[1:]:

    wb = wb.merge(

        df,

        how="outer",

        on=["Target_Country", "Year"]

    )


# =============================================================================
# CLEAN
# =============================================================================

wb = (

    wb

    .drop_duplicates(

        subset=["Target_Country", "Year"]

    )

    .sort_values(

        ["Target_Country", "Year"]

    )

    .reset_index(drop=True)

)


# =============================================================================
# DATA QUALITY REPORT
# =============================================================================

print("\n")
print("=" * 72)
print("WORLD BANK DATA SUMMARY")
print("=" * 72)

print(f"Rows      : {len(wb):,}")

print(f"Countries : {wb['Target_Country'].nunique():,}")

print(
    f"Years     : "
    f"{wb['Year'].min()} - {wb['Year'].max()}"
)

print("\nMissing Values")

print("-" * 40)

for col in [

    "GDP_per_Capita_PPP",

    "Population",

    "Trade_pct_GDP"

]:

    print(f"{col:25s}{wb[col].isna().sum():>8}")


# =============================================================================
# EXPORT
# =============================================================================

wb.to_csv(

    OUTPUT_FILE,

    index=False

)

print("\nDataset saved successfully.")

print(OUTPUT_FILE)

print("\n" + "=" * 72)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 72)