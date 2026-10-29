import os
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_DIR = r"D:\PHD notes\BRICS Cohesion\Required dataset"

print("=" * 70)
print("AUDIT OF CSV FILES & COUNTRY COUNTS IN REQUIRED DATASET FOLDER")
print("=" * 70)

for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    name = os.path.basename(path)
    if "IdealPointDyads" in name or "V-Dem" in name or "WDI" in name:
        continue
    try:
        temp = pd.read_csv(path)
        c_col = [c for c in temp.columns if 'Country' in c or 'iso' in c.lower()]
        n_countries = temp[c_col[0]].nunique() if c_col else "N/A"
        print(f"{name:35s} -> Rows: {len(temp):5d} | Countries: {n_countries}")
    except Exception as e:
        print(f"{name:35s} -> Error: {e}")

# Compare BRICS_IPD_Panel_2018_2025.csv vs BRICS_Analysis_Panel.csv
p1_path = os.path.join(DATA_DIR, "BRICS_IPD_Panel_2018_2025.csv")
p2_path = os.path.join(DATA_DIR, "BRICS_Analysis_Panel.csv")

if os.path.exists(p1_path) and os.path.exists(p2_path):
    p1 = pd.read_csv(p1_path)
    p2 = pd.read_csv(p2_path)
    print("\nCountries in BRICS_IPD_Panel_2018_2025 vs BRICS_Analysis_Panel:")
    diff1 = set(p1['Target_Country']) - set(p2['Target_Country'])
    diff2 = set(p2['Target_Country']) - set(p1['Target_Country'])
    print("In IPD_Panel but not Analysis_Panel:", diff1)
    print("In Analysis_Panel but not IPD_Panel:", diff2)

# Check countries in IdealPointDyads1946-2025 (2018-2025) vs BRICS_Analysis_Panel
dy = pd.read_csv(os.path.join(DATA_DIR, "IdealPointDyads1946-2025.csv"), encoding="cp1252")
dy18 = dy[(dy['year'] >= 2018) & (dy['year'] <= 2025)]
all_dy_isos = set(dy18['iso3c1']).union(set(dy18['iso3c2']))
core_brics = {'BRA', 'RUS', 'IND', 'CHN', 'ZAF'}
panel_isos = set(p2['Target_Country'])

print("\nTotal unique countries in Voeten 2018-2025 dyads:", len(all_dy_isos))
print("Minus 5 Core BRICS countries:", len(all_dy_isos - core_brics))
print("Countries in Voeten (non-core) but NOT in BRICS_Analysis_Panel:", sorted(list((all_dy_isos - core_brics) - panel_isos)))

# Check which countries in BRICS_Analysis_Panel have fewer than 8 years of observations
yr_counts = p2.groupby('Target_Country')['Year'].count()
print("\nCountries in BRICS_Analysis_Panel with < 8 years of data:")
print(yr_counts[yr_counts < 8])