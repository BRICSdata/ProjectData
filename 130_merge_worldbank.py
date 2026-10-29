"""
===============================================================================
SCRIPT NAME : 130_merge_worldbank.py

PURPOSE
-------
Merges the BRICS panel (already containing V-Dem political controls) with the
locally downloaded World Bank macroeconomic indicators.

INPUT FILES
-----------
1. BRICS_Panel_VDem.csv
2. WorldBank_Controls_2018_2025.csv

OUTPUT FILE
-----------
BRICS_Master_Panel.csv

NOTES
-----
- Performs ONLY dataset merging.
- No observations are removed.
- No missing values are imputed.
- No variables are transformed.
- Prints merge diagnostics and missing-value summary.
===============================================================================
"""

import pandas as pd

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_PANEL = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
    r"\BRICS_Panel_VDem.csv"
)

INPUT_WB = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
    r"\WorldBank_Controls_2018_2025.csv"
)

OUTPUT_FILE = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
    r"\BRICS_Master_Panel.csv"
)

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 72)
print("SCRIPT 130 : Merge World Bank Controls")
print("=" * 72)

print("\nLoading BRICS Panel...")

panel = pd.read_csv(INPUT_PANEL)

print(f"✓ Panel observations : {len(panel):,}")

print("\nLoading World Bank Controls...")

wb = pd.read_csv(INPUT_WB)

print(f"✓ World Bank observations : {len(wb):,}")

# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required_panel = [
    "Target_Country",
    "Year"
]

required_wb = [
    "Target_Country",
    "Year",
    "GDP_per_Capita_PPP",
    "Population",
    "Trade_pct_GDP"
]

missing_panel = [c for c in required_panel if c not in panel.columns]
missing_wb = [c for c in required_wb if c not in wb.columns]

if missing_panel:
    raise ValueError(f"Missing columns in panel: {missing_panel}")

if missing_wb:
    raise ValueError(f"Missing columns in World Bank dataset: {missing_wb}")

# =============================================================================
# CHECK DUPLICATES
# =============================================================================

dup_panel = panel.duplicated(
    subset=["Target_Country", "Year"]
).sum()

dup_wb = wb.duplicated(
    subset=["Target_Country", "Year"]
).sum()

if dup_panel > 0:
    raise ValueError(
        f"Panel contains {dup_panel} duplicate country-year observations."
    )

if dup_wb > 0:
    raise ValueError(
        f"World Bank dataset contains {dup_wb} duplicate country-year observations."
    )

# =============================================================================
# MERGE
# =============================================================================

print("\nMerging datasets...")

merged = panel.merge(

    wb,

    how="left",

    on=["Target_Country", "Year"],

    indicator=True

)

# =============================================================================
# MERGE REPORT
# =============================================================================

matched = (merged["_merge"] == "both").sum()
unmatched = (merged["_merge"] == "left_only").sum()

print("\nMerge Summary")
print("-" * 40)

print(f"Matched observations   : {matched:,}")
print(f"Unmatched observations : {unmatched:,}")
print(f"Match rate             : {matched/len(merged)*100:.2f}%")

if unmatched > 0:

    print("\nCountry-years without World Bank data:")

    print(

        merged.loc[
            merged["_merge"] == "left_only",
            ["Target_Country", "Year"]
        ]
        .drop_duplicates()
        .sort_values(["Target_Country", "Year"])
        .to_string(index=False)

    )

# =============================================================================
# CLEAN UP
# =============================================================================

merged.drop(columns="_merge", inplace=True)

# =============================================================================
# MISSING VALUE REPORT
# =============================================================================

print("\nMissing Values")
print("-" * 40)

for col in [

    "GDP_per_Capita_PPP",

    "Population",

    "Trade_pct_GDP",

    "v2x_polyarchy"

]:

    print(f"{col:25s}{merged[col].isna().sum():>8}")

# =============================================================================
# EXPORT
# =============================================================================

merged.to_csv(

    OUTPUT_FILE,

    index=False

)

print("\nOutput saved successfully.")

print(OUTPUT_FILE)

print("\nFinal Dataset")

print("-" * 40)

print(f"Observations : {len(merged):,}")
print(f"Countries    : {merged['Target_Country'].nunique()}")
print(f"Years        : {merged['Year'].min()} - {merged['Year'].max()}")

print("\n" + "=" * 72)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 72)