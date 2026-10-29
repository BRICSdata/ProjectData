"""
===============================================================================
SCRIPT NAME : 110_merge_vdem.py

PURPOSE
-------
Merges the BRICS voting-distance panel with the V-Dem Electoral Democracy Index
(v2x_polyarchy) to create a unified country-year panel containing the dependent
variable and political institutional control variable.

INPUT FILES
-----------
1. BRICS_IPD_Panel_2018_2025.csv
2. V-Dem-CY-Core-v16.csv

OUTPUT FILE
-----------
BRICS_Panel_VDem.csv

NOTES
-----
- Performs ONLY data merging.
- No observations are removed.
- No variables are transformed.
- Merge diagnostics are printed to the console.
===============================================================================
"""

import pandas as pd

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_PANEL = r"D:\PHD notes\BRICS Cohesion\Required dataset\BRICS_IPD_Panel_2018_2025.csv"

INPUT_VDEM = r"D:\PHD notes\BRICS Cohesion\Required dataset\V-Dem-CY-Core-v16.csv"

OUTPUT_FILE = r"D:\PHD notes\BRICS Cohesion\Required dataset\BRICS_Panel_VDem.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 70)
print("SCRIPT 110 : Merge V-Dem Political Controls")
print("=" * 70)

print("\nLoading BRICS Panel...")

panel = pd.read_csv(INPUT_PANEL)

print(f"✓ Panel observations : {len(panel):,}")

print("\nLoading V-Dem dataset...")

vdem = pd.read_csv(INPUT_VDEM)

print(f"✓ V-Dem observations : {len(vdem):,}")


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required_panel = [
    "Target_Country",
    "Year"
]

required_vdem = [
    "country_text_id",
    "year",
    "v2x_polyarchy"
]

missing_panel = [c for c in required_panel if c not in panel.columns]
missing_vdem = [c for c in required_vdem if c not in vdem.columns]

if missing_panel:
    raise ValueError(
        f"Missing columns in BRICS panel: {missing_panel}"
    )

if missing_vdem:
    raise ValueError(
        f"Missing columns in V-Dem dataset: {missing_vdem}"
    )


# =============================================================================
# KEEP ONLY REQUIRED V-DEM VARIABLES
# =============================================================================

print("\nSelecting required V-Dem variables...")

vdem = vdem[
    [
        "country_text_id",
        "year",
        "v2x_polyarchy"
    ]
].copy()

vdem.rename(
    columns={
        "country_text_id": "Target_Country",
        "year": "Year"
    },
    inplace=True
)


# =============================================================================
# MERGE
# =============================================================================

print("\nMerging datasets...")

merged = panel.merge(
    vdem,
    how="left",
    on=["Target_Country", "Year"],
    indicator=True
)


# =============================================================================
# MERGE DIAGNOSTICS
# =============================================================================

matched = (merged["_merge"] == "both").sum()
unmatched = (merged["_merge"] == "left_only").sum()

print("\nMerge completed.")

print(f"\nMatched observations   : {matched:,}")
print(f"Unmatched observations : {unmatched:,}")

print(f"\nMatch rate : {matched / len(merged) * 100:.2f}%")

if unmatched > 0:

    print("\nCountries/Years with no V-Dem match:")

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
# EXPORT
# =============================================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput saved successfully:")

print(f"{OUTPUT_FILE}")

print("\nFinal Dataset Summary")

print("-" * 40)

print(f"Observations : {len(merged):,}")

print(f"Countries    : {merged['Target_Country'].nunique()}")

print(f"Years        : {merged['Year'].min()} - {merged['Year'].max()}")

print(f"Missing Democracy Values : {merged['v2x_polyarchy'].isna().sum():,}")

print("\n" + "=" * 70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)