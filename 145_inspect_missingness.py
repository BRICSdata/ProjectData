"""
========================================================================
SCRIPT 145 : Prepare Analysis Dataset
========================================================================

Purpose
-------
Creates the final analysis dataset used by all regression models.

Missing-data treatment follows transparent and reviewer-defensible rules:

1. GDP per capita
       Country-wise forward fill
       Country-wise backward fill

2. Population
       Country-wise forward fill
       Country-wise backward fill

3. Trade (% GDP)
       Country-wise forward fill
       Country-wise backward fill

4. Democracy (V-Dem)
       Left unchanged
       (Missing observations are removed during estimation.)

Inputs
------
BRICS_Master_Panel.csv

Outputs
-------
BRICS_Analysis_Panel.csv
Imputation_Report.csv

========================================================================
"""

import pandas as pd

print("="*72)
print("SCRIPT 145 : PREPARE ANALYSIS DATASET")
print("="*72)

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(
    r"D:\PHD notes\BRICS Cohesion\Required dataset\BRICS_Master_Panel.csv"
)

print("\nDataset Loaded")
print(f"Observations : {len(df):,}")

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

df = df.sort_values(
    ["Target_Country","Year"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Remove Excluded Entities (Reviewer Revisions)
# ---------------------------------------------------------
print("\nRemoving Excluded Entities...")
# YUG: Obsolete
# SAU: Did not formally join, excluded from both treatment and control
# IDN: Joined in 2025, excluded from control group to prevent contamination
excluded_entities = ["YUG", "SAU", "IDN"] 

removed_count = df["Target_Country"].isin(excluded_entities).sum()
df = df[~df["Target_Country"].isin(excluded_entities)].copy()
print(f"Removed {removed_count} observations belonging to {excluded_entities}")
# ---------------------------------------------------------
# Missing values BEFORE
# ---------------------------------------------------------

print("\nMissing Values BEFORE")

cols = [
    "GDP_per_Capita_PPP",
    "Population",
    "Trade_pct_GDP",
    "v2x_polyarchy"
]

before = df[cols].isna().sum()

print(before)

# ---------------------------------------------------------
# GDP
# ---------------------------------------------------------

df["GDP_per_Capita_PPP"] = (

    df.groupby("Target_Country")["GDP_per_Capita_PPP"]

      .transform(lambda x: x.ffill().bfill())

)

# ---------------------------------------------------------
# Population
# ---------------------------------------------------------

df["Population"] = (

    df.groupby("Target_Country")["Population"]

      .transform(lambda x: x.ffill().bfill())

)

# ---------------------------------------------------------
# Trade
# ---------------------------------------------------------

df["Trade_pct_GDP"] = (

    df.groupby("Target_Country")["Trade_pct_GDP"]

      .transform(lambda x: x.ffill().bfill())

)

# ---------------------------------------------------------
# Missing values AFTER
# ---------------------------------------------------------

after = df[cols].isna().sum()

report = pd.DataFrame({

    "Variable": cols,

    "Before": before.values,

    "After": after.values

})

print("\nMissing Values AFTER")
print(report)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

report.to_csv(

    r"D:\PHD notes\BRICS Cohesion\Required dataset\Imputation_Report.csv",

    index=False

)

# ---------------------------------------------------------
# Save analysis panel
# ---------------------------------------------------------

output = r"D:\PHD notes\BRICS Cohesion\Required dataset\BRICS_Analysis_Panel.csv"

df.to_csv(

    output,

    index=False

)

print("\nOutput Created")

print("--------------------------------")

print("BRICS_Analysis_Panel.csv")

print("Imputation_Report.csv")

print("\nDataset Summary")

print("--------------------------------")

print(f"Observations : {len(df):,}")

print(f"Countries    : {df.Target_Country.nunique()}")

print(f"Years        : {df.Year.min()} - {df.Year.max()}")

print("\n")

print("="*72)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*72)