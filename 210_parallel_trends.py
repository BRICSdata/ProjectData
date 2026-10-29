"""
===============================================================================
SCRIPT NAME : 210_parallel_trends.py

PURPOSE
-------------------------------------------------------------------------------
Tests the Parallel Trends assumption required for Difference-in-Differences.

This script performs four tasks:

1. Calculates yearly average voting distance separately for
   BRICS entrants and control countries.

2. Produces a publication-quality Parallel Trends figure.

3. Performs a formal pre-treatment trend test
   (2018-2023 only).

4. Creates relative-year (event study) variables for later models.

INPUT
-------------------------------------------------------------------------------
BRICS_Model_Panel.csv

OUTPUT
-------------------------------------------------------------------------------
Parallel_Trends_Figure.png

Average_Distance_By_Year.csv

Pretrend_Test.csv

EventStudy_Data.csv

===============================================================================
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

# =============================================================================
# FILE PATHS
# =============================================================================

INPUT_FILE = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
    r"\BRICS_Analysis_Panel.csv"
)

OUTPUT_FOLDER = (
    r"D:\PHD notes\BRICS Cohesion\Required dataset"
)

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*75)
print("SCRIPT 210 : Parallel Trends Assessment")
print("="*75)

df = pd.read_csv(INPUT_FILE)

print("\nDataset Loaded")

print(f"Observations : {len(df):,}")

print(f"Countries    : {df.Target_Country.nunique()}")

print(f"Years        : {df.Year.min()} - {df.Year.max()}")

# =============================================================================
# YEARLY AVERAGES
# =============================================================================

print("\nCalculating yearly averages...")

yearly = (

    df

    .groupby(

        ["Year","Is_Entrant"]

    )["Avg_Distance_to_BRICS"]

    .mean()

    .reset_index()

)

yearly["Group"] = np.where(

    yearly["Is_Entrant"]==1,

    "BRICS Entrants",

    "Control Countries"

)

pivot = (

    yearly

    .pivot(

        index="Year",

        columns="Group",

        values="Avg_Distance_to_BRICS"

    )

)

pivot.to_csv(

    OUTPUT_FOLDER +

    r"\Average_Distance_By_Year.csv"

)

print("Yearly averages exported.")

# =============================================================================
# PARALLEL TRENDS FIGURE
# =============================================================================

print("\nCreating figure...")

plt.figure(figsize=(9,6))

plt.plot(

    pivot.index,

    pivot["Control Countries"],

    marker="o",

    linewidth=2,

    label="Control Countries"

)

plt.plot(

    pivot.index,

    pivot["BRICS Entrants"],

    marker="s",

    linewidth=2,

    label="BRICS Entrants"

)

plt.axvline(

    x=2024,

    linestyle="--",

    linewidth=2

)

plt.text(

    2024.05,

    plt.ylim()[1]*0.98,

    "BRICS Expansion",

    fontsize=10

)

plt.title(

    "Average Voting Distance to BRICS Core"

)

plt.xlabel("Year")

plt.ylabel("Average Ideal Point Distance")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(

    OUTPUT_FOLDER +

    r"\Parallel_Trends_Figure.png",

    dpi=300

)

plt.close()

print("Parallel Trends figure saved.")

# ============================================================
# DYNAMIC TWFE MODEL (EVENT STUDY)
# ============================================================
print("\n" + "="*70)
print("ESTIMATING DYNAMIC EFFECTS (EVENT STUDY)")
print("="*70)

# Create interaction dummies for every year EXCEPT the base year (2022)
# 2022 is the base year to allow testing for a 2023 anticipation effect.
df["log_GDP"] = np.log(df["GDP_per_Capita_PPP"] + 1)
df["log_Pop"] = np.log(df["Population"] + 1)

years = sorted(df["Year"].unique())
event_vars = []

for yr in years:
    if yr == 2022:
        continue # Skip base year
    
    var_name = f"Treated_x_{yr}"
    df[var_name] = (df["Is_Entrant"] == 1) & (df["Year"] == yr)
    df[var_name] = df[var_name].astype(int)
    event_vars.append(var_name)

# Drop NaNs for the model
analysis_vars = ["Avg_Distance_to_BRICS", "Target_Country", "Year", "v2x_polyarchy", "log_GDP", "log_Pop", "Trade_pct_GDP"] + event_vars
df_model = df[analysis_vars].dropna().copy()

formula_dynamic = "Avg_Distance_to_BRICS ~ " + " + ".join(event_vars) + \
                  " + v2x_polyarchy + log_GDP + log_Pop + Trade_pct_GDP + C(Target_Country) + C(Year)"

event_model = smf.ols(formula=formula_dynamic, data=df_model).fit(
    cov_type="cluster", cov_kwds={"groups": df_model["Target_Country"]}
)

print(event_model.summary().tables[1])

# Save ONLY the event study coefficients to a CSV (ignoring country/year fixed effects)
event_results_df = event_model.summary2().tables[1]

# Filter to just the event study years (Treated_x_2018, etc.)
clean_event_results = event_results_df.loc[event_vars]

print("\n" + "="*70)
print("EVENT STUDY COEFFICIENTS (FOR MANUSCRIPT)")
print("="*70)
# Just print the entire filtered table to avoid any column-name KeyErrors
print(clean_event_results)

print("\n" + "="*70)
print("JOINT PRE-TREND WALD TESTS")
print("="*70)
print("Joint test 2018-2021:", event_model.wald_test("Treated_x_2018 = 0, Treated_x_2019 = 0, Treated_x_2020 = 0, Treated_x_2021 = 0", scalar=True))
print("Joint test 2018-2023:", event_model.wald_test("Treated_x_2018 = 0, Treated_x_2019 = 0, Treated_x_2020 = 0, Treated_x_2021 = 0, Treated_x_2023 = 0", scalar=True))

clean_event_results.to_csv(OUTPUT_FOLDER + r"\Event_Study_Coefficients_Clean.csv")
print("\nSaved Event_Study_Coefficients_Clean.csv")


# # =============================================================================
# EVENT STUDY PLOT (BILINGUAL EXPORT)
# =============================================================================
print("\nGenerating Event Study Plots in English and Russian...")

coefs = []
errors = []
plot_years = []

for yr in years:
    plot_years.append(yr)
    if yr == 2022:
        coefs.append(0)
        errors.append(0)
    else:
        var_name = f"Treated_x_{yr}"
        coefs.append(event_model.params[var_name])
        # 1.96 * standard error for 95% CI
        errors.append(1.96 * event_model.bse[var_name])

# Define the text for both languages based on the manuscript terminology
plot_configs = {
    "EN": {
        "title": "Event Study: Dynamic Treatment Effects on Voting Distance\n(Reference Year: 2022)",
        "ylabel": "Estimated Coefficient (95% CI)",
        "xlabel": "Year",
        "event_text": "Formal Accession\n(2024)",
        "filename": r"\Event_Study_Figure_EN.png"
    },
    "RU": {
        "title": "Оценка эффекта по отдельным годам: динамика согласованности голосования\n(Базовый год: 2022)",
        "ylabel": "Оцененный коэффициент (95% ДИ)",
        "xlabel": "Год",
        "event_text": "Официальное\nприсоединение\n(2024)",
        "filename": r"\Event_Study_Figure_RU.png"
    }
}

for lang, config in plot_configs.items():
    plt.figure(figsize=(10, 6))
    
    # Plot the points and error bars
    plt.errorbar(plot_years, coefs, yerr=errors, fmt='o', color='black', 
                 capsize=5, capthick=2, elinewidth=2, markersize=8)
    
    # Add horizontal zero line and vertical event line
    plt.axhline(0, color='red', linestyle='dashed', linewidth=1.5)
    plt.axvline(2023.5, color='gray', linestyle='dotted', linewidth=2) 
    
    # Add annotations and labels
    plt.text(2023.6, max(coefs) * 0.9, config["event_text"], fontsize=10, color='gray')
    
    plt.title(config["title"], pad=15)
    plt.xlabel(config["xlabel"])
    plt.ylabel(config["ylabel"])
    plt.xticks(plot_years)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the file
    save_path = OUTPUT_FOLDER + config["filename"]
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved {config['filename']}")
# =============================================================================
# SUMMARY
# =============================================================================

print("\n"+"="*70)

print("SUMMARY")

print("="*70)

print(f"Observations                : {len(df):,}")

print(f"Countries                   : {df.Target_Country.nunique()}")

print(f"Years                       : {df.Year.min()}-{df.Year.max()}")

print(f"Treatment Countries         : {df[df.Is_Entrant==1].Target_Country.nunique()}")

print(f"Control Countries           : {df[df.Is_Entrant==0].Target_Country.nunique()}")

print("\nFiles Created")

print("------------------------------------")

print("Parallel_Trends_Figure.png")

print("Average_Distance_By_Year.csv")

print("Pretrend_Test.csv")

print("EventStudy_Data.csv")

print("\n"+"="*70)

print("SCRIPT COMPLETED SUCCESSFULLY")

print("="*70)