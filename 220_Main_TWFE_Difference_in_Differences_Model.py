"""
========================================================================
SCRIPT 220 : Main Difference-in-Differences (TWFE)
========================================================================

Purpose
-------
Estimates the main Two-Way Fixed Effects Difference-in-Differences model
testing whether BRICS+ accession reduced UNGA voting distance.

Inputs
------
BRICS_Analysis_Panel.csv

Outputs
-------
DiD_Coefficients.csv
DiD_Model_Summary.txt

========================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

print("=" * 72)
print("SCRIPT 220 : MAIN DIFFERENCE-IN-DIFFERENCES MODEL")
print("=" * 72)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    r"D:\PHD notes\BRICS Cohesion\Required dataset\BRICS_Analysis_Panel.csv"
)

print("\nDataset Loaded")
print(f"Observations : {len(df):,}")
print(f"Countries    : {df.Target_Country.nunique()}")
print(f"Years        : {df.Year.min()} - {df.Year.max()}")

# ==========================================================
# LOG TRANSFORM
# ==========================================================

df["log_GDP"] = np.log(df["GDP_per_Capita_PPP"] + 1)
df["log_Pop"] = np.log(df["Population"] + 1)

# ==========================================================
# VARIABLES REQUIRED
# ==========================================================

model_vars = [

    "Avg_Distance_to_BRICS",

    "Treated_x_Post",

    "v2x_polyarchy",

    "log_GDP",

    "log_Pop",

    "Trade_pct_GDP",

    "Target_Country",

    "Year"

]

df = df[model_vars].dropna().copy()

print("\nObservations Used")
print(f"{len(df):,}")

# ==========================================================
# MAIN TWFE MODEL
# ==========================================================

formula = """

Avg_Distance_to_BRICS ~

Treated_x_Post

+ v2x_polyarchy

+ log_GDP

+ log_Pop

+ Trade_pct_GDP

+ C(Target_Country)

+ C(Year)

"""

print("\nEstimating model...")

model = smf.ols(

    formula=formula,

    data=df

).fit(

    cov_type="cluster",

    cov_kwds={

        "groups": df["Target_Country"]

    }

)

# ==========================================================
# RESULTS
# ==========================================================

results = model.summary2().tables[1]

variables = [

    "Treated_x_Post",

    "v2x_polyarchy",

    "log_GDP",

    "log_Pop",

    "Trade_pct_GDP"

]

results = results.loc[variables]

print("\n")
print("=" * 72)
print("MAIN TWFE RESULTS")
print("=" * 72)

print(results)

# ==========================================================
# MODEL FIT
# ==========================================================

print("\n")
print("=" * 72)

print("MODEL FIT")

print("=" * 72)

print(f"R²            : {model.rsquared:.4f}")

print(f"Adj. R²       : {model.rsquared_adj:.4f}")

print(f"N             : {int(model.nobs)}")

print(f"Countries     : {df.Target_Country.nunique()}")

print(f"Years         : {df.Year.nunique()}")

# ==========================================================
# HYPOTHESIS TEST
# ==========================================================

coef = model.params["Treated_x_Post"]

p = model.pvalues["Treated_x_Post"]

ci_low = model.conf_int().loc["Treated_x_Post",0]

ci_high = model.conf_int().loc["Treated_x_Post",1]

print("\n")
print("=" * 72)
print("HYPOTHESIS TEST")
print("=" * 72)

print(f"Coefficient : {coef:.5f}")

print(f"P-value     : {p:.4f}")

print(f"95% CI      : ({ci_low:.5f}, {ci_high:.5f})")

if p < 0.05:

    if coef < 0:

        print("\nResult")

        print("------------------------------------------")

        print("Evidence supports H1.")

        print("Following BRICS+ accession,")

        print("entrant countries exhibited")

        print("lower voting distance from")

        print("the BRICS core.")

    else:

        print("\nResult")

        print("------------------------------------------")

        print("Significant effect detected,")

        print("but voting distance increased.")

else:

    print("\nResult")

    print("------------------------------------------")

    print("No statistically significant")

    print("treatment effect detected.")

# ==========================================================
# PERMUTATION TEST (RANDOMIZATION INFERENCE)
# ==========================================================
print("\n" + "=" * 72)
print("EXECUTING PERMUTATION TEST (1,000 ITERATIONS)")
print("=" * 72)
print("This will take a minute. Please wait...")

np.random.seed(42) # For reproducibility
n_iterations = 1000
placebo_coefs = []

# Identify unique countries and the number of actually treated countries
all_countries = df["Target_Country"].unique()
# Our actual treated countries in the filtered sample
treated_countries = ["ARE", "EGY", "ETH", "IRN"]
n_treated = len(treated_countries)

# Pre-calculate the Post_2024 indicator to speed up the loop
df["Post_2024_Perm"] = (df["Year"] >= 2024).astype(int)

for i in range(n_iterations):
    # Randomly select n_treated countries to be "pseudo-treated"
    pseudo_treated_countries = np.random.choice(all_countries, size=n_treated, replace=False)
    
    # Create the pseudo-treatment interaction
    df["Pseudo_Treated"] = df["Target_Country"].isin(pseudo_treated_countries).astype(int)
    df["Pseudo_Treated_x_Post"] = df["Pseudo_Treated"] * df["Post_2024_Perm"]
    
    # Run the model
    placebo_formula = formula.replace("Treated_x_Post", "Pseudo_Treated_x_Post")
    
    try:
        placebo_model = smf.ols(formula=placebo_formula, data=df).fit(disp=0)
        placebo_coefs.append(placebo_model.params["Pseudo_Treated_x_Post"])
    except:
        continue # Skip if convergence fails for a specific random draw

# Calculate the empirical p-value
placebo_coefs = np.array(placebo_coefs)
empirical_p = np.mean(np.abs(placebo_coefs) >= np.abs(coef))

print(f"Main Model Coefficient       : {coef:.5f}")
print(f"Empirical P-value (Perm Test): {empirical_p:.4f}")

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(placebo_coefs, bins=50, color='lightgray', edgecolor='black', alpha=0.7)
plt.axvline(coef, color='red', linestyle='dashed', linewidth=2, label=f'True Effect: {coef:.3f}')
plt.axvline(np.percentile(placebo_coefs, 2.5), color='blue', linestyle='dotted', label='95% CI bounds')
plt.axvline(np.percentile(placebo_coefs, 97.5), color='blue', linestyle='dotted')
plt.title("Permutation Test: Distribution of Placebo Treatment Effects")
plt.xlabel("Estimated Coefficient (Pseudo_Treated_x_Post)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("Permutation_Test_Histogram.png", dpi=300)
plt.close()
print("Saved Permutation_Test_Histogram.png")

# ==========================================================
# SAVE OUTPUTS
# ==========================================================

results.to_csv(

    "DiD_Coefficients.csv"

)

with open(

    "DiD_Model_Summary.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(model.summary().as_text())

print("\nOutputs Saved")

print("-----------------------------")

print("DiD_Coefficients.csv")

print("DiD_Model_Summary.txt")

print("\n")

print("=" * 72)

print("SCRIPT COMPLETED SUCCESSFULLY")

print("=" * 72)