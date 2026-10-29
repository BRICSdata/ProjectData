import pandas as pd
import numpy as np

# 1. Load the Tab-Separated File
# Using 'sep="\t"' is crucial because it's a .tab file
print("Loading dataset...")
df = pd.read_csv('IdealPointDyads1946-2025.csv',  encoding="cp1252")

print("\n--- DATA AUDIT ---")
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Maximum year in dataset: {df['year'].max()}")
print("------------------\n")

# 2. Filter the Timeline (2018 to 2025)
df_recent = df[(df['year'] >= 2018) & (df['year'] <= 2025)].copy()

# 3. Define the ISO3 Codes for your cohorts
core_brics = ['BRA', 'RUS', 'IND', 'CHN', 'ZAF']
# Assuming UAE (ARE), Egypt (EGY), Ethiopia (ETH), Iran (IRN)
new_entrants = ['ARE', 'EGY', 'ETH', 'IRN'] 

# 4. Filter for Core-to-Target Dyads
# We only want rows where ONE country is a Core member, and the OTHER is not.
# Because the Voeten dataset can list a country in either the '1' or '2' column, we check both.

# Condition A: Country 1 is Core, Country 2 is Target
cond_A = df_recent['iso3c1'].isin(core_brics) & ~df_recent['iso3c2'].isin(core_brics)
# Condition B: Country 2 is Core, Country 1 is Target
cond_B = df_recent['iso3c2'].isin(core_brics) & ~df_recent['iso3c1'].isin(core_brics)

# Combine and apply filter
df_brics_dyads = df_recent[cond_A | cond_B].copy()

# 5. Standardize the Columns
# We want a unified structure: 'Core_Country', 'Target_Country', 'Year', 'IPD'
def standardize_row(row):
    if row['iso3c1'] in core_brics:
        return pd.Series({
            'Core_Country': row['iso3c1'],
            'Target_Country': row['iso3c2'],
            'Year': row['year'],
            'IPD': row['AbsIdealDiff'] # Your primary Dependent Variable
        })
    else:
        return pd.Series({
            'Core_Country': row['iso3c2'],
            'Target_Country': row['iso3c1'],
            'Year': row['year'],
            'IPD': row['AbsIdealDiff']
        })

print("Restructuring dyads...")
df_clean = df_brics_dyads.apply(standardize_row, axis=1)

# 6. Create the Difference-in-Differences Variables
df_clean['Is_Entrant'] = df_clean['Target_Country'].isin(new_entrants).astype(int)
df_clean['Post_2024'] = (df_clean['Year'] >= 2024).astype(int)
df_clean['Treated_x_Post'] = df_clean['Is_Entrant'] * df_clean['Post_2024']

# === NEW CODE STARTS HERE ===
# 6.5 Create Sub-Core Distances (China & Russia vs. Brazil, India, South Africa)
df_clean['IPD_CN_RU'] = np.where(df_clean['Core_Country'].isin(['CHN', 'RUS']), df_clean['IPD'], np.nan)
df_clean['IPD_BR_IN_ZA'] = np.where(df_clean['Core_Country'].isin(['BRA', 'IND', 'ZAF']), df_clean['IPD'], np.nan)

# 7. Aggregate to a Single BRICS Target Score 
df_bloc_distance = df_clean.groupby(['Target_Country', 'Year']).agg({
    'IPD': 'mean',
    'IPD_CN_RU': 'mean',       # Aggregate the new CN-RU column
    'IPD_BR_IN_ZA': 'mean',    # Aggregate the new BR-IN-ZA column
    'Is_Entrant': 'max',
    'Post_2024': 'max',
    'Treated_x_Post': 'max'
}).reset_index()

# Rename for clarity
df_bloc_distance.rename(columns={
    'IPD': 'Avg_Distance_to_BRICS',
    'IPD_CN_RU': 'Avg_Distance_to_CN_RU',       # Rename for output
    'IPD_BR_IN_ZA': 'Avg_Distance_to_BR_IN_ZA'  # Rename for output
}, inplace=True)
# === NEW CODE ENDS HERE ===

# 8. Export the Final Cleaned Data
df_bloc_distance.to_csv('BRICS_IPD_Panel_2018_2025.csv', index=False)
print("Data prepped and exported successfully!")
