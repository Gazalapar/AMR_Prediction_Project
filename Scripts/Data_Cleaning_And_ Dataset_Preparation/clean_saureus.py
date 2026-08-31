import pandas as pd
import os

print("\nCleaning S.aureus dataset for ERYTHROMYCIN")

# ------------------------------------------------
# Project path
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

R_FILE = os.path.join(BASE_DIR, "Data", "isolates_saureus_R.csv")
S_FILE = os.path.join(BASE_DIR, "Data", "isolates_saureus_S.csv")

print("Loading files...")

df_R = pd.read_csv(R_FILE)
df_S = pd.read_csv(S_FILE)

# ------------------------------------------------
# Clean column names
# ------------------------------------------------

df_R.columns = df_R.columns.str.strip()
df_S.columns = df_S.columns.str.strip()

print("\nOriginal rows")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Keep only Staphylococcus aureus
# ------------------------------------------------

df_R = df_R[df_R["#Organism group"].str.contains("aureus", case=False, na=False)]
df_S = df_S[df_S["#Organism group"].str.contains("aureus", case=False, na=False)]

print("\nAfter organism filtering")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Filter erythromycin phenotype
# Dataset format example:
# chloramphenicol=S,ciprofloxacin=R,erythromycin=R
# ------------------------------------------------

df_R = df_R[df_R["AST phenotypes"].str.contains(r"erythromycin=R\b", na=False)]
df_S = df_S[df_S["AST phenotypes"].str.contains(r"erythromycin=S\b", na=False)]

print("\nAfter erythromycin filtering")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Remove rows without assembly accession
# ------------------------------------------------

df_R = df_R[df_R["Assembly"].notna()]
df_S = df_S[df_S["Assembly"].notna()]

# ------------------------------------------------
# Remove duplicate assemblies
# ------------------------------------------------

df_R = df_R.drop_duplicates(subset=["Assembly"])
df_S = df_S.drop_duplicates(subset=["Assembly"])

print("\nAfter removing duplicate assemblies")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Remove conflicting assemblies (present in both R and S)
# ------------------------------------------------

common = set(df_R["Assembly"]).intersection(set(df_S["Assembly"]))

df_R = df_R[~df_R["Assembly"].isin(common)]
df_S = df_S[~df_S["Assembly"].isin(common)]

print("\nRemoved conflicting assemblies:", len(common))

# ------------------------------------------------
# Final dataset size
# ------------------------------------------------

print("\nAfter cleaning")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Save cleaned datasets
# ------------------------------------------------

clean_folder = os.path.join(BASE_DIR, "cleaned_data")
os.makedirs(clean_folder, exist_ok=True)

df_R.to_csv(
    os.path.join(clean_folder, "saureus_R_clean.csv"),
    index=False
)

df_S.to_csv(
    os.path.join(clean_folder, "saureus_S_clean.csv"),
    index=False
)

print("\nCleaning completed successfully")
print("Files saved to cleaned_data/")