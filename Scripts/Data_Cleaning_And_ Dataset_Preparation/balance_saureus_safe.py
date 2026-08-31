import pandas as pd
import os

print("\nCreating balanced erythromycin dataset for S.aureus")

# ------------------------------------------------
# Project path
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

R_FILE = os.path.join(BASE_DIR, "cleaned_data", "saureus_R_clean.csv")
S_FILE = os.path.join(BASE_DIR, "cleaned_data", "saureus_S_clean.csv")

# ------------------------------------------------
# Load cleaned datasets
# ------------------------------------------------

df_R = pd.read_csv(R_FILE)
df_S = pd.read_csv(S_FILE)

print("\nClean dataset size")
print("Resistant:", len(df_R))
print("Susceptible:", len(df_S))

# ------------------------------------------------
# Ensure unique assemblies
# ------------------------------------------------

df_R = df_R.drop_duplicates(subset=["Assembly"])
df_S = df_S.drop_duplicates(subset=["Assembly"])

# ------------------------------------------------
# Balance dataset
# ------------------------------------------------

sample_size = min(len(df_R), len(df_S))

print("\nBalancing dataset with", sample_size, "isolates per class")

df_R_sample = df_R.sample(n=sample_size, random_state=42)
df_S_sample = df_S.sample(n=sample_size, random_state=42)

# Add labels
df_R_sample["label"] = 1
df_S_sample["label"] = 0

# Combine datasets
balanced = pd.concat([df_R_sample, df_S_sample])

# Shuffle dataset
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------------
# Integrity checks
# ------------------------------------------------

print("\nDataset integrity check")

print("Total isolates:", len(balanced))
print("Unique assemblies:", balanced["Assembly"].nunique())
print("Duplicate assemblies:", balanced["Assembly"].duplicated().sum())

print("\nClass distribution")
print(balanced["label"].value_counts())

# ------------------------------------------------
# Save balanced dataset
# ------------------------------------------------

out_folder = os.path.join(BASE_DIR, "balanced_data")
os.makedirs(out_folder, exist_ok=True)

balanced.to_csv(
    os.path.join(out_folder, "saureus_erythromycin_balanced.csv"),
    index=False
)

# ------------------------------------------------
# Save assembly accession lists
# (used later for genome download)
# ------------------------------------------------

balanced["Assembly"].to_csv(
    os.path.join(out_folder, "assembly_accessions.txt"),
    index=False,
    header=False
)

print("\nBalanced dataset saved to balanced_data/")
print("Pipeline stage completed successfully")