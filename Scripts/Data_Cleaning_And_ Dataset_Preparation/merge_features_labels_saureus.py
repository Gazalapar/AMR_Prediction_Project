import pandas as pd
import os

# -------------------------------
# CONFIG (centralized paths)
# -------------------------------
BASE_DIR = r"C:\AMR_Prediction_Project"

FEATURE_FILE = os.path.join(BASE_DIR, "Data", "saureus_kmer_features_20260414_093319.csv")
LABEL_FILE = os.path.join(BASE_DIR, "balanced_data", "saureus_erythromycin_balanced.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "Data", "saureus_final_ml_dataset.csv")

# -------------------------------
# Load files
# -------------------------------
features = pd.read_csv(FEATURE_FILE)
labels = pd.read_csv(LABEL_FILE)

# -------------------------------
# Rename for consistency
# -------------------------------
labels = labels.rename(columns={"Assembly": "genome"})

# -------------------------------
# Merge using INNER JOIN
# -------------------------------
merged = pd.merge(features, labels, on="genome", how="inner")

# -------------------------------
# Debug checks
# -------------------------------
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Merged shape:", merged.shape)

missing_in_features = set(labels["genome"]) - set(features["genome"])
print("Missing in features:", len(missing_in_features))

missing_in_labels = set(features["genome"]) - set(labels["genome"])
print("Missing in labels:", len(missing_in_labels))

# -------------------------------
# CLEAN FOR ML (IMPORTANT STEP)
# -------------------------------

# Keep only:
# - genome (optional but useful)
# - k-mer features (A,C,G,T patterns)
# - label

kmer_columns = [col for col in merged.columns if set(col).issubset({'A','C','G','T'})]

ml_df = merged[["genome"] + kmer_columns + ["label"]]

print("Final ML dataset shape:", ml_df.shape)

# -------------------------------
# Save final dataset
# -------------------------------
ml_df.to_csv(OUTPUT_FILE, index=False)

print(f"Final dataset saved at: {OUTPUT_FILE}")