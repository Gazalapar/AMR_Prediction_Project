import pandas as pd
import os
from sklearn.model_selection import train_test_split
 
# ===============================
ORGANISM = "ecoli"
DATA_FILE = r"C:\AMR_Prediction_Project\Data\ecoli_final_ml_dataset.csv"
# ===============================
 
BASE_DIR = r"C:\AMR_Prediction_Project"
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
print("\n==============================")
print(f"Running for: {ORGANISM}")
print("==============================")
 
print(f"\n📥 Reading dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
 
print("\nShape:", df.shape)
print("Columns:", df.columns.tolist())
 
# Validate required columns
if "label" not in df.columns:
    raise ValueError("❌ 'label' column not found in dataset")
if "genome" not in df.columns:
    raise ValueError("❌ 'genome' column not found in dataset")
 
# Check class distribution
print("\n📊 Class distribution:")
print(df["label"].value_counts())
print(f"Class balance: {df['label'].value_counts(normalize=True).round(3).to_dict()}")
 
# Separate components
genome_ids = df["genome"]
X = df.drop(columns=["label", "genome"])
y = df["label"]
 
print(f"\nFeature matrix shape: {X.shape}")
print(f"Class counts - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
 
# Stratified split
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, genome_ids,
    test_size=0.2,
    stratify=y,
    random_state=42
)
 
print(f"\n✅ Split sizes:")
print(f"  Train: {len(X_train)} samples")
print(f"  Test : {len(X_test)} samples")
print(f"  Train class dist: {y_train.value_counts().to_dict()}")
print(f"  Test  class dist: {y_test.value_counts().to_dict()}")
 
# Save with genome column
train_df = pd.concat([
    g_train.reset_index(drop=True),
    X_train.reset_index(drop=True),
    y_train.reset_index(drop=True)
], axis=1)
 
test_df = pd.concat([
    g_test.reset_index(drop=True),
    X_test.reset_index(drop=True),
    y_test.reset_index(drop=True)
], axis=1)
 
train_out = os.path.join(OUTPUT_DIR, "train_full.csv")
test_out  = os.path.join(OUTPUT_DIR, "test_full.csv")
 
train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out,  index=False)
 
print(f"\n📁 Saved train_full.csv → {train_out}")
print(f"📁 Saved test_full.csv  → {test_out}")
print("\n✅ Split done successfully!")