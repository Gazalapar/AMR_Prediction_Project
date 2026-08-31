import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
FILE_PATH = r"C:\AMR_Prediction_Project\Data\ecoli_kmer_features_20260320_092319.csv" 
# 👉 replace * with your actual filename

VALID_BASES = {"A", "C", "G", "T"}

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(FILE_PATH)

print("\nLoaded file successfully")
print("Shape:", df.shape)

# -------------------------------
# 1. CHECK SUM OF EACH ROW
# -------------------------------
print("\nChecking row-wise sum (should be ~1)...")

feature_df = df.drop(columns=["genome"])

row_sums = feature_df.sum(axis=1)

print("\nFirst 10 row sums:")
print(row_sums.head(10))

# Check if any row is far from 1
bad_rows = row_sums[(row_sums < 0.95) | (row_sums > 1.05)]

if len(bad_rows) == 0:
    print("\nPASS: All rows sum approximately to 1")
else:
    print("\nWARNING: Some rows do not sum to 1")
    print(bad_rows.head())

# -------------------------------
# 2. CHECK INVALID K-MERS
# -------------------------------
print("\nChecking for invalid k-mers...")

invalid_kmers = []

for col in df.columns:
    if col == "genome":
        continue

    if not set(col).issubset(VALID_BASES):
        invalid_kmers.append(col)

if len(invalid_kmers) == 0:
    print("PASS: All k-mers are valid (A,C,G,T only)")
else:
    print("ERROR: Invalid k-mers found!")
    print(invalid_kmers[:10])

# -------------------------------
# 3. CHECK FOR ALL-ZERO COLUMNS
# -------------------------------
print("\nChecking for zero-only columns...")

zero_cols = feature_df.columns[(feature_df.sum(axis=0) == 0)]

if len(zero_cols) == 0:
    print("PASS: No zero-only columns")
else:
    print("WARNING: Some columns are all zero")
    print(zero_cols[:10])

# -------------------------------
# 4. BASIC STATS
# -------------------------------
print("\nBasic stats:")

print("Min value:", feature_df.min().min())
print("Max value:", feature_df.max().max())

print("\nVerification COMPLETE")