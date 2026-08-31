import os
import datetime
from collections import Counter
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
K = 6

BASE_DIR = r"C:\AMR_Prediction_Project"
GENOME_FOLDER = os.path.join(BASE_DIR, "saureus_genomes")   # root folder

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(BASE_DIR, "Data", f"saureus_kmer_features_{timestamp}.csv")

VALID_BASES = {"A", "C", "G", "T"}

# -------------------------------
# FAST FASTA READER
# -------------------------------
def read_fasta_fast(file):
    seq = []

    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                continue

            line = line.strip().upper()

            if set(line).issubset({"A", "C", "G", "T", "N"}):
                seq.append(line)

    return "".join(seq)


# -------------------------------
# K-MER COUNTING
# -------------------------------
def count_kmers(seq, k):
    counts = Counter()

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]

        if set(kmer).issubset(VALID_BASES):
            counts[kmer] += 1

    total = sum(counts.values())

    # ✅ NORMALIZATION (VERY IMPORTANT)
    if total > 0:
        for kmer in counts:
            counts[kmer] /= total

    return counts


# -------------------------------
# EXTRACT GCA GENOME ID (FIXED)
# -------------------------------
def extract_genome_id(file_path):
    """
    Extract GCA ID from path
    Example:
    .../GCA_000626615.3/genomic.fna → GCA_000626615.3
    """
    parts = file_path.split(os.sep)

    for part in parts:
        if part.startswith("GCA_"):
            return part

    return None


# -------------------------------
# MAIN PIPELINE
# -------------------------------
all_rows = []
skipped = 0
total_files = 0

print("\n🔍 Scanning genome folders...\n")

# ✅ Walk through nested folders
for root, _, files in os.walk(GENOME_FOLDER):
    for file in files:

        if not file.endswith(".fna"):
            continue

        total_files += 1

        path = os.path.join(root, file)
        genome_id = extract_genome_id(path)

        if genome_id is None:
            print(f"⚠️ Skipping (no GCA ID): {file}")
            skipped += 1
            continue

        print(f"Processing {total_files}: {genome_id}")

        seq = read_fasta_fast(path)

        if len(seq) < K:
            print(f"⚠️ Skipping {genome_id} (too short)")
            skipped += 1
            continue

        kmers = count_kmers(seq, K)

        if sum(kmers.values()) == 0:
            print(f"⚠️ Skipping {genome_id} (no valid kmers)")
            skipped += 1
            continue

        kmers["genome"] = genome_id
        all_rows.append(kmers)


print(f"\n✅ Total processed: {len(all_rows)}")
print(f"⚠️ Skipped genomes: {skipped}")

# -------------------------------
# CREATE FEATURE MATRIX
# -------------------------------
df = pd.DataFrame(all_rows).fillna(0)

# move genome column first
df.insert(0, "genome", df.pop("genome"))

# -------------------------------
# ENSURE OUTPUT DIRECTORY EXISTS
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# -------------------------------
# SAVE
# -------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 DONE")
print("Saved to:", OUTPUT_FILE)