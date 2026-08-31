import os
import datetime
from collections import Counter
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
K = 6
GENOME_FOLDER = r"C:\AMR_Prediction_Project\ecoli_fna"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"Data/ecoli_kmer_features_{timestamp}.csv"

VALID_BASES = {"A", "C", "G", "T"}

# -------------------------------
# FAST FASTA READER (SAFE)
# -------------------------------
def read_fasta_fast(file):
    seq = []

    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                continue  # skip header

            line = line.strip().upper()

            # keep only valid DNA lines
            if set(line).issubset({"A", "C", "G", "T", "N"}):
                seq.append(line)

    return "".join(seq)

# -------------------------------
# KMER COUNTING
# -------------------------------
def count_kmers(seq, k):
    counts = Counter()

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]

        # keep only pure DNA kmers
        if set(kmer).issubset(VALID_BASES):
            counts[kmer] += 1

    total = sum(counts.values())

    # normalize
    if total > 0:
        for kmer in counts:
            counts[kmer] /= total

    return counts

# -------------------------------
# GENOME ID EXTRACTION
# -------------------------------
def extract_genome_id(filename):
    parts = filename.split("_")
    return parts[0] + "_" + parts[1]

# -------------------------------
# MAIN PIPELINE
# -------------------------------
all_rows = []

# ✅ Only .fna files + sorted (reproducible)
files = sorted([f for f in os.listdir(GENOME_FOLDER) if f.endswith(".fna")])

print("Total genomes:", len(files))

skipped = 0

for idx, file in enumerate(files):

    print(f"Processing {idx+1}/{len(files)}: {file}")

    path = os.path.join(GENOME_FOLDER, file)
    genome_id = extract_genome_id(file)

    seq = read_fasta_fast(path)

    # 🚨 skip empty sequences
    if len(seq) < K:
        print(f"Skipping {file} (sequence too short)")
        skipped += 1
        continue

    kmers = count_kmers(seq, K)

    # 🚨 skip genomes with no valid kmers
    if sum(kmers.values()) == 0:
        print(f"Skipping {file} (no valid kmers)")
        skipped += 1
        continue

    kmers["genome"] = genome_id
    all_rows.append(kmers)

print(f"\nSkipped genomes: {skipped}")

# -------------------------------
# SAVE FEATURE MATRIX
# -------------------------------
df = pd.DataFrame(all_rows).fillna(0)

# move genome column to front
df.insert(0, "genome", df.pop("genome"))

df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE")
print("Saved to:", OUTPUT_FILE)