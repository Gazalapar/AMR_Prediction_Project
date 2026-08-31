import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

accession_file = os.path.join(BASE_DIR, "balanced_data", "ecoli_accessions.txt")
genome_dir = os.path.join(BASE_DIR, "ecoli_genomes")

# all expected genomes
accessions = pd.read_csv(accession_file, header=None)[0].tolist()

# downloaded genomes
downloaded = set()

for f in os.listdir(genome_dir):
    if f.endswith(".zip"):
        downloaded.add(f.replace(".zip",""))

missing = [a for a in accessions if a not in downloaded]

print("Missing genomes:", len(missing))

with open(os.path.join(BASE_DIR,"missing_ecoli.txt"),"w") as f:
    for m in missing:
        f.write(m+"\n")

print("Saved list to missing_ecoli.txt")