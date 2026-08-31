import os
import pandas as pd
import subprocess

print("\nDownloading E.coli genomes from NCBI")

# ------------------------------------------------
# Project paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACCESSION_FILE = os.path.join(BASE_DIR, "balanced_data", "ecoli_accessions.txt")
DATASETS_EXE = os.path.join(BASE_DIR, "datasets.exe")
GENOME_DIR = os.path.join(BASE_DIR, "genomes")

os.makedirs(GENOME_DIR, exist_ok=True)

# ------------------------------------------------
# Load accession IDs
# ------------------------------------------------
accessions = pd.read_csv(ACCESSION_FILE, header=None)

print("Total genomes to download:", len(accessions))

# ------------------------------------------------
# Download genomes one by one
# ------------------------------------------------
for i, acc in enumerate(accessions[0]):

    print(f"\nDownloading {i+1}/{len(accessions)} : {acc}")

    cmd = [
        DATASETS_EXE,
        "download",
        "genome",
        "accession",
        acc,
        "--include",
        "genome",
        "--filename",
        os.path.join(GENOME_DIR, f"{acc}.zip")
    ]

    subprocess.run(cmd)

print("\nAll downloads completed")