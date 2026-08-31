import os
import pandas as pd
import subprocess

print("\nDownloading S. aureus genomes")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACCESSION_FILE = os.path.join(BASE_DIR, "balanced_data", "saureus_accessions.txt")
DATASETS_EXE = os.path.join(BASE_DIR, "datasets.exe")
GENOME_DIR = os.path.join(BASE_DIR, "saureus_genomes")

os.makedirs(GENOME_DIR, exist_ok=True)

accessions = pd.read_csv(ACCESSION_FILE, header=None)

print("Total genomes:", len(accessions))

for i, acc in enumerate(accessions[0]):

    print(f"Downloading {i+1}/{len(accessions)} : {acc}")

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

print("\nDownload completed")