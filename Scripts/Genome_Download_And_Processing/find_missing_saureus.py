import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

accession_file = os.path.join(BASE_DIR, "balanced_data", "saureus_accessions.txt")
download_dir = os.path.join(BASE_DIR, "saureus_genomes")

accessions = pd.read_csv(accession_file, header=None)[0].tolist()

downloaded = set()

for file in os.listdir(download_dir):
    if file.endswith(".zip"):
        downloaded.add(file.replace(".zip",""))

missing = [acc for acc in accessions if acc not in downloaded]

print("Missing genomes:", len(missing))

with open(os.path.join(BASE_DIR,"missing_saureus.txt"),"w") as f:
    for m in missing:
        f.write(m+"\n")

print("Saved missing accessions to missing_saureus.txt")