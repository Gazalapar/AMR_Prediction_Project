import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

missing_file = os.path.join(BASE_DIR,"missing_saureus.txt")
datasets = os.path.join(BASE_DIR,"datasets.exe")
genome_dir = os.path.join(BASE_DIR,"saureus_genomes")

with open(missing_file) as f:
    accessions = [line.strip() for line in f]

print("Downloading missing genomes:", len(accessions))

for i,acc in enumerate(accessions):

    print(f"{i+1}/{len(accessions)} {acc}")

    cmd = [
        datasets,
        "download",
        "genome",
        "accession",
        acc,
        "--include",
        "genome",
        "--filename",
        os.path.join(genome_dir,f"{acc}.zip")
    ]

    subprocess.run(cmd)