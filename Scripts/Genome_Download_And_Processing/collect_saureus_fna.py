import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "saureus_genomes", "ncbi_dataset", "data")
DEST_DIR = os.path.join(BASE_DIR, "saureus_fna")

os.makedirs(DEST_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(SOURCE_DIR):

    for file in files:

        if file.endswith(".fna"):

            src = os.path.join(root, file)
            dst = os.path.join(DEST_DIR, f"genome_{count}.fna")

            shutil.copy(src, dst)

            count += 1

print("Total genomes collected:", count)