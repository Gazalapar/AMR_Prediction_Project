import os
import shutil

print("\nCollecting genome FASTA files")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "ecoli_genomes", "ncbi_dataset", "data")
DEST_DIR = os.path.join(BASE_DIR, "ecoli_fna")

os.makedirs(DEST_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(SOURCE_DIR):

    for file in files:

        if file.endswith(".fna"):

            src = os.path.join(root, file)
            dst = os.path.join(DEST_DIR, file)

            if not os.path.exists(dst):

                shutil.move(src, dst)
                count += 1

print("Genomes moved:", count)