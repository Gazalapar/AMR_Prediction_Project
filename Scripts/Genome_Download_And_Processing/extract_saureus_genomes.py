import os
import zipfile

print("\nExtracting S. aureus genome zip files")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME_DIR = os.path.join(BASE_DIR, "saureus_genomes")

count = 0

for file in os.listdir(GENOME_DIR):

    if file.endswith(".zip"):

        path = os.path.join(GENOME_DIR, file)

        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(GENOME_DIR)

            count += 1

        except:
            print("Error extracting:", file)

print("\nExtraction completed")
print("Files extracted:", count)