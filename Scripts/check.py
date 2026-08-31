import pandas as pd, os

BASE = r"C:\AMR_Prediction_Project"

acc  = pd.read_csv(os.path.join(BASE, "balanced_data", "saureus_accessions.txt"), header=None)[0].tolist()
feat = pd.read_csv(os.path.join(BASE, "Data", "saureus_kmer_features_20260414_093319.csv"), usecols=["genome"])

print("Accession 0:", acc[0])
print("Feature IDs:", feat["genome"].head(3).tolist())
print("Acc 0 in features:", acc[0] in feat["genome"].values)