# AMR Prediction Project

**Organism-Aware Model Selection for Cross-Gram Antimicrobial Resistance Prediction Using WGS and K-mer-Based Machine Learning**

MTech Dissertation (MCEN-491), Department of Computer Engineering, Jamia Millia Islamia

## Overview

This project predicts antimicrobial resistance (AMR) directly from whole-genome sequencing (WGS) data using an alignment-free, k-mer-based machine learning pipeline. It compares a Gram-negative organism (*E. coli*, resistance to Ciprofloxacin) and a Gram-positive organism (*S. aureus*, resistance to Erythromycin) under an identical workflow, to test whether organism-specific resistance mechanisms call for organism-specific model selection.

**Key finding:** The same pipeline selected different best models for each organism — Logistic Regression for *E. coli* (mutation-driven resistance) and Random Forest for *S. aureus* (gene-acquisition-driven resistance) — supporting an organism-aware model selection strategy rather than a one-size-fits-all approach.

## Pipeline

1. **Data collection** — WGS genomes and AMR phenotype labels (Resistant/Susceptible) downloaded from NCBI using `datasets.exe` (NCBI Datasets CLI, Windows).
   > Linux/Mac users: use the equivalent NCBI `datasets` CLI for your OS.

2. **Data cleaning & balancing** — `Scripts/Data_Cleaning_And_Dataset_Preparation/`
   Organism filter -> antibiotic filter -> assembly quality filter -> duplicate removal -> conflict removal -> class balancing (undersampling majority class).
   - *E. coli*: 2,628 isolates after balancing (1,314 Resistant / 1,314 Susceptible)
   - *S. aureus*: 738 isolates after balancing (369 Resistant / 369 Susceptible)

3. **K-mer feature extraction** — `Scripts/Feature_Extraction/`
   Alignment-free, reference-free k-mer counting (k=6, chosen after empirical sensitivity analysis across k=4-7). Each genome becomes a normalized frequency vector across all 4,096 possible 6-mers.
   - `kmer_extraction_ecoli.py`, `kmer_extraction_saureus.py`, `verify_kmer_output.py`

4. **K-mer length sensitivity analysis** — `Scripts/k_analysis.py`, `k_value_analyis_on_fixed_features.py`, `k_value_fixed_feature_count.py`, `fixed_feature_count_analysis.py`
   Empirically compares k=4 through k=7 on AUC and false negatives to justify k=6 as optimal for both organisms. `check.py` is a supporting/scratch script from this analysis.

5. **Feature selection** — two-stage pipeline reducing 4,096 k-mers to informative subsets:
   - **Stage 1: ANOVA F-test** (alpha = 0.001 for *E. coli*, alpha = 0.0001 for *S. aureus*) -> 2,789 features (*E. coli*), 712 features (*S. aureus*)
   - **Stage 2: Cross-validated permutation importance** -> further ranks features per model (final counts: LR 80/20, RF 40/12, XGB 30/10 for *E. coli*/*S. aureus* respectively)

6. **Model training & tuning** — `Scripts/ecoli_Modelling_scripts/`, `Scripts/saureus_modelling_scripts/`
   Logistic Regression, Random Forest, and XGBoost, tuned via 5-fold stratified cross-validation with GridSearchCV (optimizing F1-score, `class_weight=balanced`).

7. **Evaluation** — AUC-ROC, MCC, F1, false negatives (clinically weighted), and train/test AUC gap (overfitting check).

8. **Results visualization** — `notebooks/`
   Jupyter notebooks visualizing pipeline results and final outputs.

## Results

| Organism | Best Model | Test AUC | MCC | AUC Gap | FN |
|---|---|---|---|---|---|
| *E. coli* (Ciprofloxacin) | Logistic Regression | 0.873 | 0.597 | 0.051 | 57/263 |
| *S. aureus* (Erythromycin) | Random Forest | 0.819 | 0.487 | 0.068 | 18/74 |

Both organisms converged on **k=6** as the optimal k-mer length despite differing resistance mechanisms (point mutations in *gyrA/parC* for *E. coli*; *erm*-gene acquisition for *S. aureus*), and both selected models achieved performance comparable to published gene/SNP-based approaches despite using a simpler, reference-free feature representation.

## Repository structure

```
├── Scripts/
│   ├── Data_Cleaning_And_Dataset_Preparation/
│   ├── Feature_Extraction/
│   ├── ecoli_Modelling_scripts/
│   ├── saureus_modelling_scripts/
│   ├── k_analysis.py, k_value_analyis_on_fixed_features.py,
│   │   k_value_fixed_feature_count.py, fixed_feature_count_analysis.py
│   │   (k-mer length sensitivity analysis — determined k=6 as optimal)
│   └── check.py   (supporting script from the k-value analysis)
├── notebooks/                  # Results visualization notebooks
├── balanced_data/              # Feature-selected / balanced datasets
├── cleaned_data/                # Cleaned intermediate datasets
├── datasets.exe                 # NCBI Datasets CLI (Windows) for genome download
└── missing_ecoli.txt / missing_saureus.txt   # Logs of missing/unavailable genome entries
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
```

Install with:
```
pip install pandas numpy scikit-learn xgboost joblib matplotlib
```