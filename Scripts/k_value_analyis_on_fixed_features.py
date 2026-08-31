"""
k_value_analysis_fixed_features.py
====================================
SAME as original k_value_analysis.py BUT:
- ALL three models use SAME fixed feature count (80 for E.coli)
- This validates that XGB wins at k=6 regardless of feature count
- Produces same plots as original for direct comparison

Change from original:
  top_k_lr  = 80  ← same
  top_k_rf  = 80  ← changed from 40
  top_k_xgb = 80  ← changed from 30

Output saved to:
  Data/results_{organism}/k_analysis_fixed/
"""

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef,
    f1_score, recall_score, precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier


# ============================================================
# CONFIG — CHANGE ONLY THIS SECTION
# ============================================================

ORGANISM = "saureus"        # "ecoli" or "saureus"
K_VALUES = [6, 7]  # same as original

BASE_DIR = r"C:\AMR_Prediction_Project"

GENOME_FOLDERS = {
    "ecoli"  : os.path.join(BASE_DIR, "ecoli_fna"),
    "saureus": os.path.join(BASE_DIR, "saureus_fna"),
}

LABEL_FILES = {
    "ecoli"  : os.path.join(BASE_DIR, "balanced_data",
                            "ecoli_ciprofloxacin_balanced.csv"),
    "saureus": os.path.join(BASE_DIR, "balanced_data",
                            "saureus_erythromycin_balanced.csv"),
}

# ── KEY CHANGE — all models use same feature count ────────────
CONFIG = {
    "ecoli": {
        "anova_alpha"   : 0.001,
        "top_k_lr"      : 80,   # same as original
        "top_k_rf"      : 80,   # CHANGED from 40 → 80
        "top_k_xgb"     : 80,   # CHANGED from 30 → 80
        "rf_max_depth"  : 4,
        "rf_min_leaf"   : 25,
        "xgb_lambda"    : 30,
        "xgb_min_child" : 25,
        "train_n"       : 2101,
        "antibiotic"    : "Ciprofloxacin",
    },
    "saureus": {
        "anova_alpha"   : 0.0001,
        "top_k_lr"      : 20,   # same as original
        "top_k_rf"      : 20,   # CHANGED from 12 → 20
        "top_k_xgb"     : 20,   # CHANGED from 10 → 20
        "rf_max_depth"  : 4,
        "rf_min_leaf"   : 20,
        "xgb_lambda"    : 20,
        "xgb_min_child" : 20,
        "train_n"       : 590,
        "antibiotic"    : "Erythromycin",
    },
}

# ── Derived values ────────────────────────────────────────────
cfg           = CONFIG[ORGANISM]
GENOME_FOLDER = GENOME_FOLDERS[ORGANISM]
LABEL_FILE    = LABEL_FILES[ORGANISM]

# ── SAVE TO DIFFERENT FOLDER — keeps original safe ────────────
OUTPUT_DIR = os.path.join(BASE_DIR, "Data",
                          f"results_{ORGANISM}", "k_analysis_fixed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CACHE FROM ORIGINAL — reuse existing extracted features ───
# Point to original k_analysis folder to reuse cached CSVs
CACHE_DIR = os.path.join(BASE_DIR, "Data",
                         f"results_{ORGANISM}", "k_analysis")

# ── S. aureus genome ID mapping ───────────────────────────────
SAUREUS_ID_MAP = {}
if ORGANISM == "saureus":
    acc_file = os.path.join(BASE_DIR, "balanced_data",
                            "saureus_accessions.txt")
    if os.path.exists(acc_file):
        acc_list = pd.read_csv(acc_file, header=None)[0].tolist()
        for i, acc in enumerate(acc_list):
            SAUREUS_ID_MAP[f"genome_{i}.fna"] = acc
        print(f"Loaded {len(SAUREUS_ID_MAP)} S. aureus ID mappings")

VALID_BASES = {"A", "C", "G", "T"}
COLORS      = {"LR": "#1D9E75", "RF": "#534AB7", "XGB": "#D85A30"}
MARKERS     = {"LR": "o",       "RF": "s",        "XGB": "^"}

# ============================================================
# STARTUP
# ============================================================
fixed_n = cfg["top_k_lr"]  # all models use this count
print(f"\n{'='*60}")
print(f"  K-VALUE ANALYSIS — FIXED {fixed_n} FEATURES ALL MODELS")
print(f"  Organism  : {ORGANISM.upper()}")
print(f"  Antibiotic: {cfg['antibiotic']}")
print(f"  K values  : {K_VALUES}")
print(f"  Features  : ALL models = {fixed_n} (fixed)")
print(f"  Output    : {OUTPUT_DIR}")
print(f"  Cache dir : {CACHE_DIR}")
print(f"{'='*60}\n")


# ============================================================
# GENOME READING — same as original
# ============================================================
def read_fasta(filepath):
    seq = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                continue
            line = line.strip().upper()
            if set(line).issubset({"A","C","G","T","N"}):
                seq.append(line)
    return "".join(seq)


def count_kmers_normalized(seq, k):
    counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if set(kmer).issubset(VALID_BASES):
            counts[kmer] += 1
    total = sum(counts.values())
    if total > 0:
        for kmer in counts:
            counts[kmer] /= total
    return counts


def get_genome_id(filename):
    if ORGANISM == "saureus" and filename in SAUREUS_ID_MAP:
        return SAUREUS_ID_MAP[filename]
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return os.path.splitext(filename)[0]


# ============================================================
# FEATURE EXTRACTION — uses original cache if available
# ============================================================
def extract_features(k):
    # Check original cache first — no need to re-extract
    orig_cache = os.path.join(CACHE_DIR, f"kmer_features_k{k}.csv")
    new_cache  = os.path.join(OUTPUT_DIR, f"kmer_features_k{k}.csv")

    if os.path.exists(orig_cache):
        print(f"[k={k}] Using original cache → {orig_cache}")
        df = pd.read_csv(orig_cache)
        print(f"[k={k}] Loaded: {df.shape[0]} genomes, "
              f"{df.shape[1]-1} k-mer features")
        return df

    if os.path.exists(new_cache):
        print(f"[k={k}] Cache found → loading...")
        df = pd.read_csv(new_cache)
        return df

    print(f"[k={k}] Extracting features "
          f"(4^{k} = {4**k:,} possible k-mers)...")

    fna_files = sorted([f for f in os.listdir(GENOME_FOLDER)
                        if f.endswith(".fna")])
    print(f"[k={k}] {len(fna_files)} .fna files found")

    all_rows = []
    skipped  = 0

    for idx, fname in enumerate(fna_files):
        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx+1}/{len(fna_files)}")
        fpath     = os.path.join(GENOME_FOLDER, fname)
        genome_id = get_genome_id(fname)
        seq       = read_fasta(fpath)
        if len(seq) < k:
            skipped += 1
            continue
        kmers = count_kmers_normalized(seq, k)
        if len(kmers) == 0:
            skipped += 1
            continue
        kmers["genome"] = genome_id
        all_rows.append(kmers)

    df = pd.DataFrame(all_rows).fillna(0)
    df.insert(0, "genome", df.pop("genome"))
    df.to_csv(new_cache, index=False)
    print(f"[k={k}] Extracted {len(df)} genomes → cached")
    return df


# ============================================================
# MERGE LABELS — same as original
# ============================================================
def merge_labels(kmer_df):
    labels = pd.read_csv(LABEL_FILE)
    if "Assembly" in labels.columns:
        labels = labels.rename(columns={"Assembly": "genome"})
    merged = pd.merge(
        kmer_df, labels[["genome","label"]],
        on="genome", how="inner"
    )
    kmer_cols = [c for c in merged.columns
                 if set(c).issubset({"A","C","G","T"})]
    ml_df = merged[["genome"] + kmer_cols + ["label"]]
    print(f"Merged: {len(ml_df)} genomes  "
          f"Resistant={(ml_df['label']==1).sum()}  "
          f"Susceptible={(ml_df['label']==0).sum()}")
    return ml_df


# ============================================================
# ML PIPELINE — fixed features for all models
# ============================================================
def run_for_k(k):
    fixed_n = cfg["top_k_lr"]  # same for all models

    print(f"\n{'='*60}")
    print(f"  k={k} | {ORGANISM.upper()} | "
          f"ALL models = {fixed_n} features")
    print(f"{'='*60}")

    kmer_df = extract_features(k)
    ml_df   = merge_labels(kmer_df)

    if len(ml_df) < 100:
        print(f"SKIP: only {len(ml_df)} samples after merge")
        return None

    X = ml_df.drop(columns=["genome","label"])
    y = ml_df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    n_train = len(X_tr)
    print(f"Split: {n_train} train / {len(X_te)} test")

    # ANOVA filter
    sel   = SelectFpr(f_classif, alpha=cfg["anova_alpha"])
    Xtr_a = sel.fit_transform(X_tr, y_tr)
    Xte_a = sel.transform(X_te)
    n_anova = Xtr_a.shape[1]
    print(f"ANOVA: {4**k:,} → {n_anova} features")

    if n_anova == 0:
        print(f"SKIP: no features passed ANOVA for k={k}")
        return None

    # Cap at available features
    top_n = min(fixed_n, n_anova)
    if top_n < fixed_n:
        print(f"NOTE: features reduced to {top_n} "
              f"(ANOVA only found {n_anova})")

    print(f"ALL models use {top_n} features "
          f"(ratio = {n_train}/{top_n} = "
          f"{n_train/top_n:.0f}x)")

    # Rank by RF importance — same top features for all
    ranker = RandomForestClassifier(
        n_estimators=100, max_depth=4,
        random_state=42, n_jobs=-1
    )
    ranker.fit(Xtr_a, y_tr)
    idx_sorted = np.argsort(
        ranker.feature_importances_
    )[::-1][:top_n]

    Xtr_fixed = Xtr_a[:, idx_sorted]
    Xte_fixed = Xte_a[:, idx_sorted]

    res = {}

    # ── LR ───────────────────────────────────────────────────
    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr_fixed)
    Xte_sc = sc.transform(Xte_fixed)
    lr = LogisticRegression(
        C=0.1, penalty="l2", solver="liblinear",
        max_iter=2000, class_weight="balanced"
    )
    lr.fit(Xtr_sc, y_tr)
    tr_p = lr.predict_proba(Xtr_sc)[:,1]
    te_p = lr.predict_proba(Xte_sc)[:,1]
    pred = lr.predict(Xte_sc)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["LR"] = dict(
        train_auc=tr_auc, test_auc=te_auc,
        gap=tr_auc-te_auc,
        mcc=matthews_corrcoef(y_te, pred),
        f1=f1_score(y_te, pred),
        recall=recall_score(y_te, pred),
        precision=precision_score(y_te, pred),
        fn=int(fn), features=top_n,
        ratio=round(n_train/top_n, 1),
        n_anova=n_anova
    )
    print(f"LR  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={res['LR']['mcc']:.3f} FN={fn}")

    # ── RF ───────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=cfg["rf_max_depth"],
        min_samples_leaf=cfg["rf_min_leaf"],
        max_features="sqrt",
        random_state=42, n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(Xtr_fixed, y_tr)
    tr_p = rf.predict_proba(Xtr_fixed)[:,1]
    te_p = rf.predict_proba(Xte_fixed)[:,1]
    pred = rf.predict(Xte_fixed)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["RF"] = dict(
        train_auc=tr_auc, test_auc=te_auc,
        gap=tr_auc-te_auc,
        mcc=matthews_corrcoef(y_te, pred),
        f1=f1_score(y_te, pred),
        recall=recall_score(y_te, pred),
        precision=precision_score(y_te, pred),
        fn=int(fn), features=top_n,
        ratio=round(n_train/top_n, 1),
        n_anova=n_anova
    )
    print(f"RF  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={res['RF']['mcc']:.3f} FN={fn}")

    # ── XGB ──────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.01, subsample=0.6,
        colsample_bytree=0.5,
        reg_lambda=cfg["xgb_lambda"],
        reg_alpha=1.0,
        min_child_weight=cfg["xgb_min_child"],
        eval_metric="logloss",
        random_state=42, use_label_encoder=False
    )
    xgb.fit(Xtr_fixed, y_tr)
    tr_p = xgb.predict_proba(Xtr_fixed)[:,1]
    te_p = xgb.predict_proba(Xte_fixed)[:,1]
    pred = xgb.predict(Xte_fixed)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["XGB"] = dict(
        train_auc=tr_auc, test_auc=te_auc,
        gap=tr_auc-te_auc,
        mcc=matthews_corrcoef(y_te, pred),
        f1=f1_score(y_te, pred),
        recall=recall_score(y_te, pred),
        precision=precision_score(y_te, pred),
        fn=int(fn), features=top_n,
        ratio=round(n_train/top_n, 1),
        n_anova=n_anova
    )
    print(f"XGB AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={res['XGB']['mcc']:.3f} FN={fn}")

    return res


# ============================================================
# RUN ALL K VALUES
# ============================================================
all_results = {}
for k in K_VALUES:
    try:
        res = run_for_k(k)
        if res is not None:
            all_results[k] = res
    except Exception as e:
        print(f"\nERROR for k={k}: {e}")
        print(f"Skipping k={k} and continuing...\n")

if not all_results:
    print("No results. Check paths.")
    exit()


# ============================================================
# BUILD RESULTS TABLE
# ============================================================
rows = []
for k, models in all_results.items():
    for mname, m in models.items():
        rows.append({
            "k"             : k,
            "Raw k-mers"    : 4**k,
            "ANOVA features": m["n_anova"],
            "Model"         : mname,
            "Features used" : m["features"],
            "Ratio"         : m["ratio"],
            "Train AUC"     : round(m["train_auc"], 4),
            "Test AUC"      : round(m["test_auc"],  4),
            "AUC Gap"       : round(m["gap"],        4),
            "MCC"           : round(m["mcc"],        4),
            "F1"            : round(m["f1"],         4),
            "FN"            : m["fn"],
        })

df_res = pd.DataFrame(rows)

print(f"\n\n{'='*70}")
print(f"  RESULTS — {ORGANISM.upper()} K-VALUE (FIXED {cfg['top_k_lr']} FEATURES)")
print(f"{'='*70}")
for k in sorted(all_results.keys()):
    print(f"\n  k={k}  (Raw: {4**k:,} features):")
    sub = df_res[df_res["k"]==k][
        ["Model","Features used","Test AUC","AUC Gap","MCC","FN"]
    ]
    print(sub.to_string(index=False))

    # Print winner
    best = sub.loc[sub["Test AUC"].idxmax()]
    print(f"  → Winner at k={k}: {best['Model']} "
          f"(AUC={best['Test AUC']})")

csv_path = os.path.join(OUTPUT_DIR,
    f"{ORGANISM}_k_value_fixed_results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nResults CSV → {csv_path}")


# ============================================================
# PLOT — same 3-panel as original
# ============================================================
valid_ks = sorted(all_results.keys())
fixed_n  = cfg["top_k_lr"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"K-mer Length Sensitivity Analysis — {ORGANISM.upper()} "
    f"({cfg['antibiotic']})\n"
    f"Gold band = k=6 (main study). "
    f"ALL models fixed at {fixed_n} features — fair comparison.",
    fontsize=12, fontweight="bold"
)

for ax, metric, title, ymin, ymax in [
    (axes[0], "Test AUC", "Test AUC vs K",       0.55, 1.00),
    (axes[1], "AUC Gap",  "Overfitting Gap vs K", 0.00, 0.20),
    (axes[2], "MCC",      "MCC vs K",             0.10, 0.75),
]:
    for mname, color in COLORS.items():
        sub = df_res[df_res["Model"]==mname].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub[metric],
                marker=MARKERS[mname], color=color,
                lw=2, ms=8, label=mname)
        for _, row in sub.iterrows():
            ax.text(row["k"],
                    row[metric] + (ymax-ymin)*0.025,
                    f'{row[metric]:.3f}',
                    ha="center", fontsize=8,
                    color=color, fontweight="bold")

    ax.axvspan(5.7, 6.3, alpha=0.15, color="gold", zorder=0)

    if metric == "AUC Gap":
        ax.axhline(0.08, color="#E24B4A",
                   linestyle="--", lw=1.5,
                   label="Overfit threshold (0.08)")

    ax.set_xlabel("K-mer length (k)", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(valid_ks)
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for k in valid_ks:
        ax.annotate(f"{4**k:,}\nfeatures",
                    xy=(k, ymin + (ymax-ymin)*0.02),
                    ha="center", fontsize=6.5,
                    color="gray", style="italic")

plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR,
    f"{ORGANISM}_k_value_fixed_comparison.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot → {p1}")


# ============================================================
# RATIO TABLE PLOT
# ============================================================
fig3, ax3 = plt.subplots(figsize=(13, 3.5))
ax3.axis("off")

tbl_rows = []
for k in valid_ks:
    m = all_results[k]
    fn_lr  = m["LR"]["features"]
    fn_rf  = m["RF"]["features"]
    fn_xgb = m["XGB"]["features"]
    tbl_rows.append([
        f"k={k}",
        f"{4**k:,}",
        f"{m['LR']['n_anova']}",
        f"{fn_lr}",
        f"{m['LR']['ratio']}x",
        f"{fn_rf}",
        f"{m['RF']['ratio']}x",
        f"{fn_xgb}",
        f"{m['XGB']['ratio']}x",
    ])

col_labels = [
    "K", "Raw k-mers", "After ANOVA",
    "LR features", "LR ratio",
    "RF features", "RF ratio",
    "XGB features", "XGB ratio",
]

tbl = ax3.table(
    cellText=tbl_rows, colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1E2761")
        cell.set_text_props(color="white", fontweight="bold")
    elif c in [4, 6, 8]:
        cell.set_facecolor("#EAF4FF")
    elif r > 0 and tbl_rows[r-1][0] == "k=6":
        cell.set_facecolor("#FFF3CC")

ax3.set_title(
    f"Sample/Feature Ratios — {ORGANISM.upper()} "
    f"(ALL models fixed at {fixed_n} features)\n"
    f"Fair comparison — identical feature count across all models and k values",
    fontsize=11, fontweight="bold", pad=12
)
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR,
    f"{ORGANISM}_k_value_fixed_ratio_table.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Table → {p3}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n\n{'='*60}")
print(f"  SUMMARY — {ORGANISM.upper()} FIXED FEATURE K-VALUE")
print(f"{'='*60}")
print(f"\n  Winner at each k (by Test AUC):")
for k in sorted(all_results.keys()):
    sub = df_res[df_res["k"]==k]
    best = sub.loc[sub["Test AUC"].idxmax()]
    print(f"  k={k}  →  {best['Model']}  "
          f"AUC={best['Test AUC']}  FN={best['FN']}")

print(f"\n  Output files:")
print(f"  {csv_path}")
print(f"  {p1}")
print(f"  {p3}")
print(f"\n{'='*60}")
print(f"  DONE — change ORGANISM to run for other organism")
print(f"{'='*60}\n")