"""
k_value_analysis.py
===================
Complete self-contained k-mer length sensitivity analysis.
 
What this script does for EACH k in [4, 5, 6, 7]:
  1. Extracts k-mer features from all .fna genome files
     (skips extraction if cached CSV already exists)
  2. Merges k-mer features with balanced phenotype labels
     (same logic as your merge_labels.py script)
  3. Applies ANOVA filter to remove non-significant k-mers
  4. Selects top N features by RF importance ranking
  5. Trains LR, RF, XGBoost with your regularization settings
  6. Evaluates Test AUC, AUC Gap, MCC, FN per model
  7. Saves results CSV and 3 comparison plots
 
Output saved to:
  Data/results_{organism}/k_analysis/
 
HOW TO RUN:
  1. Set ORGANISM = "ecoli" or "saureus" below
  2. Set K_VALUES (recommend [4, 5, 6] first, add 7 if time allows)
  3. py k_value_analysis.py
 
SAVE TIME — use your existing k=6 file as cache:
  Copy your existing features file:
    FROM: Data/ecoli_kmer_features_YYYYMMDD_HHMMSS.csv
    TO:   Data/results_ecoli/k_analysis/kmer_features_k6.csv
  Script will detect the cache and skip re-extraction for k=6.
 
NOTE FOR S. AUREUS:
  S. aureus genomes were downloaded as genome_0.fna, genome_1.fna etc.
  The script automatically maps these to GCA accession IDs using
  balanced_data/saureus_accessions.txt so the merge works correctly.
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
K_VALUES = [4, 5, 6, 7]    # k values to compare
 
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
 
# Per-organism ML configuration
# Feature counts are FIXED across all k values
# so that sample/feature ratios stay comparable.
CONFIG = {
    "ecoli": {
        "anova_alpha"   : 0.001,
        "top_k_lr"      : 80,
        "top_k_rf"      : 40,
        "top_k_xgb"     : 30,
        "rf_max_depth"  : 4,
        "rf_min_leaf"   : 25,
        "xgb_lambda"    : 30,
        "xgb_min_child" : 25,
        "train_n"       : 2101,
        "antibiotic"    : "Ciprofloxacin",
    },
    "saureus": {
        "anova_alpha"   : 0.0001,
        "top_k_lr"      : 20,
        "top_k_rf"      : 12,
        "top_k_xgb"     : 10,
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
OUTPUT_DIR    = os.path.join(BASE_DIR, "Data",
                             f"results_{ORGANISM}", "k_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── S. aureus genome ID mapping ───────────────────────────────
# S. aureus genomes downloaded as genome_0.fna, genome_1.fna etc.
# We map each index to the GCA accession from saureus_accessions.txt
# so the merge with labels works correctly.
SAUREUS_ID_MAP = {}
if ORGANISM == "saureus":
    acc_file = os.path.join(BASE_DIR, "balanced_data",
                            "saureus_accessions.txt")
    if os.path.exists(acc_file):
        acc_list = pd.read_csv(acc_file, header=None)[0].tolist()
        for i, acc in enumerate(acc_list):
            SAUREUS_ID_MAP[f"genome_{i}.fna"] = acc
        print(f"Loaded {len(SAUREUS_ID_MAP)} S. aureus ID mappings")
    else:
        print(f"WARNING: saureus_accessions.txt not found at {acc_file}")
        print("S. aureus genome IDs may not merge correctly.")
 
VALID_BASES = {"A", "C", "G", "T"}
COLORS      = {"LR": "#1D9E75", "RF": "#534AB7", "XGB": "#D85A30"}
MARKERS     = {"LR": "o",       "RF": "s",        "XGB": "^"}
 
# ============================================================
# STARTUP
# ============================================================
print(f"\n{'='*60}")
print(f"  K-VALUE SENSITIVITY ANALYSIS")
print(f"  Organism  : {ORGANISM.upper()}")
print(f"  Antibiotic: {cfg['antibiotic']}")
print(f"  K values  : {K_VALUES}")
print(f"  ANOVA α   : {cfg['anova_alpha']}")
print(f"  Features  : LR={cfg['top_k_lr']}  "
      f"RF={cfg['top_k_rf']}  XGB={cfg['top_k_xgb']}")
print(f"  Output    : {OUTPUT_DIR}")
print(f"{'='*60}\n")
 
 
# ============================================================
# GENOME READING FUNCTIONS
# ============================================================
def read_fasta(filepath):
    seq = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                continue
            line = line.strip().upper()
            if set(line).issubset({"A", "C", "G", "T", "N"}):
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
    """
    Extract genome accession ID from filename.
 
    S. aureus: genome_0.fna → GCA_XXXXXXX.X  (via accessions map)
    E. coli:   GCA_000026305.1_ASM2630v1.fna → GCA_000026305
    """
    # S. aureus — use pre-built mapping from accessions file
    if ORGANISM == "saureus" and filename in SAUREUS_ID_MAP:
        return SAUREUS_ID_MAP[filename]
 
    # E. coli and fallback — split on underscore
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return os.path.splitext(filename)[0]
 
 
# ============================================================
# STEP 1 — FEATURE EXTRACTION WITH CACHING
# ============================================================
def extract_features(k):
    cache = os.path.join(OUTPUT_DIR, f"kmer_features_k{k}.csv")
 
    if os.path.exists(cache):
        print(f"[k={k}] Cache found → loading...")
        df = pd.read_csv(cache)
        print(f"[k={k}] Loaded: {df.shape[0]} genomes, "
              f"{df.shape[1]-1} k-mer features")
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
    df.to_csv(cache, index=False)
 
    print(f"[k={k}] Extracted {len(df)} genomes "
          f"({skipped} skipped) → cached to {cache}")
    return df
 
 
# ============================================================
# STEP 2 — MERGE WITH LABELS
# ============================================================
def merge_labels(kmer_df):
    labels = pd.read_csv(LABEL_FILE)
 
    if "Assembly" in labels.columns:
        labels = labels.rename(columns={"Assembly": "genome"})
 
    merged = pd.merge(
        kmer_df,
        labels[["genome", "label"]],
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
# STEP 3 — FULL ML PIPELINE FOR ONE K
# ============================================================
def run_for_k(k):
    print(f"\n{'='*60}")
    print(f"  k={k}  |  {ORGANISM.upper()}  |  "
          f"{cfg['antibiotic']}")
    print(f"{'='*60}")
 
    kmer_df = extract_features(k)
    ml_df   = merge_labels(kmer_df)
 
    if len(ml_df) < 100:
        print(f"SKIP: only {len(ml_df)} samples after merge")
        return None
 
    X = ml_df.drop(columns=["genome", "label"])
    y = ml_df["label"]
 
    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    n_train = len(X_tr)
    print(f"Split: {n_train} train / {len(X_te)} test")
 
    # ANOVA filter
    sel     = SelectFpr(f_classif, alpha=cfg["anova_alpha"])
    Xtr_a   = sel.fit_transform(X_tr, y_tr)
    Xte_a   = sel.transform(X_te)
    n_anova = Xtr_a.shape[1]
    print(f"ANOVA: {4**k:,} → {n_anova} features "
          f"(alpha={cfg['anova_alpha']})")
 
    if n_anova == 0:
        print(f"SKIP: no features passed ANOVA for k={k}")
        return None
 
    # Feature counts — capped by what ANOVA found
    top_lr  = min(cfg["top_k_lr"],  n_anova)
    top_rf  = min(cfg["top_k_rf"],  n_anova)
    top_xgb = min(cfg["top_k_xgb"], n_anova)
 
    if top_lr < cfg["top_k_lr"]:
        print(f"NOTE: LR features reduced to {top_lr} "
              f"(ANOVA only found {n_anova})")
 
    print(f"Ratios: LR={n_train}/{top_lr}={n_train/top_lr:.0f}x  "
          f"RF={n_train}/{top_rf}={n_train/top_rf:.0f}x  "
          f"XGB={n_train}/{top_xgb}={n_train/top_xgb:.0f}x")
 
    # RF importance ranking
    ranker = RandomForestClassifier(
        n_estimators=100, max_depth=4,
        random_state=42, n_jobs=-1
    )
    ranker.fit(Xtr_a, y_tr)
    idx_sorted = np.argsort(ranker.feature_importances_)[::-1]
 
    idx_lr  = idx_sorted[:top_lr]
    idx_rf  = idx_sorted[:top_rf]
    idx_xgb = idx_sorted[:top_xgb]
 
    res = {}
 
    # ── LR ───────────────────────────────────────────────────
    sc     = StandardScaler()
    Xtr_lr = sc.fit_transform(Xtr_a[:, idx_lr])
    Xte_lr = sc.transform(Xte_a[:, idx_lr])
    lr = LogisticRegression(C=0.1, penalty="l2",
                            solver="liblinear", max_iter=2000,
                            class_weight="balanced")
    lr.fit(Xtr_lr, y_tr)
    tr_p = lr.predict_proba(Xtr_lr)[:, 1]
    te_p = lr.predict_proba(Xte_lr)[:, 1]
    pred = lr.predict(Xte_lr)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["LR"] = dict(train_auc=tr_auc, test_auc=te_auc,
                     gap=tr_auc-te_auc,
                     mcc=matthews_corrcoef(y_te, pred),
                     f1=f1_score(y_te, pred),
                     recall=recall_score(y_te, pred),
                     precision=precision_score(y_te, pred),
                     fn=int(fn), features=top_lr,
                     ratio=round(n_train/top_lr, 1),
                     n_anova=n_anova)
    print(f"LR  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={res['LR']['mcc']:.3f} FN={fn}")
 
    # ── RF ───────────────────────────────────────────────────
    Xtr_rf = Xtr_a[:, idx_rf]
    Xte_rf = Xte_a[:, idx_rf]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=cfg["rf_max_depth"],
        min_samples_leaf=cfg["rf_min_leaf"],
        max_features="sqrt",
        random_state=42, n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(Xtr_rf, y_tr)
    tr_p = rf.predict_proba(Xtr_rf)[:, 1]
    te_p = rf.predict_proba(Xte_rf)[:, 1]
    pred = rf.predict(Xte_rf)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["RF"] = dict(train_auc=tr_auc, test_auc=te_auc,
                     gap=tr_auc-te_auc,
                     mcc=matthews_corrcoef(y_te, pred),
                     f1=f1_score(y_te, pred),
                     recall=recall_score(y_te, pred),
                     precision=precision_score(y_te, pred),
                     fn=int(fn), features=top_rf,
                     ratio=round(n_train/top_rf, 1),
                     n_anova=n_anova)
    print(f"RF  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={res['RF']['mcc']:.3f} FN={fn}")
 
    # ── XGB ──────────────────────────────────────────────────
    Xtr_xgb = Xtr_a[:, idx_xgb]
    Xte_xgb = Xte_a[:, idx_xgb]
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
    xgb.fit(Xtr_xgb, y_tr)
    tr_p = xgb.predict_proba(Xtr_xgb)[:, 1]
    te_p = xgb.predict_proba(Xte_xgb)[:, 1]
    pred = xgb.predict(Xte_xgb)
    _, fp, fn, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    res["XGB"] = dict(train_auc=tr_auc, test_auc=te_auc,
                      gap=tr_auc-te_auc,
                      mcc=matthews_corrcoef(y_te, pred),
                      f1=f1_score(y_te, pred),
                      recall=recall_score(y_te, pred),
                      precision=precision_score(y_te, pred),
                      fn=int(fn), features=top_xgb,
                      ratio=round(n_train/top_xgb, 1),
                      n_anova=n_anova)
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
    print("No results. Check genome folder and label file paths.")
    exit()
 
 
# ============================================================
# BUILD RESULTS TABLE
# ============================================================
rows = []
for k, models in all_results.items():
    for mname, m in models.items():
        rows.append({
            "k"               : k,
            "Raw k-mers"      : 4**k,
            "ANOVA features"  : m["n_anova"],
            "Model"           : mname,
            "Features used"   : m["features"],
            "Ratio"           : m["ratio"],
            "Train AUC"       : round(m["train_auc"], 4),
            "Test AUC"        : round(m["test_auc"],  4),
            "AUC Gap"         : round(m["gap"],        4),
            "MCC"             : round(m["mcc"],        4),
            "F1"              : round(m["f1"],         4),
            "Recall"          : round(m["recall"],     4),
            "Precision"       : round(m["precision"],  4),
            "FN"              : m["fn"],
        })
 
df_res = pd.DataFrame(rows)
 
print(f"\n\n{'='*70}")
print(f"  RESULTS — {ORGANISM.upper()} K-VALUE ANALYSIS")
print(f"{'='*70}")
for k in sorted(all_results.keys()):
    print(f"\n  k={k}  (Raw: {4**k:,} features):")
    sub = df_res[df_res["k"]==k][
        ["Model","Features used","Test AUC","AUC Gap","MCC","FN"]
    ]
    print(sub.to_string(index=False))
 
csv_path = os.path.join(OUTPUT_DIR,
                        f"{ORGANISM}_k_value_results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nResults CSV → {csv_path}")
 
 
# ============================================================
# PLOT 1 — MAIN 3-PANEL COMPARISON
# ============================================================
valid_ks = sorted(all_results.keys())
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"K-mer Length Sensitivity Analysis — {ORGANISM.upper()} "
    f"({cfg['antibiotic']})\n"
    f"Gold band = k=6 (main study). "
    f"Feature counts fixed across k → ratios stay stable.",
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
                  f"{ORGANISM}_k_value_comparison.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot 1 → {p1}")
 
 
# ============================================================
# PLOT 2 — DETAILED PER-MODEL BAR CHARTS
# ============================================================
fig2, axes2 = plt.subplots(3, 4, figsize=(16, 10))
fig2.suptitle(
    f"Detailed K-value Analysis — {ORGANISM.upper()} "
    f"({cfg['antibiotic']})\n"
    f"Rows = models. Columns = AUC, Gap, MCC, FN. "
    f"Gold border = k=6 (main study).",
    fontsize=12, fontweight="bold"
)
 
metrics_d = [
    ("Test AUC", "Test AUC"),
    ("AUC Gap",  "AUC Gap"),
    ("MCC",      "MCC"),
    ("FN",       "FN (missed resistant)"),
]
 
for ri, mname in enumerate(["LR", "RF", "XGB"]):
    sub   = df_res[df_res["Model"]==mname].sort_values("k")
    color = COLORS[mname]
    ks    = sub["k"].values
 
    for ci, (metric, label) in enumerate(metrics_d):
        ax   = axes2[ri, ci]
        vals = sub[metric].values
        bars = ax.bar(range(len(ks)), vals,
                      color=color, alpha=0.75,
                      edgecolor="white", linewidth=1.5)
 
        if 6 in ks:
            k6i = list(ks).index(6)
            bars[k6i].set_alpha(1.0)
            bars[k6i].set_edgecolor("gold")
            bars[k6i].set_linewidth(3)
 
        for bar, val in zip(bars, vals):
            label_str = (str(int(val)) if metric == "FN"
                         else f'{val:.3f}')
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.03,
                    label_str,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
 
        if metric == "AUC Gap":
            ax.axhline(0.08, color="#E24B4A",
                       linestyle="--", lw=1.2,
                       label="Threshold 0.08")
            ax.legend(fontsize=7)
 
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
 
        if ri == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{mname}", fontsize=11,
                          color=color, fontweight="bold")
 
plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR,
                  f"{ORGANISM}_k_value_detailed.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot 2 → {p2}")
 
 
# ============================================================
# PLOT 3 — SAMPLE/FEATURE RATIO TABLE
# ============================================================
fig3, ax3 = plt.subplots(figsize=(13, 3.5))
ax3.axis("off")
 
tbl_rows = []
for k in valid_ks:
    m = all_results[k]
    tbl_rows.append([
        f"k={k}",
        f"{4**k:,}",
        f"{m['LR']['n_anova']}",
        f"{m['LR']['features']}",
        f"{m['LR']['ratio']}x",
        f"{m['RF']['features']}",
        f"{m['RF']['ratio']}x",
        f"{m['XGB']['features']}",
        f"{m['XGB']['ratio']}x",
    ])
 
col_labels = [
    "K", "Raw k-mers", "After ANOVA",
    "LR features", "LR ratio",
    "RF features", "RF ratio",
    "XGB features", "XGB ratio",
]
 
tbl = ax3.table(
    cellText=tbl_rows,
    colLabels=col_labels,
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
    f"Sample/Feature Ratios by K Value — {ORGANISM.upper()}\n"
    f"Feature counts are FIXED across k values → "
    f"ratios stay stable → fair comparison across k",
    fontsize=11, fontweight="bold", pad=12
)
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR,
                  f"{ORGANISM}_k_ratio_table.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot 3 → {p3}")
 
 
# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n\n{'='*60}")
print(f"  SUMMARY — {ORGANISM.upper()} K-VALUE ANALYSIS")
print(f"{'='*60}")
 
best_auc = df_res.loc[df_res["Test AUC"].idxmax()]
best_mcc = df_res.loc[df_res["MCC"].idxmax()]
best_fn  = df_res.loc[df_res["FN"].idxmin()]
best_gap = df_res.loc[df_res["AUC Gap"].idxmin()]
 
print(f"\n  Best per metric:")
print(f"  Test AUC  : k={int(best_auc['k'])}  "
      f"model={best_auc['Model']}  "
      f"AUC={best_auc['Test AUC']:.4f}")
print(f"  MCC       : k={int(best_mcc['k'])}  "
      f"model={best_mcc['Model']}  "
      f"MCC={best_mcc['MCC']:.4f}")
print(f"  Fewest FN : k={int(best_fn['k'])}  "
      f"model={best_fn['Model']}  "
      f"FN={best_fn['FN']}")
print(f"  Best gap  : k={int(best_gap['k'])}  "
      f"model={best_gap['Model']}  "
      f"gap={best_gap['AUC Gap']:.4f}")
 
k6_rows = df_res[df_res["k"] == 6]
if not k6_rows.empty:
    k6_best_auc  = k6_rows["Test AUC"].max()
    overall_best = df_res["Test AUC"].max()
    if k6_best_auc >= overall_best - 0.005:
        print(f"\n  k=6 VALIDATED — AUC within 0.005 of best "
              f"(k=6 best={k6_best_auc:.4f}, "
              f"overall best={overall_best:.4f})")
    else:
        print(f"\n  Note: k={int(best_auc['k'])} gives higher AUC "
              f"than k=6 ({best_auc['Test AUC']:.4f} vs "
              f"{k6_best_auc:.4f})")
 
print(f"\n  Output files:")
print(f"  {csv_path}")
print(f"  {p1}")
print(f"  {p2}")
print(f"  {p3}")
print(f"\n  Next: change ORGANISM to run for the other organism.")
print(f"{'='*60}")
print(f"  DONE")
print(f"{'='*60}\n")
 















