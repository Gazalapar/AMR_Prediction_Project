"""
SCRIPT 2 - K-VALUE ANALYSIS AT FIXED FEATURES
===============================================
Validates that LR wins for E.coli and RF wins for S.aureus
at every k value when all models use the SAME feature count.

THRESHOLD: 0.5 (default) - same as your existing Chapter 4.
This ensures numbers are directly comparable to thesis.

E. coli  : k=4,5,6,7 | all models at fixed 80 features
S. aureus: k=6,7     | all models at fixed 20 features

EXPECTED RESULTS (LR numbers must match Chapter 4 exactly):
  E. coli  k=6 LR: AUC~=0.871 gap~=0.037 FN=57  <- same as thesis
  S. aureus k=6 LR: AUC~=0.836 gap~=0.021 FN=19  <- same as thesis

GRAPHS PRODUCED (matching your thesis figures exactly):
  Fig 4.1 style: 3-panel line chart (AUC, Gap, MCC vs k)
  Fig 4.2 style: detailed 3x4 bar chart per model
  Ratio table  : sample/feature ratios per k value

OUTPUT - results_{organism}/fixed_features_complete/
  k_value_fixed_results_{organism}.csv
  {organism}_k_value_fixed_comparison.png   <- matches Fig 4.1/4.3
  {organism}_k_value_fixed_detailed.png     <- matches Fig 4.2/4.4
  {organism}_k_value_fixed_ratio_table.png  <- matches ratio tables

Run AFTER script1_fixed_feature_comparison.py
Run: py script2_kvalue_fixed_features.py
Time: ~15 minutes (reuses cached k-mer CSV files)
"""

import pandas as pd
import numpy as np
import os, warnings
from collections import Counter
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFpr, f_classif
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef,
    f1_score, confusion_matrix
)

# ======================================================
BASE_DIR = r"C:\AMR_Prediction_Project"
# ======================================================

# -- VERIFIED expected values from your thesis Chapter 4 ------
# LR numbers at fixed features must match these exactly
# RF and XGB will differ because they now use more features
THESIS_CHAPTER4 = {
    "ecoli": {
        # Original Chapter 4 used LR=80, RF=40, XGB=30
        # Fixed script uses ALL=80 so LR numbers stay same
        "LR": {4:{"auc":0.804,"gap":0.075,"mcc":0.464,"fn":69},
               5:{"auc":0.830,"gap":0.048,"mcc":0.541,"fn":69},
               6:{"auc":0.871,"gap":0.037,"mcc":0.582,"fn":57},
               7:{"auc":0.870,"gap":0.028,"mcc":0.606,"fn":61}},
    },
    "Saureus": {
        # Original Chapter 4 used LR=20, RF=12, XGB=10
        # Fixed script uses ALL=20 so LR numbers stay same
        "LR": {6:{"auc":0.836,"gap":0.021,"mcc":0.527,"fn":19},
               7:{"auc":0.577,"gap":0.016,"mcc":0.122,"fn":36}},
    },
}

ORGANISM_CONFIG = {
    "ecoli": {
        "data_dir"     : os.path.join(BASE_DIR, "Data", "results_ecoli"),
        "label_file"   : os.path.join(BASE_DIR, "balanced_data",
                                      "ecoli_ciprofloxacin_balanced.csv"),
        "genome_folder": os.path.join(BASE_DIR, "ecoli_fna"),
        "k_values"     : [4, 5, 6, 7],
        "fixed_feat"   : 80,   # ALL models use this count
        "anova_alpha"  : 0.001,
        "rf_max_depth" : 4,
        "rf_min_leaf"  : 25,
        "xgb_lambda"   : 30,
        "xgb_min_child": 25,
        "train_n"      : 2101,
        "antibiotic"   : "Ciprofloxacin",
        "organism_id"  : "ecoli",
    },
    "Saureus": {
        "data_dir"     : os.path.join(BASE_DIR, "Data", "results_Saureus"),
        "label_file"   : os.path.join(BASE_DIR, "balanced_data",
                                      "saureus_erythromycin_balanced.csv"),
        "genome_folder": os.path.join(BASE_DIR, "saureus_fna"),
        "k_values"     : [6, 7],
        "fixed_feat"   : 20,   # ALL models use this count
        "anova_alpha"  : 0.0001,
        "rf_max_depth" : 4,
        "rf_min_leaf"  : 40,
        "xgb_lambda"   : 20,
        "xgb_min_child": 20,
        "train_n"      : 590,
        "antibiotic"   : "Erythromycin",
        "organism_id"  : "Saureus",
    },
}

# S. aureus genome ID mapping
SAUREUS_ID_MAP = {}
acc_file = os.path.join(BASE_DIR, "balanced_data",
                        "saureus_accessions.txt")
if os.path.exists(acc_file):
    acc_list = pd.read_csv(acc_file, header=None)[0].tolist()
    for i, acc in enumerate(acc_list):
        SAUREUS_ID_MAP[f"genome_{i}.fna"] = acc

VALID_BASES = {"A","C","G","T"}
COLORS  = {"LR":"#534AB7","RF":"#1D9E75","XGB":"#D85A30"}
MARKERS = {"LR":"o","RF":"s","XGB":"^"}


# -- Genome reading --------------------------------------------
def read_fasta(filepath):
    seq = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"): continue
            line = line.strip().upper()
            if set(line).issubset({"A","C","G","T","N"}):
                seq.append(line)
    return "".join(seq)


def count_kmers_normalized(seq, k):
    counts = Counter()
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if set(kmer).issubset(VALID_BASES):
            counts[kmer] += 1
    total = sum(counts.values())
    if total > 0:
        for km in counts: counts[km] /= total
    return counts


def get_genome_id(filename, organism):
    if organism == "Saureus" and filename in SAUREUS_ID_MAP:
        return SAUREUS_ID_MAP[filename]
    parts = filename.split("_")
    if len(parts) >= 2: return parts[0]+"_"+parts[1]
    return os.path.splitext(filename)[0]


# -- Load k-mer cache (original first, then new) ---------------
def load_cache(k, cfg):
    organism = cfg["organism_id"]
    # Try original k_analysis folder first
    orig = os.path.join(cfg["data_dir"], "k_analysis",
                        f"kmer_features_k{k}.csv")
    new  = os.path.join(cfg["data_dir"], "fixed_features_complete",
                        f"kmer_features_k{k}.csv")
    if os.path.exists(orig):
        print(f"    [k={k}] Using original cache")
        return pd.read_csv(orig)
    if os.path.exists(new):
        print(f"    [k={k}] Using new cache")
        return pd.read_csv(new)
    # Extract fresh
    print(f"    [k={k}] Extracting (4^{k}={4**k:,} k-mers)...")
    fna_files = sorted([f for f in os.listdir(cfg["genome_folder"])
                        if f.endswith(".fna")])
    all_rows = []
    for idx, fname in enumerate(fna_files):
        if (idx+1)%300==0:
            print(f"      {idx+1}/{len(fna_files)}")
        fpath = os.path.join(cfg["genome_folder"], fname)
        gid   = get_genome_id(fname, organism)
        seq   = read_fasta(fpath)
        if len(seq) < k: continue
        kmers = count_kmers_normalized(seq, k)
        if not kmers: continue
        kmers["genome"] = gid
        all_rows.append(kmers)
    df = pd.DataFrame(all_rows).fillna(0)
    df.insert(0, "genome", df.pop("genome"))
    df.to_csv(new, index=False)
    return df


def merge_labels(kmer_df, label_file):
    labels = pd.read_csv(label_file)
    if "Assembly" in labels.columns:
        labels = labels.rename(columns={"Assembly":"genome"})
    merged = pd.merge(kmer_df, labels[["genome","label"]],
                      on="genome", how="inner")
    kmer_cols = [c for c in merged.columns
                 if set(c).issubset({"A","C","G","T"})]
    return merged[["genome"]+kmer_cols+["label"]]


# -- Run one k value -------------------------------------------
def run_k(k, cfg, organism):
    fixed_n = cfg["fixed_feat"]
    print(f"\n    -- k={k} | ALL models = {fixed_n} features --")

    kmer_df = load_cache(k, cfg)
    ml_df   = merge_labels(kmer_df, cfg["label_file"])
    if len(ml_df) < 100:
        print(f"    SKIP: only {len(ml_df)} samples")
        return None

    X = ml_df.drop(columns=["genome","label"])
    y = ml_df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    n_train = len(X_tr)

    # ANOVA
    sel   = SelectFpr(f_classif, alpha=cfg["anova_alpha"])
    Xtr_a = sel.fit_transform(X_tr, y_tr)
    Xte_a = sel.transform(X_te)
    n_anova = Xtr_a.shape[1]
    print(f"    ANOVA: {4**k:,} -> {n_anova} features")

    if n_anova == 0:
        print(f"    SKIP: 0 features passed ANOVA")
        return {"collapsed": True, "n_anova": 0, "k": k}

    top_n = min(fixed_n, n_anova)
    if top_n < fixed_n:
        print(f"    NOTE: reduced to {top_n} - ANOVA only found {n_anova}")

    print(f"    ALL models: {top_n} features (ratio {n_train}/{top_n}={n_train//top_n}x)")

    # Rank by RF importance - same features for all models
    ranker = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    ranker.fit(Xtr_a, y_tr)
    idx_sorted = np.argsort(ranker.feature_importances_)[::-1][:top_n]
    Xtr_f = Xtr_a[:, idx_sorted]
    Xte_f = Xte_a[:, idx_sorted]

    results = {}

    # -- LR - threshold 0.5 (same as Chapter 4) ---------------
    sc = StandardScaler()
    lr = LogisticRegression(C=0.1, penalty="l2", solver="liblinear",
                             max_iter=2000, class_weight="balanced")
    lr.fit(sc.fit_transform(Xtr_f), y_tr)
    tr_p = lr.predict_proba(sc.transform(Xtr_f))[:,1]
    te_p = lr.predict_proba(sc.transform(Xte_f))[:,1]
    pred = (te_p >= 0.5).astype(int)   # <- threshold 0.5
    _, _, fn_v, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    results["LR"] = {"auc":round(te_auc,4),"gap":round(tr_auc-te_auc,4),
                      "mcc":round(matthews_corrcoef(y_te,pred),4),
                      "fn":int(fn_v),"f1":round(f1_score(y_te,pred),4),
                      "features":top_n,"n_anova":n_anova,
                      "ratio":round(n_train/top_n,1)}
    print(f"    LR  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={results['LR']['mcc']:.3f} FN={fn_v}")

    # Verify LR matches Chapter 4 if available
    if k in THESIS_CHAPTER4.get(organism,{}).get("LR",{}):
        exp = THESIS_CHAPTER4[organism]["LR"][k]
        auc_ok = abs(round(te_auc,3) - exp["auc"]) <= 0.002
        fn_ok  = fn_v == exp["fn"]
        status = "OK matches Chapter 4" if (auc_ok and fn_ok) else "WARNING  differs from Chapter 4"
        print(f"    LR consistency check: {status} "
              f"(expected AUC~={exp['auc']} FN={exp['fn']})")

    # -- RF - threshold 0.5 ------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=cfg["rf_max_depth"],
        min_samples_leaf=cfg["rf_min_leaf"], max_features="sqrt",
        random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr_f, y_tr)
    tr_p = rf.predict_proba(Xtr_f)[:,1]
    te_p = rf.predict_proba(Xte_f)[:,1]
    pred = (te_p >= 0.5).astype(int)   # <- threshold 0.5
    _, _, fn_v, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    results["RF"] = {"auc":round(te_auc,4),"gap":round(tr_auc-te_auc,4),
                      "mcc":round(matthews_corrcoef(y_te,pred),4),
                      "fn":int(fn_v),"f1":round(f1_score(y_te,pred),4),
                      "features":top_n,"n_anova":n_anova,
                      "ratio":round(n_train/top_n,1)}
    print(f"    RF  AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={results['RF']['mcc']:.3f} FN={fn_v}")

    # -- XGB - threshold 0.5 -----------------------------------
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.5,
        reg_lambda=cfg["xgb_lambda"], reg_alpha=1.0,
        min_child_weight=cfg["xgb_min_child"],
        eval_metric="logloss", random_state=42,
        use_label_encoder=False)
    xgb.fit(Xtr_f, y_tr)
    tr_p = xgb.predict_proba(Xtr_f)[:,1]
    te_p = xgb.predict_proba(Xte_f)[:,1]
    pred = (te_p >= 0.5).astype(int)   # <- threshold 0.5
    _, _, fn_v, _ = confusion_matrix(y_te, pred).ravel()
    tr_auc = roc_auc_score(y_tr, tr_p)
    te_auc = roc_auc_score(y_te, te_p)
    results["XGB"] = {"auc":round(te_auc,4),"gap":round(tr_auc-te_auc,4),
                       "mcc":round(matthews_corrcoef(y_te,pred),4),
                       "fn":int(fn_v),"f1":round(f1_score(y_te,pred),4),
                       "features":top_n,"n_anova":n_anova,
                       "ratio":round(n_train/top_n,1)}
    print(f"    XGB AUC={te_auc:.3f} gap={tr_auc-te_auc:.3f} "
          f"MCC={results['XGB']['mcc']:.3f} FN={fn_v}")

    winner = max(results, key=lambda m: results[m]["auc"])
    best_g = min(results, key=lambda m: results[m]["gap"])
    print(f"    -> AUC winner: {winner} | Best gen: {best_g}")

    return {"collapsed":False, "k":k, "n_anova":n_anova,
            "top_n":top_n, "models":results,
            "winner":winner, "best_gen":best_g}


# -- Build and save graphs -------------------------------------
def save_graphs(df_k, cfg, organism, output_dir):
    fixed_n  = cfg["fixed_feat"]
    valid_ks = sorted(df_k[
        df_k["Test_AUC"]!="COLLAPSED"]["k"].unique())

    if not valid_ks:
        print("  No valid k values - skipping graphs")
        return

    # -- FIGURE 1: 3-panel line chart (matches Fig 4.1/4.3) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"K-mer Length Sensitivity Analysis - {organism.upper()} "
        f"({cfg['antibiotic']})\n"
        f"Gold band = k=6 (main study). "
        f"ALL models fixed at {fixed_n} features - supervisor validation.",
        fontsize=12, fontweight="bold"
    )
    plot_specs = [
        (axes[0],"Test_AUC","Test AUC vs K",    0.55,1.00),
        (axes[1],"AUC_Gap", "Overfitting Gap vs K",0.00,0.20),
        (axes[2],"MCC",     "MCC vs K",          0.10,0.80),
    ]
    for ax, col, title, ymin, ymax in plot_specs:
        for mname in ["LR","RF","XGB"]:
            sub = df_k[
                (df_k["Model"]==mname) &
                (df_k["Test_AUC"]!="COLLAPSED")
            ].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col]).sort_values("k")
            if sub.empty: continue
            ax.plot(sub["k"], sub[col],
                    marker=MARKERS[mname], color=COLORS[mname],
                    lw=2, ms=8, label=mname)
            for _, row in sub.iterrows():
                ax.text(row["k"], row[col]+(ymax-ymin)*0.025,
                        f'{row[col]:.3f}', ha="center", fontsize=8,
                        color=COLORS[mname], fontweight="bold")
        ax.axvspan(5.7, 6.3, alpha=0.15, color="gold", zorder=0)
        if col == "AUC_Gap":
            ax.axhline(0.08, color="#E24B4A", linestyle="--",
                       lw=1.5, label="Overfit threshold (0.08)")
        ax.set_xlabel("K-mer length (k)", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(valid_ks)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for k_val in valid_ks:
            ax.annotate(
                f"{4**k_val:,}\nfeatures",
                xy=(k_val, ymin+(ymax-ymin)*0.02),
                ha="center", fontsize=6.5,
                color="gray", style="italic")
    plt.tight_layout()
    p1 = os.path.join(
        output_dir, f"{organism}_k_value_fixed_comparison.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot 1 (line chart) saved -> {p1}")

    # -- FIGURE 2: detailed bar chart (matches Fig 4.2/4.4) ---
    fig2, axes2 = plt.subplots(3, 4, figsize=(16, 10))
    fig2.suptitle(
        f"Detailed K-value Analysis - {organism.upper()} "
        f"({cfg['antibiotic']})\n"
        f"Rows = models. Columns = AUC, Gap, MCC, FN. "
        f"Gold border = k=6. ALL models fixed at {fixed_n} features.",
        fontsize=12, fontweight="bold"
    )
    metrics_d = [
        ("Test_AUC","Test AUC"),
        ("AUC_Gap", "AUC Gap"),
        ("MCC",     "MCC"),
        ("FN",      "FN (missed resistant)"),
    ]
    for ri, mname in enumerate(["LR","RF","XGB"]):
        sub = df_k[
            (df_k["Model"]==mname) &
            (df_k["Test_AUC"]!="COLLAPSED")
        ].copy()
        for col_n in ["Test_AUC","AUC_Gap","MCC","FN"]:
            sub[col_n] = pd.to_numeric(sub[col_n], errors="coerce")
        sub = sub.sort_values("k")
        ks    = sub["k"].values
        color = COLORS[mname]

        for ci, (metric, label) in enumerate(metrics_d):
            ax2  = axes2[ri, ci]
            vals = sub[metric].dropna().values
            if len(vals) == 0: continue

            bars = ax2.bar(range(len(ks)), vals,
                           color=color, alpha=0.75,
                           edgecolor="white", linewidth=1.5)
            # Gold border on k=6
            if 6 in ks:
                k6i = list(ks).index(6)
                if k6i < len(bars):
                    bars[k6i].set_alpha(1.0)
                    bars[k6i].set_edgecolor("gold")
                    bars[k6i].set_linewidth(3)

            for bar, val in zip(bars, vals):
                label_s = (str(int(val)) if metric=="FN"
                           else f'{val:.3f}')
                ax2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+max(vals)*0.03,
                         label_s, ha="center", va="bottom",
                         fontsize=8, fontweight="bold")

            if metric == "AUC_Gap":
                ax2.axhline(0.08, color="#E24B4A", linestyle="--",
                            lw=1.2, label="Threshold 0.08")
                ax2.legend(fontsize=7)

            ax2.set_xticks(range(len(ks)))
            ax2.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
            ax2.grid(True, alpha=0.2, axis="y")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            if ri == 0:
                ax2.set_title(label, fontsize=10, fontweight="bold")
            if ci == 0:
                ax2.set_ylabel(mname, fontsize=11,
                               color=color, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(
        output_dir, f"{organism}_k_value_fixed_detailed.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot 2 (bar chart) saved -> {p2}")

    # -- FIGURE 3: ratio table (matches thesis ratio tables) ---
    fig3, ax3 = plt.subplots(figsize=(13, max(3.5, len(valid_ks)*0.8)))
    ax3.axis("off")

    tbl_rows = []
    for k_val in valid_ks:
        sub = df_k[(df_k["k"]==k_val) & (df_k["Model"]=="LR")]
        if sub.empty: continue
        n_anova = sub["ANOVA_features"].values[0]
        n_used  = sub["Features_used"].values[0]
        n_train = cfg["train_n"]
        ratio   = f"{round(n_train/n_used,1)}x" if n_used>0 else "-"
        tbl_rows.append([
            f"k={k_val}", f"{4**k_val:,}", str(n_anova),
            str(n_used), ratio,
            str(n_used), ratio,
            str(n_used), ratio,
        ])

    col_labels = [
        "K","Raw k-mers","After ANOVA",
        "LR features","LR ratio",
        "RF features","RF ratio",
        "XGB features","XGB ratio",
    ]
    if not tbl_rows:
        plt.close()
    else:
        tbl = ax3.table(
            cellText=tbl_rows, colLabels=col_labels,
            cellLoc="center", loc="center", bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r,c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1E2761")
                cell.set_text_props(color="white", fontweight="bold")
            elif c in [4,6,8]:
                cell.set_facecolor("#EAF4FF")
            elif r > 0 and tbl_rows[r-1][0] == "k=6":
                cell.set_facecolor("#FFF3CC")
        ax3.set_title(
            f"Sample/Feature Ratios - {organism.upper()} "
            f"(ALL models fixed at {fixed_n} features)\n"
            f"Fair comparison - identical feature count across all models",
            fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        p3 = os.path.join(
            output_dir, f"{organism}_k_value_fixed_ratio_table.png")
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot 3 (ratio table) saved -> {p3}")


# ======================================================
# MAIN
# ======================================================
all_k_rows = []

for organism, cfg in ORGANISM_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"  K-VALUE FIXED - {organism.upper()} ({cfg['antibiotic']})")
    print(f"  k values: {cfg['k_values']}")
    print(f"  Fixed features: ALL models = {cfg['fixed_feat']}")
    print(f"  Threshold: 0.5 (consistent with Chapter 4)")
    print(f"{'='*60}")

    OUTPUT_DIR = os.path.join(
        cfg["data_dir"], "fixed_features_complete")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    k_rows  = []
    k_store = {}

    for k in cfg["k_values"]:
        res = run_k(k, cfg, organism)
        if res is None:
            continue
        if res.get("collapsed"):
            for mname in ["LR","RF","XGB"]:
                k_rows.append({
                    "Organism":"COLLAPSED" if res["n_anova"]==0
                               else organism,
                    "Experiment": f"K_value_fixed_{cfg['fixed_feat']}_features",
                    "k":k,"Raw_kmers":4**k,
                    "ANOVA_features":res["n_anova"],
                    "Model":mname,"Features_used":0,
                    "Threshold":"0.5 (default)",
                    "Test_AUC":"COLLAPSED","AUC_Gap":"COLLAPSED",
                    "MCC":"COLLAPSED","FN":"COLLAPSED","F1":"COLLAPSED",
                    "AUC_Winner_at_k":"N/A","Best_Gen_at_k":"N/A",
                    "Generalisation":"COLLAPSED",
                    "Note":"0 features passed ANOVA - dataset too small for this k",
                })
            continue

        k_store[k] = res
        winner = res["winner"]
        best_g = res["best_gen"]
        for mname, mr in res["models"].items():
            k_rows.append({
                "Organism"         : organism,
                "Experiment"       : f"K_value_fixed_{cfg['fixed_feat']}_features",
                "k"                : k,
                "Raw_kmers"        : 4**k,
                "ANOVA_features"   : res["n_anova"],
                "Model"            : mname,
                "Features_used"    : mr["features"],
                "Threshold"        : "0.5 (default - consistent with Chapter 4)",
                "Test_AUC"         : mr["auc"],
                "AUC_Gap"          : mr["gap"],
                "MCC"              : mr["mcc"],
                "F1"               : mr["f1"],
                "FN"               : mr["fn"],
                "AUC_Winner_at_k"  : "YES" if mname==winner else "no",
                "Best_Gen_at_k"    : "YES" if mname==best_g else "no",
                "Generalisation"   : (
                    "WARNING - exceeds 0.08" if mr["gap"]>0.08
                    else "CAUTION - approaching" if mr["gap"]>0.065
                    else "OK - below threshold"),
                "Note"             : "threshold=0.5 for fair k comparison",
            })

    df_k = pd.DataFrame(k_rows)
    csv_path = os.path.join(
        OUTPUT_DIR, f"k_value_fixed_results_{organism}.csv")
    df_k.to_csv(csv_path, index=False)
    print(f"\n  CSV saved -> {csv_path}")

    # Save graphs
    save_graphs(df_k, cfg, organism, OUTPUT_DIR)

    # Winner summary
    print(f"\n  K-VALUE WINNER SUMMARY - {organism}")
    print(f"  {'-'*50}")
    for k in cfg["k_values"]:
        if k not in k_store:
            print(f"  k={k}: COLLAPSED")
            continue
        res = k_store[k]
        print(f"  k={k}: AUC winner={res['winner']} "
              f"({res['models'][res['winner']]['auc']}) | "
              f"Best gen={res['best_gen']} "
              f"(gap={res['models'][res['best_gen']]['gap']})")

    all_k_rows.extend(k_rows)

# Save combined
combined_path = os.path.join(
    BASE_DIR, "Data",
    "K_VALUE_FIXED_FEATURES_BOTH_ORGANISMS.csv")
pd.DataFrame(all_k_rows).to_csv(combined_path, index=False)

print(f"""
{'='*60}
  SCRIPT 2 COMPLETE
{'='*60}
  Per organism (in fixed_features_complete folder):
    k_value_fixed_results_ecoli.csv
    k_value_fixed_results_Saureus.csv
    ecoli_k_value_fixed_comparison.png   <- matches Fig 4.1
    ecoli_k_value_fixed_detailed.png     <- matches Fig 4.2
    ecoli_k_value_fixed_ratio_table.png  <- matches ratio table
    Saureus_k_value_fixed_comparison.png <- matches Fig 4.3
    Saureus_k_value_fixed_detailed.png   <- matches Fig 4.4
    Saureus_k_value_fixed_ratio_table.png

  Combined:
    {combined_path}

  KEY CONSISTENCY CHECK printed above:
    LR numbers should match your Chapter 4 exactly
    RF and XGB differ (they now have more features)
    - this difference IS the validation your supervisor asked for
{'='*60}
""")