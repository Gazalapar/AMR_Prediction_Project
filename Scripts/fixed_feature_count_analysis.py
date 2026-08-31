"""
SCRIPT 1 - FIXED FEATURE COMPARISON
=====================================
Supervisor validation experiment.
Runs LR, RF, XGB on SAME fixed feature counts.
Proves LR wins E.coli and RF wins S.aureus
regardless of feature count.

Uses Youden's J threshold - same as your final results.

E. coli  : fixed 30, 40, 80 features
S. aureus: fixed 10, 12, 20 features

OUTPUT - results_{organism}/fixed_features_complete/
  all_results_{organism}.csv     <- every metric at every count
  confusion_matrix_{model}_{n}feat.png
  roc_curve_{model}_{n}feat.png
  SUPERVISOR_FIXED_FEATURE_SUMMARY.csv  <- show this to supervisor

Run: py script1_fixed_feature_comparison.py
Time: ~8 minutes for both organisms
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    confusion_matrix, recall_score, precision_score,
    roc_curve, classification_report
)

# ======================================================
BASE_DIR = r"C:\AMR_Prediction_Project"
# ======================================================

ORGANISMS = {
    "ecoli": {
        "data_dir"      : os.path.join(BASE_DIR, "Data", "results_ecoli"),
        "fixed_counts"  : [30, 40, 80],
        "test_resistant": 263,
        "label"         : "E. coli - Ciprofloxacin",
        # Original results for comparison table
        "orig": {
            "LR" : {"features":80,"threshold":0.45,"train_auc":0.9249,
                    "test_auc":0.8734,"gap":0.0515,"mcc":0.5972,
                    "f1":0.7954,"accuracy":0.7985,"recall":0.7833,
                    "precision":0.8093,"fn":57,
                    "tn":214,"fp":49,"tp":206},
            "RF" : {"features":40,"threshold":0.47,"train_auc":0.9411,
                    "test_auc":0.8602,"gap":0.0814,"mcc":0.5577,
                    "f1":0.7665,"accuracy":0.7776,"recall":0.7300,
                    "precision":0.8067,"fn":71,
                    "tn":217,"fp":46,"tp":192},
            "XGB": {"features":30,"threshold":0.48,"train_auc":0.9447,
                    "test_auc":0.8753,"gap":0.0694,"mcc":0.5555,
                    "f1":0.7737,"accuracy":0.7776,"recall":0.7605,
                    "precision":0.7874,"fn":63,
                    "tn":209,"fp":54,"tp":200},
        },
    },
    "Saureus": {
        "data_dir"      : os.path.join(BASE_DIR, "Data", "results_Saureus"),
        "fixed_counts"  : [10, 12, 20],
        "test_resistant": 74,
        "label"         : "S. aureus - Erythromycin",
        "orig": {
            "LR" : {"features":20,"threshold":0.50,"train_auc":0.8776,
                    "test_auc":0.8187,"gap":0.0626,"mcc":0.3789,
                    "f1":0.6974,"accuracy":0.7432,"recall":0.7162,
                    "precision":0.6818,"fn":21,
                    "tn":49,"fp":25,"tp":53},
            "RF" : {"features":12,"threshold":0.50,"train_auc":0.8869,
                    "test_auc":0.8187,"gap":0.0682,"mcc":0.4867,
                    "f1":0.7467,"accuracy":0.7635,"recall":0.7568,
                    "precision":0.7368,"fn":18,
                    "tn":54,"fp":20,"tp":56},
            "XGB": {"features":10,"threshold":0.50,"train_auc":0.8561,
                    "test_auc":0.7878,"gap":0.0681,"mcc":0.4867,
                    "f1":0.7467,"accuracy":0.7635,"recall":0.7568,
                    "precision":0.7368,"fn":18,
                    "tn":54,"fp":20,"tp":56},
        },
    },
}

CLASS_NAMES = ["Susceptible", "Resistant"]


# -- Youden's J threshold --------------------------------------
def youdens_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    idx = np.argmax(j)
    return round(float(thresholds[idx]), 4), round(float(j[idx]), 4)


# -- Generalisation flag ---------------------------------------
def gen_flag(gap):
    if gap > 0.08:   return "WARNING - exceeds 0.08 threshold"
    if gap > 0.065:  return "CAUTION - approaching threshold"
    return "OK - below threshold"


# -- Save confusion matrix -------------------------------------
def save_cm(cm, model_name, n_feat, experiment,
            thresh, auc, gap, fn, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(
        f"{model_name} | {experiment}\n"
        f"{n_feat} features | threshold={thresh}\n"
        f"AUC={auc} | Gap={gap} | FN={fn}",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_xticks([0,1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticks([0,1])
    ax.set_yticklabels(CLASS_NAMES, fontsize=8,
                        rotation=90, va="center")
    tc = cm.max() / 2
    lbl = {(0,0):"TN",(0,1):"FP",
           (1,0):"FN\n(missed)",(1,1):"TP"}
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i,j] > tc else "black"
            ax.text(j, i, f"{cm[i,j]}\n{lbl[(i,j)]}",
                    ha="center", va="center", fontsize=10,
                    color=col,
                    fontweight="bold" if (i,j)==(1,0) else "normal")
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"confusion_matrix_{model_name}_{n_feat}feat.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# -- Save ROC curve --------------------------------------------
def save_roc(y_te, te_prob, y_tr, tr_prob,
             model_name, n_feat, organism, experiment,
             thresh, output_dir):
    fpr_te, tpr_te, _ = roc_curve(y_te, te_prob)
    fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, tr_prob)
    j_idx = np.argmax(tpr_tr - fpr_tr)
    auc_te = round(roc_auc_score(y_te, te_prob), 3)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_te, tpr_te, color="#C55A11", lw=2,
            label=f"{model_name} Test (AUC={auc_te})")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.5)")
    ax.scatter([fpr_tr[j_idx]], [tpr_tr[j_idx]],
               color="#1F3864", s=80, zorder=5,
               label=f"Youden J={thresh}")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(
        f"ROC - {model_name} | {n_feat} features\n"
        f"{organism} | {experiment}",
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"roc_curve_{model_name}_{n_feat}feat.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# -- Full evaluation -------------------------------------------
# fixed_thresh: pass the ORIGINAL saved threshold (e.g. 0.45 for LR)
# This ensures FN is comparable to your thesis Table 5.1
def evaluate(model, X_tr, X_te, y_tr, y_te,
             model_name, n_feat, organism,
             experiment, output_dir, scale=False,
             fixed_thresh=None):
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    model.fit(X_tr, y_tr)
    tr_prob = model.predict_proba(X_tr)[:,1]
    te_prob = model.predict_proba(X_te)[:,1]

    # Always compute Youden's J for AUC reporting
    youdens_t, j_val = youdens_threshold(y_tr, tr_prob)

    # Use original fixed threshold for FN/confusion matrix
    # so FN is directly comparable to thesis Table 5.1
    if fixed_thresh is not None:
        thresh = fixed_thresh
        thresh_note = f"{thresh} (original Youden)"
    else:
        thresh = youdens_t
        thresh_note = f"{thresh} (recalculated Youden)"

    te_pred = (te_prob >= thresh).astype(int)

    cm = confusion_matrix(y_te, te_pred)
    tn, fp, fn, tp = cm.ravel()

    tr_auc = roc_auc_score(y_tr, tr_prob)
    te_auc = roc_auc_score(y_te, te_prob)
    gap    = round(tr_auc - te_auc, 4)
    mcc    = round(matthews_corrcoef(y_te, te_pred), 4)
    f1     = round(f1_score(y_te, te_pred), 4)
    acc    = round(accuracy_score(y_te, te_pred), 4)
    rec    = round(recall_score(y_te, te_pred), 4)
    prec   = round(precision_score(y_te, te_pred, zero_division=0), 4)
    prauc  = round(average_precision_score(y_te, te_prob), 4)
    fn_pct = round(fn / (fn + tp) * 100, 1)

    cm_path  = save_cm(cm, model_name, n_feat, experiment,
                        thresh, round(te_auc,4), gap, fn, output_dir)
    roc_path = save_roc(y_te, te_prob, y_tr, tr_prob,
                         model_name, n_feat, organism,
                         experiment, thresh, output_dir)

    return {
        "Organism"            : organism,
        "Experiment"          : experiment,
        "Model"               : model_name,
        "Features"            : n_feat,
        "Threshold"           : thresh,
        "Threshold_Note"      : thresh_note,
        "Youdens_J_recalc"    : youdens_t,
        "Train_AUC"           : round(tr_auc, 4),
        "Test_AUC"            : round(te_auc, 4),
        "AUC_Gap"             : gap,
        "MCC"                 : mcc,
        "F1"                  : f1,
        "Accuracy"            : acc,
        "Recall"              : rec,
        "Precision"           : prec,
        "PR_AUC"              : prauc,
        "TN"                  : int(tn),
        "FP"                  : int(fp),
        "FN"                  : int(fn),
        "TP"                  : int(tp),
        "FN_Percent"          : fn_pct,
        "Generalisation"      : gen_flag(gap),
        "CM_image"            : os.path.basename(cm_path),
        "ROC_image"           : os.path.basename(roc_path),
    }


# ======================================================
# MAIN
# ======================================================
all_rows = []

for organism, cfg in ORGANISMS.items():
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    DATA_DIR   = cfg["data_dir"]
    OUTPUT_DIR = os.path.join(DATA_DIR, "fixed_features_complete")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "best_params.json")) as f:
        params = json.load(f)
    lr_p  = params["LR"]
    rf_p  = params["RF"]
    xgb_p = params["XGB"]

    # Load original Youden thresholds - keeps FN comparable to thesis
    thresh_file = os.path.join(DATA_DIR, "thresholds.json")
    with open(thresh_file) as f:
        saved_thresholds = json.load(f)
    thresh_lr  = saved_thresholds["LR"]
    thresh_rf  = saved_thresholds["RF"]
    thresh_xgb = saved_thresholds["XGB"]
    print(f"  Original thresholds: LR={thresh_lr} RF={thresh_rf} XGB={thresh_xgb}")

    train_lr = pd.read_csv(
        os.path.join(DATA_DIR, "train_selected_LR.csv"))
    test_lr  = pd.read_csv(
        os.path.join(DATA_DIR, "test_selected_LR.csv"))
    X_train = train_lr.drop(columns=["label","genome"], errors="ignore")
    y_train = train_lr["label"]
    X_test  = test_lr.drop(columns=["label","genome"], errors="ignore")
    y_test  = test_lr["label"]

    print(f"  Train {X_train.shape} | Test {X_test.shape}")
    print(f"  Train dist: {y_train.value_counts().to_dict()}")
    print(f"  Test dist:  {y_test.value_counts().to_dict()}")

    # Rank features by permutation importance
    print("\n  Ranking features by permutation importance...")
    rf_r = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_r.fit(X_train, y_train)
    perm = permutation_importance(
        rf_r, X_train, y_train,
        n_repeats=5, random_state=42, n_jobs=-1)
    ranked = X_train.columns[
        np.argsort(perm.importances_mean)[::-1]
    ].tolist()
    print(f"  Top 5 features: {ranked[:5]}")

    org_rows = []

    # Add original results
    for mname, orig in cfg["orig"].items():
        org_rows.append({
            "Organism"            : organism,
            "Experiment"          : "Original_model-specific",
            "Model"               : mname,
            "Features"            : orig["features"],
            "Threshold"           : orig["threshold"],
            "Threshold_Note"      : f"{orig['threshold']} (original Youden from thesis)",
            "Youdens_J_recalc"    : "original_study",
            "Train_AUC"           : orig["train_auc"],
            "Test_AUC"            : orig["test_auc"],
            "AUC_Gap"             : orig["gap"],
            "MCC"                 : orig["mcc"],
            "F1"                  : orig["f1"],
            "Accuracy"            : orig["accuracy"],
            "Recall"              : orig["recall"],
            "Precision"           : orig["precision"],
            "PR_AUC"              : "see_original",
            "TN"                  : orig["tn"],
            "FP"                  : orig["fp"],
            "FN"                  : orig["fn"],
            "TP"                  : orig["tp"],
            "FN_Percent"          : round(
                orig["fn"]/cfg["test_resistant"]*100, 1),
            "Generalisation"      : gen_flag(orig["gap"]),
            "CM_image"            : "see_original_results_folder",
            "ROC_image"           : "see_original_results_folder",
        })

    # Run fixed feature experiments
    for n in cfg["fixed_counts"]:
        print(f"\n  -- Fixed {n} features ----------------------")
        top = ranked[:n]
        X_tr_n = X_train[top].values
        X_te_n = X_test[top].values

        for mname, scale in [("LR",True),("RF",False),("XGB",False)]:
            print(f"    {mname}...", end=" ", flush=True)
            # Use original saved threshold so FN matches thesis
            fixed_t = thresh_lr if mname=="LR" else (
                      thresh_rf if mname=="RF" else thresh_xgb)
            if mname == "LR":
                model = LogisticRegression(
                    C=lr_p["C"], penalty=lr_p["penalty"],
                    solver=lr_p.get("solver","liblinear"),
                    max_iter=2000, class_weight="balanced",
                    n_jobs=-1)
            elif mname == "RF":
                model = RandomForestClassifier(
                    n_estimators=rf_p.get("n_estimators",300),
                    max_depth=rf_p.get("max_depth",5),
                    min_samples_leaf=rf_p.get("min_samples_leaf",15),
                    max_features=rf_p.get("max_features",0.3),
                    random_state=42, n_jobs=-1,
                    class_weight="balanced")
            else:
                model = XGBClassifier(
                    n_estimators=xgb_p.get("n_estimators",200),
                    max_depth=xgb_p.get("max_depth",3),
                    learning_rate=xgb_p.get("learning_rate",0.05),
                    subsample=xgb_p.get("subsample",0.6),
                    colsample_bytree=xgb_p.get("colsample_bytree",0.4),
                    reg_lambda=xgb_p.get("reg_lambda",10),
                    reg_alpha=xgb_p.get("reg_alpha",1.0),
                    eval_metric="logloss", random_state=42,
                    use_label_encoder=False)

            r = evaluate(
                model, X_tr_n, X_te_n,
                y_train.values, y_test.values,
                mname, n, organism,
                f"Fixed_{n}_features",
                OUTPUT_DIR, scale=scale,
                fixed_thresh=fixed_t)
            org_rows.append(r)
            print(f"AUC={r['Test_AUC']} Gap={r['AUC_Gap']} "
                  f"MCC={r['MCC']} FN={r['FN']} "
                  f"thresh={r['Threshold']}")

    # Save organism CSV
    df_org = pd.DataFrame(org_rows)
    csv_path = os.path.join(
        OUTPUT_DIR, f"all_results_{organism}.csv")
    df_org.to_csv(csv_path, index=False)
    print(f"\n  CSV saved -> {csv_path}")

    # Winner summary
    print(f"\n  WINNER SUMMARY - {organism}")
    print(f"  {'-'*52}")
    orig_row = df_org[df_org["Experiment"]=="Original_model-specific"]
    print(f"  Original: LR={orig_row[orig_row['Model']=='LR']['Test_AUC'].values[0]} "
          f"| RF={orig_row[orig_row['Model']=='RF']['Test_AUC'].values[0]} "
          f"| XGB={orig_row[orig_row['Model']=='XGB']['Test_AUC'].values[0]}")
    for n in cfg["fixed_counts"]:
        sub = df_org[df_org["Features"]==n].copy()
        sub["Test_AUC"] = pd.to_numeric(sub["Test_AUC"])
        sub["AUC_Gap"]  = pd.to_numeric(sub["AUC_Gap"])
        win_auc = sub.loc[sub["Test_AUC"].idxmax()]
        win_gen = sub.loc[sub["AUC_Gap"].idxmin()]
        print(f"  Fixed {n:2d}: AUC winner={win_auc['Model']} "
              f"({win_auc['Test_AUC']}) | "
              f"Best generalisation={win_gen['Model']} "
              f"(gap={win_gen['AUC_Gap']})")

    all_rows.extend(org_rows)

# Save combined supervisor CSV
summary_path = os.path.join(
    BASE_DIR, "Data",
    "SUPERVISOR_FIXED_FEATURE_SUMMARY.csv")
pd.DataFrame(all_rows).to_csv(summary_path, index=False)

print(f"""
{'='*60}
  SCRIPT 1 COMPLETE
{'='*60}
  Per organism (in fixed_features_complete folder):
    all_results_ecoli.csv
    all_results_Saureus.csv
    confusion_matrix_{{model}}_{{n}}feat.png
    roc_curve_{{model}}_{{n}}feat.png

  Combined:
    {summary_path}
    <- SHOW THIS TO SUPERVISOR

  Next: run script2_kvalue_fixed_features.py
{'='*60}
""")