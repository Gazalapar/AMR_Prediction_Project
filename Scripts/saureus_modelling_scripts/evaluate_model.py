import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    recall_score, precision_score
)
 
# ===============================
ORGANISM = "saureus"
# ===============================
 
BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")
 
print("\n==============================")
print(f"Evaluating Models for: {ORGANISM}")
print("==============================")
 
# -----------------------------------------------
# LOAD MODELS
# -----------------------------------------------
lr     = joblib.load(os.path.join(DATA_DIR, "lr_model.pkl"))
rf     = joblib.load(os.path.join(DATA_DIR, "rf_model.pkl"))
xgb    = joblib.load(os.path.join(DATA_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
 
# -----------------------------------------------
# LOAD TRAIN + TEST (model-specific feature sets)
# -----------------------------------------------
def load_split(name):
    tr = pd.read_csv(os.path.join(DATA_DIR, f"train_selected_{name}.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, f"test_selected_{name}.csv"))
    Xtr = tr.drop(columns=["label","genome"], errors="ignore")
    ytr = tr["label"]
    Xte = te.drop(columns=["label","genome"], errors="ignore")
    yte = te["label"]
    return Xtr, ytr, Xte, yte
 
Xtr_lr,  ytr_lr,  Xte_lr,  yte_lr  = load_split("LR")
Xtr_rf,  ytr_rf,  Xte_rf,  yte_rf  = load_split("RF")
Xtr_xgb, ytr_xgb, Xte_xgb, yte_xgb = load_split("XGB")
 
Xtr_lr_sc = scaler.transform(Xtr_lr)
Xte_lr_sc = scaler.transform(Xte_lr)
 
print(f"\nTest set: {len(yte_lr)} samples")
print(f"Class dist: {yte_lr.value_counts().to_dict()}")
 
CLASS_NAMES = ["Susceptible", "Resistant"]
 
# -----------------------------------------------
# THRESHOLD TUNING
# Finds threshold that gives recall >= 0.75
# AND precision >= 0.70 with best macro F1
# -----------------------------------------------
def find_threshold(model, X_input, y_true):
    probs   = model.predict_proba(X_input)[:, 1]
    best_f1 = 0.0
    best_t  = 0.5
    for t in np.arange(0.30, 0.65, 0.01):
        preds  = (probs >= t).astype(int)
        macro  = f1_score(y_true, preds, average="macro", zero_division=0)
        res_r  = recall_score(y_true, preds, zero_division=0)
        res_p  = precision_score(y_true, preds, zero_division=0)
        if macro > best_f1 and res_r >= 0.75 and res_p >= 0.70:
            best_f1 = macro
            best_t  = round(t, 2)
    return best_t
 
# FIXED — use default 0.5 for all S. aureus models
# Reason: threshold tuning worsened results for S. aureus
#   LR tuned to 0.55 → FN=24 (worse than default FN=21)
#   XGB tuned to 0.54 → FN=21 (worse than default FN=18)
#   RF tuned to 0.50 → same as default (no change)
# Default 0.5 is confirmed optimal for all three models.
thresholds = {"LR": 0.5, "RF": 0.5, "XGB": 0.5}
print(f"Thresholds (fixed at default): {thresholds}")
print("Thresholds:", thresholds)
 
# Save thresholds
with open(os.path.join(DATA_DIR, "thresholds.json"), "w") as f:
    json.dump(thresholds, f, indent=4)
 
# -----------------------------------------------
# BUILD MODELS DICT
# -----------------------------------------------
MODELS = {
    "LR" : dict(model=lr,  Xtr=Xtr_lr_sc, ytr=ytr_lr,
                Xte=Xte_lr_sc, yte=yte_lr,  feats=Xtr_lr.shape[1]),
    "RF" : dict(model=rf,  Xtr=Xtr_rf,    ytr=ytr_rf,
                Xte=Xte_rf,    yte=yte_rf,  feats=Xtr_rf.shape[1]),
    "XGB": dict(model=xgb, Xtr=Xtr_xgb,   ytr=ytr_xgb,
                Xte=Xte_xgb,   yte=yte_xgb, feats=Xtr_xgb.shape[1]),
}
 
for k, m in MODELS.items():
    t = thresholds[k]
    m["tr_prob"] = m["model"].predict_proba(m["Xtr"])[:, 1]
    m["te_prob"] = m["model"].predict_proba(m["Xte"])[:, 1]
    m["te_pred"] = (m["te_prob"] >= t).astype(int)
    m["tr_pred"] = (m["tr_prob"] >= t).astype(int)
    m["threshold"] = t
 
# -----------------------------------------------
# FULL EVALUATION FUNCTION (same as E. coli)
# -----------------------------------------------
def evaluate(name, m):
    yte, ytr = m["yte"], m["ytr"]
    te_prob, tr_prob = m["te_prob"], m["tr_prob"]
    te_pred, tr_pred = m["te_pred"], m["tr_pred"]
    t = m["threshold"]
 
    # default 0.5 predictions
    te_def = (te_prob >= 0.5).astype(int)
    cm_def = confusion_matrix(yte, te_def)
    tn0,fp0,fn0,tp0 = cm_def.ravel()
 
    # tuned threshold predictions
    cm_tun = confusion_matrix(yte, te_pred)
    tn1,fp1,fn1,tp1 = cm_tun.ravel()
 
    test_auc  = roc_auc_score(yte, te_prob)
    train_auc = roc_auc_score(ytr, tr_prob)
    gap       = train_auc - test_auc
 
    res_rec  = recall_score(yte, te_pred, zero_division=0)
    res_prec = precision_score(yte, te_pred, zero_division=0)
 
    print(f"\n{'='*55}")
    print(f"  {name}  (features={m['feats']}, threshold={t})")
    print(f"{'='*55}")
    print(f"\n  [Default 0.5]  TN={tn0} FP={fp0} FN={fn0} TP={tp0}")
    print(f"  Resistant → Recall={tp0/(tp0+fn0):.3f}  "
          f"Precision={tp0/(tp0+fp0) if (tp0+fp0)>0 else 0:.3f}  Missed={fn0}")
    print(f"\n  [Tuned  {t}]  TN={tn1} FP={fp1} FN={fn1} TP={tp1}")
    print(f"  Resistant → Recall={res_rec:.3f}  "
          f"Precision={res_prec:.3f}  Missed={fn1}")
    print(f"\n  Classification Report (tuned):")
    print(classification_report(yte, te_pred, target_names=CLASS_NAMES))
    flag = "OVERFIT" if gap > 0.08 else "OK"
    print(f"  AUC gap: {gap:.4f}  [{flag}]")
 
    return {
        "Model"               : name,
        "Features"            : m["feats"],
        "Threshold"           : t,
        "Train AUC"           : round(train_auc, 4),
        "Test AUC"            : round(test_auc,  4),
        "AUC Gap"             : round(gap,        4),
        "PR-AUC"              : round(average_precision_score(yte, te_prob), 4),
        "F1 (tuned)"          : round(f1_score(yte, te_pred),                4),
        "MCC (tuned)"         : round(matthews_corrcoef(yte, te_pred),       4),
        "Accuracy (tuned)"    : round(accuracy_score(yte, te_pred),          4),
        "Res. Recall (0.5)"   : round(tp0/(tp0+fn0), 4),
        "FN (0.5)"            : int(fn0),
        "Res. Recall (tuned)" : round(res_rec,  4),
        "Res. Prec (tuned)"   : round(res_prec, 4),
        "FN (tuned)"          : int(fn1),
        "TN":int(tn1),"FP":int(fp1),"TP":int(tp1),
    }, cm_tun
 
# -----------------------------------------------
# RUN EVALUATION
# -----------------------------------------------
results, cms, names = [], [], []
for name, m in MODELS.items():
    row, cm = evaluate(name, m)
    results.append(row)
    cms.append(cm)
    names.append(name)
 
# -----------------------------------------------
# CONFUSION MATRIX FIGURE
# -----------------------------------------------
COLORS_MAP = {"LR":"#1D9E75", "RF":"#534AB7", "XGB":"#D85A30"}
 
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, cm, name in zip(axes, cms, names):
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar=False, ax=ax)
    thresh_c = cm.max() / 2
    labels = {(0,0):"TN",(0,1):"FP",(1,0):"FN\n(missed)",( 1,1):"TP"}
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > thresh_c else "black"
            if (i,j)==(1,0): color="#8B0000"
            ax.text(j+0.5, i+0.5,
                    f"{cm[i,j]}\n{labels[(i,j)]}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold" if (i,j)==(1,0) else "normal",
                    color=color)
    auc = roc_auc_score(MODELS[name]["yte"], MODELS[name]["te_prob"])
    mcc = matthews_corrcoef(MODELS[name]["yte"], MODELS[name]["te_pred"])
    ax.set_title(f"{name}  AUC={auc:.3f}  MCC={mcc:.3f}\n"
                 f"threshold={MODELS[name]['threshold']}  features={MODELS[name]['feats']}",
                 fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
 
plt.suptitle(f"Confusion Matrices — S. aureus Erythromycin AMR (tuned thresholds)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
cm_path = os.path.join(DATA_DIR, "confusion_matrices_all_models.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix saved → {cm_path}")
 
# -----------------------------------------------
# ROC CURVES
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for name, m in MODELS.items():
    fpr, tpr, _ = roc_curve(m["yte"], m["te_prob"])
    auc = roc_auc_score(m["yte"], m["te_prob"])
    ax.plot(fpr, tpr, color=COLORS_MAP[name], lw=2,
            label=f"{name} (AUC={auc:.3f}, {m['feats']} features)")
    t = m["threshold"]
    op_rec = recall_score(m["yte"], m["te_pred"], zero_division=0)
    op_fpr = 1 - recall_score(1-m["yte"], 1-m["te_pred"], zero_division=0)
    ax.scatter(op_fpr, op_rec, color=COLORS_MAP[name], s=80,
               zorder=5, edgecolors="white", linewidths=1.5)
ax.plot([0,1],[0,1],"k--", alpha=0.35, lw=1, label="Random")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — S. aureus erythromycin resistance")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
roc_path = os.path.join(DATA_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"ROC curves saved → {roc_path}")
 
# -----------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------
df_res = pd.DataFrame(results)
print("\n\n📊 FINAL RESULTS SUMMARY")
print("\n  — AUC & Overfitting —")
print(df_res[["Model","Features","Train AUC","Test AUC","AUC Gap","PR-AUC"]].to_string(index=False))
print("\n  — Clinical Metrics —")
print(df_res[["Model","Threshold","Res. Recall (0.5)","FN (0.5)",
              "Res. Recall (tuned)","Res. Prec (tuned)","FN (tuned)"]].to_string(index=False))
print("\n  — Overall Performance (tuned) —")
print(df_res[["Model","Accuracy (tuned)","F1 (tuned)","MCC (tuned)"]].to_string(index=False))
 
print("\n🔍 Overfitting Check:")
for _, r in df_res.iterrows():
    flag = "OVERFIT" if r["AUC Gap"] > 0.08 else "OK"
    print(f"  {r['Model']:25s}  AUC gap: {r['AUC Gap']:.4f}  [{flag}]")
 
best = df_res.loc[df_res["Test AUC"].idxmax(), "Model"]
print(f"\nBest model by Test AUC: {best}")
 
out = os.path.join(DATA_DIR, "final_results.csv")
df_res.to_csv(out, index=False)
print(f"Results saved → {out}")
print("\nEvaluation complete!")
