import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    recall_score, precision_score
)

# ===============================
ORGANISM = "ecoli"
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n==============================")
print(f"Evaluating Models for: {ORGANISM}")
print("==============================")

# Load models
lr     = joblib.load(os.path.join(DATA_DIR, "lr_model.pkl"))
rf     = joblib.load(os.path.join(DATA_DIR, "rf_model.pkl"))
xgb    = joblib.load(os.path.join(DATA_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

with open(os.path.join(DATA_DIR, "thresholds.json")) as f:
    thresholds = json.load(f)

print(f"\n🎯 Tuned thresholds: {thresholds}")

# Load model-specific feature sets
def load_split(suffix):
    tr = pd.read_csv(os.path.join(DATA_DIR, f"train_selected_{suffix}.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, f"test_selected_{suffix}.csv"))
    X_tr = tr.drop(columns=["label","genome"], errors="ignore")
    y_tr = tr["label"]
    X_te = te.drop(columns=["label","genome"], errors="ignore")
    y_te = te["label"]
    return X_tr, y_tr, X_te, y_te

X_tr_lr,  y_tr_lr,  X_te_lr,  y_te_lr  = load_split("LR")
X_tr_rf,  y_tr_rf,  X_te_rf,  y_te_rf  = load_split("RF")
X_tr_xgb, y_tr_xgb, X_te_xgb, y_te_xgb = load_split("XGB")

# Scale LR inputs
X_tr_lr_sc = scaler.transform(X_tr_lr)
X_te_lr_sc = scaler.transform(X_te_lr)

print(f"\nTest set: {len(y_te_lr)} samples  |  class dist: {y_te_lr.value_counts().to_dict()}")

CLASS_NAMES = ["Susceptible", "Resistant"]

# -----------------------------------------------
# EVALUATION FUNCTION
# -----------------------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te, thresh_key):
    # ✅ FIX: Use default threshold for Logistic Regression
    if name == "Logistic Regression":
        t = 0.5
    else:
        t = thresholds[thresh_key]

    tr_prob = model.predict_proba(X_tr)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]

    # Default threshold (0.5)
    te_def = (te_prob >= 0.5).astype(int)
    cm_def = confusion_matrix(y_te, te_def)
    tn0,fp0,fn0,tp0 = cm_def.ravel()

    # Tuned threshold
    te_tun = (te_prob >= t).astype(int)
    tr_tun = (tr_prob >= t).astype(int)
    cm_tun = confusion_matrix(y_te, te_tun)
    tn1,fp1,fn1,tp1 = cm_tun.ravel()

    res_rec_tun  = recall_score(y_te, te_tun)
    res_prec_tun = precision_score(y_te, te_tun, zero_division=0)
    prec_warn    = " ⚠️ PRECISION LOW" if res_prec_tun < 0.70 else ""

    auc_gap = roc_auc_score(y_tr, tr_prob) - roc_auc_score(y_te, te_prob)

    print(f"\n{'='*62}")
    print(f"  {name}  (features used: {X_tr.shape[1]}, threshold: {t})")
    print(f"{'='*62}")

    print(f"\n  [Default 0.5]  TN={tn0} FP={fp0} FN={fn0} TP={tp0}")
    print(f"  Resistant → Recall={tp0/(tp0+fn0):.3f}  Precision={tp0/(tp0+fp0):.3f}  "
          f"Missed={fn0}")

    print(f"\n  [Tuned  {t}]  TN={tn1} FP={fp1} FN={fn1} TP={tp1}")
    print(f"  Resistant → Recall={res_rec_tun:.3f}  Precision={res_prec_tun:.3f}  "
          f"Missed={fn1}{prec_warn}")

    print(f"\n  📋 Classification Report (tuned):")
    print(classification_report(y_te, te_tun, target_names=CLASS_NAMES))

    overfit_flag = "🚨 OVERFIT" if auc_gap > 0.05 else "✅ OK"
    print(f"  Overfitting — AUC gap: {auc_gap:.4f}  {overfit_flag}")

    return {
        "Model":               name,
        "Features":            X_tr.shape[1],
        "Threshold":           t,
        "Test AUC":            round(roc_auc_score(y_te, te_prob), 4),
        "Test PR-AUC":         round(average_precision_score(y_te, te_prob), 4),
        "AUC Gap":             round(auc_gap, 4),

        # Default threshold
        "Res Recall (0.5)":    round(tp0/(tp0+fn0), 4),
        "FN (0.5)":            int(fn0),

        # Tuned threshold
        "Res Recall (tuned)":  round(res_rec_tun, 4),
        "Res Prec (tuned)":    round(res_prec_tun, 4),
        "FN (tuned)":          int(fn1),

        "Test Acc (tuned)":    round(accuracy_score(y_te, te_tun), 4),
        "Test F1 (tuned)":     round(f1_score(y_te, te_tun), 4),
        "Test MCC (tuned)":    round(matthews_corrcoef(y_te, te_tun), 4),
    }, cm_tun

# -----------------------------------------------
# RUN
# -----------------------------------------------
results, cms, names = [], [], []

for name, model, X_tr, y_tr, X_te, y_te, key in [
    ("Logistic Regression", lr,  X_tr_lr_sc, y_tr_lr,  X_te_lr_sc, y_te_lr,  "LR"),
    ("Random Forest",        rf,  X_tr_rf,   y_tr_rf,  X_te_rf,    y_te_rf,  "RF"),
    ("XGBoost",              xgb, X_tr_xgb,  y_tr_xgb, X_te_xgb,   y_te_xgb, "XGB"),
]:
    row, cm = evaluate(name, model, X_tr, y_tr, X_te, y_te, key)
    results.append(row)
    cms.append(cm)
    names.append(name)

# -----------------------------------------------
# CONFUSION MATRIX FIGURE
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, cm, name in zip(axes, cms, names):
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_xticks([0,1]); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(CLASS_NAMES, fontsize=8,
                                               rotation=90, va="center")
    thresh_c = cm.max() / 2
    labels = {(0,0):"TN", (0,1):"FP", (1,0):"FN\n(missed\nresistant)", (1,1):"TP"}
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > thresh_c else "black"
            ax.text(j, i, f"{cm[i,j]}\n{labels[(i,j)]}",
                    ha="center", va="center", fontsize=10, color=color,
                    fontweight="bold" if (i,j)==(1,0) else "normal")

plt.suptitle(f"Confusion Matrices — {ORGANISM.upper()} AMR (tuned thresholds)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

cm_path = os.path.join(DATA_DIR, "confusion_matrices_all_models.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📁 Confusion matrix saved → {cm_path}")

# -----------------------------------------------
# SUMMARY
# -----------------------------------------------
df_res = pd.DataFrame(results)

print("\n\n📊 FINAL RESULTS SUMMARY")
print("\n  — AUC & Overfitting —")
print(df_res[["Model","Features","Test AUC","Test PR-AUC","AUC Gap"]].to_string(index=False))

print("\n  — Resistant Class (default vs tuned threshold) —")
print(df_res[["Model","Threshold",
              "Res Recall (0.5)","FN (0.5)",
              "Res Recall (tuned)","Res Prec (tuned)","FN (tuned)"]].to_string(index=False))

print("\n  — Overall Performance (tuned threshold) —")
print(df_res[["Model","Test Acc (tuned)","Test F1 (tuned)","Test MCC (tuned)"]].to_string(index=False))

print("\n🔍 Overfitting Check:")
for _, r in df_res.iterrows():
    flag = "🚨 OVERFIT" if r["AUC Gap"] > 0.05 else "✅ OK"
    prec_flag = "  ⚠️ precision < 0.70!" if r["Res Prec (tuned)"] < 0.70 else ""
    print(f"  {r['Model']:25s}  AUC gap: {r['AUC Gap']:.4f}  {flag}{prec_flag}")

best = df_res.loc[df_res["Test AUC"].idxmax(), "Model"]
print(f"\n🏆 Best model by Test AUC: {best}")

out = os.path.join(DATA_DIR, "final_results.csv")
df_res.to_csv(out, index=False)
print(f"📁 Results saved → {out}")
print("\n✅ Evaluation complete!")
