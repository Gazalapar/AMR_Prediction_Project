"""
FIXED FEATURE XGB EVALUATION — E. coli
=======================================
Retrains XGB on top 80 features (same as LR)
Applies Youden's J threshold (same as original study)
Generates confusion matrix + ROC curve
Gives final comparable numbers for thesis update

Run this AFTER fixed_feature_comparison.py
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    recall_score, precision_score, roc_curve
)

# ══════════════════════════════════════════════════════════════
ORGANISM     = "ecoli"
FIXED_FEATS  = 80        # same as LR feature count
# ══════════════════════════════════════════════════════════════

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")
FIXED_DIR = os.path.join(DATA_DIR, "fixed_features")
os.makedirs(FIXED_DIR, exist_ok=True)

print("\n" + "="*62)
print(f"  XGB FIXED FEATURE EVALUATION — {ORGANISM.upper()}")
print(f"  Features: {FIXED_FEATS} (same as LR)")
print(f"  Threshold method: Youden's J")
print("="*62)

# ── Load best XGB params ──────────────────────────────────────
with open(os.path.join(DATA_DIR, "best_params.json")) as f:
    params = json.load(f)
xgb_p = params["XGB"]
print(f"\n📋 XGB params: {xgb_p}")

# ── Load full LR feature set (80 features = base) ─────────────
print(f"\n📥 Loading LR feature set as base ({FIXED_FEATS} features)...")
train_lr = pd.read_csv(os.path.join(DATA_DIR, "train_selected_LR.csv"))
test_lr  = pd.read_csv(os.path.join(DATA_DIR, "test_selected_LR.csv"))

X_train_full = train_lr.drop(columns=["label","genome"], errors="ignore")
y_train      = train_lr["label"]
X_test_full  = test_lr.drop(columns=["label","genome"], errors="ignore")
y_test       = test_lr["label"]

print(f"  Train shape: {X_train_full.shape}")
print(f"  Test shape:  {X_test_full.shape}")
print(f"  Train class dist: {y_train.value_counts().to_dict()}")
print(f"  Test class dist:  {y_test.value_counts().to_dict()}")

# ── Rank features by permutation importance ───────────────────
print(f"\n📊 Ranking {FIXED_FEATS} features by permutation importance...")
rf_ranker = RandomForestClassifier(
    n_estimators=100, max_depth=5,
    random_state=42, n_jobs=-1
)
rf_ranker.fit(X_train_full, y_train)
perm = permutation_importance(
    rf_ranker, X_train_full, y_train,
    n_repeats=5, random_state=42, n_jobs=-1
)
ranked_idx      = np.argsort(perm.importances_mean)[::-1]
ranked_features = X_train_full.columns[ranked_idx].tolist()
top_features    = ranked_features[:FIXED_FEATS]

print(f"  Top 5 features: {top_features[:5]}")

X_train = X_train_full[top_features]
X_test  = X_test_full[top_features]

# ── Train XGB with best params ────────────────────────────────
print(f"\n🚀 Training XGB on {FIXED_FEATS} fixed features...")

xgb = XGBClassifier(
    n_estimators    = xgb_p.get("n_estimators", 200),
    max_depth       = xgb_p.get("max_depth", 3),
    learning_rate   = xgb_p.get("learning_rate", 0.05),
    subsample       = xgb_p.get("subsample", 0.6),
    colsample_bytree= xgb_p.get("colsample_bytree", 0.4),
    reg_lambda      = xgb_p.get("reg_lambda", 10),
    reg_alpha       = xgb_p.get("reg_alpha", 1.0),
    eval_metric     = "logloss",
    random_state    = 42,
    use_label_encoder = False
)
xgb.fit(X_train, y_train)
print("  ✅ XGB trained.")

# ── Youden's J threshold ──────────────────────────────────────
print("\n🎯 Finding optimal threshold using Youden's J...")

train_prob = xgb.predict_proba(X_train)[:,1]
test_prob  = xgb.predict_proba(X_test)[:,1]

# Youden's J on TRAINING set to find threshold
fpr, tpr, thresholds_roc = roc_curve(y_train, train_prob)
youdens_j   = tpr - fpr
best_idx    = np.argmax(youdens_j)
best_thresh = round(float(thresholds_roc[best_idx]), 4)
best_j      = round(float(youdens_j[best_idx]), 4)

print(f"  Youden's J = {best_j} at threshold = {best_thresh}")

# ── Evaluate at Youden's J threshold ─────────────────────────
test_pred  = (test_prob  >= best_thresh).astype(int)
train_pred = (train_prob >= best_thresh).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()

# Train AUC for gap
train_auc = roc_auc_score(y_train, train_prob)
test_auc  = roc_auc_score(y_test,  test_prob)
auc_gap   = round(train_auc - test_auc, 4)

# All metrics
mcc   = round(matthews_corrcoef(y_test, test_pred), 4)
f1    = round(f1_score(y_test, test_pred), 4)
acc   = round(accuracy_score(y_test, test_pred), 4)
rec   = round(recall_score(y_test, test_pred), 4)
prec  = round(precision_score(y_test, test_pred, zero_division=0), 4)
prauc = round(average_precision_score(y_test, test_prob), 4)

print(f"\n{'='*62}")
print(f"  XGB — {FIXED_FEATS} Fixed Features — Youden's J threshold={best_thresh}")
print(f"{'='*62}")
print(f"\n  Confusion Matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"\n  ── Key Metrics ──")
print(f"  Test AUC      : {round(test_auc, 4)}")
print(f"  Train AUC     : {round(train_auc, 4)}")
print(f"  AUC Gap       : {auc_gap}  {'✅ OK' if auc_gap <= 0.08 else '⚠️ High'}")
print(f"  MCC           : {mcc}")
print(f"  F1 (Resistant): {f1}")
print(f"  Accuracy      : {acc}")
print(f"  Recall        : {rec}")
print(f"  Precision     : {prec}")
print(f"  PR-AUC        : {prauc}")
print(f"  FN (missed)   : {fn}")

print(f"\n  Classification Report:")
CLASS_NAMES = ["Susceptible", "Resistant"]
print(classification_report(y_test, test_pred, target_names=CLASS_NAMES))

# ── Compare with original results ─────────────────────────────
print(f"\n{'='*62}")
print("  COMPARISON — Original LR vs Updated XGB (E. coli)")
print(f"{'='*62}")
print(f"  {'Metric':<20} {'Original LR':>14} {'Updated XGB':>14}")
print(f"  {'-'*50}")
orig = {
    "Test AUC"    : 0.8734,
    "MCC"         : 0.5972,
    "F1"          : 0.7953,
    "FN"          : 57,
    "AUC Gap"     : 0.0515,
    "Accuracy"    : 0.7985,
    "Recall"      : 0.7833,
    "Precision"   : 0.8093,
}
updated = {
    "Test AUC"    : round(test_auc, 4),
    "MCC"         : mcc,
    "F1"          : f1,
    "FN"          : fn,
    "AUC Gap"     : auc_gap,
    "Accuracy"    : acc,
    "Recall"      : rec,
    "Precision"   : prec,
}
for metric in orig:
    o = orig[metric]
    u = updated[metric]
    if metric == "FN":
        arrow = "✅ Lower" if u < o else ("❌ Higher" if u > o else "= Same")
    else:
        arrow = "✅ Higher" if u > o else ("⚠️ Lower" if u < o else "= Same")
    print(f"  {metric:<20} {str(o):>14} {str(u):>14}   {arrow}")

# ── FIGURE 1 — Confusion Matrix ───────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_title(f"XGB — {FIXED_FEATS} Fixed Features\n"
             f"Youden's J threshold={best_thresh}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_xticks([0,1]); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(CLASS_NAMES, fontsize=9,
                                          rotation=90, va="center")
thresh_c = cm.max() / 2
labels   = {(0,0):"TN", (0,1):"FP", (1,0):"FN\n(missed\nresistant)", (1,1):"TP"}
for i in range(2):
    for j in range(2):
        color = "white" if cm[i,j] > thresh_c else "black"
        ax.text(j, i, f"{cm[i,j]}\n{labels[(i,j)]}",
                ha="center", va="center", fontsize=11, color=color,
                fontweight="bold" if (i,j)==(1,0) else "normal")

plt.tight_layout()
cm_path = os.path.join(FIXED_DIR, "xgb_fixed80_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📁 Confusion matrix saved → {cm_path}")

# ── FIGURE 2 — ROC Curve ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
ax.plot(fpr_test, tpr_test, color="#C55A11", lw=2,
        label=f"XGB (AUC = {round(test_auc,3)})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random (AUC = 0.5)")
ax.scatter([fpr[best_idx]], [tpr[best_idx]],
           color="#1F3864", s=100, zorder=5,
           label=f"Youden's J threshold = {best_thresh}")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"ROC Curve — XGB {FIXED_FEATS} Fixed Features\n"
             f"E. coli Ciprofloxacin Resistance",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(FIXED_DIR, "xgb_fixed80_roc_curve.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📁 ROC curve saved → {roc_path}")

# ── Save updated model and threshold ─────────────────────────
xgb_path    = os.path.join(FIXED_DIR, "xgb_fixed80_model.pkl")
thresh_path = os.path.join(FIXED_DIR, "xgb_fixed80_threshold.json")
feats_path  = os.path.join(FIXED_DIR, "xgb_fixed80_features.json")

joblib.dump(xgb, xgb_path)
with open(thresh_path, "w") as f:
    json.dump({"threshold": best_thresh, "youdens_j": best_j}, f, indent=4)
with open(feats_path, "w") as f:
    json.dump({"features": top_features}, f, indent=4)

print(f"📁 XGB model saved  → {xgb_path}")
print(f"📁 Threshold saved  → {thresh_path}")
print(f"📁 Features saved   → {feats_path}")

# ── Final summary for thesis ──────────────────────────────────
print(f"\n{'='*62}")
print("  FINAL NUMBERS FOR THESIS UPDATE")
print(f"{'='*62}")
print(f"  Organism    : E. coli (Ciprofloxacin)")
print(f"  Best model  : XGB (at {FIXED_FEATS} fixed features)")
print(f"  Features    : {FIXED_FEATS} (same as LR — fair comparison)")
print(f"  Threshold   : {best_thresh} (Youden's J)")
print(f"  Test AUC    : {round(test_auc, 4)}")
print(f"  Train AUC   : {round(train_auc, 4)}")
print(f"  AUC Gap     : {auc_gap}")
print(f"  MCC         : {mcc}")
print(f"  F1          : {f1}")
print(f"  Accuracy    : {acc}")
print(f"  Recall      : {rec}")
print(f"  Precision   : {prec}")
print(f"  FN          : {fn}")
# Save final numbers to CSV
import pandas as pd
final = {
    "Organism": "E. coli",
    "Model": "XGB",
    "Features": FIXED_FEATS,
    "Feature_type": "Fixed (same as LR)",
    "Threshold_method": "Youden_J",
    "Threshold": best_thresh,
    "Test_AUC": round(test_auc, 4),
    "Train_AUC": round(train_auc, 4),
    "AUC_Gap": auc_gap,
    "MCC": mcc,
    "F1": f1,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "FN": fn,
    "TP": tp,
    "TN": tn,
    "FP": fp,
}
pd.DataFrame([final]).to_csv(os.path.join(FIXED_DIR, "xgb_fixed80_results.csv"), index=False)
print(f"📁 Results CSV saved → {os.path.join(FIXED_DIR, 'xgb_fixed80_results.csv')}")
print(f"\n✅ Evaluation complete! Share the numbers above with supervisor.")
print(f"\n📂 All files saved in: {FIXED_DIR}")