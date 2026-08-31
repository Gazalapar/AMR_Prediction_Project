"""
FIXED FEATURE RF EVALUATION — S. aureus
========================================
Retrains RF on top 12 features (same as original RF)
Applies Youden's J threshold (same as original study)
Generates confusion matrix + ROC curve
Gives final comparable numbers for thesis update

Run this after fixed_feature_comparison.py for saureus
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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    recall_score, precision_score, roc_curve
)

# ══════════════════════════════════════════════════════════════
ORGANISM     = "Saureus"   # match your exact folder name
FIXED_FEATS  = 12          # RF original count — lowest FN=17
# ══════════════════════════════════════════════════════════════

BASE_DIR  = r"C:\AMR_Prediction_Project"
DATA_DIR  = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")
FIXED_DIR = os.path.join(DATA_DIR, "fixed_features")
os.makedirs(FIXED_DIR, exist_ok=True)

print("\n" + "="*62)
print(f"  RF FIXED FEATURE EVALUATION — {ORGANISM.upper()}")
print(f"  Features: {FIXED_FEATS} (original RF count — lowest FN)")
print(f"  Threshold method: Youden's J")
print("="*62)

# ── Load best RF params ───────────────────────────────────────
with open(os.path.join(DATA_DIR, "best_params.json")) as f:
    params = json.load(f)
rf_p = params["RF"]
print(f"\n📋 RF params: {rf_p}")

# ── Load full LR feature set as base (20 features) ───────────
print(f"\n📥 Loading LR feature set as base (20 features)...")
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
print(f"\n📊 Ranking features by permutation importance...")
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

print(f"  Top {FIXED_FEATS} features: {top_features}")

X_train = X_train_full[top_features]
X_test  = X_test_full[top_features]

# ── Train RF with best params ─────────────────────────────────
print(f"\n🌲 Training RF on {FIXED_FEATS} fixed features...")

rf = RandomForestClassifier(
    n_estimators    = rf_p.get("n_estimators", 200),
    max_depth       = rf_p.get("max_depth", 4),
    min_samples_leaf= rf_p.get("min_samples_leaf", 40),
    max_features    = rf_p.get("max_features", 0.2),
    random_state    = 42,
    n_jobs          = -1,
    class_weight    = "balanced"
)
rf.fit(X_train, y_train)
print("  ✅ RF trained.")

# ── Youden's J threshold ──────────────────────────────────────
print("\n🎯 Finding optimal threshold using Youden's J...")

train_prob = rf.predict_proba(X_train)[:,1]
test_prob  = rf.predict_proba(X_test)[:,1]

# Youden's J on TRAINING set
fpr_tr, tpr_tr, thresholds_roc = roc_curve(y_train, train_prob)
youdens_j   = tpr_tr - fpr_tr
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
print(f"  RF — {FIXED_FEATS} Fixed Features — Youden's J threshold={best_thresh}")
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

CLASS_NAMES = ["Susceptible", "Resistant"]
print(f"\n  Classification Report:")
print(classification_report(y_test, test_pred, target_names=CLASS_NAMES))

# ── Compare with original results ─────────────────────────────
print(f"\n{'='*62}")
print("  COMPARISON — Original RF vs Fixed Feature RF (S. aureus)")
print(f"{'='*62}")
print(f"  {'Metric':<20} {'Original RF':>14} {'Fixed RF':>14}")
print(f"  {'-'*50}")

orig = {
    "Test AUC"  : 0.8187,
    "MCC"       : 0.4867,
    "F1"        : 0.7470,
    "FN"        : 18,
    "AUC Gap"   : 0.0682,
    "Accuracy"  : 0.7432,
    "Recall"    : 0.7432,
    "Precision" : 0.7500,
}
updated = {
    "Test AUC"  : round(test_auc, 4),
    "MCC"       : mcc,
    "F1"        : f1,
    "FN"        : fn,
    "AUC Gap"   : auc_gap,
    "Accuracy"  : acc,
    "Recall"    : rec,
    "Precision" : prec,
}
for metric in orig:
    o = orig[metric]
    u = updated[metric]
    if metric == "FN":
        arrow = "✅ Lower" if u < o else ("⚠️ Higher" if u > o else "= Same")
    else:
        arrow = "✅ Higher" if u > o else ("⚠️ Lower" if u < o else "= Same")
    print(f"  {metric:<20} {str(o):>14} {str(u):>14}   {arrow}")

# ── All fixed counts summary ───────────────────────────────────
print(f"\n{'='*62}")
print("  RF AUC AND FN AT ALL FIXED FEATURE COUNTS (S. aureus)")
print(f"{'='*62}")
print(f"  {'Features':<12} {'AUC':>8} {'FN':>6} {'MCC':>8} {'F1':>8}")
print(f"  {'-'*46}")
# From fixed feature comparison script output
fixed_summary = [
    {"n":10, "AUC":0.8278, "FN":22, "MCC":0.5165, "F1":0.7429},
    {"n":12, "AUC":0.8234, "FN":17, "MCC":0.5271, "F1":0.7651},
    {"n":20, "AUC":0.8307, "FN":20, "MCC":0.5282, "F1":0.7552},
]
for r in fixed_summary:
    star = " ← lowest FN" if r["n"]==12 else (" ← highest AUC" if r["n"]==20 else "")
    print(f"  {r['n']:<12} {r['AUC']:>8} {r['FN']:>6} {r['MCC']:>8} {r['F1']:>8}{star}")

print(f"\n  Note: Youden's J results above (fixed RF {FIXED_FEATS}) "
      f"are the official comparable numbers.")

# ── FIGURE 1 — Confusion Matrix ───────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_title(f"RF — {FIXED_FEATS} Fixed Features\n"
             f"Youden's J threshold={best_thresh}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_xticks([0,1]); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(CLASS_NAMES, fontsize=9,
                                          rotation=90, va="center")
thresh_c = cm.max() / 2
labels   = {(0,0):"TN",(0,1):"FP",
            (1,0):"FN\n(missed\nresistant)",(1,1):"TP"}
for i in range(2):
    for j in range(2):
        color = "white" if cm[i,j] > thresh_c else "black"
        ax.text(j, i, f"{cm[i,j]}\n{labels[(i,j)]}",
                ha="center", va="center", fontsize=11,
                color=color,
                fontweight="bold" if (i,j)==(1,0) else "normal")
plt.tight_layout()
cm_path = os.path.join(FIXED_DIR, "rf_fixed12_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📁 Confusion matrix saved → {cm_path}")

# ── FIGURE 2 — ROC Curve ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fpr_te, tpr_te, _ = roc_curve(y_test, test_prob)
ax.plot(fpr_te, tpr_te, color="#1F3864", lw=2,
        label=f"RF (AUC = {round(test_auc,3)})")
ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC = 0.5)")
ax.scatter([fpr_tr[best_idx]], [tpr_tr[best_idx]],
           color="#C55A11", s=100, zorder=5,
           label=f"Youden's J threshold = {best_thresh}")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"ROC Curve — RF {FIXED_FEATS} Fixed Features\n"
             f"S. aureus Erythromycin Resistance",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(FIXED_DIR, "rf_fixed12_roc_curve.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📁 ROC curve saved → {roc_path}")

# ── Save model and results ─────────────────────────────────────
rf_path     = os.path.join(FIXED_DIR, "rf_fixed12_model.pkl")
thresh_path = os.path.join(FIXED_DIR, "rf_fixed12_threshold.json")
feats_path  = os.path.join(FIXED_DIR, "rf_fixed12_features.json")

joblib.dump(rf, rf_path)
with open(thresh_path, "w") as f:
    json.dump({"threshold":best_thresh,"youdens_j":best_j}, f, indent=4)
with open(feats_path, "w") as f:
    json.dump({"features":top_features}, f, indent=4)

final = {
    "Organism"        : "S. aureus",
    "Model"           : "RF",
    "Features"        : FIXED_FEATS,
    "Feature_type"    : "Fixed (same count as original)",
    "Threshold_method": "Youden_J",
    "Threshold"       : best_thresh,
    "Test_AUC"        : round(test_auc, 4),
    "Train_AUC"       : round(train_auc, 4),
    "AUC_Gap"         : auc_gap,
    "MCC"             : mcc,
    "F1"              : f1,
    "Accuracy"        : acc,
    "Recall"          : rec,
    "Precision"       : prec,
    "FN"              : fn,
    "TP"              : tp,
    "TN"              : tn,
    "FP"              : fp,
}
pd.DataFrame([final]).to_csv(
    os.path.join(FIXED_DIR, "rf_fixed12_results.csv"), index=False
)

print(f"📁 Model saved    → {rf_path}")
print(f"📁 Threshold saved→ {thresh_path}")
print(f"📁 Features saved → {feats_path}")
print(f"📁 Results CSV    → {os.path.join(FIXED_DIR, 'rf_fixed12_results.csv')}")

# ── Final summary ─────────────────────────────────────────────
print(f"\n{'='*62}")
print("  FINAL NUMBERS FOR THESIS UPDATE — S. aureus")
print(f"{'='*62}")
print(f"  Organism    : S. aureus (Erythromycin)")
print(f"  Best model  : RF (at {FIXED_FEATS} fixed features)")
print(f"  Features    : {FIXED_FEATS} (fixed — fair comparison)")
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
print(f"\n📂 All files saved in: {FIXED_DIR}")
print(f"\n✅ Evaluation complete! Share numbers above with supervisor.")