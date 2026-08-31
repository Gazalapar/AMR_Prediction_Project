"""
FIXED FEATURE COMPARISON — Supervisor Validation Experiment
============================================================
Runs LR, RF, XGB on the SAME number of features
to prove that LR wins for E.coli and RF wins for S.aureus
regardless of feature count.

Run this for ORGANISM = "ecoli" first, then "saureus"
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                             f1_score, recall_score, precision_score)
from sklearn.inspection import permutation_importance

# ══════════════════════════════════════════════════════════════
# CHANGE THIS ONLY
# ══════════════════════════════════════════════════════════════
ORGANISM = "Saureus"   # change to "saureus" for second run
# ══════════════════════════════════════════════════════════════

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n" + "="*60)
print(f"FIXED FEATURE COMPARISON — {ORGANISM.upper()}")
print("="*60)

# ── Load best params ──────────────────────────────────────────
with open(os.path.join(DATA_DIR, "best_params.json")) as f:
    params = json.load(f)
lr_p  = params["LR"]
rf_p  = params["RF"]
xgb_p = params["XGB"]

# ── Load ANOVA filtered data (full set before model split) ────
# We use the LR file as base since it has the most features
# All features in RF and XGB are a SUBSET of LR features
print("\n📥 Loading ANOVA-filtered feature set (LR as base)...")
train_lr = pd.read_csv(os.path.join(DATA_DIR, "train_selected_LR.csv"))
test_lr  = pd.read_csv(os.path.join(DATA_DIR, "test_selected_LR.csv"))

X_train_full = train_lr.drop(columns=["label","genome"], errors="ignore")
y_train      = train_lr["label"]
X_test_full  = test_lr.drop(columns=["label","genome"], errors="ignore")
y_test       = test_lr["label"]

print(f"  Full feature set shape: {X_train_full.shape}")
print(f"  Test set shape: {X_test_full.shape}")
print(f"  Class distribution — Train: {y_train.value_counts().to_dict()}")
print(f"  Class distribution — Test:  {y_test.value_counts().to_dict()}")

# ── Rank features by permutation importance ───────────────────
print("\n📊 Ranking features by permutation importance (RF on full set)...")
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
print(f"  Top 5 features: {ranked_features[:5]}")

# ── Define fixed feature counts to test ──────────────────────
if ORGANISM == "ecoli":
    # Original: LR=80, RF=40, XGB=30
    fixed_counts = [30, 40, 80]
    orig_results = {
        "LR" : {"features":80, "AUC":0.8734, "MCC":0.5972,
                "F1":0.7953, "FN":57,  "Gap":0.0515},
        "RF" : {"features":40, "AUC":0.8602, "MCC":0.5412,
                "F1":0.7800, "FN":71,  "Gap":0.0652},
        "XGB": {"features":30, "AUC":0.8752, "MCC":0.5113,
                "F1":0.7850, "FN":63,  "Gap":0.0612},
    }
else:
    # Original: LR=20, RF=12, XGB=10
    fixed_counts = [10, 12, 20]
    orig_results = {
        "LR" : {"features":20, "AUC":0.8187, "MCC":0.4866,
                "F1":0.7470, "FN":21,  "Gap":0.0681},
        "RF" : {"features":12, "AUC":0.8187, "MCC":0.4867,
                "F1":0.7470, "FN":18,  "Gap":0.0682},
        "XGB": {"features":10, "AUC":0.7876, "MCC":0.4213,
                "F1":0.7130, "FN":20,  "Gap":0.0743},
    }

# ── Model definitions ─────────────────────────────────────────
def get_models():
    lr = LogisticRegression(
        C=lr_p["C"],
        penalty=lr_p["penalty"],
        solver=lr_p.get("solver","liblinear"),
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=rf_p.get("n_estimators",300),
        max_depth=rf_p.get("max_depth",5),
        min_samples_leaf=rf_p.get("min_samples_leaf",15),
        max_features=rf_p.get("max_features",0.3),
        random_state=42, n_jobs=-1,
        class_weight="balanced"
    )
    xgb = XGBClassifier(
        n_estimators=xgb_p.get("n_estimators",200),
        max_depth=xgb_p.get("max_depth",3),
        learning_rate=xgb_p.get("learning_rate",0.05),
        subsample=xgb_p.get("subsample",0.6),
        colsample_bytree=xgb_p.get("colsample_bytree",0.4),
        reg_lambda=xgb_p.get("reg_lambda",10),
        reg_alpha=xgb_p.get("reg_alpha",1.0),
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    return lr, rf, xgb

# ── Evaluate function ─────────────────────────────────────────
def evaluate(model, X_tr, X_te, y_tr, y_te, scale=False):
    if scale:
        sc     = StandardScaler()
        X_tr   = sc.fit_transform(X_tr)
        X_te   = sc.transform(X_te)

    model.fit(X_tr, y_tr)

    # threshold tuning — macro F1
    probs    = model.predict_proba(X_tr)[:,1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.65, 0.01):
        preds = (probs >= t).astype(int)
        mf1   = f1_score(y_tr, preds, average="macro")
        rec   = recall_score(y_tr, preds)
        prec  = precision_score(y_tr, preds, zero_division=0)
        if mf1 > best_f1 and rec >= 0.75 and prec >= 0.70:
            best_f1, best_t = mf1, round(t, 2)

    # test evaluation
    te_prob  = model.predict_proba(X_te)[:,1]
    te_pred  = (te_prob >= best_t).astype(int)
    tr_prob  = model.predict_proba(
        StandardScaler().fit_transform(X_tr)
        if scale else X_tr
    )[:,1] if not scale else probs

    auc  = roc_auc_score(y_te, te_prob)
    mcc  = matthews_corrcoef(y_te, te_pred)
    f1   = f1_score(y_te, te_pred)
    fn   = int(((y_te==1) & (te_pred==0)).sum())
    tr_auc = roc_auc_score(y_tr, probs)
    gap  = round(tr_auc - auc, 4)

    return {
        "AUC" :round(auc,4),
        "MCC" :round(mcc,4),
        "F1"  :round(f1,4),
        "FN"  :fn,
        "Gap" :gap,
        "Thr" :best_t
    }

# ── Run experiment ────────────────────────────────────────────
all_results = []

for n in fixed_counts:
    print(f"\n{'─'*50}")
    print(f"Testing all models at FIXED {n} features")
    print(f"{'─'*50}")

    top_feats  = ranked_features[:n]
    X_tr_n     = X_train_full[top_feats]
    X_te_n     = X_test_full[top_feats]

    lr, rf, xgb = get_models()

    print(f"  Running LR  (fixed {n} features)...")
    r_lr  = evaluate(lr,  X_tr_n, X_te_n, y_train, y_test, scale=True)

    print(f"  Running RF  (fixed {n} features)...")
    r_rf  = evaluate(rf,  X_tr_n, X_te_n, y_train, y_test, scale=False)

    print(f"  Running XGB (fixed {n} features)...")
    r_xgb = evaluate(xgb, X_tr_n, X_te_n, y_train, y_test, scale=False)

    for name, r in [("LR",r_lr),("RF",r_rf),("XGB",r_xgb)]:
        r["Model"]    = name
        r["Features"] = n
        r["Type"]     = "Fixed"
        all_results.append(r)
        print(f"    {name}: AUC={r['AUC']}  MCC={r['MCC']}  "
              f"F1={r['F1']}  FN={r['FN']}  Gap={r['Gap']}")

# ── Add original results ──────────────────────────────────────
for name, r in orig_results.items():
    row = r.copy()
    row["Model"] = name
    row["Type"]  = "Original"
    row["Thr"]   = "Youden-J"
    all_results.append(row)

# ── Print final comparison table ─────────────────────────────
df = pd.DataFrame(all_results)
df = df.sort_values(["Features","Model"])

print("\n" + "="*80)
print(f"FINAL COMPARISON TABLE — {ORGANISM.upper()}")
print("="*80)
print(df[["Type","Features","Model","AUC","MCC","F1","FN","Gap"]
        ].to_string(index=False))

# ── Print winner at each feature count ───────────────────────
print("\n" + "="*80)
print("WINNER BY AUC AT EACH FEATURE COUNT")
print("="*80)
fixed_df = df[df["Type"]=="Fixed"]
for n in fixed_counts:
    subset = fixed_df[fixed_df["Features"]==n]
    winner = subset.loc[subset["AUC"].idxmax()]
    print(f"  Fixed {n:3d} features → Winner: {winner['Model']} "
          f"(AUC={winner['AUC']})")

print(f"\n  Original (model-specific) → LR AUC={orig_results['LR']['AUC']} "
      f"| RF AUC={orig_results['RF']['AUC']} "
      f"| XGB AUC={orig_results['XGB']['AUC']}")

# ── Save results ──────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "fixed_feature_comparison.csv")
df.to_csv(out_path, index=False)
print(f"\n✅ Results saved to: {out_path}")
print("\nShare the WINNER BY AUC section above with your supervisor.")