import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

# ===============================
ORGANISM = "ecoli"
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n==============================")
print(f"Training Final Models for: {ORGANISM}")
print("==============================")

# Load tuned params
with open(os.path.join(DATA_DIR, "best_params.json")) as f:
    params = json.load(f)
print("\n📋 Tuned params:")
print(json.dumps(params, indent=4))

lr_p  = params["LR"]
rf_p  = params["RF"]
xgb_p = params["XGB"]

# -----------------------------------------------
# LOAD MODEL-SPECIFIC FEATURE SETS
# -----------------------------------------------
def load_xy(suffix):
    df = pd.read_csv(os.path.join(DATA_DIR, f"train_selected_{suffix}.csv"))
    X  = df.drop(columns=["label","genome"], errors="ignore")
    y  = df["label"]
    print(f"  [{suffix}] shape: {X.shape}")
    return X, y

print("\n📥 Loading feature sets:")
X_lr,  y_lr  = load_xy("LR")
X_rf,  y_rf  = load_xy("RF")
X_xgb, y_xgb = load_xy("XGB")

# -----------------------------------------------
# LOGISTIC REGRESSION
# -----------------------------------------------
scaler   = StandardScaler()
X_lr_sc  = scaler.fit_transform(X_lr)

lr = LogisticRegression(
    C=lr_p["C"],
    penalty=lr_p["penalty"],
    solver=lr_p.get("solver", "liblinear"),
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

# -----------------------------------------------
# RANDOM FOREST — hard floors on regularization
# -----------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,              # hard cap at 4 — not from tuner
    min_samples_leaf=25,      # up from 15
    max_features=0.2,         # only 20% features per split
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# -----------------------------------------------
# XGBOOST — hard floors on regularization
# -----------------------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=3,              # hard cap at 3
    learning_rate=0.01,       # slower
    subsample=0.5,            # down from 0.6
    colsample_bytree=0.4,     # down from 0.5
    reg_lambda=30,            # up from 10-20
    reg_alpha=2.0,            # up from 1.0
    min_child_weight=25,      # up from 15
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)
# -----------------------------------------------
# TRAIN
# -----------------------------------------------
print("\n🚀 Training Logistic Regression...")
lr.fit(X_lr_sc, y_lr)

print("🌲 Training Random Forest...")
rf.fit(X_rf, y_rf)

print("🚀 Training XGBoost...")
xgb.fit(X_xgb, y_xgb)

# -----------------------------------------------
# THRESHOLD TUNING — target: both classes F1 >= threshold
# v3 change: uses macro F1 as objective (not recall alone)
# to stop precision from collapsing like it did in v2 (LR precision=0.68)
# -----------------------------------------------
print("\n🎯 Threshold tuning (optimise macro F1)...")

thresholds_out = {}

def find_best_threshold(name, model, X_input, y_true):
    probs     = model.predict_proba(X_input)[:, 1]
    best_f1   = 0.0
    best_t    = 0.5

    for t in np.arange(0.30, 0.65, 0.01):
        preds  = (probs >= t).astype(int)
        macro  = f1_score(y_true, preds, average="macro")
        res_r  = recall_score(y_true, preds)
        res_p  = precision_score(y_true, preds, zero_division=0)
        # Only accept threshold if resistant recall >= 0.75
        # AND resistant precision >= 0.70 (no precision collapse)
        if macro > best_f1 and res_r >= 0.75 and res_p >= 0.70:
            best_f1 = macro
            best_t  = round(t, 2)

    preds_best = (probs >= best_t).astype(int)
    print(f"  {name:25s}  threshold={best_t:.2f}  "
          f"macro-F1={f1_score(y_true,preds_best,average='macro'):.3f}  "
          f"recall={recall_score(y_true,preds_best):.3f}  "
          f"precision={precision_score(y_true,preds_best):.3f}")
    return best_t

thresholds_out["LR"]  = find_best_threshold("Logistic Regression", lr,  X_lr_sc, y_lr)
thresholds_out["RF"]  = find_best_threshold("Random Forest",        rf,  X_rf,    y_rf)
thresholds_out["XGB"] = find_best_threshold("XGBoost",              xgb, X_xgb,   y_xgb)

# -----------------------------------------------
# SAVE
# -----------------------------------------------
paths = {
    "lr":     os.path.join(DATA_DIR, "lr_model.pkl"),
    "rf":     os.path.join(DATA_DIR, "rf_model.pkl"),
    "xgb":    os.path.join(DATA_DIR, "xgb_model.pkl"),
    "scaler": os.path.join(DATA_DIR, "scaler.pkl"),
    "thresh": os.path.join(DATA_DIR, "thresholds.json"),
}
for p in paths.values():
    if os.path.exists(p): os.remove(p)

joblib.dump(lr,     paths["lr"])
joblib.dump(rf,     paths["rf"])
joblib.dump(xgb,    paths["xgb"])
joblib.dump(scaler, paths["scaler"])
with open(paths["thresh"], "w") as f:
    json.dump(thresholds_out, f, indent=4)

print("\n✅ All models + thresholds saved!")
for k, p in paths.items():
    print(f"  📁 {p}")
