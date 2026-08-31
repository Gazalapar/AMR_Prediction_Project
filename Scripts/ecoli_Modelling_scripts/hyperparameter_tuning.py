
import pandas as pd
import os
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
ORGANISM = "ecoli"
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n==============================")
print(f"Hyperparameter Tuning for: {ORGANISM}")
print("==============================")

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = "f1"   # balances recall + precision; avoids threshold collapse
best_params = {}

# =========================================================
# 1. LOGISTIC REGRESSION — tuned on LR feature set (80 feat)
# =========================================================
print("\n🔍 Tuning Logistic Regression (80 features)...")

df_lr = pd.read_csv(os.path.join(DATA_DIR, "train_selected_LR.csv"))
X_lr  = df_lr.drop(columns=["label","genome"], errors="ignore")
y_lr  = df_lr["label"]
print(f"  Shape: {X_lr.shape}")

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    param_grid={
        "C":       [0.001, 0.01, 0.1, 1],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear"]
    },
    cv=cv, scoring=SCORING, n_jobs=-1, verbose=1
)
lr_grid.fit(X_lr, y_lr)
best_params["LR"] = lr_grid.best_params_
print(f"  Best LR: {lr_grid.best_params_}  |  CV F1: {lr_grid.best_score_:.4f}")

# =========================================================
# 2. RANDOM FOREST — tuned on RF feature set (40 feat)
# Heavy regularization grid to close the 0.10 AUC gap
# =========================================================
print("\n🌲 Tuning Random Forest (40 features)...")

df_rf = pd.read_csv(os.path.join(DATA_DIR, "train_selected_RF.csv"))
X_rf  = df_rf.drop(columns=["label","genome"], errors="ignore")
y_rf  = df_rf["label"]
print(f"  Shape: {X_rf.shape}")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    param_grid={
        "n_estimators":     [200, 300],
        "max_depth":        [3, 4, 5],         # max 5 (was going up to 8)
        "min_samples_leaf": [15, 25, 40],       # much higher floor (was 10)
        "max_features":     [0.2, 0.3, "sqrt"], # 0.2 = very few features/split
    },
    cv=cv, scoring=SCORING, n_jobs=-1, verbose=1
)
rf_grid.fit(X_rf, y_rf)
best_params["RF"] = rf_grid.best_params_
print(f"  Best RF: {rf_grid.best_params_}  |  CV F1: {rf_grid.best_score_:.4f}")

# =========================================================
# 3. XGBOOST — tuned on XGB feature set (30 feat)
# Very strong regularization to close the 0.10 AUC gap
# =========================================================
print("\n🚀 Tuning XGBoost (30 features)...")

df_xgb = pd.read_csv(os.path.join(DATA_DIR, "train_selected_XGB.csv"))
X_xgb  = df_xgb.drop(columns=["label","genome"], errors="ignore")
y_xgb  = df_xgb["label"]
print(f"  Shape: {X_xgb.shape}")

xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
    param_grid={
        "n_estimators":     [100, 200],
        "max_depth":        [2, 3],             # shallower (was 3,4)
        "learning_rate":    [0.01, 0.05],
        "subsample":        [0.5, 0.6],         # more aggressive subsampling
        "colsample_bytree": [0.4, 0.5],         # fewer features per tree
        "reg_lambda":       [10, 20, 50],        # strong L2 (was 5,10,20)
        "reg_alpha":        [1.0, 5.0],          # strong L1
        "min_child_weight": [15, 25],            # large leaves only (was 10,20)
    },
    cv=cv, scoring=SCORING, n_jobs=-1, verbose=1
)
xgb_grid.fit(X_xgb, y_xgb)
best_params["XGB"] = xgb_grid.best_params_
print(f"  Best XGB: {xgb_grid.best_params_}  |  CV F1: {xgb_grid.best_score_:.4f}")

# Save
output_file = os.path.join(DATA_DIR, "best_params.json")
with open(output_file, "w") as f:
    json.dump(best_params, f, indent=4)

print("\n✅ Tuning complete!")
print(f"📁 {output_file}")
print(json.dumps(best_params, indent=4))
