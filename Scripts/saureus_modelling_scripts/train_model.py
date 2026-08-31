import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
ORGANISM = "saureus"
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n==============================")
print(f"Training Models for: {ORGANISM}")
print("==============================")

# -------------------------------
# LOAD BEST PARAMETERS
# -------------------------------
params_path = os.path.join(DATA_DIR, "best_params.json")

if not os.path.exists(params_path):
    raise FileNotFoundError("best_params.json not found — run tuning first")

with open(params_path) as f:
    params = json.load(f)

print("\nBest parameters:")
print(json.dumps(params, indent=4))

lrp  = params["LR"]
rfp  = params["RF"]
xgbp = params["XGB"]

# -------------------------------
# LOAD DATA
# -------------------------------
def load_dataset(name):
    path = os.path.join(DATA_DIR, f"train_selected_{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run feature selection first")

    df = pd.read_csv(path)
    X = df.drop(columns=["label", "genome"], errors="ignore")
    y = df["label"]

    print(f"[{name}] Shape: {X.shape}")
    return X, y

print("\nLoading datasets...")
X_lr,  y_lr  = load_dataset("LR")
X_rf,  y_rf  = load_dataset("RF")
X_xgb, y_xgb = load_dataset("XGB")

# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------
print("\nTraining Logistic Regression...")

scaler = StandardScaler()
X_lr_scaled = scaler.fit_transform(X_lr)

lr = LogisticRegression(
    C=lrp["C"],
    penalty=lrp["penalty"],
    solver=lrp.get("solver", "liblinear"),
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

lr.fit(X_lr_scaled, y_lr)

# -------------------------------
# RANDOM FOREST (FIXED PROPERLY)
# -------------------------------
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=rfp["n_estimators"],
    max_depth=min(rfp["max_depth"], 4),              # safety cap
    min_samples_leaf=max(rfp["min_samples_leaf"], 20),  # safety floor
    max_features=rfp["max_features"],
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_rf, y_rf)

# -------------------------------
# XGBOOST (FIXED PROPERLY)
# -------------------------------
print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=xgbp["n_estimators"],
    max_depth=min(xgbp["max_depth"], 3),             # safety cap
    learning_rate=xgbp["learning_rate"],
    subsample=xgbp["subsample"],
    colsample_bytree=xgbp["colsample_bytree"],
    reg_lambda=max(xgbp["reg_lambda"], 20),          # strong regularization
    reg_alpha=xgbp["reg_alpha"],
    min_child_weight=max(xgbp["min_child_weight"], 20),
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

xgb.fit(X_xgb, y_xgb)

# -------------------------------
# SAVE MODELS
# -------------------------------
print("\nSaving models...")

paths = {
    "lr": os.path.join(DATA_DIR, "lr_model.pkl"),
    "rf": os.path.join(DATA_DIR, "rf_model.pkl"),
    "xgb": os.path.join(DATA_DIR, "xgb_model.pkl"),
    "scaler": os.path.join(DATA_DIR, "scaler.pkl"),
}

for p in paths.values():
    if os.path.exists(p):
        os.remove(p)

joblib.dump(lr, paths["lr"])
joblib.dump(rf, paths["rf"])
joblib.dump(xgb, paths["xgb"])
joblib.dump(scaler, paths["scaler"])

print("\n✅ Training COMPLETE")
for name, path in paths.items():
    print(f"{name.upper()} saved → {path}")