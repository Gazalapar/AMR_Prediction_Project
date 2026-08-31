import pandas as pd
import os
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
ORGANISM = "saureus"
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_params = {}

# ---------------- LR ----------------
df = pd.read_csv(os.path.join(DATA_DIR, "train_selected_LR.csv"))
X = df.drop(columns=["label","genome"])
y = df["label"]

lr = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    {
        "C":[0.001,0.01,0.1,1],
        "penalty":["l1","l2"],
        "solver":["liblinear"]
    },
    cv=cv, scoring="f1", n_jobs=-1
)
lr.fit(X,y)
best_params["LR"] = lr.best_params_

# ---------------- RF ----------------
df = pd.read_csv(os.path.join(DATA_DIR, "train_selected_RF.csv"))
X = df.drop(columns=["label","genome"])
y = df["label"]

rf = GridSearchCV(
    RandomForestClassifier(class_weight="balanced"),
    {
        "n_estimators":[200,300],
        "max_depth":[3,4],
        "min_samples_leaf":[20,30,40],
        "max_features":["sqrt",0.2]
    },
    cv=cv, scoring="f1", n_jobs=-1
)
rf.fit(X,y)
best_params["RF"] = rf.best_params_

# ---------------- XGB ----------------
df = pd.read_csv(os.path.join(DATA_DIR, "train_selected_XGB.csv"))
X = df.drop(columns=["label","genome"])
y = df["label"]

xgb = GridSearchCV(
    XGBClassifier(eval_metric="logloss", use_label_encoder=False),
    {
        "n_estimators":[100,200],
        "max_depth":[2,3],
        "learning_rate":[0.01,0.05],
        "subsample":[0.5,0.6],
        "colsample_bytree":[0.4,0.5],
        "reg_lambda":[10,20,50],
        "reg_alpha":[1,5],
        "min_child_weight":[20,30]
    },
    cv=cv, scoring="f1", n_jobs=-1
)
xgb.fit(X,y)
best_params["XGB"] = xgb.best_params_

# Save
with open(os.path.join(DATA_DIR,"best_params.json"),"w") as f:
    json.dump(best_params,f,indent=4)

print("Tuning DONE")
print(best_params)