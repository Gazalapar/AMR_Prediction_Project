import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

# ===============================
ORGANISM = "saureus"
TOP_K = {"LR": 20, "RF": 12, "XGB": 10}
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print(f"\nFeature Selection for: {ORGANISM}")

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_full.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_full.csv"))

X_train = train_df.drop(columns=["label", "genome"], errors="ignore")
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label", "genome"], errors="ignore")
y_test  = test_df["label"]

# -------------------------------
# STEP 1: ANOVA
# -------------------------------
print("\nStep 1: ANOVA (alpha=0.0001)")
selector = SelectFpr(score_func=f_classif, alpha=0.0001)

X_train_anova = selector.fit_transform(X_train, y_train)
X_test_anova  = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()].tolist()
print("Features after ANOVA:", len(selected_features))

if len(selected_features) == 0:
    raise ValueError("No features selected — increase alpha")

# -------------------------------
# STEP 2: CV Permutation Importance
# -------------------------------
print("\nStep 2: CV Permutation Importance")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importance = np.zeros(X_train_anova.shape[1])

for i, (tr, val) in enumerate(cv.split(X_train_anova, y_train)):
    print(f"Fold {i+1}/5")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train_anova[tr], y_train.iloc[tr])

    perm = permutation_importance(
        model,
        X_train_anova[val],
        y_train.iloc[val],
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importance += perm.importances_mean

importance /= 5

importance_df = pd.DataFrame({
    "feature": selected_features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

# -------------------------------
# STEP 3: SAVE FEATURE SETS
# -------------------------------
def save_dataset(k, name):
    features = importance_df.head(k)["feature"]

    X_tr = pd.DataFrame(X_train_anova, columns=selected_features)[features]
    X_te = pd.DataFrame(X_test_anova,  columns=selected_features)[features]

    train_out = pd.concat([train_df["genome"], X_tr, y_train], axis=1)
    test_out  = pd.concat([test_df["genome"],  X_te, y_test], axis=1)

    train_out.to_csv(os.path.join(DATA_DIR, f"train_selected_{name}.csv"), index=False)
    test_out.to_csv(os.path.join(DATA_DIR, f"test_selected_{name}.csv"), index=False)

    print(f"{name}: {k} features saved")

save_dataset(TOP_K["LR"], "LR")
save_dataset(TOP_K["RF"], "RF")
save_dataset(TOP_K["XGB"], "XGB")

importance_df.to_csv(os.path.join(DATA_DIR, "feature_importances.csv"), index=False)

print("\nFeature selection DONE")