import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

# ===============================
ORGANISM = "ecoli"

# Feature counts per model — tune these if needed
TOP_K = {
    "LR":  80,   # logistic regression
    "RF":  40,   # random forest
    "XGB": 30,   # xgboost
}
# ===============================

BASE_DIR = r"C:\AMR_Prediction_Project"
DATA_DIR = os.path.join(BASE_DIR, "Data", f"results_{ORGANISM}")

print("\n==============================")
print(f"Feature Selection for: {ORGANISM}")
print(f"Target features — LR:{TOP_K['LR']}  RF:{TOP_K['RF']}  XGB:{TOP_K['XGB']}")
print("==============================")

# Load
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_full.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_full.csv"))

X_train = train_df.drop(columns=["label", "genome"], errors="ignore")
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label", "genome"], errors="ignore")
y_test  = test_df["label"]

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# -----------------------------------------------
# STEP 1: ANOVA — strict pre-filter
# -----------------------------------------------
print("\n🔍 Step 1: ANOVA F-test (alpha=0.001)...")

selector = SelectFpr(score_func=f_classif, alpha=0.001)
X_train_anova = selector.fit_transform(X_train, y_train)
X_test_anova  = selector.transform(X_test)
selected_anova = X_train.columns[selector.get_support()].tolist()

print(f"  Features after ANOVA: {len(selected_anova)}")
if len(selected_anova) == 0:
    raise ValueError("ANOVA removed ALL features. Try alpha=0.005.")

max_needed = max(TOP_K.values())
if len(selected_anova) < max_needed:
    print(f"  ⚠️  Only {len(selected_anova)} features passed ANOVA — "
          f"reducing LR top_k to {len(selected_anova)}")
    TOP_K["LR"] = min(TOP_K["LR"], len(selected_anova))
    TOP_K["RF"]  = min(TOP_K["RF"],  len(selected_anova))
    TOP_K["XGB"] = min(TOP_K["XGB"], len(selected_anova))

# -----------------------------------------------
# STEP 2: CV PERMUTATION IMPORTANCE
# -----------------------------------------------
print("\n🌲 Step 2: 5-fold CV permutation importance (leak-free)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importance_accumulator = np.zeros(X_train_anova.shape[1])

for fold_num, (tr_idx, val_idx) in enumerate(cv.split(X_train_anova, y_train), 1):
    print(f"  Fold {fold_num}/5 ...", end=" ", flush=True)

    rf_cv = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=15, max_features="sqrt",
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf_cv.fit(X_train_anova[tr_idx], y_train.iloc[tr_idx])

    perm = permutation_importance(
        rf_cv, X_train_anova[val_idx], y_train.iloc[val_idx],
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importance_accumulator += perm.importances_mean
    print("done")

avg_importances = importance_accumulator / cv.get_n_splits()

importance_df = pd.DataFrame({
    "feature":    selected_anova,
    "importance": avg_importances
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

print(f"\n  Top 10 features:")
print(importance_df.head(10).to_string(index=False))

# -----------------------------------------------
# STEP 3: BUILD THREE FEATURE SETS
# -----------------------------------------------
anova_cols = np.array(selected_anova)

def make_dataset(top_k, suffix):
    k = min(top_k, (importance_df["importance"] > 0).sum())
    features = importance_df.head(k)["feature"].tolist()

    X_tr = pd.DataFrame(X_train_anova, columns=anova_cols)[features]
    X_te = pd.DataFrame(X_test_anova,  columns=anova_cols)[features]

    tr_out = pd.concat([train_df["genome"].reset_index(drop=True),
                        X_tr.reset_index(drop=True),
                        y_train.reset_index(drop=True)], axis=1)
    te_out = pd.concat([test_df["genome"].reset_index(drop=True),
                        X_te.reset_index(drop=True),
                        y_test.reset_index(drop=True)], axis=1)

    tr_path = os.path.join(DATA_DIR, f"train_selected_{suffix}.csv")
    te_path = os.path.join(DATA_DIR, f"test_selected_{suffix}.csv")
    ft_path = os.path.join(DATA_DIR, f"selected_features_{suffix}.csv")

    for p in [tr_path, te_path, ft_path]:
        if os.path.exists(p): os.remove(p)

    tr_out.to_csv(tr_path, index=False)
    te_out.to_csv(te_path, index=False)
    pd.Series(features, name="feature").to_csv(ft_path, index=False)

    print(f"  [{suffix}] {k} features → {tr_path}")
    return k

print("\n📦 Saving model-specific feature sets...")
make_dataset(TOP_K["LR"],  "LR")
make_dataset(TOP_K["RF"],  "RF")
make_dataset(TOP_K["XGB"], "XGB")

# Also save full importance table
imp_path = os.path.join(DATA_DIR, "feature_importances.csv")
if os.path.exists(imp_path): os.remove(imp_path)
importance_df.to_csv(imp_path, index=False)
print(f"  [ALL] Importances → {imp_path}")

print("\n✅ Feature selection complete!")
print(f"  Sample/feature ratios:")
print(f"    LR : 2101/{TOP_K['LR']}  = {2101/TOP_K['LR']:.0f}x")
print(f"    RF : 2101/{TOP_K['RF']}  = {2101/TOP_K['RF']:.0f}x")
print(f"    XGB: 2101/{TOP_K['XGB']} = {2101/TOP_K['XGB']:.0f}x")
