import os
import random
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# XGBoost + Random Forest ensemble
# Competition-safe:
# - train.csv: model fitting only
# - validation.csv: loaded only for schema compatibility, not used for fitting
# - test.csv: final prediction only
# - no external data
# - fixed seeds
# ============================================================

SEED = 20260916
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
train = pd.read_csv("C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/train.csv")
validation = pd.read_csv("C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/validation.csv")
test = pd.read_csv("C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/test.csv")
sample_submission = pd.read_csv("C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/sample_submission.csv")

TARGET = "target"
ID_COL = "row_id"

# Use exactly the feature columns from train.
feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]

X_train = train[feature_cols]
y_train = train[TARGET]

X_test = test[feature_cols]

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")
print(f"Features: {len(feature_cols)}")

# ------------------------------------------------------------
# Model 1: XGBoost
# ------------------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=900,
    max_depth=3,
    learning_rate=0.025,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=SEED,
)

# ------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    n_jobs=4,
    random_state=SEED,
)

print("\nTraining XGBoost...")
xgb_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
print("\nPredicting test set...")

xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict_proba(X_test)[:, 1]

# XGBoost gets the larger weight because it is the primary model.
# Random Forest adds a more diverse tree-based signal.
final_pred = 0.70 * xgb_pred + 0.30 * rf_pred

# ------------------------------------------------------------
# Submission
# ------------------------------------------------------------
submission = sample_submission.copy()

# Preserve the exact sample_submission column structure.
prediction_col = [c for c in submission.columns if c != ID_COL][0]

submission[prediction_col] = final_pred

submission.to_csv("C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/submission.csv", index=False)

print(f"\nSaved submission to: {"C:/Users/Safiu/Downloads/week_03_lab/week_03_lab/submission.csv"}")
print(submission.head())
print("\nDone.")
