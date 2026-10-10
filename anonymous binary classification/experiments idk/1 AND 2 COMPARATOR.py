import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ============================================================
# COMPARISON:
#   1) CatBoost + Random Forest
#   2) XGBoost + CatBoost + LightGBM
#
# Rules:
#   - train.csv      -> model fitting
#   - validation.csv -> evaluation / comparison only
#   - test.csv       -> final prediction only
#   - no external data
#   - fixed random seeds
# ============================================================

SEED = 20260916
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = r"C:/Users/Safiu/Downloads/week_03_lab/week_03_lab"

TRAIN_PATH = BASE_DIR + "/train.csv"
VALID_PATH = BASE_DIR + "/validation.csv"
TEST_PATH = BASE_DIR + "/test.csv"
SAMPLE_PATH = BASE_DIR + "/sample_submission.csv"

SUBMISSION_CAT_RF = BASE_DIR + "/submission_cat_rf.csv"
SUBMISSION_XGB_CAT_LGBM = BASE_DIR + "/submission_xgb_cat_lgbm.csv"


# ============================================================
# Load data
# ============================================================

print("Loading data...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALID_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)

TARGET = "target"
ID_COL = "row_id"

feature_cols = [
    c for c in train.columns
    if c not in [TARGET, ID_COL]
]

X_train = train[feature_cols]
y_train = train[TARGET]

X_valid = validation[feature_cols]
y_valid = validation[TARGET]

X_test = test[feature_cols]

print(f"Train:      {X_train.shape}")
print(f"Validation: {X_valid.shape}")
print(f"Test:       {X_test.shape}")
print(f"Features:   {len(feature_cols)}")


# ============================================================
# Model factories
# ============================================================

def make_cat():
    return CatBoostClassifier(
        iterations=700,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def make_xgb():
    return XGBClassifier(
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


def make_rf():
    return RandomForestClassifier(
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


# ============================================================
# 1. CatBoost + Random Forest
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE 1: CatBoost + Random Forest")
print("=" * 60)

cat_rf_model = make_cat()
rf_model = make_rf()

print("Training CatBoost...")
cat_rf_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Predicting validation...")

cat_rf_valid = cat_rf_model.predict_proba(X_valid)[:, 1]
rf_valid = rf_model.predict_proba(X_valid)[:, 1]

# CatBoost 70% + RF 30%
cat_rf_valid_blend = (
    0.70 * cat_rf_valid +
    0.30 * rf_valid
)

auc_cat_rf = roc_auc_score(
    y_valid,
    cat_rf_valid_blend
)

print(f"\nCatBoost + RF validation ROC-AUC: {auc_cat_rf:.6f}")


# ============================================================
# 2. XGBoost + CatBoost + LightGBM
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE 2: XGBoost + CatBoost + LightGBM")
print("=" * 60)

xgb_model = make_xgb()
cat_lgbm_model = make_cat()

lgbm_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.05,
    reg_lambda=2.0,
    objective="binary",
    random_state=SEED,
    n_jobs=4,
    verbosity=-1,
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

print("Training CatBoost...")
cat_lgbm_model.fit(X_train, y_train)

print("Training LightGBM...")
lgbm_model.fit(X_train, y_train)

print("Predicting validation...")

xgb_valid = xgb_model.predict_proba(X_valid)[:, 1]
cat_valid = cat_lgbm_model.predict_proba(X_valid)[:, 1]
lgbm_valid = lgbm_model.predict_proba(X_valid)[:, 1]

# XGB 50% + CatBoost 30% + LightGBM 20%
xgb_cat_lgbm_valid_blend = (
    0.50 * xgb_valid +
    0.30 * cat_valid +
    0.20 * lgbm_valid
)

auc_xgb_cat_lgbm = roc_auc_score(
    y_valid,
    xgb_cat_lgbm_valid_blend
)

print(
    f"\nXGBoost + CatBoost + LightGBM validation ROC-AUC: "
    f"{auc_xgb_cat_lgbm:.6f}"
)


# ============================================================
# Comparison
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPARISON")
print("=" * 60)

print(f"CatBoost + Random Forest:       {auc_cat_rf:.6f}")
print(f"XGB + CatBoost + LightGBM:      {auc_xgb_cat_lgbm:.6f}")

difference = auc_cat_rf - auc_xgb_cat_lgbm

print(f"Difference (Cat+RF - 3 models): {difference:+.6f}")


# ============================================================
# Test predictions
# ============================================================

print("\n" + "=" * 60)
print("Creating test predictions")
print("=" * 60)

print("Predicting test with CatBoost + RF...")

cat_rf_test = cat_rf_model.predict_proba(X_test)[:, 1]
rf_test = rf_model.predict_proba(X_test)[:, 1]

final_cat_rf = (
    0.70 * cat_rf_test +
    0.30 * rf_test
)

print("Predicting test with XGB + CatBoost + LightGBM...")

xgb_test = xgb_model.predict_proba(X_test)[:, 1]
cat_test = cat_lgbm_model.predict_proba(X_test)[:, 1]
lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]

final_xgb_cat_lgbm = (
    0.50 * xgb_test +
    0.30 * cat_test +
    0.20 * lgbm_test
)


# ============================================================
# Save submissions
# ============================================================

prediction_col = [
    c for c in sample_submission.columns
    if c != ID_COL
][0]

submission_1 = sample_submission.copy()
submission_1[prediction_col] = final_cat_rf
submission_1.to_csv(SUBMISSION_CAT_RF, index=False)

submission_2 = sample_submission.copy()
submission_2[prediction_col] = final_xgb_cat_lgbm
submission_2.to_csv(SUBMISSION_XGB_CAT_LGBM, index=False)

print("\nSaved:")
print(SUBMISSION_CAT_RF)
print(SUBMISSION_XGB_CAT_LGBM)

print("\nCatBoost + RF submission:")
print(submission_1.head())

print("\nXGB + CatBoost + LightGBM submission:")
print(submission_2.head())

print("\nDone.")
