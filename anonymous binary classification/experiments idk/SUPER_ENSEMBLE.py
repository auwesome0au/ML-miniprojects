"""
SVM ENSEMBLE LAB
================

Focus: many SVMs, different representations + hyperparameters.
No XGBoost / Random Forest.

The goal is to find whether several geometrically different SVMs
can beat the current ~0.943 single SVM.

Assumption:
512 anonymous features = [A(256) | B(256)].

Models:
1. RAW PCA + RBF SVM, several PCA sizes
2. ABS(A-B) + RBF SVM
3. SQUARED(A-B) + RBF SVM
4. ABS + SQUARED(A-B) + RBF SVM
5. A+B + ABS(A-B) + RBF SVM
6. A*B + ABS(A-B) + RBF SVM
7. cosine / correlation / norm / distance aggregate features + RBF SVM
8. RAW + aggregate features + RBF SVM
9. symmetric representation + RBF SVM

For the strongest representations, several C values are tested.

Then:
- individual AUC leaderboard
- pairwise ensemble search
- 3-model ensemble search
- final submission.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

SEED = 20260916

TRAIN_PATH = "train.csv"
VALID_PATH = "validation.csv"
TEST_PATH = "test.csv"

SUBMISSION_PATH = "submission.csv"

TARGET = "target"
ID_COL = "row_id"

# Main PCA grid.
PCA_GRID = [128, 192, 256, 320, 384]

# SVM C grid.
C_GRID = [1.0, 3.0, 10.0, 30.0]

# Gamma:
# "scale" is generally the first thing to try.
# A few explicit multipliers are applied to sklearn's scale gamma.
GAMMA_MULTIPLIERS = [0.5, 1.0, 2.0]

# Ensemble grid.
ENSEMBLE_STEP = 0.05

# Number of best models allowed into 3-way ensemble search.
TOP_FOR_3WAY = 8


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("LOADING")
print("=" * 90)

train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VALID_PATH)
test = pd.read_csv(TEST_PATH)

features = [c for c in train.columns if c not in [TARGET, ID_COL]]

X_train = train[features].values.astype(np.float32)
X_valid = valid[features].values.astype(np.float32)
X_test = test[features].values.astype(np.float32)

y_train = train[TARGET].astype(int).values
y_valid = valid[TARGET].astype(int).values

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)

if X_train.shape[1] % 2 != 0:
    raise ValueError("Feature count is not even; cannot split A/B.")

HALF = X_train.shape[1] // 2

A_train = X_train[:, :HALF]
B_train = X_train[:, HALF:]

A_valid = X_valid[:, :HALF]
B_valid = X_valid[:, HALF:]

A_test = X_test[:, :HALF]
B_test = X_test[:, HALF:]

print(f"A/B split: {HALF} + {HALF}")


# ============================================================
# FEATURE REPRESENTATIONS
# ============================================================

def aggregate(A, B):
    eps = 1e-8

    d = A - B
    ad = np.abs(d)

    an = np.linalg.norm(A, axis=1)
    bn = np.linalg.norm(B, axis=1)

    cosine = np.sum(A * B, axis=1) / (an * bn + eps)

    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)

    corr = np.sum(Ac * Bc, axis=1) / (
        np.linalg.norm(Ac, axis=1) *
        np.linalg.norm(Bc, axis=1) + eps
    )

    return np.column_stack([
        np.linalg.norm(d, axis=1),
        np.sum(ad, axis=1),
        np.mean(ad, axis=1),
        np.median(ad, axis=1),
        np.std(ad, axis=1),
        np.min(ad, axis=1),
        np.max(ad, axis=1),
        np.sum(d * d, axis=1),
        np.mean(d, axis=1),
        np.std(d, axis=1),
        cosine,
        corr,
        an,
        bn,
        np.abs(an - bn),
        np.mean(A, axis=1),
        np.mean(B, axis=1),
        np.std(A, axis=1),
        np.std(B, axis=1),
        np.abs(A.mean(axis=1) - B.mean(axis=1)),
        np.abs(A.std(axis=1) - B.std(axis=1)),
        np.mean(ad > 0.25, axis=1),
        np.mean(ad > 0.5, axis=1),
        np.mean(ad > 1.0, axis=1),
        np.mean(ad > 2.0, axis=1),
    ]).astype(np.float32)


def build_representation(name, A, B):
    d = A - B
    ad = np.abs(d)

    if name == "RAW":
        return np.hstack([A, B]).astype(np.float32)

    if name == "ABS":
        return ad.astype(np.float32)

    if name == "SQUARED":
        return (d * d).astype(np.float32)

    if name == "ABS_SQUARED":
        return np.hstack([
            ad,
            d * d
        ]).astype(np.float32)

    if name == "SUM_ABS":
        return np.hstack([
            A + B,
            ad
        ]).astype(np.float32)

    if name == "PRODUCT_ABS":
        return np.hstack([
            A * B,
            ad
        ]).astype(np.float32)

    if name == "ABS_SUM_PRODUCT":
        return np.hstack([
            ad,
            A + B,
            A * B
        ]).astype(np.float32)

    if name == "AGG":
        return aggregate(A, B)

    if name == "RAW_AGG":
        return np.hstack([
            A,
            B,
            aggregate(A, B)
        ]).astype(np.float32)

    if name == "ABS_AGG":
        return np.hstack([
            ad,
            aggregate(A, B)
        ]).astype(np.float32)

    raise ValueError(f"Unknown representation: {name}")


# ============================================================
# PREBUILD REPRESENTATIONS
# ============================================================

REP_NAMES = [
    "RAW",
    "ABS",
    "SQUARED",
    "ABS_SQUARED",
    "SUM_ABS",
    "PRODUCT_ABS",
    "ABS_SUM_PRODUCT",
    "AGG",
    "RAW_AGG",
    "ABS_AGG",
]

print("\n" + "=" * 90)
print("BUILDING REPRESENTATIONS")
print("=" * 90)

representations = {}

for name in REP_NAMES:
    print(f"Building {name}...")

    representations[name] = {
        "train": build_representation(name, A_train, B_train),
        "valid": build_representation(name, A_valid, B_valid),
        "test": build_representation(name, A_test, B_test),
    }

    print(
        " ",
        representations[name]["train"].shape
    )


# ============================================================
# MODEL STORAGE
# ============================================================

results = {}
valid_predictions = {}
test_predictions = {}


# ============================================================
# SVM RUNNER
# ============================================================

def run_svm(rep_name, pca_n, C, gamma_multiplier):
    key = (
        f"{rep_name}"
        f"__pca{pca_n}"
        f"__C{C:g}"
        f"__g{gamma_multiplier:g}"
    )

    data = representations[rep_name]

    Xtr = data["train"]
    Xva = data["valid"]
    Xte = data["test"]

    print("\n" + "-" * 90)
    print(key)
    print("-" * 90)

    scaler = StandardScaler()

    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    ncomp = min(
        pca_n,
        Xtr_s.shape[0] - 1,
        Xtr_s.shape[1]
    )

    pca = PCA(
        n_components=ncomp,
        random_state=SEED
    )

    Xtr_p = pca.fit_transform(Xtr_s)
    Xva_p = pca.transform(Xva_s)
    Xte_p = pca.transform(Xte_s)

    # sklearn's gamma='scale':
    # 1 / (n_features * variance)
    base_gamma = 1.0 / (
        Xtr_p.shape[1] * Xtr_p.var()
    )

    gamma = base_gamma * gamma_multiplier

    print(
        f"shape={Xtr_p.shape}, "
        f"explained={pca.explained_variance_ratio_.sum():.4f}, "
        f"gamma={gamma:.6g}"
    )

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        random_state=SEED,
        cache_size=2048
    )

    model.fit(Xtr_p, y_train)

    pv = model.predict_proba(Xva_p)[:, 1]
    pt = model.predict_proba(Xte_p)[:, 1]

    auc = roc_auc_score(y_valid, pv)

    results[key] = auc
    valid_predictions[key] = pv
    test_predictions[key] = pt

    print(f"AUC = {auc:.6f}")

    return auc


# ============================================================
# EXPERIMENT PLAN
# ============================================================

# First: broad representation scan with C=10, gamma=scale.
# This answers the most important question:
# WHICH GEOMETRY WORKS?
#
# Then: tune C/PCA/gamma only around promising representations.

print("\n" + "=" * 90)
print("PHASE 1: REPRESENTATION SCAN")
print("=" * 90)

phase1 = [
    ("RAW", 256),
    ("ABS", 256),
    ("SQUARED", 256),
    ("ABS_SQUARED", 256),
    ("SUM_ABS", 256),
    ("PRODUCT_ABS", 256),
    ("ABS_SUM_PRODUCT", 256),
    ("AGG", min(24, 256)),
    ("RAW_AGG", 256),
    ("ABS_AGG", 256),
]

for rep, pca_n in phase1:
    run_svm(
        rep_name=rep,
        pca_n=pca_n,
        C=10.0,
        gamma_multiplier=1.0
    )


# ============================================================
# PHASE 2: TUNE BEST REPRESENTATIONS
# ============================================================

phase1_ranked = sorted(
    [
        (name, auc)
        for name, auc in results.items()
    ],
    key=lambda x: x[1],
    reverse=True
)

print("\n" + "=" * 90)
print("PHASE 1 LEADERBOARD")
print("=" * 90)

for name, auc in phase1_ranked:
    print(f"{auc:.6f}  {name}")


# Select top representations, not top models.
top_reps = []

for model_key, auc in phase1_ranked:
    rep = model_key.split("__")[0]

    if rep not in top_reps:
        top_reps.append(rep)

    if len(top_reps) >= 5:
        break

print("\nRepresentations selected for tuning:")
for rep in top_reps:
    print(" ", rep)


print("\n" + "=" * 90)
print("PHASE 2: HYPERPARAMETER TUNING")
print("=" * 90)

for rep in top_reps:

    for pca_n in PCA_GRID:

        for C in C_GRID:

            for gm in GAMMA_MULTIPLIERS:

                # Avoid duplicating the Phase 1 model.
                if (
                    pca_n == 256
                    and C == 10.0
                    and gm == 1.0
                ):
                    continue

                # Aggregate representations have very few dimensions.
                if rep == "AGG":
                    actual_pca = min(pca_n, 24)
                else:
                    actual_pca = pca_n

                run_svm(
                    rep_name=rep,
                    pca_n=actual_pca,
                    C=C,
                    gamma_multiplier=gm
                )


# ============================================================
# FINAL LEADERBOARD
# ============================================================

print("\n" + "=" * 90)
print("FINAL SVM LEADERBOARD")
print("=" * 90)

leaderboard = (
    pd.DataFrame([
        {
            "model": name,
            "auc": auc
        }
        for name, auc in results.items()
    ])
    .sort_values("auc", ascending=False)
    .reset_index(drop=True)
)

print(
    leaderboard.head(30).to_string(index=False)
)


# ============================================================
# ENSEMBLE SEARCH
# ============================================================

print("\n" + "=" * 90)
print("SVM ENSEMBLE SEARCH")
print("=" * 90)

# Use top models, but avoid having 10 almost identical SVMs
# dominate the ensemble.
#
# We select at most the best 2 models per representation.

selected = []

for rep in REP_NAMES:

    candidates = [
        (name, auc)
        for name, auc in results.items()
        if name.startswith(rep + "__")
    ]

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for name, auc in candidates[:2]:
        selected.append(name)


selected.sort(
    key=lambda x: results[x],
    reverse=True
)

selected = selected[:12]

print("Models entering ensemble search:")

for name in selected:
    print(f"{results[name]:.6f}  {name}")


# ------------------------------------------------------------
# Single / pair / triple search
# ------------------------------------------------------------

best_auc = -np.inf
best_names = None
best_weights = None
best_valid = None
best_test = None


# Singles
for name in selected:

    auc = results[name]

    if auc > best_auc:
        best_auc = auc
        best_names = [name]
        best_weights = {name: 1.0}
        best_valid = valid_predictions[name].copy()
        best_test = test_predictions[name].copy()


# Pairs
for i in range(len(selected)):

    for j in range(i + 1, len(selected)):

        n1 = selected[i]
        n2 = selected[j]

        for w in np.arange(
            0.0,
            1.0001,
            ENSEMBLE_STEP
        ):

            pv = (
                w * valid_predictions[n1]
                + (1.0 - w) * valid_predictions[n2]
            )

            auc = roc_auc_score(y_valid, pv)

            if auc > best_auc:

                best_auc = auc
                best_names = [n1, n2]

                best_weights = {
                    n1: float(w),
                    n2: float(1.0 - w)
                }

                best_valid = pv.copy()

                best_test = (
                    w * test_predictions[n1]
                    + (1.0 - w) * test_predictions[n2]
                )


# Triples
top3 = selected[:TOP_FOR_3WAY]

for i in range(len(top3)):

    for j in range(i + 1, len(top3)):

        for k in range(j + 1, len(top3)):

            n1 = top3[i]
            n2 = top3[j]
            n3 = top3[k]

            for w1 in np.arange(
                0.0,
                1.0001,
                ENSEMBLE_STEP
            ):

                for w2 in np.arange(
                    0.0,
                    1.0001 - w1 + 1e-9,
                    ENSEMBLE_STEP
                ):

                    w3 = 1.0 - w1 - w2

                    if w3 < -1e-9:
                        continue

                    pv = (
                        w1 * valid_predictions[n1]
                        + w2 * valid_predictions[n2]
                        + w3 * valid_predictions[n3]
                    )

                    auc = roc_auc_score(y_valid, pv)

                    if auc > best_auc:

                        best_auc = auc

                        best_names = [
                            n1,
                            n2,
                            n3
                        ]

                        best_weights = {
                            n1: float(w1),
                            n2: float(w2),
                            n3: float(w3)
                        }

                        best_valid = pv.copy()

                        best_test = (
                            w1 * test_predictions[n1]
                            + w2 * test_predictions[n2]
                            + w3 * test_predictions[n3]
                        )


# ============================================================
# BEST ENSEMBLE
# ============================================================

print("\n" + "=" * 90)
print("BEST SVM ENSEMBLE")
print("=" * 90)

print(f"AUC: {best_auc:.6f}")

for name, weight in best_weights.items():
    print(f"{weight:.2f}  {name}")


# ============================================================
# SUBMISSION
# ============================================================

print("\n" + "=" * 90)
print("SUBMISSION")
print("=" * 90)

pred = np.asarray(best_test, dtype=np.float64)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET: pred
})

assert len(submission) == len(test)
assert submission[ID_COL].equals(test[ID_COL])
assert submission[TARGET].notna().all()
assert submission[TARGET].between(0, 1).all()

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)

print(f"Saved: {os.path.abspath(SUBMISSION_PATH)}")

print("\nSubmission:")
print(submission.head(10))

print("\nDONE.")
