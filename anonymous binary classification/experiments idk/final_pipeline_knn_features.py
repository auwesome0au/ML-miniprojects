import os
import random
from functools import partial
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def find_data_dir():
    path = Path(__file__).resolve().parent
    files = ["train.csv", "validation.csv", "test.csv", "sample_submission.csv"]
    if all((path / f).exists() for f in files):
        return path
    raise FileNotFoundError("Competition CSV files not found")


DATA_DIR = find_data_dir()
FEATURES = [f"f{i:04d}" for i in range(512)]


def load(name, labeled=False):
    df = pd.read_csv(DATA_DIR / name, dtype={"row_id": "string"})
    columns = {"row_id", *FEATURES}
    if labeled:
        columns.add("target")
    assert set(df.columns) == columns
    return df


train = load("train.csv", True)
valid = load("validation.csv", True)
test = load("test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype={"row_id": "string"})

X_train = train[FEATURES].to_numpy(np.float64)
X_valid = valid[FEATURES].to_numpy(np.float64)
X_test = test[FEATURES].to_numpy(np.float64)
y_train = train["target"].to_numpy()
y_valid = valid["target"].to_numpy()


class KNNFeatureAugmenter(TransformerMixin, BaseEstimator):
    def __init__(self, ks=(1, 3, 5, 10, 20), metric="euclidean"):
        self.ks = tuple(ks)
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.max_k_ = max(self.ks)
        self.knn_ = NearestNeighbors(
            n_neighbors=self.max_k_ + 1,
            metric=self.metric,
            n_jobs=1,
        )
        self.knn_.fit(X)
        return self

    def _features(self, distances):
        selected = np.column_stack([distances[:, k - 1] for k in self.ks])
        return np.column_stack([
            selected,
            selected[:, :3].mean(axis=1),
            selected.mean(axis=1),
            1.0 / (selected[:, -1] + 1e-12),
            selected[:, 0] / (selected[:, -1] + 1e-12),
        ])

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        distances = self.knn_.kneighbors(
            X, n_neighbors=self.max_k_ + 1, return_distance=True
        )[0]
        return self._features(distances[:, 1:])

    def transform(self, X):
        distances = self.knn_.kneighbors(
            X, n_neighbors=self.max_k_, return_distance=True
        )[0]
        return self._features(distances)


KNN_KS = (1, 3, 5, 10, 20)


class ScaledGammaSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, multiplier=1.0, C=10.0):
        self.multiplier = multiplier
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        gamma = self.multiplier / (X.shape[1] * X.var())
        self.model_ = SVC(
            kernel=partial(rbf_kernel, gamma=gamma),
            C=self.C,
            tol=1e-4,
            cache_size=512,
            random_state=SEED,
        )
        with threadpool_limits(1):
            self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def decision_function(self, X):
        return self.model_.decision_function(np.asarray(X, dtype=np.float64))


class PolynomialSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=100.0, coef0=0.0):
        self.C = C
        self.coef0 = coef0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        gamma = 1.0 / (X.shape[1] * X.var())
        self.model_ = SVC(
            kernel=partial(polynomial_kernel, degree=2, gamma=gamma, coef0=self.coef0),
            C=self.C,
            tol=1e-4,
            cache_size=512,
            random_state=SEED,
        )
        with threadpool_limits(1):
            self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def decision_function(self, X):
        return self.model_.decision_function(np.asarray(X, dtype=np.float64))


def calibrated(model):
    return CalibratedClassifierCV(
        model,
        method="sigmoid",
        ensemble=False,
        n_jobs=1,
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    )


class SpreadScaler(TransformerMixin, BaseEstimator):
    def __init__(self, alpha=-0.5):
        self.alpha = alpha

    def fit(self, X, y=None):
        X = np.sqrt(np.asarray(X, dtype=np.float64))
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0) ** self.alpha
        return self

    def transform(self, X):
        X = np.sqrt(np.asarray(X, dtype=np.float64))
        return (X - self.mean_) / self.scale_


def svm_member(multiplier, pca_components=None):
    steps = [
        ("spread", SpreadScaler()),
    ]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    steps += [
        ("knn", KNNFeatureAugmenter(KNN_KS)),
        ("model", ScaledGammaSVC(multiplier=multiplier, C=10.0)),
    ]
    return calibrated(Pipeline(steps))


def polynomial_member():
    return calibrated(Pipeline([
        ("sqrt", FunctionTransformer(np.sqrt, validate=True)),
        ("scaler", StandardScaler()),
        ("model", PolynomialSVC()),
    ]))


def row_summaries(X):
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    return np.column_stack([
        mean,
        np.median(X, axis=1),
        np.log1p(X).mean(axis=1),
        std,
        X.max(axis=1),
        cv,
    ])


class LightGBMMember:
    params = dict(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        reg_lambda=10.0,
        reg_alpha=0.1,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        max_bin=127,
        objective="binary",
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=64, svd_solver="full").fit(scaled)
        self.knn_ = KNNFeatureAugmenter(KNN_KS)
        knn = self.knn_.fit_transform(scaled)
        features = np.hstack([X, row_summaries(X), self.pca_.transform(scaled), knn])
        self.model_ = LGBMClassifier(**self.params, random_state=SEED, n_jobs=1)
        self.model_.fit(X=features, y=y)
        return self

    def predict_proba(self, X):
        scaled = self.scaler_.transform(X)
        features = np.hstack([
            X,
            row_summaries(X),
            self.pca_.transform(scaled),
            self.knn_.transform(scaled),
        ])
        return self.model_.predict_proba(features)


MEMBERS = {
    "Full SVM": lambda: svm_member(1.25),
    "PCA SVM": lambda: svm_member(1.5, 255),
    "LightGBM": LightGBMMember,
    "Polynomial SVM": polynomial_member,
}

WEIGHTS = np.array([0.405, 0.405, 0.09, 0.10])


def rank_position(reference, values):
    ordered = np.sort(reference)
    levels = (np.arange(len(ordered)) + 0.5) / len(ordered)
    return np.interp(values, ordered, levels)


def rank_pool(probabilities, reference):
    ranks = np.column_stack([
        rank_position(reference[:, i], probabilities[:, i])
        for i in range(probabilities.shape[1])
    ])
    return ranks @ WEIGHTS


folds = list(
    RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    .split(X_train, y_train)
)[:5]

oof = np.zeros((len(X_train), len(MEMBERS)))

for train_idx, valid_idx in folds:
    for i, factory in enumerate(MEMBERS.values()):
        model = factory()
        model.fit(X_train[train_idx], y_train[train_idx])
        oof[valid_idx, i] = model.predict_proba(X_train[valid_idx])[:, 1]

models = {}
valid_pred = []
test_pred = []

for name, factory in MEMBERS.items():
    model = factory()
    model.fit(X_train, y_train)
    models[name] = model
    valid_pred.append(model.predict_proba(X_valid)[:, 1])
    test_pred.append(model.predict_proba(X_test)[:, 1])

valid_pred = np.column_stack(valid_pred)
test_pred = np.column_stack(test_pred)

final_valid = rank_pool(valid_pred, oof)
final_test = rank_pool(test_pred, oof)

results = []

for i, name in enumerate(MEMBERS):
    auc = roc_auc_score(y_valid, valid_pred[:, i])
    results.append(f"{name}: {auc:.6f}")
    print(f"{name}: {auc:.6f}")

final_auc = roc_auc_score(y_valid, final_valid)
results.append(f"Final ROC-AUC: {final_auc:.6f}")
print(f"Final ROC-AUC: {final_auc:.6f}")

submission = pd.DataFrame({"row_id": test["row_id"], "target": final_test})
submission = submission.set_index("row_id").loc[sample["row_id"]].reset_index()
submission.to_csv(DATA_DIR / "final_submission.csv", index=False)
submission.to_csv(DATA_DIR / "submission_kaggle.csv", index=False)

with open(DATA_DIR / "roc_auc.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
