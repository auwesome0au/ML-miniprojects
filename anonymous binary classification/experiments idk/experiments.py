"""Rule-compliant experiments that aim to raise validation ROC-AUC.

Only decision trees, random forests (including ExtraTrees) and gradient
boosting (XGBoost) are used, following the user's ruling on allowed models.
No linear or other non-tree model is fitted anywhere.

Run:        .venv/bin/python experiments.py
Smoke test: .venv/bin/python experiments.py --quick --output-dir <scratch dir>

Reads train.csv and validation.csv through preprocessing.load_data and never
reads test.csv. Every learned step (models and monotone-constraint directions)
is fitted on training rows only. Validation is used once, at the end, to
evaluate the pre-declared finalists and select among them. Importing this
module does not load data, fit models, or write files.

Stages and selection rules are fixed in code; every choice uses CV on train:
1. CV calibration: the three baseline configurations under repeated
   stratified 5-fold CV on train, later compared with their validation AUC.
2. Feature ladder at baseline settings, one change per step: row mean, six row
   summaries, a wider forest split search, monotone constraints.
3. Per-family hyperparameter grids on the best ladder step. Boosting rounds
   are chosen by CV on train; there is no early stopping on validation.
4. Feature re-check at tuned settings: row summaries and monotone
   constraints are tried again on each family's tuned configuration.
5. Seed averaging for forest and boosting, and their equal-weight average of
   probabilities (row by row, no fitted weights).
6. Finalists refitted on all training rows and scored once on validation with
   stratified bootstrap intervals; the highest validation AUC is recorded.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from boosting import MODEL_PARAMS as BOOSTING_BASELINE
from decision_tree import MODEL_PARAMS as TREE_BASELINE
from preprocessing import FEATURE_COLUMNS, load_data
from random_forest import MODEL_PARAMS as FOREST_BASELINE


PROJECT_DIR = Path(__file__).resolve().parent
SEED = 42
THRESHOLD = 0.5
SUMMARY_COLUMNS = (
    "row_mean", "row_median", "row_log1p_mean", "row_std", "row_max", "row_cv",
)
ALL_COLUMNS = (*FEATURE_COLUMNS, *SUMMARY_COLUMNS)
FEATURE_SETS = {
    "raw": (),
    "raw+mean": ("row_mean",),
    "raw+summaries": SUMMARY_COLUMNS,
}
# Boosting settings shared by every stage-3 configuration.
BOOSTING_FIXED = {
    "learning_rate": 0.03, "subsample": 0.8, "reg_alpha": 0.0,
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
    "device": "cpu", "importance_type": "gain",
}
FAMILIES = ("tree", "forest", "boosting")


@dataclass(frozen=True)
class Settings:
    """Run size. QUICK exists only to smoke-test the code path."""
    n_splits: int = 5
    n_repeats: int = 2
    forest_trees: int = 300
    max_rounds: int = 3000
    round_step: int = 50
    seeds: tuple = (42, 43, 44, 45, 46)
    grid_limit: int | None = None
    bootstrap: int = 2000


FULL = Settings()
QUICK = Settings(n_repeats=1, forest_trees=60, max_rounds=200, seeds=(42, 43),
                 grid_limit=3, bootstrap=200)


@dataclass(frozen=True)
class Config:
    """One model configuration. params never hold random_state or n_jobs."""
    family: str                 # "tree", "forest", "boosting" or "ensemble"
    name: str
    stage: str
    params: dict
    feature_set: str = "raw"
    monotone: bool = False
    seeds: tuple = (SEED,)
    checkpoints: tuple = ()     # boosting round counts scored from one fit


@dataclass
class CVResult:
    config: Config
    rounds: int | None          # CV-chosen boosting rounds, else None
    fold_aucs: np.ndarray
    oof: np.ndarray             # shape (n_repeats, n_train_rows)


# ----------------------------------------------------------------------------
# Features. Row summaries use only each row's own values, so the same code can
# run on any split without fitting anything.
# ----------------------------------------------------------------------------

def row_summaries(X: pd.DataFrame) -> np.ndarray:
    """Six statistics of each row's 512 values, in SUMMARY_COLUMNS order."""
    values = X.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    # An all-zero row would make std/mean undefined; such a row gets 0.
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    return np.column_stack([
        mean, np.median(values, axis=1), np.log1p(values).mean(axis=1),
        std, values.max(axis=1), cv,
    ])


def feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """Raw features followed by the row summaries, in ALL_COLUMNS order."""
    raw = X.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    return np.hstack([raw, row_summaries(X)])


def column_index(feature_set: str) -> np.ndarray:
    position = {name: i for i, name in enumerate(ALL_COLUMNS)}
    return np.array([position[n] for n in (*FEATURE_COLUMNS, *FEATURE_SETS[feature_set])])


def column_names(config: Config) -> list[str]:
    return [*FEATURE_COLUMNS, *FEATURE_SETS[config.feature_set]]


def monotone_directions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constraint per column, derived only from the rows a model is fitted on.

    -1 when lower values go with matches, +1 when higher values do, and 0 when
    the column's AUC is within two null standard errors of 0.5, meaning the
    direction is unclear and the column is left unconstrained. This is a rank
    statistic, not a fitted model.
    """
    positives = y == 1
    n1, n0 = int(positives.sum()), int((~positives).sum())
    ranks = rankdata(X, axis=0)
    auc = (ranks[positives].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    null_se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    directions = np.where(auc > 0.5, 1, -1)
    directions[np.abs(auc - 0.5) <= 2 * null_se] = 0
    return directions.astype(int)


# ----------------------------------------------------------------------------
# Models: decision trees, random forests / ExtraTrees, XGBoost only.
# ----------------------------------------------------------------------------

def without_runtime(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in ("random_state", "n_jobs")}


def make_model(config: Config, directions, seed: int, n_jobs: int):
    params = dict(config.params)
    if config.family == "tree":
        return DecisionTreeClassifier(**params, monotonic_cst=directions, random_state=seed)
    if config.family == "forest":
        forest = RandomForestClassifier if params.pop("kind") == "rf" else ExtraTreesClassifier
        return forest(**params, monotonic_cst=directions, random_state=seed, n_jobs=n_jobs)
    if config.family == "boosting":
        constraints = None if directions is None else tuple(int(d) for d in directions)
        return XGBClassifier(**params, monotone_constraints=constraints,
                             random_state=seed, n_jobs=n_jobs)
    raise ValueError(f"Unknown model family: {config.family}")


def fit_and_score(config: Config, X_fit, y_fit, eval_sets: dict, n_jobs: int):
    """Fit every seed of a configuration and average its probabilities.

    Returns ({eval name: {round count or 0: probabilities}}, models, directions).
    Monotone directions come from X_fit and y_fit only.
    """
    directions = monotone_directions(X_fit, y_fit) if config.monotone else None
    points = config.checkpoints or (0,)
    scores = {name: {k: np.zeros(len(X)) for k in points} for name, X in eval_sets.items()}
    models = []
    for seed in config.seeds:
        model = make_model(config, directions, seed, n_jobs).fit(X_fit, y_fit)
        models.append(model)
        for name, X in eval_sets.items():
            for k in points:
                extra = {"iteration_range": (0, k)} if k else {}
                scores[name][k] += model.predict_proba(X, **extra)[:, 1] / len(config.seeds)
    return scores, models, directions


# ----------------------------------------------------------------------------
# Cross-validation on train. Every fit sees training-fold rows only.
# ----------------------------------------------------------------------------

def cv_task(X, y, fit_idx, eval_idx, config: Config):
    cols = column_index(config.feature_set)
    X_fit, X_eval = X[np.ix_(fit_idx, cols)], X[np.ix_(eval_idx, cols)]
    scores, _, _ = fit_and_score(config, X_fit, y[fit_idx], {"eval": X_eval}, n_jobs=1)
    return scores["eval"]


def collect(config: Config, per_fold: list, y, folds, n_splits: int) -> CVResult:
    """Fold AUCs and out-of-fold scores; for boosting, the best round count."""
    best = None
    for k in per_fold[0]:
        aucs = np.array([roc_auc_score(y[e], per_fold[f][k]) for f, (_, e) in enumerate(folds)])
        if best is None or aucs.mean() > best[1].mean():
            best = (k, aucs)
    k, aucs = best
    oof = np.full((len(folds) // n_splits, len(y)), np.nan)
    for f, (_, eval_idx) in enumerate(folds):
        oof[f // n_splits, eval_idx] = per_fold[f][k]
    return CVResult(config, k or None, aucs, oof)


def run_cv(configs, X, y, folds, n_splits, n_jobs=-1) -> list[CVResult]:
    # verbose=5 prints task progress to stderr so long runs can be monitored.
    outputs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(cv_task)(X, y, fit_idx, eval_idx, config)
        for config in configs for fit_idx, eval_idx in folds
    )
    n = len(folds)
    return [collect(c, outputs[i * n:(i + 1) * n], y, folds, n_splits)
            for i, c in enumerate(configs)]


def average_cv(config: Config, members: list[CVResult], y, folds, n_splits) -> CVResult:
    """Equal-weight probability average of out-of-fold predictions."""
    oof = np.mean([m.oof for m in members], axis=0)
    aucs = np.array([roc_auc_score(y[e], oof[f // n_splits, e]) for f, (_, e) in enumerate(folds)])
    return CVResult(config, None, aucs, oof)


def pick(results: list[CVResult]) -> CVResult:
    return max(results, key=lambda r: r.fold_aucs.mean())


def freeze(result: CVResult) -> Config:
    """Replace a boosting round search with the CV-chosen round count."""
    config = result.config
    if not config.checkpoints:
        return config
    return replace(config, params={**config.params, "n_estimators": result.rounds}, checkpoints=())


def limit(grid: list, settings: Settings) -> list:
    """Quick mode keeps evenly spaced grid entries so every code path still runs."""
    if settings.grid_limit is None or len(grid) <= settings.grid_limit:
        return grid
    keep = np.unique(np.linspace(0, len(grid) - 1, settings.grid_limit).round().astype(int))
    return [grid[i] for i in keep]


# ----------------------------------------------------------------------------
# Configuration lists
# ----------------------------------------------------------------------------

def baseline_ladders() -> dict[str, list[Config]]:
    """Stage 1 baselines and stage 2 feature steps, all at baseline settings."""
    forest = {"kind": "rf", **without_runtime(FOREST_BASELINE)}
    ladders = {}
    for family, params in (("tree", without_runtime(TREE_BASELINE)), ("forest", forest),
                           ("boosting", without_runtime(BOOSTING_BASELINE))):
        steps = [
            Config(family, "0 baseline", "1-calibration", params, "raw"),
            Config(family, "1 +row mean", "2-ladder", params, "raw+mean"),
            Config(family, "2 +six row summaries", "2-ladder", params, "raw+summaries"),
        ]
        if family == "forest":
            # Lets the few summary columns be considered at more splits.
            steps.append(Config(family, "3 +max_features 0.2", "2-ladder",
                                {**params, "max_features": 0.2}, "raw+summaries"))
        steps.append(replace(steps[-1], name="4 +monotone constraints", monotone=True))
        ladders[family] = steps
    return ladders


def tree_grid(base: Config) -> list[Config]:
    return [replace(base, stage="3-grid", name=f"depth {d}, leaf {leaf}, ccp {a}",
                    params={"max_depth": d, "min_samples_leaf": leaf, "ccp_alpha": a})
            for d in (3, 4, 5, 6) for leaf in (10, 20, 30) for a in (0.0, 0.001, 0.003)]


def forest_grid(base: Config, trees: int) -> list[Config]:
    grid = []
    for kind in ("rf", "et"):
        for max_features in ("sqrt", 0.2):
            for depth in (3, 6, 10, None):
                for leaf in (1, 10, 30):
                    params = {"kind": kind, "n_estimators": trees, "max_depth": depth,
                              "min_samples_leaf": leaf, "max_features": max_features}
                    if kind == "rf":
                        params["bootstrap"] = True
                    grid.append(replace(base, stage="3-grid", params=params,
                                        name=f"{kind}, max_features {max_features}, "
                                             f"depth {depth}, leaf {leaf}"))
    return grid


def boosting_grid(base: Config, settings: Settings) -> list[Config]:
    checkpoints = tuple(range(settings.round_step, settings.max_rounds + 1, settings.round_step))
    return [replace(base, stage="3-grid", checkpoints=checkpoints,
                    name=f"depth {d}, colsample {c}, {label} regularisation",
                    params={**BOOSTING_FIXED, "n_estimators": settings.max_rounds,
                            "max_depth": d, "colsample_bytree": c,
                            "min_child_weight": mcw, "reg_lambda": lam})
            for d in (1, 2, 3) for c in (0.3, 0.7)
            for label, mcw, lam in (("weak", 1, 1.0), ("strong", 10, 10.0))]


def feature_variants(tuned: Config) -> list[Config]:
    """Stage 4: the other combinations of {raw, raw+summaries} x {monotone off, on}.

    Stage 2 tested these only at baseline settings; tuned models may use them
    differently. Boosting keeps its round search so rounds are re-chosen by CV.
    """
    variants = []
    for feature_set in ("raw", "raw+summaries"):
        for monotone in (False, True):
            if (feature_set, monotone) == (tuned.feature_set, tuned.monotone):
                continue
            label = f"{feature_set}{', monotone' if monotone else ''}"
            variants.append(replace(tuned, stage="4-feature-recheck", feature_set=feature_set,
                                    monotone=monotone, name=f"{tuned.name} [{label}]"))
    return variants


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------

def cv_row(result: CVResult, reference: CVResult, max_rounds: int) -> dict:
    diff = result.fold_aucs - reference.fold_aucs
    config = result.config
    return {
        "stage": config.stage, "family": config.family, "name": config.name,
        "feature_set": config.feature_set, "monotone": config.monotone,
        "seeds": len(config.seeds), "boosting_rounds": result.rounds,
        "rounds_at_grid_edge": result.rounds is not None and result.rounds == max_rounds,
        "cv_auc_mean": result.fold_aucs.mean(), "cv_auc_std": result.fold_aucs.std(ddof=1),
        "reference": f"{reference.config.family}: {reference.config.name}",
        "delta_vs_reference": diff.mean(),
        "share_folds_better_than_reference": (diff > 0).mean(),
        "params": json.dumps(config.params, sort_keys=True, default=str),
    }


def split_metrics(y, probabilities) -> dict:
    predicted = (probabilities >= THRESHOLD).astype(int)
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predicted)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def stratified_bootstrap(y, scores: dict, n_resamples: int, seed: int) -> dict:
    """Resample positives and negatives separately; same indices for every model."""
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = {name: np.empty(n_resamples) for name in scores}
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(positives, positives.size),
                              rng.choice(negatives, negatives.size)])
        for name, p in scores.items():
            samples[name][b] = roc_auc_score(y[idx], p[idx])
    return samples


def describe(config: Config, cv_result: CVResult | None = None) -> dict:
    info = asdict(config)
    info["columns"] = len(column_names(config)) if config.family != "ensemble" else None
    if cv_result is not None:
        info["cv_auc_mean"] = float(cv_result.fold_aucs.mean())
        info["cv_auc_std"] = float(cv_result.fold_aucs.std(ddof=1))
        info["boosting_rounds"] = cv_result.rounds
    return info


def label(config: Config) -> str:
    return f"{config.name} [{config.feature_set}" + (", monotone" if config.monotone else "") + "]"


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

def run_experiments(data_dir: Path, output_dir: Path, settings: Settings = FULL,
                    n_jobs: int = -1) -> dict:
    started = time.perf_counter()
    timings, rows, fold_rows = {}, [], []

    def done(stage: str, since: float) -> None:
        timings[stage] = round(time.perf_counter() - since, 1)
        print(f"  [{stage} finished in {timings[stage]:.0f}s]", flush=True)

    def log(results: list[CVResult], reference: CVResult) -> None:
        for r in results:
            rows.append(cv_row(r, reference, settings.max_rounds))
            fold_rows.extend({"stage": r.config.stage, "family": r.config.family,
                              "name": r.config.name, "fold": f, "auc": auc}
                             for f, auc in enumerate(r.fold_aucs))

    def show(result: CVResult, prefix: str = "") -> None:
        rounds = f", rounds {result.rounds}" if result.rounds else ""
        print(f"  {prefix}{result.config.family:8s} {result.config.name:55s} "
              f"CV AUC {result.fold_aucs.mean():.4f}{rounds}", flush=True)

    data = load_data(data_dir)
    X, y = feature_matrix(data.X_train), data.y_train.to_numpy()
    X_val, y_val = feature_matrix(data.X_val), data.y_val.to_numpy()
    folds = list(RepeatedStratifiedKFold(
        n_splits=settings.n_splits, n_repeats=settings.n_repeats, random_state=SEED,
    ).split(X, y))

    def cv(configs):
        return run_cv(configs, X, y, folds, settings.n_splits, n_jobs)

    def by_family(configs, results):
        grouped = {}
        for config, result in zip(configs, results):
            grouped.setdefault(config.family, []).append(result)
        return grouped

    # Stages 1-2: baselines under CV, then one feature change per step.
    t = time.perf_counter()
    print(f"Stages 1-2: calibration and feature ladder ({len(folds)} CV folds)", flush=True)
    ladder_configs = [c for steps in baseline_ladders().values() for c in steps]
    ladder = by_family(ladder_configs, cv(ladder_configs))
    baseline_cv = {family: results[0] for family, results in ladder.items()}
    for family, results in ladder.items():
        log(results, baseline_cv[family])
        for r in results:
            show(r)
    chosen_ladder = {family: pick(results) for family, results in ladder.items()}
    done("stages 1-2", t)

    # Stage 3: hyperparameter grids on each family's best ladder step.
    t = time.perf_counter()
    grids = {
        "tree": limit(tree_grid(chosen_ladder["tree"].config), settings),
        "forest": limit(forest_grid(chosen_ladder["forest"].config, settings.forest_trees), settings),
        "boosting": limit(boosting_grid(chosen_ladder["boosting"].config, settings), settings),
    }
    grid_configs = [c for grid in grids.values() for c in grid]
    print(f"Stage 3: {len(grid_configs)} grid configurations", flush=True)
    grid_results = by_family(grid_configs, cv(grid_configs))
    for family, results in grid_results.items():
        log(results, baseline_cv[family])
    tuned = {family: pick(results) for family, results in grid_results.items()}
    for result in tuned.values():
        show(result, "best ")
    done("stage 3", t)

    # Stage 4: row summaries and monotone constraints re-checked at tuned settings.
    t = time.perf_counter()
    variant_configs = [v for f in FAMILIES for v in feature_variants(tuned[f].config)]
    print(f"Stage 4: {len(variant_configs)} feature variants of the tuned configurations",
          flush=True)
    variants = by_family(variant_configs, cv(variant_configs))
    final_single = {}
    for family in FAMILIES:
        log(variants[family], baseline_cv[family])
        for r in variants[family]:
            show(r)
        final_single[family] = pick([tuned[family], *variants[family]])
        show(final_single[family], "chosen ")
    done("stage 4", t)

    # Stage 5: seed averaging, then an equal-weight forest + boosting average.
    t = time.perf_counter()
    seeded_configs = [replace(freeze(final_single[f]), stage="5-seeds", seeds=settings.seeds,
                              name=f"{final_single[f].config.name} x{len(settings.seeds)} seeds")
                      for f in ("forest", "boosting")]
    seeded = {c.family: r for c, r in zip(seeded_configs, cv(seeded_configs))}
    ensemble_config = Config(
        "ensemble", "equal-weight forest + boosting probabilities", "5-ensemble",
        {"members": [seeded["forest"].config.name, seeded["boosting"].config.name],
         "weights": [0.5, 0.5]},
    )
    ensemble_cv = average_cv(ensemble_config, [seeded["forest"], seeded["boosting"]],
                             y, folds, settings.n_splits)
    for family in ("forest", "boosting"):
        log([seeded[family]], baseline_cv[family])
        show(seeded[family])
    log([ensemble_cv], baseline_cv["boosting"])
    show(ensemble_cv)
    done("stage 5", t)

    finals = {"tree": freeze(final_single["tree"]),
              "forest": seeded["forest"].config, "boosting": seeded["boosting"].config}
    final_cv = {"tree": final_single["tree"], "forest": seeded["forest"],
                "boosting": seeded["boosting"], "ensemble": ensemble_cv}

    # Stage 6: refit on all training rows; score validation once.
    t = time.perf_counter()
    print("Stage 6: validation evaluation of baselines and finalists", flush=True)
    probabilities, descriptions = {}, {}
    baseline_models = {
        "tree": DecisionTreeClassifier(**TREE_BASELINE),
        "forest": RandomForestClassifier(**FOREST_BASELINE),
        "boosting": XGBClassifier(**BOOSTING_BASELINE),
    }
    for family, model in baseline_models.items():
        # Same inputs and parameters as the baseline scripts.
        model.fit(data.X_train, data.y_train)
        probabilities[f"{family}_baseline"] = (model.predict_proba(data.X_train)[:, 1],
                                               model.predict_proba(data.X_val)[:, 1])
        descriptions[f"{family}_baseline"] = f"{type(model).__name__} baseline, raw features"

    final_details, importances, tree_rules = {}, {}, ""
    for family, config in finals.items():
        cols = column_index(config.feature_set)
        X_fit, X_eval = X[:, cols], X_val[:, cols]
        scores, models, directions = fit_and_score(
            config, X_fit, y, {"train": X_fit, "validation": X_eval}, n_jobs)
        probabilities[f"{family}_enhanced"] = (scores["train"][0], scores["validation"][0])
        descriptions[f"{family}_enhanced"] = label(config)
        names = column_names(config)
        details = describe(config, final_cv[family])
        if directions is not None:
            details["monotone_directions_full_train"] = {
                "decreasing": int((directions == -1).sum()),
                "increasing": int((directions == 1).sum()),
                "unconstrained": int((directions == 0).sum()),
                "summary_columns": {name: int(d) for name, d in zip(names, directions)
                                    if name not in FEATURE_COLUMNS},
            }
        final_details[family] = details
        if family == "tree":
            tree_rules = export_text(models[0], feature_names=names, decimals=4, show_weights=True)
        else:
            importances[family] = pd.Series(
                np.mean([m.feature_importances_ for m in models], axis=0), index=names)
    probabilities["ensemble_enhanced"] = tuple(
        (probabilities["forest_enhanced"][i] + probabilities["boosting_enhanced"][i]) / 2
        for i in (0, 1))
    descriptions["ensemble_enhanced"] = ensemble_config.name
    final_details["ensemble"] = describe(ensemble_config, ensemble_cv)

    validation_scores = {name: p[1] for name, p in probabilities.items()}
    boot = stratified_bootstrap(y_val, validation_scores, settings.bootstrap, SEED)
    reference = {"tree_enhanced": "tree_baseline", "forest_enhanced": "forest_baseline",
                 "boosting_enhanced": "boosting_baseline",
                 "ensemble_enhanced": "boosting_baseline"}
    table = []
    for name, (p_train, p_val) in probabilities.items():
        family, role = name.split("_")
        train_m, val_m = split_metrics(y, p_train), split_metrics(y_val, p_val)
        cv_source = baseline_cv[family] if role == "baseline" else final_cv[family]
        row = {
            "model": name, "family": family, "role": role, "description": descriptions[name],
            "cv_auc_mean": float(cv_source.fold_aucs.mean()),
            "train_auc": train_m["auc"], "validation_auc": val_m["auc"],
            "validation_auc_se": float(boot[name].std(ddof=1)),
            "validation_ci_low": float(np.percentile(boot[name], 2.5)),
            "validation_ci_high": float(np.percentile(boot[name], 97.5)),
            "cv_minus_validation_auc": float(cv_source.fold_aucs.mean()) - val_m["auc"],
            "train_minus_validation_auc": train_m["auc"] - val_m["auc"],
            "train_accuracy": train_m["accuracy"], "validation_accuracy": val_m["accuracy"],
            "validation_confusion_matrix": json.dumps(val_m["confusion_matrix"]),
        }
        for tag, other in (("family_baseline", reference.get(name)),
                           ("xgboost_baseline", "boosting_baseline")):
            if other and other != name:
                diff = boot[name] - boot[other]
                row[f"diff_vs_{tag}"] = val_m["auc"] - roc_auc_score(y_val, validation_scores[other])
                row[f"diff_vs_{tag}_ci_low"] = float(np.percentile(diff, 2.5))
                row[f"diff_vs_{tag}_ci_high"] = float(np.percentile(diff, 97.5))
                row[f"diff_vs_{tag}_share_le_0"] = float((diff <= 0).mean())
        table.append(row)
    results = pd.DataFrame(table)
    enhanced = results[results.role == "enhanced"]
    selected = enhanced.loc[enhanced.validation_auc.idxmax()]
    done("stage 6", t)

    summary = {
        "purpose": "Rule-compliant experiments to raise validation ROC-AUC",
        "allowed_models": "Decision tree, random forest / ExtraTrees, XGBoost only "
                          "(user ruling); equal-weight averaging of forest and boosting",
        "data": {"train_rows": len(y), "validation_rows": len(y_val), "raw_features": 512,
                 "summary_features": list(SUMMARY_COLUMNS), "test_csv_read": False},
        "settings": asdict(settings),
        "cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": settings.n_splits,
               "n_repeats": settings.n_repeats, "random_state": SEED, "folds": len(folds)},
        "selection_rule": "Highest mean fold AUC on train CV at every stage; validation "
                          "is used only for the final evaluation and selection.",
        "chosen_ladder_steps": {f: describe(r.config, r) for f, r in chosen_ladder.items()},
        "chosen_grid_configs": {f: describe(r.config, r) for f, r in tuned.items()},
        "chosen_after_feature_recheck": {f: describe(r.config, r) for f, r in final_single.items()},
        "finals": final_details,
        "calibration": {f: {"cv_auc": float(baseline_cv[f].fold_aucs.mean()),
                            "validation_auc": float(results.loc[results.model == f"{f}_baseline",
                                                                "validation_auc"].iloc[0])}
                        for f in FAMILIES},
        "validation": table,
        "selected_candidate": {
            "model": selected.model, "description": selected.description,
            "validation_auc": float(selected.validation_auc),
            "note": "Highest validation AUC among the four enhanced finalists. Selecting "
                    "on validation makes this score slightly optimistic. No test "
                    "predictions were generated.",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__, "xgboost": xgboost.__version__,
                     "joblib": joblib.__version__},
        "runtime_seconds": {**timings, "total": round(time.perf_counter() - started, 1)},
    }
    for info in summary["calibration"].values():
        info["cv_minus_validation"] = info["cv_auc"] - info["validation_auc"]
    summary_json = json.dumps(summary, indent=2, default=json_default, allow_nan=False) + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_aucs.csv", index=False)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    pd.DataFrame({"row_id": data.val_ids.to_numpy(), "target": y_val,
                  **validation_scores}).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(importances).rename_axis("feature").to_csv(
        output_dir / "final_feature_importances.csv")
    (output_dir / "final_tree_rules.txt").write_text(
        f"Final decision tree: {label(finals['tree'])}\n"
        "Weights are [nonmatches, matches]: training rows in the leaf times its class\n"
        "probabilities. Monotone constraints can clip leaf probabilities, so weights\n"
        "may be fractional.\n\n" + tree_rules,
        encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")

    print("\nValidation results (bootstrap 95% intervals)")
    for row in table:
        print(f"  {row['model']:18s} CV {row['cv_auc_mean']:.4f}  validation "
              f"{row['validation_auc']:.4f} [{row['validation_ci_low']:.4f}, "
              f"{row['validation_ci_high']:.4f}]")
    print(f"Selected by validation AUC: {selected.model} ({selected.validation_auc:.4f})")
    print(f"Artifacts: {output_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default outputs/experiments, or outputs/experiments_quick "
                             "with --quick. Reruns replace the files.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced smoke test of the code path; not for reporting.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    output_dir = args.output_dir or PROJECT_DIR / "outputs" / (
        "experiments_quick" if args.quick else "experiments")
    run_experiments(args.data_dir, output_dir, settings, args.n_jobs)


if __name__ == "__main__":
    main()
