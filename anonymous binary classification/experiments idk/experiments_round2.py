"""Round 2 of the tree-only experiments: work-plan phases 1-4 (approved 2026-09-19).

Run after experiments.py:  .venv/bin/python experiments_round2.py
Smoke test: .venv/bin/python experiments_round2.py --quick --round1-dir <dir> --output-dir <dir>

Only decision trees, random forests / ExtraTrees and gradient boosting
(XGBoost and scikit-learn HistGradientBoosting) are fitted, following the
user's ruling on allowed models. Reads train.csv and validation.csv through
preprocessing.load_data, plus the round-1 summary.json and saved validation
predictions written by experiments.py. Never reads test.csv. Every learned
step (models, monotone directions, train-quantile features) is fitted on
training rows only. Importing this module does not load data, fit models, or
write files.

Uses the same folds as experiments.py: RepeatedStratifiedKFold(5, 2, seed 42).
Screening uses the first 5 folds; the best few are confirmed on all 10. Every
choice is made by the mean fold AUC of CV on train.

Phase 1  Additive boosting: each tree may use only one feature, so the model
         is a sum of per-feature effects. XGBoost with one interaction group
         per feature, and HistGradientBoosting with no interactions, each with
         and without monotone constraints. Rounds are chosen by CV.
Phase 2  Extra unsupervised aggregates re-tested on the best models: row
         quantiles and a trimmed mean (row-wise), plus the mean train-ECDF
         value and the shares of values below / above each feature's train
         quartiles (fitted on the fitting rows only; no labels).
Phase 3  Forest refinement around the best forest: max_features, leaf size,
         row subsampling (random forest only) and more trees.
Phase 4  Seed averaging and equal-weight probability averages of the best
         forest, standard boosting and additive boosting.
Validation: at most three finalists (best forest, best boosting, best
average), plus the best decision tree for the required comparison only.
Paired bootstrap against the XGBoost baseline and the round-1 candidate uses
round-1's saved validation predictions. The highest validation AUC among the
enhanced finalists of both rounds is written to selected_model.json.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from joblib import Parallel, delayed
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from experiments import (SEED, SUMMARY_COLUMNS, average_cv, collect, json_default, limit,
                         monotone_directions, pick, split_metrics, stratified_bootstrap)
from preprocessing import FEATURE_COLUMNS, load_data


PROJECT_DIR = Path(__file__).resolve().parent
ROWWISE_EXTRA = ("row_q10", "row_q25", "row_q75", "row_q90", "row_trimmed_mean")
FITTED_EXTRA = ("mean_train_ecdf", "share_below_train_q25", "share_above_train_q75")
FEATURE_SETS = {
    "raw": (),
    "raw+mean": ("row_mean",),
    "raw+summaries": SUMMARY_COLUMNS,
    "raw+aggregates": (*SUMMARY_COLUMNS, *ROWWISE_EXTRA, *FITTED_EXTRA),
    # Added for round 3: each feature divided by its row mean replaces the raw
    # values; the six summaries (including the row mean) keep the scale.
    "profile+summaries": SUMMARY_COLUMNS,
}
PROFILE_SETS = ("profile+summaries",)
XGB_FIXED = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
             "device": "cpu", "importance_type": "gain", "reg_alpha": 0.0}


@dataclass(frozen=True)
class Settings:
    """Run size. QUICK exists only to smoke-test the code path."""
    n_splits: int = 5
    n_repeats: int = 2
    max_rounds: int = 2000          # additive boosting, learning rate 0.1
    round_step: int = 50
    additive_lr: float = 0.1
    # Round-1 XGBoost is re-searched up to 2x its rounds. 6000 because round 1's
    # boosting choice sat at its 3000-round edge in train CV.
    standard_round_cap: int = 6000
    forest_trees: int = 500
    seeds: tuple = (42, 43, 44, 45, 46)
    top_k: int = 2
    grid_limit: int | None = None
    bootstrap: int = 2000


FULL = Settings()
QUICK = Settings(n_repeats=1, max_rounds=200, standard_round_cap=200, forest_trees=60,
                 seeds=(42, 43), top_k=1, grid_limit=3, bootstrap=200)


@dataclass(frozen=True)
class Spec:
    """One model configuration. params never hold random_state or n_jobs."""
    family: str                 # "tree", "forest", "xgb", "hgb" or "ensemble"
    name: str
    phase: str
    params: dict
    feature_set: str = "raw"
    monotone: bool = False
    additive: bool = False      # xgb: one feature per tree; hgb: no interactions
    seeds: tuple = (SEED,)
    checkpoints: tuple = ()     # boosting round counts scored from one fit


# ----------------------------------------------------------------------------
# Features. Row-wise statistics use only each row's own values. The three
# fitted aggregates use the distribution of the fitting rows (no labels).
# ----------------------------------------------------------------------------

def rowwise_features(values: np.ndarray) -> dict:
    mean, std = values.mean(axis=1), values.std(axis=1)
    q10, q25, q75, q90 = np.quantile(values, [0.1, 0.25, 0.75, 0.9], axis=1)
    trim = int(round(0.1 * values.shape[1]))
    ordered = np.sort(values, axis=1)
    return {
        # Same definitions as experiments.row_summaries.
        "row_mean": mean, "row_median": np.median(values, axis=1),
        "row_log1p_mean": np.log1p(values).mean(axis=1), "row_std": std,
        "row_max": values.max(axis=1),
        "row_cv": np.divide(std, mean, out=np.zeros_like(std), where=mean > 0),
        "row_q10": q10, "row_q25": q25, "row_q75": q75, "row_q90": q90,
        "row_trimmed_mean": ordered[:, trim:values.shape[1] - trim].mean(axis=1),
    }


def row_profile(values: np.ndarray) -> np.ndarray:
    """Scale-free profile: each value divided by its row mean (0 for an all-zero row)."""
    mean = values.mean(axis=1, keepdims=True)
    return np.divide(values, mean, out=np.zeros_like(values), where=mean > 0)


def fitted_features(reference: np.ndarray, values: np.ndarray) -> dict:
    """Where each value sits within its feature's distribution in the fitting rows."""
    ordered = np.sort(reference, axis=0)
    ecdf = np.column_stack([np.searchsorted(ordered[:, j], values[:, j], side="right")
                            for j in range(values.shape[1])]) / len(ordered)
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    return {"mean_train_ecdf": ecdf.mean(axis=1),
            "share_below_train_q25": (values < q25).mean(axis=1),
            "share_above_train_q75": (values > q75).mean(axis=1)}


def design(feature_set: str, raw_fit: np.ndarray, raw_eval: list) -> tuple:
    """Model inputs for the fitting rows and each evaluation block.

    Anything fitted (the train quantiles) comes from raw_fit only; evaluation
    rows, including validation or test rows, are only transformed.
    """
    extra = FEATURE_SETS[feature_set]

    def build(values):
        base = row_profile(values) if feature_set in PROFILE_SETS else values
        if not extra:
            return base
        stats = rowwise_features(values)  # summaries always use the raw values
        if any(name in FITTED_EXTRA for name in extra):
            stats.update(fitted_features(raw_fit, values))
        return np.hstack([base, np.column_stack([stats[name] for name in extra])])

    return build(raw_fit), [build(values) for values in raw_eval]


def column_names(spec: Spec) -> list[str]:
    base = ([f"{name}_over_row_mean" for name in FEATURE_COLUMNS]
            if spec.feature_set in PROFILE_SETS else list(FEATURE_COLUMNS))
    return [*base, *FEATURE_SETS[spec.feature_set]]


# ----------------------------------------------------------------------------
# Models: decision trees, random forests / ExtraTrees, XGBoost, HistGradientBoosting.
# ----------------------------------------------------------------------------

def make_model(spec: Spec, directions, seed: int, n_jobs: int, n_features: int):
    params = dict(spec.params)
    if spec.family == "tree":
        return DecisionTreeClassifier(**params, monotonic_cst=directions, random_state=seed)
    if spec.family == "forest":
        forest = RandomForestClassifier if params.pop("kind") == "rf" else ExtraTreesClassifier
        return forest(**params, monotonic_cst=directions, random_state=seed, n_jobs=n_jobs)
    if spec.family == "xgb":
        # One interaction group per column index. The string form is used because
        # XGBoost maps list entries through feature names, which arrays lack.
        groups = "[" + ",".join(f"[{j}]" for j in range(n_features)) + "]"
        extra = {"interaction_constraints": groups} if spec.additive else {}
        constraints = None if directions is None else tuple(int(d) for d in directions)
        return XGBClassifier(**params, **extra, monotone_constraints=constraints,
                             random_state=seed, n_jobs=n_jobs)
    if spec.family == "hgb":
        return HistGradientBoostingClassifier(
            **params, interaction_cst="no_interactions" if spec.additive else None,
            monotonic_cst=directions, early_stopping=False, random_state=seed)
    raise ValueError(f"Unknown model family: {spec.family}")


def predictions(model, spec: Spec, X):
    """(round count or 0, probabilities) for every checkpoint of one fitted model."""
    if not spec.checkpoints:
        yield 0, model.predict_proba(X)[:, 1]
    elif spec.family == "xgb":
        for k in spec.checkpoints:
            yield k, model.predict_proba(X, iteration_range=(0, k))[:, 1]
    else:  # HistGradientBoosting: one pass over the boosting stages
        wanted = set(spec.checkpoints)
        for i, proba in enumerate(model.staged_predict_proba(X), start=1):
            if i in wanted:
                yield i, proba[:, 1]


def fit_and_score(spec: Spec, X_fit, y_fit, eval_sets: dict, n_jobs: int):
    """Fit every seed and average probabilities; directions come from X_fit, y_fit."""
    directions = monotone_directions(X_fit, y_fit) if spec.monotone else None
    points = spec.checkpoints or (0,)
    scores = {name: {k: np.zeros(len(X)) for k in points} for name, X in eval_sets.items()}
    models = []
    for seed in spec.seeds:
        model = make_model(spec, directions, seed, n_jobs, X_fit.shape[1]).fit(X_fit, y_fit)
        models.append(model)
        for name, X in eval_sets.items():
            for k, p in predictions(model, spec, X):
                scores[name][k] += p / len(spec.seeds)
    return scores, models, directions


def freeze(result) -> Spec:
    """Replace a boosting round search with the CV-chosen round count."""
    spec = result.config
    if not spec.checkpoints:
        return spec
    key = "n_estimators" if spec.family == "xgb" else "max_iter"
    return replace(spec, params={**spec.params, key: result.rounds}, checkpoints=())


# ----------------------------------------------------------------------------
# Cross-validation on train, cached per (configuration, fold).
# ----------------------------------------------------------------------------

def cv_task(raw, y, fit_idx, eval_idx, spec: Spec):
    X_fit, (X_eval,) = design(spec.feature_set, raw[fit_idx], [raw[eval_idx]])
    scores, _, _ = fit_and_score(spec, X_fit, y[fit_idx], {"eval": X_eval}, n_jobs=1)
    return scores["eval"]


class CrossValidator:
    def __init__(self, raw, y, folds, n_splits, n_jobs):
        self.raw, self.y, self.folds, self.n_splits, self.n_jobs = raw, y, folds, n_splits, n_jobs
        self.cache, self.specs = {}, {}

    def run(self, specs: list, fold_ids: list) -> list:
        """Score specs on the given folds, reusing folds already computed."""
        todo = []
        for spec in specs:
            if self.specs.setdefault(spec.name, spec) != spec:
                raise ValueError(f"Two different configurations share the name {spec.name!r}.")
            done = self.cache.setdefault(spec.name, {})
            todo += [(spec, f) for f in fold_ids if f not in done]
        outputs = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(cv_task)(self.raw, self.y, *self.folds[f], spec) for spec, f in todo)
        for (spec, f), output in zip(todo, outputs):
            self.cache[spec.name][f] = output
        return [self.result(spec, fold_ids) for spec in specs]

    def result(self, spec: Spec, fold_ids: list):
        folds = [self.folds[f] for f in fold_ids]
        return collect(spec, [self.cache[spec.name][f] for f in fold_ids],
                       self.y, folds, self.n_splits)


# ----------------------------------------------------------------------------
# Configuration lists
# ----------------------------------------------------------------------------

def checkpoints(limit_rounds: int, step: int) -> tuple:
    return tuple(range(step, limit_rounds + 1, step))


def additive_grid(settings: Settings) -> list[Spec]:
    cps = checkpoints(settings.max_rounds, settings.round_step)
    grid = []
    for depth in (2, 3):
        for reg, mcw, lam in (("weak", 1, 1.0), ("strong", 10, 10.0)):
            for monotone in (False, True):
                grid.append(Spec(
                    "xgb", f"additive xgb depth {depth}, {reg} reg"
                           + (", monotone" if monotone else ""), "1-additive",
                    {**XGB_FIXED, "learning_rate": settings.additive_lr,
                     "n_estimators": settings.max_rounds, "max_depth": depth,
                     "subsample": 0.8, "colsample_bytree": 0.5,
                     "min_child_weight": mcw, "reg_lambda": lam},
                    monotone=monotone, additive=True, checkpoints=cps))
    for leaves in (4, 8):
        for l2 in (0.0, 10.0):
            for monotone in (False, True):
                grid.append(Spec(
                    "hgb", f"additive hgb leaves {leaves}, l2 {l2}"
                           + (", monotone" if monotone else ""), "1-additive",
                    {"learning_rate": settings.additive_lr, "max_iter": settings.max_rounds,
                     "max_leaf_nodes": leaves, "min_samples_leaf": 20,
                     "l2_regularization": l2, "max_bins": 255},
                    monotone=monotone, additive=True, checkpoints=cps))
    return grid


def spec_from_round1(info: dict, name: str, phase: str, seeds: tuple) -> Spec:
    family = {"tree": "tree", "forest": "forest", "boosting": "xgb"}[info["family"]]
    return Spec(family, name, phase, dict(info["params"]), info["feature_set"],
                bool(info["monotone"]), seeds=tuple(seeds))


def forest_refinement(base: Spec, trees: int) -> list[Spec]:
    """Grid around the best forest: max_features, leaf size, row subsampling."""
    p = base.params
    n_columns = len(column_names(base))
    mf = p["max_features"]
    fraction = 1 / np.sqrt(n_columns) if mf == "sqrt" else float(mf)
    fractions = sorted({round(min(1.0, fraction * m), 4) for m in (0.5, 1.0, 2.0)})
    leaf = int(p["min_samples_leaf"])
    leaves = sorted({max(1, leaf // 2), leaf, leaf * 2 if leaf > 1 else 3})
    samples = (None, 0.7) if p["kind"] == "rf" else (None,)
    grid = []
    for f in fractions:
        for leaf_size in leaves:
            for s in samples:
                params = {**p, "n_estimators": trees, "max_features": f,
                          "min_samples_leaf": leaf_size}
                if p["kind"] == "rf":
                    params["max_samples"] = s
                grid.append(replace(base, phase="3-forest", params=params,
                                    name=f"{p['kind']}, max_features {f}, leaf {leaf_size}, "
                                         f"max_samples {s}, depth {p['max_depth']}, "
                                         f"{trees} trees [{base.feature_set}"
                                         + (", monotone]" if base.monotone else "]")))
    return grid


def seeded(result, settings: Settings) -> Spec:
    spec = freeze(result)
    # HistGradientBoosting without subsampling is deterministic, so extra seeds add nothing.
    seeds = (SEED,) if spec.family == "hgb" else settings.seeds
    return replace(spec, phase="4-seeds", seeds=seeds, name=f"{spec.name} x{len(seeds)} seeds")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def cv_row(result, phase: str, reference=None) -> dict:
    spec = result.config
    row = {
        "phase": phase, "family": spec.family, "name": spec.name,
        "feature_set": spec.feature_set, "monotone": spec.monotone, "additive": spec.additive,
        "seeds": len(spec.seeds), "folds": len(result.fold_aucs), "rounds": result.rounds,
        "rounds_at_grid_edge": bool(result.rounds is not None and spec.checkpoints
                                    and result.rounds == max(spec.checkpoints)),
        "cv_auc_mean": result.fold_aucs.mean(), "cv_auc_std": result.fold_aucs.std(ddof=1),
        "params": json.dumps(spec.params, sort_keys=True, default=str),
    }
    if reference is not None and len(reference.fold_aucs) == len(result.fold_aucs):
        diff = result.fold_aucs - reference.fold_aucs
        row.update({"reference": reference.config.name, "delta_vs_reference": diff.mean(),
                    "share_folds_better_than_reference": (diff > 0).mean()})
    return row


def describe(spec: Spec, result=None) -> dict:
    info = asdict(spec)
    if result is not None:
        info["cv_auc_mean"] = float(result.fold_aucs.mean())
        info["cv_folds"] = len(result.fold_aucs)
    return info


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

def run_round2(data_dir: Path, round1_dir: Path, output_dir: Path,
               settings: Settings = FULL, n_jobs: int = -1) -> dict:
    started = time.perf_counter()
    timings, rows = {}, []

    def done(label: str, since: float) -> None:
        timings[label] = round(time.perf_counter() - since, 1)
        print(f"  [{label} finished in {timings[label]:.0f}s]", flush=True)

    def show(result, prefix: str = "") -> None:
        rounds = f", rounds {result.rounds}" if result.rounds else ""
        print(f"  {prefix}{result.config.name:78s} CV AUC {result.fold_aucs.mean():.4f} "
              f"({len(result.fold_aucs)} folds){rounds}", flush=True)

    round1_dir = Path(round1_dir)
    round1 = json.loads((round1_dir / "summary.json").read_text(encoding="utf-8"))
    round1_val = pd.read_csv(round1_dir / "validation_predictions.csv",
                             dtype={"row_id": "string"}, float_precision="round_trip")
    data = load_data(data_dir)
    raw, y = data.X_train.to_numpy(np.float64), data.y_train.to_numpy()
    raw_val, y_val = data.X_val.to_numpy(np.float64), data.y_val.to_numpy()
    if not (np.array_equal(round1_val.row_id.to_numpy(), data.val_ids.to_numpy())
            and np.array_equal(round1_val.target.to_numpy(), y_val)):
        raise ValueError("Round-1 validation predictions do not match validation.csv.")
    cv_info = round1["cv"]
    if (cv_info["n_splits"], cv_info["n_repeats"], cv_info["random_state"]) != (
            settings.n_splits, settings.n_repeats, SEED):
        raise ValueError("Round-1 CV folds differ from this run; rerun experiments.py first.")

    folds = list(RepeatedStratifiedKFold(n_splits=settings.n_splits, n_repeats=settings.n_repeats,
                                         random_state=SEED).split(raw, y))
    cv = CrossValidator(raw, y, folds, settings.n_splits, n_jobs)
    screen_ids, all_ids = list(range(settings.n_splits)), list(range(len(folds)))

    # Phase 1: additive boosting, screened on 5 folds, top configs confirmed on all.
    t = time.perf_counter()
    grid = limit(additive_grid(settings), settings)
    print(f"Phase 1: {len(grid)} additive boosting configurations, "
          f"screened on {len(screen_ids)} folds", flush=True)
    screened = cv.run(grid, screen_ids)
    rows += [cv_row(r, "1-screen") for r in screened]
    shortlist = []
    for family in ("xgb", "hgb"):
        ranked = sorted((r for r in screened if r.config.family == family),
                        key=lambda r: -r.fold_aucs.mean())
        shortlist += [r.config for r in ranked[:settings.top_k]]
    confirmed = cv.run(shortlist, all_ids)
    rows += [cv_row(r, "1-confirm") for r in confirmed]
    for r in confirmed:
        show(r)
    best_additive = pick(confirmed)
    show(best_additive, "best additive: ")
    done("phase 1", t)

    # Phase 2: extra aggregates on the best round-1 models and the best additive model.
    t = time.perf_counter()
    finals1 = round1["finals"]
    xgb1 = spec_from_round1(finals1["boosting"], "round-1 xgb (1 seed)", "2-aggregates", (SEED,))
    rounds1 = int(xgb1.params["n_estimators"])
    search = min(settings.standard_round_cap, 2 * rounds1)
    search -= search % settings.round_step
    bases = {
        "tree": spec_from_round1(finals1["tree"], "round-1 tree", "2-aggregates", (SEED,)),
        "forest": spec_from_round1(finals1["forest"], "round-1 forest (1 seed)",
                                   "2-aggregates", (SEED,)),
        "xgb": replace(xgb1, params={**xgb1.params, "n_estimators": max(search, rounds1)},
                       checkpoints=checkpoints(max(search, rounds1), settings.round_step)),
        "additive": best_additive.config,
    }
    pairs = {key: (base, replace(base, phase="2-aggregates", feature_set="raw+aggregates",
                                 name=f"{base.name} +aggregates"))
             for key, base in bases.items() if base.feature_set != "raw+aggregates"}
    print(f"Phase 2: aggregates re-tested on {len(pairs)} models over {len(all_ids)} folds",
          flush=True)
    specs = [s for pair in pairs.values() for s in pair]
    results = dict(zip([s.name for s in specs], cv.run(specs, all_ids)))
    winners = {}
    for key, (base, variant) in pairs.items():
        base_result, variant_result = results[base.name], results[variant.name]
        rows += [cv_row(base_result, "2-base"), cv_row(variant_result, "2-aggregates", base_result)]
        show(base_result)
        show(variant_result)
        winners[key] = pick([base_result, variant_result])
    done("phase 2", t)

    # Phase 3: forest refinement around the phase-2 forest winner.
    t = time.perf_counter()
    grid = limit(forest_refinement(winners["forest"].config, settings.forest_trees), settings)
    print(f"Phase 3: {len(grid)} forest refinements, screened on {len(screen_ids)} folds",
          flush=True)
    screened = cv.run(grid, screen_ids)
    rows += [cv_row(r, "3-screen") for r in screened]
    shortlist = [r.config for r in sorted(screened, key=lambda r: -r.fold_aucs.mean())
                 [:settings.top_k]]
    confirmed = cv.run(shortlist, all_ids)
    rows += [cv_row(r, "3-confirm", winners["forest"]) for r in confirmed]
    for r in confirmed:
        show(r)
    best_forest = pick([winners["forest"], *confirmed])
    show(best_forest, "best forest: ")
    done("phase 3", t)

    # Phase 4: seed averaging, then equal-weight averages of the three members.
    t = time.perf_counter()
    members = {"forest": best_forest, "xgb": winners["xgb"], "additive": winners["additive"]}
    seeded_specs = {key: seeded(result, settings) for key, result in members.items()}
    print("Phase 4: seed averaging and equal-weight averages", flush=True)
    seeded_results = dict(zip(seeded_specs, cv.run(list(seeded_specs.values()), all_ids)))
    for key, result in seeded_results.items():
        rows.append(cv_row(result, "4-seeds", members[key]))
        show(result)
    averages = []
    for size in (2, 3):
        for combo in combinations(seeded_results, size):
            spec = Spec("ensemble", "average of " + " + ".join(combo), "4-average",
                        {"members": list(combo), "weights": [1 / size] * size})
            averages.append(average_cv(spec, [seeded_results[k] for k in combo],
                                       y, folds, settings.n_splits))
    rows += [cv_row(r, "4-average") for r in averages]
    for r in averages:
        show(r)
    best_average = pick(averages)
    boosting_key = max(("xgb", "additive"), key=lambda k: seeded_results[k].fold_aucs.mean())
    done("phase 4", t)

    # Validation: fit on all train rows; at most three finalists plus the tree.
    t = time.perf_counter()
    print("Validation: finalists fitted on train and scored once", flush=True)
    fitted, importances = {}, {}
    to_fit = {**seeded_specs, "tree": freeze(winners["tree"])}
    for key, spec in to_fit.items():
        X_fit, (X_train_eval, X_val_eval) = design(spec.feature_set, raw, [raw, raw_val])
        scores, models, _ = fit_and_score(spec, X_fit, y,
                                          {"train": X_train_eval, "validation": X_val_eval}, n_jobs)
        fitted[key] = (scores["train"][0], scores["validation"][0])
        values = [getattr(m, "feature_importances_", None) for m in models]
        if values[0] is not None:
            importances[key] = pd.Series(np.mean(values, axis=0), index=column_names(spec))
    average_members = best_average.config.params["members"]
    finalists = {
        "round2_forest": ("forest", fitted["forest"], seeded_specs["forest"]),
        "round2_boosting": ("boosting", fitted[boosting_key], seeded_specs[boosting_key]),
        "round2_average": ("ensemble",
                           tuple(np.mean([fitted[k][i] for k in average_members], axis=0)
                                 for i in (0, 1)),
                           best_average.config),
        "round2_tree": ("tree", fitted["tree"], to_fit["tree"]),
    }
    cv_means = {"round2_forest": seeded_results["forest"], "round2_boosting":
                seeded_results[boosting_key], "round2_average": best_average,
                "round2_tree": winners["tree"]}
    round1_candidate = round1["selected_candidate"]["model"]
    reference_columns = ["boosting_baseline", "tree_enhanced", "forest_enhanced",
                         "boosting_enhanced", "ensemble_enhanced"]
    scores = {name: round1_val[name].to_numpy() for name in reference_columns}
    scores.update({name: value[1][1] for name, value in finalists.items()})
    boot = stratified_bootstrap(y_val, scores, settings.bootstrap, SEED)
    table = []
    for name, p in scores.items():
        round2 = name.startswith("round2_")
        row = {
            "model": name, "round": 2 if round2 else 1,
            "role": ("report-only" if name == "round2_tree" else "finalist") if round2 else (
                "baseline" if name.endswith("baseline") else "round-1 finalist"),
            "description": finalists[name][2].name if round2 else name,
            "cv_auc_mean": float(cv_means[name].fold_aucs.mean()) if round2 else None,
            "validation_auc": float(roc_auc_score(y_val, p)),
            "validation_ci_low": float(np.percentile(boot[name], 2.5)),
            "validation_ci_high": float(np.percentile(boot[name], 97.5)),
            "validation_accuracy": split_metrics(y_val, p)["accuracy"],
        }
        if round2:
            train_auc = split_metrics(y, finalists[name][1][0])["auc"]
            row.update({"train_auc": train_auc,
                        "train_minus_validation_auc": train_auc - row["validation_auc"]})
        for tag, other in (("xgboost_baseline", "boosting_baseline"),
                           ("round1_candidate", round1_candidate)):
            if other != name:
                diff = boot[name] - boot[other]
                row[f"diff_vs_{tag}"] = row["validation_auc"] - roc_auc_score(y_val, scores[other])
                row[f"diff_vs_{tag}_ci_low"] = float(np.percentile(diff, 2.5))
                row[f"diff_vs_{tag}_ci_high"] = float(np.percentile(diff, 97.5))
        table.append(row)
    results_table = pd.DataFrame(table)
    pool = results_table[results_table.role.isin(["finalist", "round-1 finalist"])]
    selected = pool.loc[pool.validation_auc.idxmax()]
    done("validation", t)

    # Selected model, in a form make_submission.py can rebuild from train only.
    if selected.model.startswith("round2_"):
        _, _, spec = finalists[selected.model]
        member_specs = ([seeded_specs[k] for k in spec.params["members"]]
                        if spec.family == "ensemble" else [spec])
    else:
        family = selected.model.split("_")[0]
        keys = ["forest", "boosting"] if family == "ensemble" else [family]
        member_specs = [spec_from_round1(round1["finals"][k], f"round-1 {k}", "round-1",
                                         round1["finals"][k]["seeds"]) for k in keys]
    selected_model = {
        "model": selected.model, "description": selected.description,
        "validation_auc": float(selected.validation_auc),
        "members": [asdict(s) for s in member_specs],
        "weights": [1 / len(member_specs)] * len(member_specs),
        "note": "Rebuild each member on train.csv only with its spec, average probabilities "
                "with equal weights. Chosen as the highest validation AUC among the "
                "enhanced finalists of both rounds, so the score is slightly optimistic.",
    }

    summary = {
        "purpose": "Round-2 tree-only experiments (work-plan phases 1-4)",
        "allowed_models": "Decision tree, random forest / ExtraTrees, XGBoost, "
                          "HistGradientBoosting; equal-weight averages of these",
        "round1_dir": str(round1_dir), "settings": asdict(settings),
        "cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": settings.n_splits,
               "n_repeats": settings.n_repeats, "random_state": SEED,
               "screening_folds": len(screen_ids), "confirm_folds": len(all_ids)},
        "best_additive": describe(best_additive.config, best_additive),
        "phase2_winners": {k: describe(r.config, r) for k, r in winners.items()},
        "best_forest": describe(best_forest.config, best_forest),
        "seeded_members": {k: {**describe(r.config, r), "rounds": r.rounds}
                           for k, r in seeded_results.items()},
        "best_average": describe(best_average.config, best_average),
        "validation": table,
        "selected_model": selected_model,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__, "xgboost": xgboost.__version__,
                     "joblib": joblib.__version__},
        "runtime_seconds": {**timings, "total": round(time.perf_counter() - started, 1)},
    }
    summary_json = json.dumps(summary, indent=2, default=json_default, allow_nan=False) + "\n"
    selected_json = json.dumps(selected_model, indent=2, default=json_default,
                               allow_nan=False) + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "cv_results.csv", index=False)
    results_table.to_csv(output_dir / "validation_results.csv", index=False)
    pd.DataFrame({"row_id": data.val_ids.to_numpy(), "target": y_val,
                  **{name: value[1][1] for name, value in finalists.items()}}).to_csv(
        output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(importances).rename_axis("feature").to_csv(
        output_dir / "feature_importances.csv")
    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (output_dir / "selected_model.json").write_text(selected_json, encoding="utf-8")

    print("\nValidation (bootstrap 95% intervals)")
    for row in table:
        print(f"  {row['model']:18s} {row['role']:16s} validation {row['validation_auc']:.4f} "
              f"[{row['validation_ci_low']:.4f}, {row['validation_ci_high']:.4f}]")
    print(f"Selected: {selected.model} ({selected.validation_auc:.4f}) - {selected.description}")
    print(f"Artifacts: {output_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--round1-dir", type=Path, default=PROJECT_DIR / "outputs" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default outputs/experiments_round2 (outputs/experiments_round2_quick "
                             "with --quick). Reruns replace the files.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced smoke test of the code path; not for reporting.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    output_dir = args.output_dir or PROJECT_DIR / "outputs" / (
        "experiments_round2_quick" if args.quick else "experiments_round2")
    run_round2(args.data_dir, args.round1_dir, output_dir, settings, args.n_jobs)


if __name__ == "__main__":
    main()
