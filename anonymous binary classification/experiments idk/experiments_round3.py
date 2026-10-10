"""Round 3 of the tree-only experiments (approved 2026-09-19).

Deeper XGBoost, RandomForest-only tuning, a scale-free profile feature test,
seed averaging and averages. No ExtraTrees: the user is not sure it is
allowed, so every forest here is RandomForestClassifier. Only decision trees,
random forests and XGBoost are fitted.

Run after experiments.py:  .venv/bin/python experiments_round3.py
Smoke test: .venv/bin/python experiments_round3.py --quick --round1-dir <dir> --output-dir <dir>

Reads train.csv and validation.csv through preprocessing.load_data, plus
experiments.py's summary.json, cv_fold_aucs.csv and validation predictions.
Never reads test.csv. Every learned step is fitted on training rows only.
Importing this module does not load data, fit models, or write files.

Same folds as experiments.py: RepeatedStratifiedKFold(5, 2, seed 42). Grids are
screened on the first 5 folds; the top configs are confirmed on all 10. Every
choice uses the mean fold AUC of CV on train.

Phase 1  Deeper XGBoost: depth 4-6, colsample 0.3 or 0.5, subsample 0.6 or
         0.8, learning rate 0.05, rounds chosen by CV up to 4000, raw
         features plus six row summaries (round 1's best feature set).
Phase 2  RandomForest: max_features 0.2, 0.3 or 0.5; no depth limit or 15;
         leaf 1 or 2; 500 trees; raw features (round 1's best for forests).
Phase 3  Scale-free profile test on both winners: each feature divided by its
         row mean, plus the six summaries, instead of the raw values.
Phase 4  XGBoost seed averaging (5 seeds). RandomForest keeps one seed: with
         500 trees, extra seeds only add trees. Averages: RandomForest +
         seeded XGBoost, and the top two XGBoost settings.
Validation: three finalists (XGBoost, RandomForest, best average), compared
by paired bootstrap with round 1's saved predictions, including its tuned
XGBoost (no ExtraTrees). The best validation AUC among the round-3 finalists
and round 1's tuned XGBoost is written to selected_model.json for
make_submission.py --model selected.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from experiments import average_cv, json_default, limit, pick, split_metrics, stratified_bootstrap
from experiments_round2 import (SEED, XGB_FIXED, CrossValidator, Spec, checkpoints, column_names,
                                design, fit_and_score, freeze, spec_from_round1)
from preprocessing import load_data


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Run size. QUICK exists only to smoke-test the code path."""
    n_splits: int = 5
    n_repeats: int = 2
    learning_rate: float = 0.05
    max_rounds: int = 4000
    round_step: int = 50
    forest_trees: int = 500
    seeds: tuple = (42, 43, 44, 45, 46)
    top_k: int = 2
    grid_limit: int | None = None
    bootstrap: int = 2000


FULL = Settings()
QUICK = Settings(n_repeats=1, max_rounds=200, forest_trees=60, seeds=(42, 43), top_k=1,
                 grid_limit=3, bootstrap=200)


def xgb_grid(settings: Settings) -> list[Spec]:
    cps = checkpoints(settings.max_rounds, settings.round_step)
    return [Spec("xgb", f"xgb depth {d}, colsample {c}, subsample {ss}", "1-xgb",
                 {**XGB_FIXED, "learning_rate": settings.learning_rate,
                  "n_estimators": settings.max_rounds, "max_depth": d, "colsample_bytree": c,
                  "subsample": ss, "min_child_weight": 1, "reg_lambda": 1.0},
                 feature_set="raw+summaries", checkpoints=cps)
            for d in (4, 5, 6) for c in (0.3, 0.5) for ss in (0.6, 0.8)]


def rf_grid(settings: Settings) -> list[Spec]:
    return [Spec("forest", f"rf max_features {mf}, depth {d}, leaf {leaf}, "
                           f"{settings.forest_trees} trees", "2-rf",
                 {"kind": "rf", "n_estimators": settings.forest_trees, "max_features": mf,
                  "max_depth": d, "min_samples_leaf": leaf, "bootstrap": True},
                 feature_set="raw")
            for mf in (0.2, 0.3, 0.5) for d in (None, 15) for leaf in (1, 2)]


def cv_row(result, phase: str, reference: np.ndarray | None = None,
           reference_name: str | None = None) -> dict:
    spec = result.config
    row = {
        "phase": phase, "family": spec.family, "name": spec.name,
        "feature_set": spec.feature_set, "seeds": len(spec.seeds),
        "folds": len(result.fold_aucs), "rounds": result.rounds,
        "rounds_at_grid_edge": bool(result.rounds is not None and spec.checkpoints
                                    and result.rounds == max(spec.checkpoints)),
        "cv_auc_mean": result.fold_aucs.mean(), "cv_auc_std": result.fold_aucs.std(ddof=1),
        "params": json.dumps(spec.params, sort_keys=True, default=str),
    }
    if reference is not None and len(reference) == len(result.fold_aucs):
        diff = result.fold_aucs - reference
        row.update({"reference": reference_name, "delta_vs_reference": diff.mean(),
                    "share_folds_better_than_reference": (diff > 0).mean()})
    return row


def describe(spec: Spec, result=None) -> dict:
    info = asdict(spec)
    if result is not None:
        info.update({"cv_auc_mean": float(result.fold_aucs.mean()),
                     "cv_folds": len(result.fold_aucs), "rounds": result.rounds})
    return info


def run_round3(data_dir: Path, round1_dir: Path, output_dir: Path,
               settings: Settings = FULL, n_jobs: int = -1) -> dict:
    started = time.perf_counter()
    timings, rows = {}, []

    def done(label: str, since: float) -> None:
        timings[label] = round(time.perf_counter() - since, 1)
        print(f"  [{label} finished in {timings[label]:.0f}s]", flush=True)

    def show(result, prefix: str = "") -> None:
        rounds = f", rounds {result.rounds}" if result.rounds else ""
        print(f"  {prefix}{result.config.name:70s} CV AUC {result.fold_aucs.mean():.4f} "
              f"({len(result.fold_aucs)} folds){rounds}", flush=True)

    round1_dir = Path(round1_dir)
    round1 = json.loads((round1_dir / "summary.json").read_text(encoding="utf-8"))
    round1_val = pd.read_csv(round1_dir / "validation_predictions.csv",
                             dtype={"row_id": "string"}, float_precision="round_trip")
    round1_folds = pd.read_csv(round1_dir / "cv_fold_aucs.csv")
    data = load_data(data_dir)
    raw, y = data.X_train.to_numpy(np.float64), data.y_train.to_numpy()
    raw_val, y_val = data.X_val.to_numpy(np.float64), data.y_val.to_numpy()
    if not (np.array_equal(round1_val.row_id.to_numpy(), data.val_ids.to_numpy())
            and np.array_equal(round1_val.target.to_numpy(), y_val)):
        raise ValueError("Round-1 validation predictions do not match validation.csv.")
    info = round1["cv"]
    if (info["n_splits"], info["n_repeats"], info["random_state"]) != (
            settings.n_splits, settings.n_repeats, SEED):
        raise ValueError("Round-1 CV folds differ from this run; rerun experiments.py first.")
    # Round 1's seeded XGBoost fold AUCs (same folds) are the paired CV reference.
    reference_rows = round1_folds[(round1_folds.stage == "5-seeds")
                                  & (round1_folds.family == "boosting")].sort_values("fold")
    reference = reference_rows.auc.to_numpy()
    reference_name = "round-1 XGBoost x5 seeds: " + str(reference_rows.name.iloc[0])

    folds = list(RepeatedStratifiedKFold(n_splits=settings.n_splits, n_repeats=settings.n_repeats,
                                         random_state=SEED).split(raw, y))
    if len(reference) != len(folds):
        raise ValueError("Round-1 reference fold AUCs do not cover every fold.")
    cv = CrossValidator(raw, y, folds, settings.n_splits, n_jobs)
    screen_ids, all_ids = list(range(settings.n_splits)), list(range(len(folds)))

    def screen_and_confirm(grid: list, phase: str) -> list:
        screened = cv.run(grid, screen_ids)
        rows.extend(cv_row(r, f"{phase}-screen") for r in screened)
        shortlist = [r.config for r in sorted(screened, key=lambda r: -r.fold_aucs.mean())
                     [:settings.top_k]]
        confirmed = sorted(cv.run(shortlist, all_ids), key=lambda r: -r.fold_aucs.mean())
        rows.extend(cv_row(r, f"{phase}-confirm", reference, reference_name) for r in confirmed)
        for r in confirmed:
            show(r)
        return confirmed

    # Phase 1: deeper XGBoost.
    t = time.perf_counter()
    grid = limit(xgb_grid(settings), settings)
    print(f"Phase 1: {len(grid)} XGBoost configurations, screened on {len(screen_ids)} folds",
          flush=True)
    xgb_confirmed = screen_and_confirm(grid, "1-xgb")
    done("phase 1", t)

    # Phase 2: RandomForest only.
    t = time.perf_counter()
    grid = limit(rf_grid(settings), settings)
    print(f"Phase 2: {len(grid)} RandomForest configurations, screened on {len(screen_ids)} folds",
          flush=True)
    rf_confirmed = screen_and_confirm(grid, "2-rf")
    done("phase 2", t)

    # Phase 3: scale-free profile features on both winners.
    t = time.perf_counter()
    bests = {"xgb": xgb_confirmed[0], "forest": rf_confirmed[0]}
    variants = {key: replace(r.config, phase="3-profile", feature_set="profile+summaries",
                             name=f"{r.config.name} [profile+summaries]")
                for key, r in bests.items()}
    print("Phase 3: scale-free profile features on both winners", flush=True)
    variant_results = dict(zip(variants, cv.run(list(variants.values()), all_ids)))
    winners = {}
    for key, result in variant_results.items():
        rows.append(cv_row(result, "3-profile", bests[key].fold_aucs, bests[key].config.name))
        show(bests[key])
        show(result)
        winners[key] = pick([bests[key], result])
    done("phase 3", t)

    # Phase 4: XGBoost seed averaging and two averages.
    t = time.perf_counter()
    xgb_seeded = replace(freeze(winners["xgb"]), phase="4-seeds", seeds=settings.seeds,
                         name=f"{winners['xgb'].config.name} x{len(settings.seeds)} seeds")
    print("Phase 4: XGBoost seed averaging and averages", flush=True)
    (xgb_seeded_result,) = cv.run([xgb_seeded], all_ids)
    rows.append(cv_row(xgb_seeded_result, "4-seeds", reference, reference_name))
    show(xgb_seeded_result)
    rf_final = winners["forest"]
    averages = {
        "rf+xgb": average_cv(Spec("ensemble", "average of RandomForest + XGBoost x5 seeds",
                                  "4-average", {"members": ["forest", "xgb_seeded"]}),
                             [rf_final, xgb_seeded_result], y, folds, settings.n_splits),
    }
    if len(xgb_confirmed) > 1:
        averages["xgb_top2"] = average_cv(
            Spec("ensemble", "average of the top two XGBoost settings", "4-average",
                 {"members": ["xgb_top1", "xgb_top2"]}),
            xgb_confirmed[:2], y, folds, settings.n_splits)
    for result in averages.values():
        rows.append(cv_row(result, "4-average", reference, reference_name))
        show(result)
    best_average_key = max(averages, key=lambda k: averages[k].fold_aucs.mean())
    done("phase 4", t)

    # Validation: fit on all train rows; three finalists scored once.
    t = time.perf_counter()
    print("Validation: finalists fitted on train and scored once", flush=True)
    member_specs = {"xgb_seeded": xgb_seeded, "forest": freeze(rf_final)}
    if best_average_key == "xgb_top2":
        member_specs.update({"xgb_top1": freeze(xgb_confirmed[0]),
                             "xgb_top2": freeze(xgb_confirmed[1])})
    fitted, importances = {}, {}
    for key, spec in member_specs.items():
        X_fit, (X_train_eval, X_val_eval) = design(spec.feature_set, raw, [raw, raw_val])
        scores, models, _ = fit_and_score(spec, X_fit, y,
                                          {"train": X_train_eval, "validation": X_val_eval}, n_jobs)
        fitted[key] = (scores["train"][0], scores["validation"][0])
        importances[key] = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0),
                                     index=column_names(spec))
    average_members = averages[best_average_key].config.params["members"]
    finalists = {
        "round3_xgb": (fitted["xgb_seeded"], [member_specs["xgb_seeded"]], xgb_seeded_result),
        "round3_rf": (fitted["forest"], [member_specs["forest"]], rf_final),
        "round3_average": (tuple(np.mean([fitted[k][i] for k in average_members], axis=0)
                                 for i in (0, 1)),
                           [member_specs[k] for k in average_members], averages[best_average_key]),
    }
    reference_columns = ["boosting_baseline", "tree_enhanced", "forest_enhanced",
                         "boosting_enhanced", "ensemble_enhanced"]
    scores = {name: round1_val[name].to_numpy() for name in reference_columns}
    scores.update({name: value[0][1] for name, value in finalists.items()})
    boot = stratified_bootstrap(y_val, scores, settings.bootstrap, SEED)
    uses_extratrees = {"forest_enhanced", "ensemble_enhanced"}
    table = []
    for name, p in scores.items():
        round3 = name.startswith("round3_")
        row = {
            "model": name, "round": 3 if round3 else 1,
            "role": "finalist" if round3 else ("baseline" if name.endswith("baseline")
                                               else "round-1 finalist"),
            "uses_extratrees": name in uses_extratrees,
            "description": (" + ".join(s.name for s in finalists[name][1]) if round3 else name),
            "cv_auc_mean": float(finalists[name][2].fold_aucs.mean()) if round3 else None,
            "validation_auc": float(roc_auc_score(y_val, p)),
            "validation_ci_low": float(np.percentile(boot[name], 2.5)),
            "validation_ci_high": float(np.percentile(boot[name], 97.5)),
            "validation_accuracy": split_metrics(y_val, p)["accuracy"],
        }
        if round3:
            row["train_auc"] = split_metrics(y, finalists[name][0][0])["auc"]
        for tag, other in (("xgboost_baseline", "boosting_baseline"),
                           ("round1_xgboost", "boosting_enhanced")):
            if other != name:
                diff = boot[name] - boot[other]
                row[f"diff_vs_{tag}"] = row["validation_auc"] - roc_auc_score(y_val, scores[other])
                row[f"diff_vs_{tag}_ci_low"] = float(np.percentile(diff, 2.5))
                row[f"diff_vs_{tag}_ci_high"] = float(np.percentile(diff, 97.5))
        table.append(row)
    results = pd.DataFrame(table)
    pool = results[results.model.isin([*finalists, "boosting_enhanced"])]
    selected = pool.loc[pool.validation_auc.idxmax()]
    done("validation", t)

    if selected.model in finalists:
        selected_members = finalists[selected.model][1]
    else:
        selected_members = [spec_from_round1(round1["finals"]["boosting"], "round-1 boosting",
                                             "round-1", round1["finals"]["boosting"]["seeds"])]
    selected_model = {
        "model": selected.model, "description": selected.description,
        "validation_auc": float(selected.validation_auc),
        "members": [asdict(s) for s in selected_members],
        "weights": [1 / len(selected_members)] * len(selected_members),
        "note": "No ExtraTrees. Rebuild each member on train.csv only with its spec and "
                "average probabilities with equal weights. Chosen as the highest validation "
                "AUC among the round-3 finalists and round 1's tuned XGBoost, so the score is "
                "slightly optimistic.",
    }
    summary = {
        "purpose": "Round-3 tree-only experiments without ExtraTrees",
        "allowed_models": "Decision tree, RandomForest, XGBoost; equal-weight averages",
        "round1_dir": str(round1_dir), "settings": asdict(settings),
        "cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": settings.n_splits,
               "n_repeats": settings.n_repeats, "random_state": SEED,
               "screening_folds": len(screen_ids), "confirm_folds": len(all_ids)},
        "cv_reference": {"name": reference_name, "cv_auc_mean": float(reference.mean())},
        "xgb_confirmed": [describe(r.config, r) for r in xgb_confirmed],
        "rf_confirmed": [describe(r.config, r) for r in rf_confirmed],
        "profile_test": {k: {"without": float(bests[k].fold_aucs.mean()),
                             "with": float(variant_results[k].fold_aucs.mean())} for k in bests},
        "winners": {k: describe(r.config, r) for k, r in winners.items()},
        "xgb_seeded": describe(xgb_seeded, xgb_seeded_result),
        "averages": {k: describe(r.config, r) for k, r in averages.items()},
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
    results.to_csv(output_dir / "validation_results.csv", index=False)
    pd.DataFrame({"row_id": data.val_ids.to_numpy(), "target": y_val,
                  **{name: value[0][1] for name, value in finalists.items()}}).to_csv(
        output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(importances).rename_axis("feature").to_csv(output_dir / "feature_importances.csv")
    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (output_dir / "selected_model.json").write_text(selected_json, encoding="utf-8")

    print("\nValidation (bootstrap 95% intervals)")
    for row in table:
        print(f"  {row['model']:18s} {row['role']:16s} validation {row['validation_auc']:.4f} "
              f"[{row['validation_ci_low']:.4f}, {row['validation_ci_high']:.4f}]"
              + ("  (ExtraTrees)" if row["uses_extratrees"] else ""))
    print(f"Selected: {selected.model} ({selected.validation_auc:.4f}) - {selected.description}")
    print(f"Artifacts: {output_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--round1-dir", type=Path, default=PROJECT_DIR / "outputs" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default outputs/experiments_round3 (outputs/experiments_round3_quick "
                             "with --quick). Reruns replace the files.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced smoke test of the code path; not for reporting.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    output_dir = args.output_dir or PROJECT_DIR / "outputs" / (
        "experiments_round3_quick" if args.quick else "experiments_round3")
    run_round3(args.data_dir, args.round1_dir, output_dir, settings, args.n_jobs)


if __name__ == "__main__":
    main()
