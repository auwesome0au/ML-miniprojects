"""Run the first XGBoost baseline on the supplied train/validation split.

Run: .venv/bin/python boosting.py
Requires the xgboost-cpu package (imported as xgboost).
Importing this module does not load data, train, or save artifacts.
"""

import argparse
from pathlib import Path

import xgboost
from xgboost import XGBClassifier

from baseline_reporting import save_report
from preprocessing import load_data


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cpu",
    "importance_type": "gain",
    "random_state": 42,
    "n_jobs": 4,
}


def run_baseline(data_dir: Path, output_dir: Path) -> dict:
    data = load_data(data_dir)
    model = XGBClassifier(**MODEL_PARAMS)
    # Fixed boosting rounds: validation is not passed to fit or early stopping.
    # min_child_weight constrains Hessian sums, not the number of leaf samples;
    # it is not equivalent to the forest's min_samples_leaf.
    model.fit(data.X_train, data.y_train)
    return save_report(
        model, data, output_dir,
        importance_type="Average split gain (normalized)",
        extra_versions={"xgboost": xgboost.__version__},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "boosting",
        help="Artifact directory; rerunning replaces the baseline output files.",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
