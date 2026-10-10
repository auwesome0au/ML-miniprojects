"""Run the first random-forest baseline on the supplied train/validation split.

Run: .venv/bin/python random_forest.py
Importing this module does not load data, train, or save artifacts.
"""

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from baseline_reporting import save_report
from preprocessing import load_data


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}


def run_baseline(data_dir: Path, output_dir: Path) -> dict:
    data = load_data(data_dir)
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(data.X_train, data.y_train)
    return save_report(
        model, data, output_dir,
        importance_type="Mean decrease in impurity (normalized)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "random_forest",
        help="Artifact directory; rerunning replaces the baseline output files.",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
