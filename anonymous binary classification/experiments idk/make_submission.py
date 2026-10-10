"""Generate a Kaggle submission file from train-only models.

Run: .venv/bin/python make_submission.py                    (XGBoost baseline)
     .venv/bin/python make_submission.py --model round1     (round-1 selection)
     .venv/bin/python make_submission.py --model selected   (round-2 selection)
The baseline writes outputs/submissions/submission_xgboost_baseline.csv. The
round1 option rebuilds the candidate selected in outputs/experiments/summary.json
(experiments.py); the selected option rebuilds the model recorded in
outputs/experiments_round2/selected_model.json. Both write
outputs/submissions/submission_<model>.csv. Each file gets a JSON record beside
it; rerunning replaces both. A rebuilt model must reproduce its recorded
validation AUC exactly, or no file is written.

Competition rules followed (Sections 4 and 7):
- Models are fitted on train.csv only: the baseline with boosting.py's exact
  configuration, the selected model with its recorded specification. They
  are never refitted on train plus validation.
- validation.csv is only scored, to confirm the refit reproduces the result.
- Only decision trees, random forests and gradient boosting are used.
- test.csv is used only to generate predictions. Nothing is fitted on it or
  computed from it apart from input checks.
- The file is written by code with sample_submission.csv's columns and row
  order, every test row_id exactly once and unmodified, and continuous
  probabilities between 0 and 1. Manual edits to the output are prohibited.
Importing this module does not load data, fit models, or write files.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from boosting import MODEL_PARAMS
from preprocessing import FEATURE_COLUMNS, load_data


PROJECT_DIR = Path(__file__).resolve().parent
SUBMISSION_COLUMNS = ["row_id", "target"]
SUBMISSION_DIR = PROJECT_DIR / "outputs" / "submissions"
DEFAULT_OUTPUT = SUBMISSION_DIR / "submission_xgboost_baseline.csv"
DEFAULT_SELECTED = PROJECT_DIR / "outputs" / "experiments_round2" / "selected_model.json"
DEFAULT_ROUND1 = PROJECT_DIR / "outputs" / "experiments" / "summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_ids(ids: pd.Series, source: str) -> None:
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{source}: row_id contains missing or blank values.")
    if ids.duplicated().any():
        raise ValueError(f"{source}: row_id contains duplicates.")


def load_test_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read test.csv for prediction only; reject problems instead of repairing them."""
    frame = pd.read_csv(path, dtype={"row_id": "string"})
    expected = {"row_id", *FEATURE_COLUMNS}
    missing = sorted(expected - set(frame.columns))
    unexpected = sorted(set(frame.columns) - expected)
    if missing or unexpected:
        raise ValueError(f"{path.name}: schema mismatch; missing columns={missing}, "
                         f"unexpected columns={unexpected}.")
    if frame.empty:
        raise ValueError(f"{path.name}: dataset has no rows.")
    check_ids(frame["row_id"], path.name)
    # Canonical column order, identical to the order used for training.
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    nonnumeric = [name for name in FEATURE_COLUMNS
                  if not pd.api.types.is_numeric_dtype(features[name])
                  or pd.api.types.is_bool_dtype(features[name])]
    if nonnumeric:
        raise ValueError(f"{path.name}: nonnumeric feature columns={nonnumeric}.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name}: missing or infinite feature values.")
    return features, frame["row_id"]


def load_sample_ids(path: Path) -> pd.Series:
    """Row IDs in the required submission order."""
    sample = pd.read_csv(path, dtype={"row_id": "string"})
    if list(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {SUBMISSION_COLUMNS}, "
                         f"found {list(sample.columns)}.")
    check_ids(sample["row_id"], path.name)
    return sample["row_id"].reset_index(drop=True)


def check_submission(submission: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Enforce every file requirement from the competition page."""
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}.")
    ids = submission["row_id"].astype("string").reset_index(drop=True)
    if len(ids) != len(expected_ids) or not ids.equals(expected_ids):
        raise ValueError("Submission row_id values or order differ from sample_submission.csv.")
    target = submission["target"]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise ValueError("Submission target must be numeric.")
    values = target.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Submission target must be finite and between 0 and 1.")
    if np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Submission target looks like class labels; probabilities are required.")


def load_selected(path: Path) -> tuple[dict, list]:
    """Read the round-2 selection and turn its members back into Specs."""
    from experiments_round2 import Spec

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    members = [Spec(**{**m, "seeds": tuple(m["seeds"]), "checkpoints": tuple(m["checkpoints"])})
               for m in record["members"]]
    if any(spec.checkpoints for spec in members):
        raise ValueError("Selected members must have fixed round counts.")
    if len(set(record["weights"])) != 1 or len(record["weights"]) != len(members):
        raise ValueError("Only equal-weight averages are supported.")
    return record, members


def load_round1(path: Path) -> tuple[dict, list]:
    """Turn experiments.py's selected candidate into the same member format."""
    from experiments_round2 import spec_from_round1

    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = summary["selected_candidate"]
    family = chosen["model"].split("_")[0]
    keys = ["forest", "boosting"] if family == "ensemble" else [family]
    members = [spec_from_round1(summary["finals"][k], f"round-1 {k}", "round-1",
                                summary["finals"][k]["seeds"]) for k in keys]
    record = {"model": f"round1_{chosen['model']}", "description": chosen["description"],
              "validation_auc": chosen["validation_auc"],
              "members": [asdict(m) for m in members], "weights": [1 / len(members)] * len(members)}
    return record, members


def selected_predictions(data, X_test: pd.DataFrame, members: list) -> tuple:
    """Fit each member on train only; average validation and test probabilities.

    Train-quantile features are fitted on the training rows only; validation
    and test rows are just transformed.
    """
    from experiments_round2 import design, fit_and_score

    raw_train = data.X_train.to_numpy(np.float64)
    raw_eval = [data.X_val.to_numpy(np.float64), X_test.to_numpy(np.float64)]
    validation, test = [], []
    for spec in members:
        X_fit, (X_val, X_te) = design(spec.feature_set, raw_train, raw_eval)
        scores, _, _ = fit_and_score(spec, X_fit, data.y_train.to_numpy(),
                                     {"validation": X_val, "test": X_te}, n_jobs=-1)
        validation.append(scores["validation"][0])
        test.append(scores["test"][0])
    # Same arithmetic as experiments_round2.py uses for its averages.
    return np.mean(validation, axis=0), np.mean(test, axis=0)


def make_submission(data_dir: Path, output_path: Path | None = None, model_choice: str = "baseline",
                    selected_path: Path = DEFAULT_SELECTED,
                    round1_path: Path = DEFAULT_ROUND1) -> dict:
    data_dir = Path(data_dir)
    data = load_data(data_dir)

    # Test rows are used only to generate predictions.
    test_path, sample_path = data_dir / "test.csv", data_dir / "sample_submission.csv"
    X_test, test_ids = load_test_features(test_path)
    sample_ids = load_sample_ids(sample_path)
    if len(test_ids) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError("test.csv and sample_submission.csv contain different row_id values.")

    if model_choice == "baseline":
        # Fit on train only, exactly as boosting.py does; validation is only scored.
        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(data.X_train, data.y_train)
        positive = list(model.classes_).index(1)
        validation_auc = float(roc_auc_score(data.y_val,
                                             model.predict_proba(data.X_val)[:, positive]))
        test_probabilities = model.predict_proba(X_test)[:, positive].astype(np.float64)
        solution = {"solution": "XGBoost baseline (boosting.py configuration)",
                    "model_params": MODEL_PARAMS}
        output_path = Path(output_path or DEFAULT_OUTPUT)
    else:
        record, members = (load_round1(round1_path) if model_choice == "round1"
                           else load_selected(selected_path))
        validation_probabilities, test_probabilities = selected_predictions(data, X_test, members)
        validation_auc = float(roc_auc_score(data.y_val, validation_probabilities))
        if abs(validation_auc - record["validation_auc"]) > 1e-12:
            raise ValueError(f"Refit validation AUC {validation_auc:.12f} does not reproduce the "
                             f"recorded {record['validation_auc']:.12f}; no file written.")
        source = round1_path if model_choice == "round1" else selected_path
        solution = {"solution": f"{record['model']} ({record['description']})",
                    "members": record["members"], "weights": record["weights"],
                    "selection_file": str(Path(source).resolve())}
        output_path = Path(output_path or SUBMISSION_DIR / f"submission_{record['model']}.csv")
    predictions = pd.Series(test_probabilities, index=test_ids.to_numpy())
    submission = pd.DataFrame({
        "row_id": sample_ids.to_numpy(),
        "target": predictions.loc[sample_ids.to_numpy()].to_numpy(),
    })
    check_submission(submission, sample_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    # Re-read the file on disk so the checks cover exactly what will be uploaded.
    # round_trip parsing avoids the default parser's last-digit float errors.
    written = pd.read_csv(output_path, dtype={"row_id": "string"}, float_precision="round_trip")
    check_submission(written, sample_ids)
    if not np.array_equal(written["target"].to_numpy(), submission["target"].to_numpy()):
        raise ValueError("Written targets differ from the model's predictions.")

    record = {
        **solution,
        "fitted_on": f"train.csv only ({len(data.y_train)} rows)",
        "validation_auc_check": validation_auc,
        "test_usage": "Predictions only; no fitting or statistics on test rows",
        "submission_file": output_path.name,
        "submission_sha256": sha256(output_path),
        "rows": len(submission),
        "columns": SUBMISSION_COLUMNS,
        "row_order": "Same as sample_submission.csv",
        "target_range": [float(submission.target.min()), float(submission.target.max())],
        "input_sha256": {name: sha256(data_dir / name) for name in
                         ("train.csv", "validation.csv", "test.csv", "sample_submission.csv")},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Validation ROC-AUC check (train-only fit): {validation_auc:.6f}")
    print(f"Submission: {output_path.resolve()} ({len(submission)} rows)")
    print(f"SHA-256: {record['submission_sha256']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--model", choices=("baseline", "round1", "selected"), default="baseline",
                        help="baseline: XGBoost baseline; round1: experiments.py selection; "
                             "selected: the round-2 selected model.")
    parser.add_argument("--round1", type=Path, default=DEFAULT_ROUND1,
                        help="summary.json written by experiments.py.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED,
                        help="selected_model.json written by experiments_round2.py.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Submission CSV path; a JSON record is written beside it.")
    args = parser.parse_args()
    make_submission(args.data_dir, args.output, args.model, args.selected, args.round1)


if __name__ == "__main__":
    main()
