"""Shared data loading for the decision-tree, random-forest, and boosting baselines.

Only train.csv and validation.csv are read. Features are returned unchanged:
there is no fitted transformation, resplitting, filtering, or model training.

Usage from a script or notebook in the project directory::

    from preprocessing import load_data

    data = load_data(".")
    # After approving a model configuration:
    # model.fit(data.X_train, data.y_train)
    # probabilities = model.predict_proba(data.X_val)[:, 1]
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = tuple(f"f{i:04d}" for i in range(512))
EXPECTED_COLUMNS = ("row_id", *FEATURE_COLUMNS, "target")


@dataclass(frozen=True)
class BaselineData:
    """Named inputs shared by all models; row IDs are never model features.

    Row order is preserved within each split, so features, labels, and IDs
    remain aligned. Pandas objects remain mutable; callers should copy them
    before experimenting with transformations.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    train_ids: pd.Series
    val_ids: pd.Series


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read one labeled split and reject invalid inputs without repairing them."""
    frame = pd.read_csv(path, dtype={"row_id": "string"})

    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"{path.name}: schema mismatch; missing columns={missing}, "
            f"unexpected columns={unexpected}."
        )
    if frame.empty:
        raise ValueError(f"{path.name}: dataset has no rows.")

    row_ids = frame["row_id"]
    if row_ids.isna().any() or row_ids.str.strip().eq("").any():
        raise ValueError(f"{path.name}: row_id contains missing or blank values.")
    if row_ids.duplicated().any():
        raise ValueError(f"{path.name}: row_id contains duplicates.")

    target = frame["target"]
    if not target.isin([0, 1]).all():
        raise ValueError(f"{path.name}: target must contain only nonmissing 0 or 1.")
    if target.nunique() != 2:
        raise ValueError(f"{path.name}: both target classes are required for ROC-AUC.")

    # Canonical column order also handles CSVs whose headers were reordered.
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    nonnumeric = [
        name for name in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(features[name])
        or pd.api.types.is_bool_dtype(features[name])
    ]
    if nonnumeric:
        raise ValueError(f"{path.name}: nonnumeric feature columns={nonnumeric}.")
    finite = np.isfinite(features.to_numpy(dtype=np.float64))
    if not finite.all():
        invalid_columns = features.columns[~finite.all(axis=0)].tolist()
        raise ValueError(
            f"{path.name}: missing or infinite feature values in {invalid_columns}."
        )

    return features, target.astype("int64"), row_ids.copy()


def load_data(data_dir: str | Path) -> BaselineData:
    """Load the original train/validation split from an explicit directory.

    Raises ValueError for invalid data or shared row IDs. File and CSV parsing
    errors propagate to the caller. No test/submission files are accessed and
    no files are written. All feature values and input row orders are retained.
    """
    data_dir = Path(data_dir)
    X_train, y_train, train_ids = _load_split(data_dir / "train.csv")
    X_val, y_val, val_ids = _load_split(data_dir / "validation.csv")

    if train_ids.isin(val_ids).any():
        raise ValueError("train.csv and validation.csv contain overlapping row_id values.")

    return BaselineData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_ids=train_ids,
        val_ids=val_ids,
    )
