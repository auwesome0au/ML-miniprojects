"""Shared evaluation and artifacts for the two ensemble baselines."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from preprocessing import BaselineData


THRESHOLD = 0.5


def save_report(model, data: BaselineData, output_dir: Path, *,
                importance_type: str, extra_versions: dict | None = None) -> dict:
    """Evaluate an already fitted model; never fit or select a configuration.

    Predictions include IDs for auditing and preserve each split's row order.
    Importances describe model use, not causality, and different importance
    definitions should not be compared numerically across model families.
    """
    positive_index = list(model.classes_).index(1)
    metrics = {}
    prediction_tables = {}
    for split, X, y, ids in [
        ("train", data.X_train, data.y_train, data.train_ids),
        ("validation", data.X_val, data.y_val, data.val_ids),
    ]:
        probabilities = model.predict_proba(X)[:, positive_index]
        if not np.isfinite(probabilities).all() or not (
            (probabilities >= 0) & (probabilities <= 1)
        ).all():
            raise ValueError(f"{split}: model returned invalid probabilities.")
        predicted = (probabilities >= THRESHOLD).astype(int)
        metrics[split] = {
            "rows": len(y),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "accuracy": float(accuracy_score(y, predicted)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        }
        prediction_tables[split] = pd.DataFrame({
            "row_id": ids.to_numpy(),
            "target": y.to_numpy(),
            "probability_match": probabilities,
            "predicted_target": predicted,
        })

    # XGBoost's default missing-value sentinel is NaN. Represent it as text
    # in metadata so metrics.json remains standards-compliant JSON.
    parameters = {
        name: "NaN" if isinstance(value, (float, np.floating)) and np.isnan(value) else value
        for name, value in model.get_params().items()
    }
    report = {
        "model": type(model).__name__,
        "model_params": parameters,
        "features": list(data.X_train.columns),
        "preprocessing": "Original features; no transformations; supplied split preserved",
        "fit_split": "train",
        "early_stopping": False,
        "hyperparameter_search": False,
        "classification_threshold": THRESHOLD,
        "confusion_matrix_layout": "Rows: actual [0, 1]; columns: predicted [0, 1]",
        "importance_type": importance_type,
        "importance_note": "Training-based model importance, not causal importance",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            **(extra_versions or {}),
        },
        **metrics,
        "train_minus_validation_auc": metrics["train"]["roc_auc"] - metrics["validation"]["roc_auc"],
    }
    # Serialize before writing so unsupported values cannot leave a partial report.
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    importances = pd.DataFrame({
        "feature": data.X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False, kind="stable")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(report_json, encoding="utf-8")
    importances.to_csv(output_dir / "feature_importances.csv", index=False)
    for split, table in prediction_tables.items():
        table.to_csv(output_dir / f"{split}_predictions.csv", index=False)

    print(report["model"])
    for split in ["train", "validation"]:
        result = metrics[split]
        print(f"{split}: ROC-AUC={result['roc_auc']:.6f}, accuracy={result['accuracy']:.6f}")
        print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {result['confusion_matrix']}")
    print(f"ROC-AUC gap (train - validation): {report['train_minus_validation_auc']:.6f}")
    print(f"Artifacts: {output_dir.resolve()}")
    return report
