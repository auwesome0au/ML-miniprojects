"""Train and explain the approved first decision-tree baseline.

Run with: .venv/bin/python decision_tree.py
Importing this module does not load data, train a model, or write artifacts.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from preprocessing import load_data


MODEL_PARAMS = {
    "max_depth": 3,
    "min_samples_leaf": 30,
    "random_state": 42,
}
THRESHOLD = 0.5
PROJECT_DIR = Path(__file__).resolve().parent


def evaluate(model: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate ranking with probabilities and classification at a fixed threshold."""
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions = (probabilities >= THRESHOLD).astype(int)
    return {
        "rows": len(y),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def run_baseline(data_dir: Path, output_dir: Path) -> dict:
    """Fit once on train, evaluate both splits, and save interpretation artifacts."""
    data = load_data(data_dir)
    model = DecisionTreeClassifier(**MODEL_PARAMS)
    model.fit(data.X_train, data.y_train)

    train_metrics = evaluate(model, data.X_train, data.y_train)
    val_metrics = evaluate(model, data.X_val, data.y_val)

    # These probabilities and counts use training labels only.
    train_leaf_ids = model.apply(data.X_train)
    leaves = []
    for node_id in np.unique(train_leaf_ids):
        targets = data.y_train.loc[train_leaf_ids == node_id]
        leaves.append({
            "node_id": int(node_id),
            "training_samples": len(targets),
            "nonmatches": int((targets == 0).sum()),
            "matches": int((targets == 1).sum()),
            "probability_match": float(targets.mean()),
        })

    report = {
        "model": "DecisionTreeClassifier",
        "model_params": model.get_params(),
        "data_dir": str(Path(data_dir).resolve()),
        "features": list(data.X_train.columns),
        "preprocessing": "Original features; no transformations; supplied split preserved",
        "classification_threshold": THRESHOLD,
        "confusion_matrix_layout": "Rows: actual [0, 1]; columns: predicted [0, 1]",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "train": train_metrics,
        "validation": val_metrics,
        "train_minus_validation_auc": train_metrics["roc_auc"] - val_metrics["roc_auc"],
        "tree": {
            "depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
            "nodes": int(model.tree_.node_count),
            "leaf_details": leaves,
        },
    }

    rules = export_text(
        model, feature_names=list(data.X_train.columns),
        decimals=6, show_weights=True,
    )
    rules += "\nThresholds above are rounded for display. Weights are [nonmatches, matches].\n"
    rules += "Leaf node IDs correspond to tree.png; probabilities use training labels.\n"
    for leaf in leaves:
        rules += (
            f"Node {leaf['node_id']}: n={leaf['training_samples']}, "
            f"nonmatches={leaf['nonmatches']}, matches={leaf['matches']}, "
            f"P(target=1)={leaf['probability_match']:.6f}\n"
        )

    # Use a noninteractive canvas so the script also works without a display.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(24, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    plot_tree(
        model, feature_names=list(data.X_train.columns),
        class_names=["different (0)", "same (1)"],
        node_ids=True, filled=True, rounded=True, precision=3,
        fontsize=10, ax=axes,
    )
    axes.set_title(
        "Decision-tree baseline — training counts; value = [different, same]\n"
        "max_depth=3, min_samples_leaf=30, random_state=42"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "tree.png", dpi=160)
    (output_dir / "tree_rules.txt").write_text(rules, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )

    for split, metrics in [("Train", train_metrics), ("Validation", val_metrics)]:
        print(f"{split}: ROC-AUC={metrics['roc_auc']:.6f}, accuracy={metrics['accuracy']:.6f}")
        print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"Tree: depth={model.get_depth()}, leaves={model.get_n_leaves()}")
    print(f"ROC-AUC gap (train - validation): {report['train_minus_validation_auc']:.6f}")
    print(f"Artifacts: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "decision_tree",
        help="Artifact directory; rerunning replaces this baseline's output files.",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
