"""Describe train.csv only, without fitting models or changing model pipelines.

Run: .venv/bin/python training_analysis.py
Target associations are exploratory training statistics, not validation scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = [f"f{i:04d}" for i in range(512)]


def analyse(data_dir: Path, output_dir: Path) -> dict:
    path = Path(data_dir) / "train.csv"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, dtype={"row_id": "string"})
    if set(frame.columns) != {"row_id", "target", *FEATURES}:
        raise ValueError("Unexpected training schema.")
    X, y = frame[FEATURES], frame["target"]
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("Target must be binary with both classes present.")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in FEATURES):
        raise ValueError("Nonnumeric training features found.")
    missing = int(X.isna().sum().sum())
    nonfinite = int((~np.isfinite(X.to_numpy())).sum())
    if missing or nonfinite:
        raise ValueError(f"Found {missing} missing and {nonfinite} nonfinite values.")

    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr_flags = X.lt(q1 - 1.5 * (q3 - q1)) | X.gt(q3 + 1.5 * (q3 - q1))
    auc = pd.Series({name: roc_auc_score(y, X[name]) for name in FEATURES})
    feature_summary = pd.DataFrame({
        "minimum": X.min(), "maximum": X.max(), "mean": X.mean(),
        "std": X.std(), "median": X.median(), "skewness": X.skew(),
        "unique_values": X.nunique(), "zero_fraction": X.eq(0).mean(),
        "largest_value_frequency": [X[c].value_counts(normalize=True).iloc[0] for c in FEATURES],
        "iqr_flag_fraction": iqr_flags.mean(),
        "nonmatch_mean": X.loc[y == 0].mean(), "match_mean": X.loc[y == 1].mean(),
        "auc_higher_value_predicts_match": auc,
        "auc_best_direction_train_only": np.maximum(auc, 1 - auc),
    })
    feature_summary.index.name = "feature"

    correlations = X.corr().to_numpy()
    first, second = np.triu_indices(len(FEATURES), k=1)
    pair_correlations = pd.DataFrame({
        "feature_1": np.array(FEATURES)[first],
        "feature_2": np.array(FEATURES)[second],
        "pearson_r": correlations[first, second],
    })
    pair_correlations["abs_r"] = pair_correlations.pearson_r.abs()
    pair_correlations = pair_correlations.sort_values("abs_r", ascending=False)

    row_summary = pd.DataFrame({
        "mean": X.mean(axis=1), "median": X.median(axis=1),
        "std": X.std(axis=1), "l2_norm": np.sqrt(X.pow(2).sum(axis=1)),
        "maximum": X.max(axis=1), "zero_fraction": X.eq(0).mean(axis=1),
    })
    aggregate_auc = {
        name: float(roc_auc_score(y, row_summary[name])) for name in row_summary
    }
    summary = {
        "source": "train.csv only",
        "source_sha256": source_hash,
        "rows": len(frame), "features": len(FEATURES),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_feature_values": missing, "nonfinite_feature_values": nonfinite,
        "negative_feature_values": int(X.lt(0).sum().sum()),
        "missing_or_blank_ids": int((frame.row_id.isna() | frame.row_id.str.strip().eq("")).sum()),
        "duplicate_ids": int(frame.row_id.duplicated().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "duplicate_feature_rows_after_rounding_6_decimals": int(X.round(6).duplicated().sum()),
        "duplicate_feature_columns": int(X.T.duplicated().sum()),
        "constant_features": int(X.nunique().eq(1).sum()),
        "all_zero_rows": int(X.eq(0).all(axis=1).sum()),
        "zero_cell_fraction": float(X.eq(0).to_numpy().mean()),
        "max_feature_zero_fraction": float(X.eq(0).mean().max()),
        "max_feature_single_value_fraction": float(feature_summary.largest_value_frequency.max()),
        "feature_std_range": [float(X.std().min()), float(X.std().max())],
        "feature_skewness_range": [float(X.skew().min()), float(X.skew().max())],
        "max_absolute_pearson_correlation": float(pair_correlations.abs_r.max()),
        "feature_pairs_abs_pearson_above_0_8": int(pair_correlations.abs_r.gt(0.8).sum()),
        "iqr_flagged_cell_fraction": float(iqr_flags.to_numpy().mean()),
        "rows_with_any_iqr_flag": int(iqr_flags.any(axis=1).sum()),
        "features_auc_below_0_5": int(auc.lt(0.5).sum()),
        "best_single_feature": str(feature_summary.auc_best_direction_train_only.idxmax()),
        "best_single_feature_oriented_training_auc": float(feature_summary.auc_best_direction_train_only.max()),
        "aggregate_means_by_class": {
            str(label): {name: float(value) for name, value in values.items()}
            for label, values in row_summary.groupby(y).mean().iterrows()
        },
        "aggregate_auc_higher_value_predicts_match": aggregate_auc,
        "aggregate_auc_best_direction_train_only": {
            name: max(value, 1 - value) for name, value in aggregate_auc.items()
        },
        "limitations": [
            "Training-only target associations are descriptive, not generalisation estimates.",
            "Direction and feature rankings use training labels; no significance testing performed.",
            "Low pairwise Pearson correlation does not exclude multivariate or nonlinear redundancy.",
            "IQR flags identify tail values, not confirmed errors; no rows were removed.",
            "No hidden entity IDs are available to establish row independence.",
            "This audit does not measure label noise or train-validation distribution shift.",
        ],
    }

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    axes[0, 0].bar(["Different (0)", "Same (1)"], [int((y == 0).sum()), int((y == 1).sum())], color=["#ce7a3d", "#367db5"])
    axes[0, 0].set(title="Class balance", ylabel="Training rows", ylim=(0, 1500))
    for i, label in enumerate([0, 1]):
        axes[0, 0].text(i, int((y == label).sum()) + 30, str(int((y == label).sum())), ha="center")
    bins = np.linspace(row_summary["mean"].min(), row_summary["mean"].max(), 36)
    for label, color, name in [(0, "#ce7a3d", "Different (0)"), (1, "#367db5", "Same (1)")]:
        axes[0, 1].hist(row_summary.loc[y == label, "mean"], bins=bins, alpha=0.55, label=name, color=color)
    axes[0, 1].set(title="Average feature value per row, by class", xlabel="Mean of the 512 features", ylabel="Training rows")
    axes[0, 1].legend()
    axes[1, 0].hist(pair_correlations.pearson_r, bins=45, color="#367db5")
    axes[1, 0].set(title="All distinct feature pairs", xlabel="Pearson correlation", ylabel="Number of feature pairs")
    axes[1, 1].hist(iqr_flags.sum(axis=1), bins=30, color="#ce7a3d")
    axes[1, 1].set(title="Tail-value flags per row (not confirmed errors)", xlabel="Features outside 1.5 × IQR fences", ylabel="Training rows")
    figure.suptitle("Training-data audit — no validation/test data or model fitting", fontsize=15)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_summary.to_csv(output_dir / "feature_summary.csv")
    pair_correlations.head(30).to_csv(output_dir / "top_correlations.csv", index=False)
    row_summary.assign(row_id=frame.row_id, target=y).to_csv(output_dir / "row_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure.savefig(output_dir / "training_overview.png", dpi=150)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Training file changed during analysis.")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "training_analysis")
    args = parser.parse_args()
    analyse(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
