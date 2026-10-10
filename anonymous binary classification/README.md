# Entity Resolution: Binary Classification of Hidden Pairs

Binary classification project on anonymous numerical data. Each row describes a pair of hidden observations. The task is to predict the probability that both observations belong to the same hidden entity (`target=1`) or to different ones (`target=0`).

## Data

The dataset contains 512 precomputed anonymous numerical features (`f0000` - `f0511`):
* `train.csv` — Training set.
* `validation.csv` — Hold-out validation set (split at the hidden entity level, providing a more realistic generalization estimate than a random row split).
* `test.csv` — Test set for inference.
* `sample_submission.csv` — Submission format example.

## Final Pipeline Architecture

The final model is an ensemble of 4 components combined via **Rank Pooling** with fixed weights. Final Validation ROC-AUC: **0.9581**.

### Ensemble Components

| # | Model | Input Features | Weight |
|---|-------|----------------|--------|
| 1 | **Full SVM (RBF)** | All 512 features (`sqrt`-transformed, centered, scaled by spread). Params: `C=10`, `γ=1.25*scale` | 0.3115 |
| 2 | **PCA SVM (RBF)** | Same preprocessing as Full SVM, followed by PCA to 255 components. Params: `C=10`, `γ=1.5*scale` | 0.3115 |
| 3 | **Polynomial SVM** | Polynomial kernel (`degree=2`) to capture pairwise feature interactions. | 0.0769 |
| 4 | **LightGBM** | Raw 512 features + 6 row statistics + 64 PCA components + 6 distance features + 27 kNN features (match ratios and distances to k-nearest training rows). Total: 615 features. Params: 800 trees, 31 leaves. | 0.3000 |

### Rank Pooling Logic
1. Each ensemble component is trained strictly on `train.csv`.
2. Predicted probabilities from each component are converted into ranks relative to the out-of-fold (OOF) predictions on the training set (5 folds).
3. The ranks are averaged using the fixed weights specified in the table above.

## Repository Structure

* `final_pipeline.ipynb` — **The only verified and working code**. Reproduces the entire pipeline from data loading to submission generation.
* `experiments/` — Folder containing drafts, alternative ensembles, stacking attempts, and feature selection experiments.
  > ⚠️ **Disclaimer:** The code in `experiments/` is strictly for research purposes. We do not guarantee that all scripts are 100% working, reproducible, or bug-free. Treat them as drafts and exploratory notes.

## How to Run

To reproduce the final result:
1. Ensure `train.csv`, `validation.csv`, `test.csv`, and `sample_submission.csv` are located in the root directory.
2. Open `final_pipeline.ipynb`.
3. Execute **Restart & Run All**.
4. The pipeline will generate `final_submission.csv`.

*Requirements: Python 3.12+, scikit-learn, lightgbm, pandas, numpy.*