# Week 03 — Hidden Pairs

Predict whether two hidden visual entities match using anonymous numerical features.

- `train.csv`: labeled training data.
- `validation.csv`: labeled validation data; do not combine it with training data.
- `test.csv`: unlabeled competition rows.
- `sample_submission.csv`: required submission structure.

Use `row_id` only to align predictions. The evaluation metric is ROC-AUC, so submit continuous probabilities in the `target` column.
