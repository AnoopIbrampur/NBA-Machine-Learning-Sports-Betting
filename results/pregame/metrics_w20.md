# XGBoost: season-to-date (repo) vs rolling pre-game features (window=20)

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier | Home-win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Repo season-to-date features (baseline) | test | 3386 | 0.6400 | 0.6565 | 0.7580 | 0.7036 | 0.6797 | 0.6349 | 0.2221 | 0.5638 |
| Repo season-to-date features (baseline) | validation | 1681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 | 0.5449 |
| Pre-game rolling-20 + context | test | 3386 | 0.6435 | 0.6509 | 0.7931 | 0.7150 | 0.6749 | 0.6371 | 0.2229 | 0.5638 |
| Pre-game rolling-20 + context | validation | 1681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 | 0.5449 |
| Ablation: pre-game rolling-20 + context + scoring margin | test | 3386 | 0.6480 | 0.6561 | 0.7894 | 0.7166 | 0.6838 | 0.6326 | 0.2209 | 0.5638 |
| Ablation: pre-game rolling-20 + context + scoring margin | validation | 1681 | 0.6466 | 0.6507 | 0.7587 | 0.7006 | 0.7107 | 0.6207 | 0.2157 | 0.5449 |
| Parent paper, in-game H2 (first two quarters) | 10-fold CV | - | 0.720 | 0.736 | 0.775 | 0.754 | 0.783 | - | - | - |
| Parent paper, in-game H3 (first three quarters) | 10-fold CV | - | 0.798 | 0.834 | 0.807 | 0.820 | 0.876 | - | - | - |
| Parent paper, in-game Full game | 10-fold CV | - | 0.933 | 0.938 | 0.939 | 0.939 | 0.982 | - | - | - |

Positive class = home win; threshold 0.5. Parent-paper rows use box scores from the game being predicted, so they are an upper bound rather than a like-for-like comparison.
