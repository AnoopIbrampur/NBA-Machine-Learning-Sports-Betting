# XGBoost: season-to-date (repo) vs rolling pre-game features (window=10)

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier | Home-win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Repo season-to-date features (baseline) | test | 3386 | 0.6400 | 0.6565 | 0.7580 | 0.7036 | 0.6797 | 0.6349 | 0.2221 | 0.5638 |
| Repo season-to-date features (baseline) | validation | 1681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 | 0.5449 |
| Pre-game rolling-10 + context | test | 3386 | 0.6353 | 0.6502 | 0.7643 | 0.7026 | 0.6611 | 0.6454 | 0.2267 | 0.5638 |
| Pre-game rolling-10 + context | validation | 1681 | 0.6347 | 0.6409 | 0.7500 | 0.6911 | 0.6829 | 0.6373 | 0.2234 | 0.5449 |
| Ablation: pre-game rolling-10 + context + scoring margin | test | 3386 | 0.6341 | 0.6476 | 0.7700 | 0.7035 | 0.6672 | 0.6417 | 0.2252 | 0.5638 |
| Ablation: pre-game rolling-10 + context + scoring margin | validation | 1681 | 0.6449 | 0.6475 | 0.7642 | 0.7011 | 0.6952 | 0.6300 | 0.2201 | 0.5449 |
| Parent paper, in-game H2 (first two quarters) | 10-fold CV | - | 0.720 | 0.736 | 0.775 | 0.754 | 0.783 | - | - | - |
| Parent paper, in-game H3 (first three quarters) | 10-fold CV | - | 0.798 | 0.834 | 0.807 | 0.820 | 0.876 | - | - | - |
| Parent paper, in-game Full game | 10-fold CV | - | 0.933 | 0.938 | 0.939 | 0.939 | 0.982 | - | - | - |

Positive class = home win; threshold 0.5. Parent-paper rows use box scores from the game being predicted, so they are an upper bound rather than a like-for-like comparison.
