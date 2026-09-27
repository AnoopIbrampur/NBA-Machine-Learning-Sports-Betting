# XGBoost: season-to-date (repo) vs rolling pre-game features (window=10)

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier | Home-win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Repo season-to-date features (baseline) | test | 3390 | 0.6451 | 0.6594 | 0.7654 | 0.7085 | 0.6889 | 0.6302 | 0.2199 | 0.5634 |
| Repo season-to-date features (baseline) | validation | 1683 | 0.6928 | 0.6869 | 0.8015 | 0.7398 | 0.7648 | 0.5912 | 0.2020 | 0.5449 |
| Pre-game rolling-10 + context | test | 3390 | 0.6348 | 0.6495 | 0.7644 | 0.7023 | 0.6601 | 0.6460 | 0.2270 | 0.5634 |
| Pre-game rolling-10 + context | validation | 1683 | 0.6346 | 0.6406 | 0.7503 | 0.6911 | 0.6826 | 0.6376 | 0.2235 | 0.5449 |
| Ablation: pre-game rolling-10 + context + scoring margin | test | 3390 | 0.6336 | 0.6469 | 0.7702 | 0.7032 | 0.6665 | 0.6421 | 0.2254 | 0.5634 |
| Ablation: pre-game rolling-10 + context + scoring margin | validation | 1683 | 0.6447 | 0.6473 | 0.7644 | 0.7010 | 0.6950 | 0.6301 | 0.2202 | 0.5449 |
| Parent paper, in-game H2 (first two quarters) | 10-fold CV | - | 0.720 | 0.736 | 0.775 | 0.754 | 0.783 | - | - | - |
| Parent paper, in-game H3 (first three quarters) | 10-fold CV | - | 0.798 | 0.834 | 0.807 | 0.820 | 0.876 | - | - | - |
| Parent paper, in-game Full game | 10-fold CV | - | 0.933 | 0.938 | 0.939 | 0.939 | 0.982 | - | - | - |

Positive class = home win; threshold 0.5. Parent-paper rows use box scores from the game being predicted, so they are an upper bound rather than a like-for-like comparison.
