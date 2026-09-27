# XGBoost: season-to-date (repo) vs rolling pre-game features (window=20)

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier | Home-win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Repo season-to-date features (baseline) | test | 3390 | 0.6451 | 0.6594 | 0.7654 | 0.7085 | 0.6889 | 0.6302 | 0.2199 | 0.5634 |
| Repo season-to-date features (baseline) | validation | 1683 | 0.6928 | 0.6869 | 0.8015 | 0.7398 | 0.7648 | 0.5912 | 0.2020 | 0.5449 |
| Pre-game rolling-20 + context | test | 3390 | 0.6431 | 0.6502 | 0.7932 | 0.7146 | 0.6743 | 0.6375 | 0.2231 | 0.5634 |
| Pre-game rolling-20 + context | validation | 1683 | 0.6417 | 0.6451 | 0.7612 | 0.6983 | 0.6985 | 0.6277 | 0.2189 | 0.5449 |
| Ablation: pre-game rolling-20 + context + scoring margin | test | 3390 | 0.6475 | 0.6554 | 0.7895 | 0.7162 | 0.6834 | 0.6329 | 0.2210 | 0.5634 |
| Ablation: pre-game rolling-20 + context + scoring margin | validation | 1683 | 0.6465 | 0.6505 | 0.7590 | 0.7006 | 0.7105 | 0.6208 | 0.2158 | 0.5449 |
| Parent paper, in-game H2 (first two quarters) | 10-fold CV | - | 0.720 | 0.736 | 0.775 | 0.754 | 0.783 | - | - | - |
| Parent paper, in-game H3 (first three quarters) | 10-fold CV | - | 0.798 | 0.834 | 0.807 | 0.820 | 0.876 | - | - | - |
| Parent paper, in-game Full game | 10-fold CV | - | 0.933 | 0.938 | 0.939 | 0.939 | 0.982 | - | - | - |

Positive class = home win; threshold 0.5. Parent-paper rows use box scores from the game being predicted, so they are an upper bound rather than a like-for-like comparison.
