# Leakage audit of the repo's season-to-date features

Share of team rows consistent with features that exclude / include the game being predicted:

|   season |   excludes_current_game |   includes_current_game |
|---------:|------------------------:|------------------------:|
|     2012 |                   1     |                   0.545 |
|     2013 |                   1     |                   0.544 |
|     2014 |                   1     |                   0.568 |
|     2015 |                   1     |                   0.538 |
|     2016 |                   1     |                   0.528 |
|     2017 |                   1     |                   0.545 |
|     2018 |                   1     |                   0.53  |
|     2019 |                   1     |                   0.535 |
|     2020 |                   1     |                   0.536 |
|     2021 |                   1     |                   0.542 |
|     2022 |                   1     |                   0.531 |
|     2023 |                   0.998 |                   0.551 |
|     2024 |                   1     |                   0.557 |
|     2025 |                   1     |                   0.536 |

Seasons whose features include the current game: none.
Only the baseline uses these features; the pre-game models are built from game logs.

## Saved models re-scored on never-affected vs regenerated seasons

| Window | Model | Subset | n | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| 10 | Repo season-to-date (baseline) | test, never-affected seasons | 3234 | 0.6401 | 0.6813 | 0.6345 | 0.2219 |
| 10 | Repo season-to-date (baseline) | test, regenerated seasons | 152 | 0.6382 | 0.6444 | 0.6450 | 0.2266 |
| 10 | Repo season-to-date (baseline) | validation (all regenerated) | 1681 | 0.6556 | 0.7089 | 0.6221 | 0.2165 |
| 10 | Pre-game + context | test, never-affected seasons | 3234 | 0.6342 | 0.6628 | 0.6451 | 0.2266 |
| 10 | Pre-game + context | test, regenerated seasons | 152 | 0.6579 | 0.6339 | 0.6504 | 0.2291 |
| 10 | Pre-game + context | validation (all regenerated) | 1681 | 0.6347 | 0.6829 | 0.6373 | 0.2234 |
| 10 | Ablation: pre-game + context + margin | test, never-affected seasons | 3234 | 0.6336 | 0.6678 | 0.6420 | 0.2253 |
| 10 | Ablation: pre-game + context + margin | test, regenerated seasons | 152 | 0.6447 | 0.6595 | 0.6355 | 0.2223 |
| 10 | Ablation: pre-game + context + margin | validation (all regenerated) | 1681 | 0.6449 | 0.6952 | 0.6300 | 0.2201 |
| 20 | Repo season-to-date (baseline) | test, never-affected seasons | 3234 | 0.6401 | 0.6813 | 0.6345 | 0.2219 |
| 20 | Repo season-to-date (baseline) | test, regenerated seasons | 152 | 0.6382 | 0.6444 | 0.6450 | 0.2266 |
| 20 | Repo season-to-date (baseline) | validation (all regenerated) | 1681 | 0.6556 | 0.7089 | 0.6221 | 0.2165 |
| 20 | Pre-game + context | test, never-affected seasons | 3234 | 0.6432 | 0.6774 | 0.6365 | 0.2226 |
| 20 | Pre-game + context | test, regenerated seasons | 152 | 0.6513 | 0.6298 | 0.6507 | 0.2293 |
| 20 | Pre-game + context | validation (all regenerated) | 1681 | 0.6419 | 0.6987 | 0.6275 | 0.2189 |
| 20 | Ablation: pre-game + context + margin | test, never-affected seasons | 3234 | 0.6481 | 0.6852 | 0.6325 | 0.2208 |
| 20 | Ablation: pre-game + context + margin | test, regenerated seasons | 152 | 0.6447 | 0.6608 | 0.6344 | 0.2218 |
| 20 | Ablation: pre-game + context + margin | validation (all regenerated) | 1681 | 0.6466 | 0.7107 | 0.6207 | 0.2157 |

## Difference vs baseline (paired bootstrap, 95% CI)

2000 resamples of the same games for both models. An interval containing 0 means the difference is not distinguishable from noise.

| Window | Model | Games | Accuracy diff CI | AUC diff CI |
|---|---|---|---|---|
| 10 | Pre-game + context | test, never-affected seasons | [-0.0213, +0.0099] | [-0.0328, -0.0046] |
| 10 | Pre-game + context | test (all) | [-0.0201, +0.0106] | [-0.0321, -0.0058] |
| 10 | Pre-game + context | validation (all) | [-0.0416, -0.0018] | [-0.0430, -0.0090] |
| 10 | Ablation: pre-game + context + margin | test, never-affected seasons | [-0.0216, +0.0080] | [-0.0270, -0.0005] |
| 10 | Ablation: pre-game + context + margin | test (all) | [-0.0207, +0.0080] | [-0.0249, -0.0004] |
| 10 | Ablation: pre-game + context + margin | validation (all) | [-0.0315, +0.0095] | [-0.0300, +0.0030] |
| 20 | Pre-game + context | test, never-affected seasons | [-0.0108, +0.0167] | [-0.0146, +0.0065] |
| 20 | Pre-game + context | test (all) | [-0.0098, +0.0165] | [-0.0152, +0.0057] |
| 20 | Pre-game + context | validation (all) | [-0.0327, +0.0030] | [-0.0232, +0.0021] |
| 20 | Ablation: pre-game + context + margin | test, never-affected seasons | [-0.0050, +0.0207] | [-0.0061, +0.0136] |
| 20 | Ablation: pre-game + context + margin | test (all) | [-0.0047, +0.0210] | [-0.0055, +0.0136] |
| 20 | Ablation: pre-game + context + margin | validation (all) | [-0.0268, +0.0072] | [-0.0097, +0.0136] |
