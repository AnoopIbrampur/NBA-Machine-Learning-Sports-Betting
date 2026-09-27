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
|     2024 |                   0.558 |                   1     |
|     2025 |                   0.536 |                   1     |

Affected seasons (features include the current game): 2024-25, 2025-26.
Only the baseline uses these features; the pre-game models are built from game logs.

## Saved models re-scored on leak-free vs affected games

| Window | Model | Subset | n | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| 10 | Repo season-to-date (baseline) | test, leak-free seasons | 3234 | 0.6401 | 0.6813 | 0.6345 | 0.2219 |
| 10 | Repo season-to-date (baseline) | test, affected seasons | 156 | 0.7500 | 0.8453 | 0.5424 | 0.1787 |
| 10 | Repo season-to-date (baseline) | validation (all affected) | 1683 | 0.6928 | 0.7648 | 0.5912 | 0.2020 |
| 10 | Pre-game + context | test, leak-free seasons | 3234 | 0.6342 | 0.6628 | 0.6451 | 0.2266 |
| 10 | Pre-game + context | test, affected seasons | 156 | 0.6474 | 0.6118 | 0.6633 | 0.2348 |
| 10 | Pre-game + context | validation (all affected) | 1683 | 0.6346 | 0.6826 | 0.6376 | 0.2235 |
| 10 | Ablation: pre-game + context + margin | test, leak-free seasons | 3234 | 0.6336 | 0.6678 | 0.6420 | 0.2253 |
| 10 | Ablation: pre-game + context + margin | test, affected seasons | 156 | 0.6346 | 0.6438 | 0.6455 | 0.2269 |
| 10 | Ablation: pre-game + context + margin | validation (all affected) | 1683 | 0.6447 | 0.6950 | 0.6301 | 0.2202 |
| 20 | Repo season-to-date (baseline) | test, leak-free seasons | 3234 | 0.6401 | 0.6813 | 0.6345 | 0.2219 |
| 20 | Repo season-to-date (baseline) | test, affected seasons | 156 | 0.7500 | 0.8453 | 0.5424 | 0.1787 |
| 20 | Repo season-to-date (baseline) | validation (all affected) | 1683 | 0.6928 | 0.7648 | 0.5912 | 0.2020 |
| 20 | Pre-game + context | test, leak-free seasons | 3234 | 0.6432 | 0.6774 | 0.6365 | 0.2226 |
| 20 | Pre-game + context | test, affected seasons | 156 | 0.6410 | 0.6152 | 0.6585 | 0.2330 |
| 20 | Pre-game + context | validation (all affected) | 1683 | 0.6417 | 0.6985 | 0.6277 | 0.2189 |
| 20 | Ablation: pre-game + context + margin | test, leak-free seasons | 3234 | 0.6481 | 0.6852 | 0.6325 | 0.2208 |
| 20 | Ablation: pre-game + context + margin | test, affected seasons | 156 | 0.6346 | 0.6509 | 0.6407 | 0.2249 |
| 20 | Ablation: pre-game + context + margin | validation (all affected) | 1683 | 0.6465 | 0.7105 | 0.6208 | 0.2158 |

## Difference vs baseline on leak-free test games (paired bootstrap, 95% CI)

2000 resamples of the same games for both models. An interval containing 0 means the difference is not distinguishable from noise.

| Window | Model | Accuracy diff CI | AUC diff CI |
|---|---|---|---|
| 10 | Pre-game + context | [-0.0213, +0.0099] | [-0.0328, -0.0046] |
| 10 | Ablation: pre-game + context + margin | [-0.0216, +0.0080] | [-0.0270, -0.0005] |
| 20 | Pre-game + context | [-0.0108, +0.0167] | [-0.0146, +0.0065] |
| 20 | Ablation: pre-game + context + margin | [-0.0050, +0.0207] | [-0.0061, +0.0136] |
