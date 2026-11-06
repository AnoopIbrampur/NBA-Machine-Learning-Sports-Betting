# SHAP rankings: pre-game XGBoost (window=10, test split, n=3390)

## All features (mean |SHAP|)

|                      |   rank |   mean_abs_shap |
|:---------------------|-------:|----------------:|
| home_win_pct_at_home |      1 |          0.1926 |
| 2P%_diff             |      2 |          0.1622 |
| away_win_pct_on_road |      3 |          0.1336 |
| win_pct_home         |      4 |          0.1276 |
| win_pct_away         |      5 |          0.1086 |
| FG%_diff             |      6 |          0.0512 |
| rest_days_away       |      7 |          0.0486 |
| DRB_diff             |      8 |          0.0482 |
| BLK_diff             |      9 |          0.0417 |
| FT%_diff             |     10 |          0.0408 |
| STL_diff             |     11 |          0.0385 |
| TOV_diff             |     12 |          0.0344 |
| 3P%_diff             |     13 |          0.0297 |
| AST_diff             |     14 |          0.0162 |
| ORB_diff             |     15 |          0.0153 |
| PF_diff              |     16 |          0.0147 |
| rest_days_home       |     17 |          0.0138 |
| b2b_away             |     18 |          0.0039 |
| b2b_home             |     19 |          0.0027 |

## The paper's 11 categories only, vs. parent paper Table 10

|   Rank | Pre-game (ours)   | Paper H2   | Paper H3   | Paper Full game   |
|-------:|:------------------|:-----------|:-----------|:------------------|
|      1 | 2P%               | FG%        | FG%        | FG%               |
|      2 | FG%               | DRB        | TOV        | 3P%               |
|      3 | DRB               | AST        | 3P%        | TOV               |
|      4 | BLK               | TOV        | DRB        | DRB               |
|      5 | FT%               | FT%        | ORB        | ORB               |
|      6 | STL               | PF         | FT%        | PF                |
|      7 | TOV               | STL        | PF         | FT%               |
|      8 | 3P%               | 3P%        | AST        | 2P%               |
|      9 | AST               | ORB        | STL        | AST               |
|     10 | ORB               | 2P%        | 2P%        | STL               |
|     11 | PF                | BLK        | BLK        | BLK               |

Spearman rank correlation with the paper's order: H2 0.07, H3 -0.06, Full game -0.01
