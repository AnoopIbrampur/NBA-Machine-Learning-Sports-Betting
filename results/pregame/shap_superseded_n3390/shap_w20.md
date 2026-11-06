# SHAP rankings: pre-game XGBoost (window=20, test split, n=3390)

## All features (mean |SHAP|)

|                      |   rank |   mean_abs_shap |
|:---------------------|-------:|----------------:|
| win_pct_home         |      1 |          0.1969 |
| win_pct_away         |      2 |          0.1613 |
| home_win_pct_at_home |      3 |          0.1425 |
| away_win_pct_on_road |      4 |          0.1136 |
| 2P%_diff             |      5 |          0.113  |
| FG%_diff             |      6 |          0.064  |
| DRB_diff             |      7 |          0.0476 |
| BLK_diff             |      8 |          0.0357 |
| TOV_diff             |      9 |          0.0332 |
| STL_diff             |     10 |          0.032  |
| rest_days_away       |     11 |          0.0304 |
| FT%_diff             |     12 |          0.0271 |
| 3P%_diff             |     13 |          0.0249 |
| b2b_away             |     14 |          0.014  |
| PF_diff              |     15 |          0.0116 |
| AST_diff             |     16 |          0.0089 |
| ORB_diff             |     17 |          0.0063 |
| rest_days_home       |     18 |          0.0042 |
| b2b_home             |     19 |          0.0009 |

## The paper's 11 categories only, vs. parent paper Table 10

|   Rank | Pre-game (ours)   | Paper H2   | Paper H3   | Paper Full game   |
|-------:|:------------------|:-----------|:-----------|:------------------|
|      1 | 2P%               | FG%        | FG%        | FG%               |
|      2 | FG%               | DRB        | TOV        | 3P%               |
|      3 | DRB               | AST        | 3P%        | TOV               |
|      4 | BLK               | TOV        | DRB        | DRB               |
|      5 | TOV               | FT%        | ORB        | ORB               |
|      6 | STL               | PF         | FT%        | PF                |
|      7 | FT%               | STL        | PF         | FT%               |
|      8 | 3P%               | 3P%        | AST        | 2P%               |
|      9 | PF                | ORB        | STL        | AST               |
|     10 | AST               | 2P%        | 2P%        | STL               |
|     11 | ORB               | BLK        | BLK        | BLK               |

Spearman rank correlation with the paper's order: H2 0.09, H3 0.00, Full game 0.08
