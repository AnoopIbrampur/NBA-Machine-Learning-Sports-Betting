# DS 340W findings: pre-game NBA outcome prediction

Single reference for the course report and check-ins. Every number here comes from a committed
file under `results/pregame/`; regenerate with the commands at the end.

## In one paragraph

This fork's existing model (the baseline) was never an in-game model: it predicts from each team's
**season-to-date averages**. So the question is not "can we convert in-game to pre-game" but
"does **recent form (last N games) plus schedule context** beat season-to-date averages?".
With a 20-game window, the pre-game model **ties** the baseline on leak-free test games. A 10-game
window is significantly worse on AUC. A rolling scoring-margin feature, kept as an **ablation
only**, puts the window-20 model slightly ahead (0.6481 vs 0.6401 accuracy, 0.6852 vs 0.6813 AUC),
but the gain is **within noise** (95% CIs include 0).

SHAP shows recent win-record features outranking all 11 box-score stats. This is the same pattern
parent paper 2 reports: engineered context features rank above box-score stats. We also found that
the baseline's features **leak the outcome** in the 2024-25 and 2025-26 seasons, which inflates its
validation-split numbers.

## 1. What the repo's baseline actually is

- **Source:** `src/Process-Data/Get_Data.py` downloads stats.nba.com `leaguedashteamstats` with
  `PerMode=PerGame` and `DateTo=<game date>`. Each game row (`Data/dataset.sqlite`,
  `dataset_2012-26`) therefore holds both teams' **season-to-date per-game averages and ranks**:
  FG%, rebounds, assists, points, plus-minus, W/L record, and so on. Rest days are also included.
- **Not in-game:** these are team averages over earlier games, not the box score of the game being
  predicted. The in-game → pre-game framing does not apply to this repo.
- **Leakage from 2024-25 on:**
  - For 2012-13 through 2023-24, each row's stats exclude the game being predicted (≥99.8% of rows).
  - For 2024-25 and 2025-26 they include it (100%), so the baseline sees the result.
  - Details: `results/pregame/leakage_audit.md`, `docs/dataset-tables.md`.

## 2. What the pre-game feature set adds

Built from per-game team box scores (stats.nba.com `leaguegamelog`, 2012-13 → 2025-26, regular
season + play-in + playoffs). Every feature for a game uses only that team's games **strictly
before** it, within the same season. Unit tests check this, and they fail if leakage is introduced
(`Tests/test_pregame_features.py`).

| Group | Features |
|---|---|
| Parent paper 1's 11 stats, rolling last N, home − away | FG%, 2P%, 3P%, FT%, ORB, DRB, AST, STL, BLK, TOV, PF |
| Recent form | each team's win % over its last N games |
| Venue form | home team's win % in its last N home games; away team's in its last N road games |
| Schedule | rest days since each team's last game; back-to-back flag for each team |
| **Ablation only** | scoring margin (mean plus-minus over last N games), home − away |

The 11-stat list is the final feature set in parent paper 1's SHAP model (PLOS ONE 2024, Table 10).
Here the stats are computed as rolling pre-game averages instead of in-game totals.

### 2a. Decision: scoring margin is an ablation only

The official pre-game feature set is the 11 stats plus the context features above. Rolling scoring
margin is **not** part of it and is reported only as a labelled ablation (`pregame_margin` in
`src/Pregame/train_xgb.py`). There are two reasons:

1. **Its gain is small next to the window change.**
   - On leak-free test games, going from a 10-game to a 20-game window adds +0.009 accuracy and
     +0.015 AUC.
   - Adding margin on top of window 20 adds only +0.005 accuracy and +0.008 AUC, and that gain is
     not statistically distinguishable from zero.
2. **It would break comparability with parent paper 1.** The point of using that paper's stated
   11-stat list is that our SHAP rankings can be compared with its rankings stat for stat. Scoring
   margin is not on that list.

The ablation rows stay in every results table so the effect is visible, labelled as ablation.

### 2b. Relation to parent paper 2 (Alves & Barbosa, Computation 2025)

J. M. Alves and R. S. Barbosa, "Machine Learning for Basketball Game Outcomes: NBA and WNBA
Leagues," *Computation* 13(10):230, 2025. The feature and SHAP details below are from the project
team's reading of the paper.

- **Same style of features.** Paper 2 builds pre-game features from:
  - last-N-game rolling averages, with the window deliberately varied rather than fixed;
  - rest days between matches;
  - home/away status and next-opponent information;
  - Elo ratings (starting at 1500, k between 16 and 32).

  Like our feature set, it describes each team's recent form and schedule before tip-off. That
  makes it a fair reference point for this project. Our window comparison (10 vs 20) is in the
  same spirit as its varied window.
- **Same SHAP pattern.** In paper 2's SHAP analysis, engineered context features (`home_next`,
  `team_elo_5_y`, `team_elo`) ranked above every box-score stat in every model it tested. We see the
  same structure: recent win-record features rank above all 11 box-score stats (§5).
- **What is not claimed.** This is a structural parallel (engineered context > box score), not a
  claim that the feature sets match.
  - We use no Elo ratings and no next-opponent features.
  - Our context features are rolling win %, venue-specific win % and rest days.
  - Paper 2's data, seasons and evaluation protocol also differ from ours. Its abstract reports
    65.50% accuracy for the NBA, which is in the same range as our 0.643–0.648 but not directly
    comparable.

## 3. Results

**Common setup**
- **Model:** XGBoost for every row.
- **Games:** the same 16,946 (99.9% of the split; 23 games missing from the game logs are dropped
  from every model).
- **Split:** the fixed chronological split in `Data/splits/split_keys.csv`: train 2012-11 → 2022-01,
  test 2022-01 → 2024-11-13, validation 2024-11-13 → 2026-01-07.
- **Tuning:** 40-trial random search scored by walk-forward CV on train only; test and validation
  are never used for tuning.
- **Metrics:** positive class = home win, threshold 0.5. Home teams win 56.3% of test games and
  54.5% of validation games.

### 3a. Full split

| Model | Split | Acc | Prec | Recall | F1 | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 0.6451 | 0.6594 | 0.7654 | 0.7085 | 0.6889 | 0.6302 | 0.2199 |
| Pre-game, window=10 | test | 0.6348 | 0.6495 | 0.7644 | 0.7023 | 0.6601 | 0.6460 | 0.2270 |
| Pre-game, window=10 + margin (ablation) | test | 0.6336 | 0.6469 | 0.7702 | 0.7032 | 0.6665 | 0.6421 | 0.2254 |
| **Pre-game, window=20** | test | 0.6431 | 0.6502 | 0.7932 | 0.7146 | 0.6743 | 0.6375 | 0.2231 |
| Pre-game, window=20 + margin (ablation) | test | 0.6475 | 0.6554 | 0.7895 | 0.7162 | 0.6834 | 0.6329 | 0.2210 |
| Baseline: repo season-to-date ⚠ leaky | validation | 0.6928 | 0.6869 | 0.8015 | 0.7398 | 0.7648 | 0.5912 | 0.2020 |
| Pre-game, window=10 | validation | 0.6346 | 0.6406 | 0.7503 | 0.6911 | 0.6826 | 0.6376 | 0.2235 |
| Pre-game, window=10 + margin (ablation) | validation | 0.6447 | 0.6473 | 0.7644 | 0.7010 | 0.6950 | 0.6301 | 0.2202 |
| **Pre-game, window=20** | validation | 0.6417 | 0.6451 | 0.7612 | 0.6983 | 0.6985 | 0.6277 | 0.2189 |
| Pre-game, window=20 + margin (ablation) | validation | 0.6465 | 0.6505 | 0.7590 | 0.7006 | 0.7105 | 0.6208 | 0.2158 |

⚠ The baseline's validation row is inflated: every validation game is in the leaky 2024-26
seasons. Its test row is slightly inflated too, because 156 of its 3,390 games are from 2024-25.

The window=10 rows are the numbers reported at the previous check-in and are unchanged. Window=20
is now the primary configuration.

### 3b. Like-for-like: leak-free test games only (n = 3,234, seasons 2021-22 → 2023-24)

| Model | Acc | AUC | Log loss | Brier | Acc diff vs baseline, 95% CI | AUC diff vs baseline, 95% CI |
|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | 0.6401 | 0.6813 | 0.6345 | 0.2219 | — | — |
| Pre-game, window=10 | 0.6342 | 0.6628 | 0.6451 | 0.2266 | [−0.021, +0.010] | **[−0.033, −0.005]** |
| Pre-game, window=10 + margin | 0.6336 | 0.6678 | 0.6420 | 0.2253 | [−0.022, +0.008] | **[−0.027, −0.001]** |
| **Pre-game, window=20** | 0.6432 | 0.6774 | 0.6365 | 0.2226 | [−0.011, +0.017] | [−0.015, +0.007] |
| Pre-game, window=20 + margin | 0.6481 | 0.6852 | 0.6325 | 0.2208 | [−0.005, +0.021] | [−0.006, +0.014] |

The CIs come from a paired bootstrap (2,000 resamples of the same games for both models). Bold
intervals exclude 0.

**Reading the results**
- **Window=10 is worse than the baseline.** Its AUC shortfall is statistically clear.
- **Window=20 matches the baseline.** It is slightly better on accuracy and slightly worse on AUC,
  and neither difference is distinguishable from noise.
- **Scoring margin (ablation only, §2a) helps a little.** It lifts window=20 by about +0.005
  accuracy and +0.008 AUC on leak-free games and moves it ahead of the baseline on every metric.
  The lead is not statistically significant.
- **Margin is not the missing piece.** It narrows the gap to the baseline, but the gap was mostly
  window length (10 → 20) plus noise.
- **Behind the full-split validation gap:** the baseline's large lead there (0.765 vs ~0.70 AUC)
  comes from leakage, not from better features.

## 4. Parent paper 1 (PLOS ONE 2024) — corrected numbers

XGBoost results as printed in the paper's Tables 7–9:

| Period | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| First two quarters (H2) | 0.720 | 0.736 | 0.775 | 0.754 | 0.783 |
| First three quarters (H3) | 0.798 | 0.834 | 0.807 | 0.820 | 0.876 |
| Full game | 0.933 | 0.938 | 0.939 | 0.939 | 0.982 |

These use box scores **from the game being predicted** and are evaluated with 10-fold
cross-validation, not a chronological hold-out. They are an upper bound on what in-game
information gives, not a like-for-like target for a pre-game model. Earlier notes listed different
values (0.769/0.846, 0.818/0.897, 0.902/0.964); those do not appear in the paper.

## 5. SHAP: pre-game model vs parent paper 1

Both official pre-game models (11 stats + context, no scoring margin) were explained on the test
split (n = 3,390), ranking features by mean |SHAP|. Window=20 is the primary model. The window=10
results are from the previous check-in and unchanged. Files: `results/pregame/shap_w20.md` and
`shap_w10.md`, with bar and beeswarm plots for each window.

### 5a. All features

| Rank | Window=20 (primary) | Window=10 |
|---|---|---|
| 1 | win_pct_home | home_win_pct_at_home |
| 2 | win_pct_away | 2P%_diff |
| 3 | home_win_pct_at_home | away_win_pct_on_road |
| 4 | away_win_pct_on_road | win_pct_home |
| 5 | 2P%_diff | win_pct_away |
| 6 | FG%_diff | FG%_diff |
| 7 | DRB_diff | rest_days_away |
| 8 | BLK_diff | DRB_diff |
| 9 | TOV_diff | BLK_diff |
| 10 | STL_diff | FT%_diff |
| 11 | rest_days_away | STL_diff |
| 12 | FT%_diff | TOV_diff |
| 13 | 3P%_diff | 3P%_diff |
| 14 | b2b_away | AST_diff |
| 15 | PF_diff | ORB_diff |
| 16 | AST_diff | PF_diff |
| 17 | ORB_diff | rest_days_home |
| 18 | rest_days_home | b2b_away |
| 19 | b2b_home | b2b_home |

- **Recent win record dominates in both windows.** At window=20 the four win-record features
  (overall and venue-specific win %, for each team) take ranks 1–4, ahead of every box-score stat.
  At window=10, 2P% diff splits them at #2.
- **Win record's share grows with the window.** It is 58% of total |SHAP| at window=20, up from 50%
  at window=10. The 11 box-score stats' share falls from 44% to 38%.
- **Schedule effects point the right way but are small.** Short rest for the away team favors the
  home team. Back-to-back flags add little beyond rest days.

### 5b. The paper's 11 stats only, vs parent paper 1 (Table 10)

| Rank | Window=20 (primary) | Window=10 | Paper H2 | Paper H3 | Paper full game |
|---|---|---|---|---|---|
| 1 | 2P% | 2P% | FG% | FG% | FG% |
| 2 | FG% | FG% | DRB | TOV | 3P% |
| 3 | DRB | DRB | AST | 3P% | TOV |
| 4 | BLK | BLK | TOV | DRB | DRB |
| 5 | TOV | FT% | FT% | ORB | ORB |
| 6 | STL | STL | PF | FT% | PF |
| 7 | FT% | TOV | STL | PF | FT% |
| 8 | 3P% | 3P% | 3P% | AST | 2P% |
| 9 | PF | AST | ORB | STL | AST |
| 10 | AST | ORB | 2P% | 2P% | STL |
| 11 | ORB | PF | BLK | BLK | BLK |

**Window=10 vs window=20: the ranking barely changes.**
- Spearman rank correlation between the two windows is 0.92 over all 19 features and 0.94 over the
  11 stats.
- The top five features are the same set, and the top four box-score stats (2P%, FG%, DRB, BLK)
  are identical and in the same order.
- The only movement is within the win-record group and a few positions in the lower half (for
  example TOV 7th → 5th and PF 11th → 9th among the 11 stats).

**Our ranking vs parent paper 1:**
- **What carries over:** shooting efficiency and defensive rebounding (FG%/2P%, DRB) stay near the
  top in both windows, as in the paper.
- **What doesn't:** blocks rise from last to 4th in both windows, and 3P% falls to 8th.
- **Overall agreement is about zero.** Spearman rank correlation with the paper's order:

  | | H2 | H3 | Full game |
  |---|---|---|---|
  | Window=20 | 0.09 | 0.00 | 0.08 |
  | Window=10 | 0.07 | −0.06 | −0.01 |

  What drives winning within a game is not what best predicts a game in advance.

### 5c. Link to parent paper 2

Engineered context features outrank every box-score stat here, as they do in parent paper 2's
SHAP analysis (§2b). This holds for both windows. The parallel is in structure: the two papers'
specific context features differ.

## 6. Data provenance and housekeeping

- **Dataset tables:** `dataset_2012-24_new` and `dataset_2012-26` are the same data through 2023-24;
  `dataset_2012-26` adds 2024-25 and 2025-26. See `docs/dataset-tables.md`.
- **Split:** valid for the pre-game models. For the baseline, compare on leak-free games (§3b).
- **Earlier LR baseline** (0.6409 acc / 0.6324 log loss): not reproducible, and not caused by
  package versions. It is also scored on a different test set (the LR script's own 90/10 split).
  See `docs/lr-reproducibility.md`.
- **Repo's own XGBoost protocol** (after the scikit-learn fix): 0.6977 test accuracy on the last 10%
  of `dataset_2012-26`. That window is inside the leaky 2024-26 seasons, so this number is inflated
  and should not be cited as a baseline.

## 7. Open decisions

Still open:
1. How to present the baseline's validation numbers given the leakage. The options are to report
   leak-free test only (§3b), to regenerate 2024-26 rows of the dataset without the current game,
   or both.

Resolved:
- **Scoring margin:** stays an ablation only, not part of the official feature set (§2a).
- **SHAP for window=20:** done. The ranking is stable against window=10 (§5).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-research.txt
.venv/bin/python -m unittest Tests.test_pregame_features
.venv/bin/python -m src.Pregame.game_logs            # download game logs (cached)
for w in 10 20; do
  .venv/bin/python -m src.Pregame.features --window $w
  .venv/bin/python -m src.Pregame.train_xgb --window $w
  .venv/bin/python -m src.Pregame.shap_analysis --window $w
done
.venv/bin/python -m src.Pregame.leakage_audit
```
