# DS 340W findings: pre-game NBA outcome prediction

Single reference for the course report and check-ins. Every number here comes from a committed
file under `results/pregame/`; regenerate with the commands at the end.

## In one paragraph

This fork's existing model (the baseline) was never an in-game model: it predicts from each team's
**season-to-date averages**. So the question is not "can we convert in-game to pre-game" but
"does **recent form (last N games) plus schedule context** beat season-to-date averages?".
The baseline's data for 2024-25 and 2025-26 used to **leak the outcome**. That is now fixed and
those rows regenerated (§1).

On the corrected data, the window-20 pre-game model is **statistically tied** with the baseline
on both held-out periods:
- **Test:** 0.6435 vs 0.6400 accuracy, 0.6749 vs 0.6797 AUC.
- **Validation:** 0.6419 vs 0.6556 accuracy, 0.6987 vs 0.7089 AUC.

Every 95% CI for the difference includes 0. Point estimates slightly favour the pre-game model on
test and the baseline on validation. A 10-game window is significantly worse on AUC. A rolling
scoring-margin feature, kept as an **ablation only**, lifts window-20 to 0.6480 / 0.6838 on test
and 0.6466 / 0.7107 on validation, also within noise.

SHAP shows recent win-record features outranking all 11 box-score stats. This is the same pattern
parent paper 2 reports: engineered context features rank above box-score stats.

**Back-to-backs and travel (§7): a clean null for prediction.** We rebuilt "back-to-back" from two
flags into a full fatigue and travel feature group, following parent paper 3 (Bowman et al., 2023).
- **No gain.** Accuracy, AUC and log loss do not improve beyond noise on test or validation,
  overall or on any of six slices fixed in advance.
- **Test results:** pre-game + fatigue + travel scores 0.6412 accuracy / 0.6763 AUC, against
  0.6435 / 0.6749 for the official pre-game model.
- **The effect itself is real.** The home team wins 61.9% when only the visitor is on a b2b and
  49.9% when only the home team is.
- **Partial replication of Bowman.** On our training seasons the visitor-b2b odds ratio is
  1.31 [1.18, 1.46], against Bowman's 1.51. The models already capture this through rest days.

## 1. What the repo's baseline actually is

- **Source:** `src/Process-Data/Get_Data.py` downloads stats.nba.com `leaguedashteamstats` with
  `PerMode=PerGame` and `DateTo=<game date>`. Each game row (`Data/dataset.sqlite`,
  `dataset_2012-26`) therefore holds both teams' **season-to-date per-game averages and ranks**:
  FG%, rebounds, assists, points, plus-minus, W/L record, and so on. Rest days are also included.
- **Not in-game:** these are team averages over earlier games, not the box score of the game being
  predicted. The in-game → pre-game framing does not apply to this repo.
- **Leakage in 2024-25 and 2025-26 (found and fixed):**
  - *Symptom.* Through 2023-24, each row's stats excluded the game being predicted (≥99.8% of
    rows). In 2024-25 and 2025-26 they included it (100%), so the baseline saw the result.
  - *Cause.* The upstream rewrite of `Get_Data.py` (`8e36b0b`, Jan 2026) dropped a one-day offset.
    `DateTo` is inclusive, and the older code saved a `DateTo=D` fetch as the snapshot for D+1. The
    rewrite saved it as the snapshot for D, so the snapshot for a game's date included that game.
    Only the two seasons fetched with the rewritten code were affected.
  - *Fix.* `fetch_data` now requests `DateTo = D − 1` for date D, with a regression test in
    `Tests/test_get_data.py`.
  - *Regeneration.* `scripts/regenerate_2024_26.py` re-fetched those seasons' 323 snapshots and
    rebuilt only their rows. The 15,115 rows for 2012-13 to 2023-24 are byte-identical. Six
    early-season games (2024-10-24, 2025-10-23) no longer have a full 30-team snapshot and drop out,
    the same rule every earlier season follows.
  - *Verification.* `Tests/test_dataset_leakage.py` checks all 14 seasons. It failed on 2024-25 and
    2025-26 before the fix and passes on every season now.
  - *Unaffected.* The prediction app (`main.py`) fetches live stats itself and was never affected.
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
- **Games:** the same 16,940 for every model: the 16,969 games of the split, minus 6 early-season
  games with no pre-game snapshot and 23 missing from the game logs.
- **Split:** the fixed chronological split in `Data/splits/split_keys.csv`: train 2012-11 → 2022-01,
  test 2022-01 → 2024-11-13, validation 2024-11-13 → 2026-01-07.
- **Tuning:** 40-trial random search scored by walk-forward CV on train only; test and validation
  are never used for tuning.
- **Metrics:** positive class = home win, threshold 0.5. Home teams win 56.4% of test games and
  54.5% of validation games.

### 3a. Primary results: corrected data, full test and validation sets

| Model | Split | n | Acc | Prec | Recall | F1 | AUC | Log loss | Brier | Acc diff vs baseline, 95% CI | AUC diff vs baseline, 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 3,386 | 0.6400 | 0.6565 | 0.7580 | 0.7036 | 0.6797 | 0.6349 | 0.2221 | — | — |
| **Pre-game, window=20** | test | 3,386 | 0.6435 | 0.6509 | 0.7931 | 0.7150 | 0.6749 | 0.6371 | 0.2229 | [−0.010, +0.017] | [−0.015, +0.006] |
| Pre-game, window=20 + margin (ablation) | test | 3,386 | 0.6480 | 0.6561 | 0.7894 | 0.7166 | 0.6838 | 0.6326 | 0.2209 | [−0.005, +0.021] | [−0.005, +0.014] |
| Baseline: repo season-to-date | validation | 1,681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 | — | — |
| **Pre-game, window=20** | validation | 1,681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 | [−0.033, +0.003] | [−0.023, +0.002] |
| Pre-game, window=20 + margin (ablation) | validation | 1,681 | 0.6466 | 0.6507 | 0.7587 | 0.7006 | 0.7107 | 0.6207 | 0.2157 | [−0.027, +0.007] | [−0.010, +0.014] |

Sources:
- metrics: `results/pregame/metrics_w20.md`
- bootstrap CIs (paired, 2,000 resamples of the same games for both models):
  `results/pregame/leakage_audit.md`

The models are identical to the pre-fix run (same training data, tuning result and parameters).
Only the regenerated 2024-26 evaluation rows changed. Window=10 on the corrected data is in
`results/pregame/metrics_w10.md`. It is significantly worse than the baseline on AUC in both test
(CI [−0.032, −0.006]) and validation (CI [−0.043, −0.009]).

**Reading the results**
- **Window=20 is statistically tied with the baseline** in both held-out periods. On test it is
  slightly ahead on accuracy and behind on AUC. On validation (2024-25 and 2025-26) the baseline
  is ahead on both, but both intervals include 0, if only just (upper bounds +0.003 and +0.002).
- **Scoring margin (ablation only, §2a)** moves window=20 ahead of the baseline on test (every
  metric) and level on validation AUC (0.7107 vs 0.7089). None of this is significant.
- **The leak mattered only for the later period.** Fixing it moved the baseline's validation AUC
  from 0.7648 to 0.7089 and its test AUC from 0.6889 to 0.6797. On its 152 test games from 2024-25,
  the baseline went from 0.750 / 0.845 to 0.638 / 0.644 (accuracy / AUC).
- **Window=10 is the one clear loser**, significantly below the baseline on AUC in both periods.

### 3b. Secondary check: test games from never-affected seasons only (n = 3,234, 2021-22 → 2023-24)

This comparison does not depend on the fix at all. Its numbers are unchanged from before the fix,
which also confirms the models are the same.

| Model | Acc | AUC | Log loss | Brier | Acc diff vs baseline, 95% CI | AUC diff vs baseline, 95% CI |
|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | 0.6401 | 0.6813 | 0.6345 | 0.2219 | — | — |
| Pre-game, window=10 | 0.6342 | 0.6628 | 0.6451 | 0.2266 | [−0.021, +0.010] | **[−0.033, −0.005]** |
| Pre-game, window=10 + margin | 0.6336 | 0.6678 | 0.6420 | 0.2253 | [−0.022, +0.008] | **[−0.027, −0.0005]** |
| **Pre-game, window=20** | 0.6432 | 0.6774 | 0.6365 | 0.2226 | [−0.011, +0.017] | [−0.015, +0.007] |
| Pre-game, window=20 + margin | 0.6481 | 0.6852 | 0.6325 | 0.2208 | [−0.005, +0.021] | [−0.006, +0.014] |

Bold intervals exclude 0. The conclusions match §3a:
- window=10 is worse than the baseline on AUC;
- window=20 is tied with the baseline;
- margin helps a little, not significantly (+0.005 accuracy, +0.008 AUC on top of window=20).

### 3c. Superseded: full-split results before the leakage fix

Kept verbatim because some of these numbers were reported at earlier check-ins. The window=10 rows
are the ones reported first. Files: `results/pregame/pre_fix/`.

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

⚠ These baseline rows are inflated by the leak. Every validation game, and 156 of the 3,390 test
games, came from the leaky 2024-26 rows. The pre-game rows moved only in the 4th decimal after
the fix: they are unaffected by the leak, and the regenerated dataset drops 4 test and 2
validation games.

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
split (n = 3,390), ranking features by mean |SHAP|. This ran before the leakage fix. The fix
touched neither these models nor their features, and removed only 4 of these test games, so the
rankings stand. Window=20 is the primary model. The window=10
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

- **Dataset tables:** `dataset_2012-24_new` and `dataset_2012-26` are the same data through
  2023-24. `dataset_2012-26` adds 2024-25 and 2025-26, now regenerated without the leak. See
  `docs/dataset-tables.md`.
- **Split:** valid for all models on the corrected data. Six split games no longer have a dataset
  row (§1) and drop out of every model equally.
- **Earlier LR baseline** (0.6409 acc / 0.6324 log loss): not reproducible, and not caused by
  package versions. It is also scored on a different test set (the LR script's own 90/10 split).
  See `docs/lr-reproducibility.md`.
- **Repo's own XGBoost protocol** (after the scikit-learn fix): 0.6977 test accuracy on the last 10%
  of `dataset_2012-26`. That was measured before the fix, inside the leaky 2024-26 seasons. It is
  inflated and should not be cited as a baseline.

## 7. Schedule fatigue and travel (parent paper 3)

**Why.** Our stated USP is back-to-back games. Before this work it was two binary flags
(`b2b_home`, `b2b_away`) that ranked 14th and 19th of 19 in the window-20 SHAP ranking and carried
1.4% of total |SHAP|. This section turns it into a schedule-fatigue and travel feature group built
from parent paper 3 and a supporting paper, then measures whether it helps.

- **Parent paper 3:** Bowman, Harmon & Ashman, "Schedule inequity in the National Basketball
  Association", *Journal of Sports Analytics* 9(1), 2023.
- **Supporting paper:** Leota et al., *Frontiers in Physiology* 13:892681, 2022. It supplies the
  net jet-lag definition.

**Short answer: a clean null for prediction, and a partial replication of the paper.**
- **Prediction:** no new feature set beats the repo baseline or the pre-game model on test. Only 1
  of the 24 headline 95% CIs excludes zero, and it does not hold up on the other split or the other
  metric (§7d).
- **Replication:** the back-to-back effect itself is clearly in our data and largely replicates
  Bowman's odds ratios for the visitor (§7f). The models already capture it through rest days.

All files are in `results/pregame/schedule/`.

### 7a. Venues, and games without a real home court

- **Arena table, `Data/arenas.csv`.** Each team's home arena with latitude, longitude, IANA time zone
  and validity dates. Arena moves are covered: GSW (2019), LAC (2024), SAC (2016), MIL (2018),
  DET (2017).
- **Toronto 2020-21.** Home games in Tampa (Amalie Arena, `America/New_York`) from 2020-12-01 to
  2021-07-31.
- **Time zone offsets.** Computed with `zoneinfo` at 19:00 local on the actual game date. Both
  venues' offsets are read on the same date, so a daylight-saving switch between two games in one
  city is not a shift, and Phoenix (no DST) is +1 h from Los Angeles in January and 0 h in July.
- **2019-20 restart, from 2020-07-30.** The logs show these games as ordinary pairs ("BOS vs. POR").
  There are 172 of them: 88 seeding, 1 play-in and 83 playoff games, all in the training split.
  - All were played at one site (Lake Buena Vista), so travel between restart games is 0 km and
    0 h. The trip into the first restart game, after a 4½-month stoppage, is left missing.
  - No team counts as "at home" in the bubble, so restart games extend `consec_away`.
  - The models keep these games, because the existing feature sets already include them. The
    replication excludes them (no home court).
- **Neutral-site games flagged by `features.py`.** There are 10, all from 2024-25 onward: NBA Cup
  Las Vegas, Mexico City, Paris, Berlin and London. Travel is missing for the game and for each
  team's next game. None of these are in the split.
- **Surprise: off-site games the logs list as home games.** 24 earlier off-site games appear in
  MATCHUP as ordinary home games ("vs."), so the existing neutral flag misses them.
  - They are 22 international games from 2013–2024 and the 2023 In-Season Tournament semifinals in
    Las Vegas.
  - The list is `Data/offsite_games.csv`, taken from Wikipedia's NBA Global Games page. Every entry
    is checked against the logs (date plus both teams must match exactly one game), and the code
    raises an error otherwise.
  - Cities are not recorded, because the source extraction was unreliable on city. All 24 are
    treated like neutral sites: travel missing for the game and the next game.
  - 18 are in train and 6 in test. In the existing features these games still count as home games;
    that is left unchanged.
- **Missing travel.** 95 of 16,940 split games (63 train, 20 test, 12 validation) have travel
  missing for at least one team. The causes are season openers, the game after an off-site game,
  and the first restart game.
- **stats.nba.com status.** When this was built (2026-10-08), stats.nba.com returned an HTML page
  in place of JSON on every endpoint. Nothing needed re-downloading: the cached game logs and saved
  models reproduce `metrics_w20.md` exactly.

### 7b. Features (`src/Pregame/schedule_features.py`)

Computed per team per game, within season, from the dates and venues of that team's earlier games
plus the date of its next regular-season game. No result or box score is read. The features do not
depend on the rolling window. The existing `rest_days` / `b2b` columns are unchanged.

| Group | Feature (per team, home and away) | Definition |
|---|---|---|
| Fatigue | `rest_bucket` | days since last game, capped at 3 (1, 2, 3+) |
| Fatigue | `games_last7` | the team's games in the 7 days before game day |
| Fatigue | `three_in_four` | 2+ games in the 3 days before, i.e. this is the 3rd game in 4 days |
| Fatigue | `consec_away` | consecutive games away from its home arena immediately before this one |
| Fatigue | `b2b_prev_away` | on a back-to-back: 1 if last night's game was away, 0 if at home |
| Fatigue | `first_leg_b2b` | the next scheduled regular-season game is tomorrow (date only; 0 in the postseason, which has no back-to-backs in our data) |
| Fatigue (game) | `rest_diff` | home − away `rest_bucket` |
| Travel | `travel_km` | great-circle km from the previous game's venue |
| Travel | `tz_shift` | signed clock change in hours, east positive |
| Travel | `jet_lag` | sign × max(\|shift\| − days since previous game, 0) |
| Travel (game) | `visitor_long_trip` | Bowman: visitor > 1,609 km on a back-to-back or > 3,219 km otherwise |
| Travel (game) | `home_east_b2b` | Bowman: home team crossed 1+ time zone eastward from a game one day ago |

**Leakage tests (`Tests/test_schedule_features.py`).** Each test checks that a game's features are
unchanged when one thing is altered:
- the game's own result and box score;
- later games' results;
- the dates and venues of every game after the next one.

Each test was confirmed to fail when the matching leak was injected deliberately (current result,
a later result, a later venue). The injected leaks were then removed. Hand-computed cases cover:
- Los Angeles → Boston = +3 h, giving 2 h eastward jet lag 1 day later (Leota's example);
- Phoenix in January vs July;
- Toronto/Tampa 2020-21;
- a distance sanity check: Los Angeles → Boston ≈ 4,180 km.

**Jet lag is almost always 0.** Under this definition one zone of travel with one day of rest nets
to 0. Only 415 team-games in the split have non-zero net jet lag. That makes slice 5 tiny and leaves
the models with nothing to split on (§7e).

### 7c. Results: window 20, full test and validation sets

Same split, same 40-trial walk-forward search on train only, same seed. The three original models
are the saved ones, untouched. Source: `results/pregame/schedule/metrics_w20.md`.

| Model | Split | n | Acc | Prec | Recall | F1 | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 3,386 | 0.6400 | 0.6565 | 0.7580 | 0.7036 | 0.6797 | 0.6349 | 0.2221 |
| Pre-game w20 (official) | test | 3,386 | 0.6435 | 0.6509 | 0.7931 | 0.7150 | 0.6749 | 0.6371 | 0.2229 |
| Pre-game w20 + margin (ablation) | test | 3,386 | 0.6480 | 0.6561 | 0.7894 | 0.7166 | 0.6838 | 0.6326 | 0.2209 |
| Pre-game w20 + fatigue | test | 3,386 | 0.6409 | 0.6483 | 0.7936 | 0.7136 | 0.6754 | 0.6370 | 0.2229 |
| Pre-game w20 + fatigue + travel | test | 3,386 | 0.6412 | 0.6527 | 0.7768 | 0.7094 | 0.6763 | 0.6364 | 0.2226 |
| Repo + fatigue + travel | test | 3,386 | 0.6394 | 0.6571 | 0.7538 | 0.7021 | 0.6809 | 0.6339 | 0.2216 |
| Baseline: repo season-to-date | validation | 1,681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 |
| Pre-game w20 (official) | validation | 1,681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 |
| Pre-game w20 + margin (ablation) | validation | 1,681 | 0.6466 | 0.6507 | 0.7587 | 0.7006 | 0.7107 | 0.6207 | 0.2157 |
| Pre-game w20 + fatigue | validation | 1,681 | 0.6472 | 0.6483 | 0.7707 | 0.7042 | 0.6985 | 0.6283 | 0.2192 |
| Pre-game w20 + fatigue + travel | validation | 1,681 | 0.6496 | 0.6538 | 0.7587 | 0.7024 | 0.6998 | 0.6268 | 0.2185 |
| Repo + fatigue + travel | validation | 1,681 | 0.6526 | 0.6566 | 0.7598 | 0.7045 | 0.7095 | 0.6213 | 0.2161 |

Training CV log loss (walk-forward, train only):

| Model | Training CV log loss |
|---|---|
| repo | 0.6240 |
| repo + fatigue + travel | 0.6238 |
| pregame | 0.6308 |
| pregame + fatigue | 0.6304 |
| pregame + fatigue + travel | 0.6305 |

The new features barely move training CV either.

### 7d. Paired bootstrap 95% CIs (model − reference, 2,000 resamples)

| Model | Reference | Test acc | Test AUC | Validation acc | Validation AUC |
|---|---|---|---|---|---|
| Pre-game + fatigue | repo | [−0.012, +0.015] | [−0.015, +0.006] | [−0.027, +0.008] | [−0.023, +0.002] |
| Pre-game + fatigue + travel | repo | [−0.012, +0.014] | [−0.014, +0.007] | [−0.025, +0.011] | [−0.022, +0.003] |
| Repo + fatigue + travel | repo | [−0.006, +0.004] | [−0.0003, +0.003] | [−0.010, +0.004] | [−0.001, +0.003] |
| Pre-game + fatigue | pregame | [−0.008, +0.002] | [−0.001, +0.002] | [−0.002, +0.013] | [−0.002, +0.002] |
| Pre-game + fatigue + travel | pregame | [−0.009, +0.004] | [−0.0004, +0.003] | **[+0.0006, +0.015]** | [−0.001, +0.003] |
| Pre-game + fatigue + travel | pregame + fatigue | [−0.005, +0.006] | [−0.001, +0.002] | [−0.005, +0.010] | [−0.001, +0.003] |

**One interval excludes zero:** pre-game + fatigue + travel vs pre-game, validation accuracy,
+0.008 (0.6496 vs 0.6419). It should not be read as a gain:
- On the same model, the validation AUC interval includes zero.
- On test, the accuracy point estimate goes the other way (0.6412 vs 0.6435).
- It is 1 of 24 intervals in this table, roughly what chance alone produces at 95%.

Every comparison against the repo baseline includes zero on both splits.

### 7e. Pre-registered slices

These six slices were fixed before any result was seen, and all are reported in
`results/pregame/schedule/metrics_w20.md` for every model, with n, accuracy, AUC, log loss and
Brier. b2b uses the existing flags; a season's first game counts as not on a b2b. Slices 5 and 6
exclude games with unknown travel.

| Slice | Test n | Test home-win | Val n | Val home-win | Note |
|---|---|---|---|---|---|
| 1. neither team on a b2b | 2,445 | 0.566 | 1,193 | 0.556 | |
| 2. away team only on a b2b | 458 | 0.620 | 205 | 0.585 | |
| 3. home team only on a b2b | 339 | 0.466 | 192 | 0.453 | validation unreliable (n < 200) |
| 4. both on a b2b | 144 | 0.576 | 91 | 0.505 | unreliable (n < 200) |
| 5. home eastward net jet lag ≥ 1 h | 13 | 0.385 | 8 | 0.625 | no CI (n < 30); not interpretable |
| 6. away travelled > 1,609 km | 727 | 0.550 | 378 | 0.542 | |

Accuracy / AUC on each slice for the baseline, the official pre-game model, and the two full
schedule models:

| Slice | Split | Repo | Pre-game | Pre-game + fat. + travel | Repo + fat. + travel |
|---|---|---|---|---|---|
| 1 | test | 0.634 / 0.671 | 0.640 / 0.666 | 0.638 / 0.667 | 0.630 / 0.670 |
| 1 | val | 0.664 / 0.714 | 0.651 / 0.709 | 0.660 / 0.709 | 0.661 / 0.714 |
| 2 | test | 0.662 / 0.682 | 0.666 / 0.678 | 0.655 / 0.679 | 0.672 / 0.685 |
| 2 | val | 0.659 / 0.716 | 0.673 / 0.685 | 0.688 / 0.693 | 0.654 / 0.719 |
| 3 | test | 0.637 / 0.721 | 0.637 / 0.700 | 0.631 / 0.698 | 0.649 / 0.727 |
| 3 | val | 0.630 / 0.714 | 0.604 / 0.699 | 0.599 / 0.697 | 0.630 / 0.715 |
| 4 | test | 0.674 / 0.717 | 0.653 / 0.742 | 0.667 / 0.748 | 0.667 / 0.721 |
| 4 | val | 0.593 / 0.650 | 0.527 / 0.621 | 0.538 / 0.626 | 0.593 / 0.653 |
| 6 | test | 0.618 / 0.677 | 0.641 / 0.663 | 0.634 / 0.665 | 0.616 / 0.676 |
| 6 | val | 0.646 / 0.698 | 0.651 / 0.708 | 0.651 / 0.710 | 0.651 / 0.703 |

**What the slices show**
- **The schedule features do not help on back-to-back games specifically.** On slices 2–4 the
  schedule models are within noise of the models without them, in both directions.
- **Slice 3 (home team only on a b2b) is the hardest situation for the pre-game models.** The home
  team wins under half the time (0.466 test, 0.453 validation). The official pre-game model and
  both pre-game schedule models trail the baseline on AUC there in both splits, by 0.016–0.026.
  Adding schedule features does not close the gap.
- **Intervals that exclude zero, against the repo baseline.** 4 of 100 slice intervals (5 models ×
  5 slices with a CI × 2 splits × 2 metrics), close to the 5 expected by chance. None repeats
  across splits:
  - repo + fatigue + travel, slice 3 test AUC: [+0.0004, +0.010]; validation [−0.007, +0.008];
  - pre-game and pre-game + fatigue, slice 4 validation accuracy (worse, n = 91);
  - pre-game + margin (ablation), slice 6 validation AUC: [+0.002, +0.050]; test
    [−0.029, +0.013].
- **Slice 5 is not interpretable** (13 and 8 games).

No other slices were examined.

### 7f. Replicating Bowman et al. on our training seasons

Logistic regression of home win on Bowman's schedule variables. Source:
`results/pregame/schedule/replication.md`.

- **Games:** training split (2012-11 → 2022-01), excluding the 172 restart games, 18 off-site games
  and 33 games with missing rest or travel, which leaves n = 11,650.
- **Primary control:** team-by-season dummies for home and visitor, as in the paper. The fit is
  stable: converged, full rank (598 columns), largest SE 1.08. One home dummy per season and one
  visitor dummy are dropped for identifiability.
- **Second control (check):** both teams' season-to-date win % from the repo dataset.
- **Reference rest:** 2 days.

| Variable | Games | Bowman | Ours, team-season dummies (95% CI) | Ours, win % control (95% CI) |
|---|---|---|---|---|
| Visitor on a back-to-back | 2,930 | 1.506 | **1.311 [1.180, 1.457]** | **1.289 [1.172, 1.419]** |
| Home on a back-to-back | 1,535 | 0.806 | 0.897 [0.780, 1.031] | 0.885 [0.780, 1.004] |
| Home crossed 1+ zone eastward from a game one day ago | 185 | 0.693 | 0.864 [0.605, 1.234] | 0.905 [0.655, 1.251] |
| Visitor > 1,000 mi on a b2b or > 2,000 mi otherwise | 396 | 1.261 | **1.319 [1.035, 1.683]** | 1.155 [0.930, 1.433] |
| Visitor played 3+ games in the last week | 9,730 | 1.153 | 0.995 [0.880, 1.124] | 0.994 [0.888, 1.111] |
| Home played 4+ games in the last week | 3,540 | 0.914 | 1.041 [0.940, 1.152] | 1.000 [0.912, 1.095] |
| Visitor rested 3+ days (vs 2) | 2,141 | not different | 0.916 [0.811, 1.034] | 0.962 [0.861, 1.074] |
| Home rested 3+ days (vs 2) | 2,652 | not different | 1.070 [0.957, 1.196] | 1.066 [0.963, 1.180] |

**Replicates**
- **Visitor back-to-back** favours the home team. The direction matches Bowman and the effect is
  significant under both controls, but smaller (1.31 vs 1.51).
- **Rest of 3+ days** is not different from 2 days, as in Bowman.

**Same direction, not significant**
- **Home back-to-back:** 0.90 vs 0.81.
- **Home team crossing 1+ zone eastward on a b2b:** 0.86 vs 0.69, with only 185 games.

**Partly replicates**
- **Visitor long trip:** 1.32 vs 1.26, significant with team-season dummies but not with the
  win % control.

**Does not replicate**
- **Games-in-the-last-week variables.** Our definition counts the team's games in the 7 days
  before game day. With it, "visitor played 3+" is true in 9,730 of 11,650 games, almost a
  constant. Bowman's exact window may differ, so this is a definitional mismatch as much as a
  finding.

**Raw home-win rate by back-to-back situation, all split games.** From the dataset's Days-Rest
columns (n = 16,963), which reproduces the numbers we started from:

| Situation | n | Home-win rate |
|---|---|---|
| home rested / away b2b | 2,956 | 0.619 |
| both rested | 11,691 | 0.568 |
| both b2b | 890 | 0.583 |
| home b2b / away rested | 1,426 | 0.499 |

The same table by split, using the game-log b2b flags (16,940 games), is in
`home_win_by_b2b_situation.csv`. The pattern holds in every split; validation's "both b2b" (n = 91)
is the noisiest cell.

**Back-to-back frequency by season** (`b2b_frequency_by_season.csv`):
- Back-to-backs per team fell from about 18.5 in 2012-13 to 13–15 since 2017-18.
- The away team was on a b2b in 31% of games in 2012-13 and 17–18% since 2022-23.
- 2020-21 (compressed COVID schedule) is the exception: 19% of home teams were on a b2b.

The fall means fewer of the situations where schedule features matter in the test and validation
years than in training.

### 7g. SHAP (test split)

`pregame_sched` is the better new pre-game model by training CV log loss (0.6304 vs 0.6305). The
two are essentially tied, so `pregame_sched_travel` is also explained, as a supplement.

Share of total mean |SHAP| by group:

| Group | pregame w20 (§5) | pregame + fatigue | pregame + fatigue + travel | repo + fatigue + travel |
|---|---|---|---|---|
| Win record | 58% | 58.8% | 55.1% | 33.0% |
| Box-score stats | 38% | 33.5% | 34.5% | 60.8% (incl. plus-minus) |
| Fatigue (incl. existing rest/b2b) | 4.6% | 7.7% | 8.6% | 4.5% |
| Travel | — | — | 1.8% | 0.4% |
| Games played / minutes | — | — | — | 1.3% |

- **Fatigue's share rises (4.6% → 7.7%) but accuracy does not.** The new columns mostly share
  credit with the old rest-day columns rather than adding information.
- **Top schedule features in pregame_sched:** `rest_diff` (#13), `rest_days_away` (#14),
  `b2b_prev_away_away` (#16), `games_last7_away` (#17). Home-side fatigue features are near the
  bottom.
- **Away back-to-back.** Mean SHAP is +0.011 log-odds toward the home team when the visitor is on a
  b2b, and −0.004 otherwise. The repo + fatigue + travel model shows the same through
  `rest_bucket_away` (+0.021 at 1 day vs −0.007 at 2 or 3+). The direction matches Bowman; the size
  is small.
- **Home net jet lag has exactly zero SHAP in every model.** No tree ever splits on it, so its
  dependence plots are flat. Of the travel features, only `travel_km_away` carries any weight
  (#16 in pregame + fatigue + travel).

### 7h. Reading the result

- **The back-to-back effect is real.** It is clear in raw rates, and the visitor-b2b odds ratio
  replicates Bowman's direction and significance.
- **It adds nothing to prediction.** Both baselines already contain rest days, and the win-record
  and team-strength features carry what remains.
- **Richer fatigue and travel features do not move accuracy, AUC or log loss** beyond noise on
  either held-out period, overall or on back-to-back games.
- **What the USP can honestly claim:** a careful, leakage-tested measurement of schedule effects
  that replicates parent paper 3's main finding on 2012-22 data. It cannot claim that schedule
  features improve prediction.

**Limitations**
- **Definition choices that could matter:**
  - the 7-day window for "games in the last week";
  - the jet-lag formula as specified (with Leota's "one rest day" read as one day since the previous
    game);
  - games after postponements use the date actually played, not the originally published date.
- **Off-site list:** taken from Wikipedia's NBA Global Games page and verified against the logs, so
  it could still miss a game the page omits.

## 8. Open decisions

- **Should the official feature set change to include the schedule features (§7)?**
  **Recommendation: no.** Keep the official set as pre-game window 20 (11 stats + win-record + rest
  days + b2b flags). Report `pregame_sched` and `pregame_sched_travel` as a labelled extension, the
  way margin is reported. The evidence:
  - no new feature set differs from the repo baseline on either split;
  - against the pre-game model, the only interval excluding zero (validation accuracy, +0.008)
    disappears on AUC and reverses on test;
  - training CV log loss moves only in the 4th decimal (0.6308 → 0.6304 / 0.6305).

  Two things argue for keeping the schedule work in the report, but not in the official set:
  - the visitor back-to-back effect replicates Bowman (odds ratio 1.31 [1.18, 1.46]);
  - the pre-registered slices are the honest test of the back-to-back USP, and they show no gain.

  The team needs to decide this, and how to frame the USP to the professor (§7h).

Resolved:
- **Baseline leakage in 2024-26:** fixed at the source and the rows regenerated. The corrected
  full-split results are primary (§3a), with the never-affected-seasons comparison kept as a
  secondary check (§3b).
- **Scoring margin:** stays an ablation only, not part of the official feature set (§2a).
- **SHAP for window=20:** done. The ranking is stable against window=10 (§5).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-research.txt
.venv/bin/python -m unittest discover -s Tests -t .   # all tests
.venv/bin/python -m src.Pregame.game_logs            # download game logs (cached)
for w in 10 20; do
  .venv/bin/python -m src.Pregame.features --window $w
  .venv/bin/python -m src.Pregame.train_xgb --window $w
  .venv/bin/python -m src.Pregame.shap_analysis --window $w
done
.venv/bin/python -m src.Pregame.leakage_audit
# Schedule fatigue and travel (§7), window 20 only
.venv/bin/python -m src.Pregame.schedule_features      # Data/pregame/schedule_features.csv
.venv/bin/python -m src.Pregame.schedule_experiments   # trains the 3 new models if not saved
.venv/bin/python -m src.Pregame.schedule_replication
.venv/bin/python -m src.Pregame.schedule_shap
```
