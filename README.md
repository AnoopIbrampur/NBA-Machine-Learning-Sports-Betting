# DS 340W: Does the Schedule Help Predict NBA Games?

Penn State DS 340W, Fall 2026. Anoop Ibrampur and Sai Pardeev Reddy Daadi.

This branch (`ds340w-pregame`) is a fork of
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting).
**Our work starts at commit `a38a84a`**; everything before it is upstream. To see only our commits:
`git log a38a84a^..ds340w-pregame`.

**Research question:** do schedule-context features improve pre-game NBA win prediction beyond team
strength and recent form, and where do they rank inside the model?

**Short answer:** the back-to-back effect is real, but it adds no predictive value, and the betting
market already prices it. Every result below comes from [docs/findings.md](docs/findings.md), the
single source for this project's numbers. Paper details are checked against the papers.

### Sources

| Paper | Role here |
|---|---|
| Bowman, Harmon & Ashman (2023), *Journal of Sports Analytics* 9(1) | **Implementation parent paper**: schedule variables and the odds-multiplier replication |
| Ouyang et al. (2024), *PLOS ONE* 19(7) | XGBoost + SHAP approach and the 11 box-score stats, moved from in-game to pre-game |
| Alves & Barbosa (2025), *Computation* 13(10):230 | Pre-game framing and a reference accuracy |
| Leota et al. (2022), *Frontiers in Physiology* 13:892681 | Net jet-lag definition |
| Ashman, Bowman & Lambrinos (2010), *Journal of Sports Economics* 11(6) | Back-to-backs in wagering markets |

### Key results

XGBoost on 16,940 games with a fixed chronological split. Tuning uses a 40-trial walk-forward search
on train only; test and validation are never used for tuning.

| Model | Test acc | Test AUC | Validation acc | Validation AUC |
|---|---|---|---|---|
| Baseline: repo season-to-date | 0.6400 | 0.6793 | 0.6556 | 0.7089 |
| **Pre-game, window 20 (official)** | 0.6435 | 0.6746 | 0.6419 | 0.6987 |
| Pre-game + fatigue + travel (labelled extension) | 0.6412 | 0.6759 | 0.6496 | 0.6998 |
| *Market: devigged moneyline (reference row, not our model)* | 0.6790 | 0.7255 | 0.6811 | 0.7420 |

Test n = 3,386; validation n = 1,681.

- **Pre-game vs baseline:** statistically tied. Every 95% CI for the difference includes 0.
- **Schedule features:** no gain beyond noise on either held-out period, overall or on the slices
  fixed in advance. Only 1 of 24 headline 95% CIs excludes zero, and it does not hold up on the
  other split or the other metric.
- **The effect is real:** across all split games, the home team wins 61.9% when only the visitor is on
  a back-to-back and 50.0% when only the home team is.
- **The market:** a market line beats every model of ours (all CIs exclude zero). With its probability
  controlled for, no back-to-back situation's odds multiplier differs from 1 in train, test or
  validation. Whether it is a closing line can't be confirmed from this repo.

### Comparison with Bowman et al.: a partial match

Odds multipliers for a home win (above 1 favours the home team), reference rest 2 days.

| Variable | Games | Bowman | Ours, team-by-season dummies (95% CI) | Ours, win % control (95% CI) |
|---|---|---|---|---|
| Visitor on a back-to-back | 2,930 | 1.506 | **1.311 [1.180, 1.457]** | **1.289 [1.172, 1.419]** |
| Home on a back-to-back | 1,535 | 0.806 | 0.897 [0.780, 1.031] | 0.885 [0.780, 1.004] |
| Home crossed 1+ zone eastward from a game one day ago | 185 | 0.693 | 0.864 [0.605, 1.234] | 0.905 [0.655, 1.251] |
| Visitor > 1,000 mi on a b2b or > 2,000 mi otherwise | 396 | 1.261 | **1.319 [1.035, 1.683]** | 1.155 [0.930, 1.433] |
| Visitor played 3+ games in the last week | 9,730 | 1.153 | 0.995 [0.880, 1.124] | 0.994 [0.888, 1.111] |
| Home played 4+ games in the last week | 3,540 | 0.914 | 1.041 [0.940, 1.152] | 1.000 [0.912, 1.095] |
| Visitor rested 3+ days (vs 2) | 2,141 | not different | 0.916 [0.811, 1.034] | 0.962 [0.861, 1.074] |
| Home rested 3+ days (vs 2) | 2,652 | not different | 1.070 [0.957, 1.196] | 1.066 [0.963, 1.180] |

Bold intervals exclude 1. The "Games" column counts our games.

- **Data differ.** Ours: training split, 2012-11 → 2022-01, n = 11,650, and 8 variables adapted from
  Bowman's 20. Bowman: 2000-01 to 2018-19 with all 20.
- **Controls differ.** Bowman uses dummies for the home and visiting teams; ours are team-by-season
  dummies, with both teams' season-to-date win % as a second check.
- **Visitor back-to-back:** same direction and significant under both controls, but smaller. Our
  interval does not reach Bowman's 1.506.
- **Visitor long trip:** 1.32 vs 1.26, significant with dummies but not with the win % control.
- **Games-in-the-last-week rows are not like-for-like.** Bowman's variables exclude back-to-back games;
  these two rows do not apply that restriction.
- **Rested rows pool 3+ days.** Bowman reports each rest length separately.
- **Why the numbers differ has not been tested.** Season range, controls and variable set are likely
  reasons, not established ones.

### Setup

Use Python 3.13 and `requirements-research.txt`, **not** `requirements.txt` (its tensorflow pin has no
Python 3.13 wheels).

```bash
git clone -b ds340w-pregame https://github.com/AnoopIbrampur/NBA-Machine-Learning-Sports-Betting.git
cd NBA-Machine-Learning-Sports-Betting
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-research.txt
```

### Reproduce

```bash
.venv/bin/python -m src.Pregame.game_logs             # download game logs from stats.nba.com
.venv/bin/python -m unittest discover -s Tests -t .   # expect 62 tests, 0 skipped
for w in 10 20; do
  .venv/bin/python -m src.Pregame.features --window $w
  .venv/bin/python -m src.Pregame.train_xgb --window $w
  .venv/bin/python -m src.Pregame.shap_analysis --window $w
done
.venv/bin/python -m src.Pregame.leakage_audit
for w in 10 20; do .venv/bin/python -m src.Pregame.train_xgb --window $w --rescore; done
# Schedule fatigue and travel (window 20 only)
.venv/bin/python -m src.Pregame.schedule_features
.venv/bin/python -m src.Pregame.schedule_experiments
.venv/bin/python -m src.Pregame.schedule_replication
.venv/bin/python -m src.Pregame.schedule_shap
# Betting market (window 20 only)
.venv/bin/python -m src.Pregame.market_analysis
```

- **Runtime:** about 15 minutes in total, from one clean-clone run on Nov 6, 2026, covering a fresh
  venv, a fresh game-log download and a full retrain. That run matched the committed numbers in the
  main result JSONs exactly.
- **stats.nba.com dependency:** `Data/pregame/` is not in git, so a clean clone has to download the game
  logs. The site worked on Nov 6, 2026, but on Oct 8, 2026 it returned an HTML page instead of JSON
  on every endpoint. If it is down, `game_logs` and every step after it cannot run. The committed
  results below stay readable.
- **Tests before the download:** without the game logs, 4 of the 62 tests skip.
- **Already applied, do not rerun:** `scripts/fix_game_results.py` repaired the committed data and
  rewrites tracked database files.

### Where results live

| Path | Contents |
|---|---|
| `docs/findings.md` | Full write-up and every number |
| `results/pregame/` | Main metrics, bootstrap CIs (`leakage_audit.md`), SHAP |
| `results/pregame/schedule/` | Schedule models, slices, `replication.md` (Bowman comparison) |
| `results/pregame/market/` | Market benchmark and back-to-back tests |
| `results/pregame/pre_fix/`, `pre_label_fix/` | Superseded results, kept for the record |
| `docs/report/` | Report drafts |

---

Everything below is the upstream project's README. Its Python 3.11 and `requirements.txt`
instructions are for the upstream prediction app, not for the DS 340W pipeline above.

---

# NBA Sports Betting Using Machine Learning
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

## Overview
This project predicts NBA game winners and totals (over/under) using team stats and sportsbook odds. It pulls team data from 2007-08 through the current season, builds matchup features, and runs trained models to estimate win probabilities and totals outcomes. It also outputs expected value and optional Kelly Criterion stake sizing.

## Features
- Moneyline and totals predictions (XGBoost and Neural Net models).
- Expected value calculation and optional Kelly Criterion sizing.
- Odds ingest from supported sportsbooks or manual input.
- Data processing pipeline and model training scripts.
- Flask web app for browsing outputs.

## How it works
1. **Collect stats and odds**: `Get_Data` pulls daily team stats from NBA endpoints and stores them in SQLite. `Get_Odds_Data` pulls sportsbook odds and scores from SBR and stores them in a separate SQLite DB.
2. **Build game features**: `Create_Games` merges team stats, odds, scores, and days-rest into a training dataset.
3. **Train models**: XGBoost/NN scripts in `src/Train-Models` fit moneyline and totals models.
4. **Predict today**: `main.py` fetches today’s schedule, builds matchup features, loads trained models, and prints predictions, expected value, and optional Kelly Criterion sizing.

## Requirements
- Python 3.11
- Packages: Tensorflow, XGBoost, NumPy, Pandas, Colorama, Tqdm, Requests, Scikit-learn

Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Quick start
```bash
python3 main.py -xgb -odds=fanduel
```

Odds will be fetched automatically when `-odds` is provided. Supported books:
`fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`

If `-odds` is omitted, the script will prompt for manual odds and totals.

Optional flags:
- `-nn` run neural network model
- `-xgb` run XGBoost model
- `-A` run all models
- `-kc` show Kelly Criterion bankroll fraction

## Flask web app
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

```bash
cd Flask
flask --debug run
```

## Data pipeline and training
```bash
# Create/update datasets
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
python -m XGBoost_Model_UO --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
python -m NN_Model_ML
python -m NN_Model_UO
python -m Logistic_Regression_ML --dataset dataset_2012-26_new --trials 50 --splits 5 --calibration sigmoid
python -m Logistic_Regression_UO --dataset dataset_2012-26_new --trials 50 --splits 5 --calibration sigmoid
```

### Neural network notes
- The current NN training scripts are the original versions with hard-coded dataset and model paths.
- They train on `dataset_2012-24_new` and save into `Models/` with timestamped names.
- If you want configurable flags or feature/scaler sidecars, switch back to the newer NN scripts.

### Backfilling missing data
Get_Data normally fetches only new dates in the current season. To fill missing dates:
```bash
cd src/Process-Data
python -m Get_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Data --backfill --season 2025-26
```

### Backfilling odds data
Get_Odds_Data normally fetches only new dates in the current season. To fill missing odds dates:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill --season 2025-26
```

## Contributing
Contributions are welcome. If you change model behavior or data pipelines, add a note in the README and update any related scripts or docs.
