# Modified Implementation, round 2: analysis plan

Fixed before any player feature, Elo variant or rolling model was computed. The commit date of this file
is the record of that. Changes after results exist go under "Deviations".

**This round is not blind.** The test and validation results of round 1 (`docs/enhanced-plan.md`, findings
§11) were seen and motivated it. Any gain here should be read with that in mind. New outputs go in
`results/pregame/enhanced2/`. Existing and round 1 models and results stay as they are.

The official feature set stays pre-game window 20. Everything here is a labelled extension. The market
is a reference row only.

## Primary test
- **Row:** `pregame_elo2_player`, pre-game window 20 + Elo variant + player features.
- **Comparison:** against `repo`.
- **Gain rule:** as in round 1. The CI for the same metric excludes 0 in the positive direction on both
  test and validation. All other rows and comparisons are secondary.

## Shared rules
Same split, games, seed (42), `paired_bootstrap` and 40-trial walk-forward search on train only. Test and
validation are never used for tuning or selection.

## Elo variant
Extends round 1's Elo. Defaults reproduce round 1 exactly, and a test enforces that.
- **Grid:** k {16, 20, 24, 28, 32} × home advantage {0, 50, 100} × regression toward 1500 at each new
  season {0, 1/4, 1/3} × margin of victory {off, on}.
- **Home advantage** applies to the expected score only, and is 0 for neutral, off-site and 2019-20
  restart games.
- **Margin term:** when on, the update is multiplied by ln(|margin| + 1) × 2.2 / (0.001 × winner's rating
  edge + 2.2). This is our choice of a common form, not from our sources.
- **Choosing:** the same rule as round 1: lowest walk-forward log loss of `elo_diff` alone on train. Ties
  go to the simplest setting: lower k, then lower home advantage, lower regression, margin off.
- **Features:** `elo2_home`, `elo2_away`, `elo2_diff`.

## Player features
Downloaded from stats.nba.com `leaguegamelog` (player level) into `Data/pregame/PlayerLogs.sqlite`.
Everything is computed by date: all of a date's features are read from what was known before that date,
and only then does that date's data update the state. A game's own player rows are never read.

- **Player value:** our own box-score value, PTS + 0.4 FGM − 0.7 FGA − 0.4 (FTA − FTM) + 0.7 OREB +
  0.3 DREB + STL + 0.7 AST + 0.7 BLK − 0.4 PF − TOV.
- **Player rating:** value per 36 minutes over the player's previous 40 games, across teams and seasons,
  shrunk toward a replacement level with 200 minutes of prior weight. Replacement level is 0.75 × the
  minutes-weighted league mean per-36 value of train-split games only. Players with no history take that
  level.
- **Per team, within a season:**
  - `player_strength`: rating weighted by minutes in the team's last 5 games.
  - `player_strength_lastgame`: the same, using only players who appeared in the team's previous game.
  - `player_availability`: share of the team's last 10 games' minutes from players who appeared in the
    previous game.
  - `player_continuity`: share of the last 5 games' minutes from players who played for this team last
    season.
- **Game level:** home and away versions of all four, plus `player_strength_diff` and
  `player_strength_lastgame_diff`. Missing for a team's first game of a season.
- **Known limit:** these use the previous game, so they miss injury and rest decisions made on game day.
  The market likely already prices those.

## New rows

| Row | Definition |
|---|---|
| `pregame_elo2` | Pre-game window 20 + Elo variant |
| `pregame_elo2_player` | **Primary:** pre-game window 20 + Elo variant + player features |
| `repo_elo2_player` | Repo baseline + Elo variant + player features |
| `stack2` | Stack of `repo`, `pregame`, `pregame_elo2_player`, built as in round 1 |
| `roll_repo` | `repo` with rolling retraining |
| `roll_pregame_elo2_player` | `pregame_elo2_player` with rolling retraining |
| `roll_stack2` | `stack2`'s fixed meta-model on rolling base models |

## Rolling retraining
- **Cadence:** a refit on the first of each calendar month, on all split games dated before it. Games
  before the first refit use the full-train model.
- **Fixed from train-only searches:** each model's parameters and rounds. Nothing is re-tuned, and no
  early stopping uses held-out games.
- **Held-out labels:** later predictions train on earlier test and validation games, as a live system
  would. Validation predictions come from models trained through the test period. So rolling rows are
  not "trained on train only", though nothing is tuned on held-out data.
- `roll_stack2` uses `stack2`'s meta-model, fit on train out-of-fold rows only.

## Comparisons
Accuracy and AUC difference CIs on test and validation for:
- `pregame_elo2`, `pregame_elo2_player`, `repo_elo2_player` and `stack2` against `repo`, `pregame` and
  round 1's `stack`;
- `pregame_elo2_player` against `pregame_elo2`, the effect of player features;
- `roll_repo` against `repo`, `roll_pregame_elo2_player` against `pregame_elo2_player`, and `roll_stack2`
  against `stack2`, the effect of retraining;
- `roll_pregame_elo2_player` and `roll_stack2` against `roll_repo`, and against `repo`;
- market − `pregame_elo2_player`, market − `stack2`, market − `roll_stack2`.

## Slices
The seven round 1 slices, unchanged, for every new row against `repo`. No other slices.

## Reporting
All rows and comparisons are reported, including nulls, with the count of intervals excluding zero
against what chance would give. New SHAP group shares for `pregame_elo2_player` on test are reported,
adding a Player group. Hypothesis 2 is not retested.

## Possible fresh holdout (not scored without team approval)
The game logs hold 768 games after 2026-01-07 that no analysis has scored. `dataset_2012-26` and the odds
may not cover them, so the baseline and market may be unavailable. The primary split above stays primary.
If approved, the holdout would get a single look at `pregame`, `pregame_elo2_player` and
`roll_pregame_elo2_player`, with no change to any model after seeing it.

## Deviations
None yet.
