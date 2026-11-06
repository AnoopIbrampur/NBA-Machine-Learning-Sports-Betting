# Modified Implementation: analysis plan

Fixed before any Elo, stacking, calibration or window 5/7 result was computed. The commit date of
this file is the record of that. Any change after results exist is listed under "Deviations" at the
end, with its reason.

All new outputs go in `results/pregame/enhanced/`. Existing models and result files stay as they are.
The official feature set stays pre-game window 20. Everything here is a labelled extension. The
market is a reference row, never a model of ours and never a stacking input.

## Shared rules
- Same split (`Data/splits/split_keys.csv`), same 16,940 games, same seed (42).
- XGBoost rows use the existing protocol: 40-trial random search scored by 5-fold walk-forward CV on
  train only, then a refit on all of train.
- Test and validation are never used for tuning, selection, stacking fits or calibration fits.
- Comparisons use `paired_bootstrap` (2,000 resamples, same seed).

## Elo
- **Source:** Alves & Barbosa (2025) §3.1.4: all teams start at 1500, ratings carry over in full between
  seasons, and k is usually 16 to 32.
- **Update:** R + k × (S − E). The paper's Equation (1) prints k × (S − E + R), which we read as a typo,
  because it would multiply the rating by k.
- **Expected score:** the paper doesn't define E. We assume the standard 1 / (1 + 10^((Rb − Ra) / 400)),
  with no home-court term, since the paper gives none.
- **Games:** every logged game in date order, including playoffs, play-in, the 2019-20 restart and
  neutral-site games. Ratings are keyed by franchise `TEAM_ID`.
- **Features:** `elo_home`, `elo_away` and `elo_diff`, each from before the game.
- **Online use:** ratings keep updating through test and validation games, as rolling features do.
  Each game's value uses only earlier games.
- **Burn-in:** the logs start in 2012-13, so every team starts that season at 1500.
- **Choosing k:** from {16, 20, 24, 28, 32}, the one with the lowest mean log loss of a logistic
  regression of home win on `elo_diff`, under 5-fold walk-forward CV on train only. Lower k wins a tie.

## New rows

| Row | Definition |
|---|---|
| `elo_logit` | Logistic regression of home win on `elo_diff`, fit on train |
| `pregame_elo` | Pre-game window 20 + Elo |
| `repo_elo` | Repo baseline + Elo |
| `pregame_sched_travel_elo` | Pre-game window 20 + fatigue + travel + Elo |
| `stack` | Stack of `repo`, `pregame` and `pregame_elo` |
| `pregame_w5`, `pregame_w7` | Ablations: official feature set with rolling windows 5 and 7 |

## Stacking
- **Base models:** `repo`, `pregame`, `pregame_elo`, using each one's saved best parameters and rounds.
- **Out-of-fold predictions:** 5-fold `TimeSeriesSplit` on train in date order. Each prediction comes
  from a model fit on strictly earlier rows. The first block gets no prediction and isn't used.
- **Meta-model:** logistic regression on logit(p) of each base model, C = 1e6, fit on out-of-fold rows
  only.
- **Scoring:** test and validation use the full-train base models and the fitted meta-model.
- **Known limit:** base parameters were tuned on the same train folds, so out-of-fold predictions may be
  slightly optimistic. This stays within train.

## Comparisons
Accuracy and AUC difference CIs on test and validation for:
- every new row against `repo` and against `pregame`;
- `pregame_sched_travel_elo` against `pregame_sched_travel`;
- market − `pregame_elo` and market − `stack`.

**What counts as a gain:** the CI for the same metric excludes 0 in the positive direction on both test
and validation. Anything else is reported as no demonstrated gain. Log loss and Brier are reported
without CIs.

## Slices
The six slices of findings §7e, unchanged, plus one new slice:

7. Either team has played fewer than 20 games this season (`GP` or `GP.1` below 20).

Carryover Elo should help most there, because the rolling window is still filling. Each new XGBoost row
and `stack` get CIs against `repo` on each slice, with the same rules: no CI below n = 30, unreliable
below n = 200. No other slices will be examined.

## Hypothesis 2: "Elo strongest, schedule in the middle"
Tested on `pregame_sched_travel_elo`, test split, by mean |SHAP|. Validation is reported as a check.
- **Elo strongest:** an Elo feature ranks #1, and the Elo group has the largest share of total |SHAP|
  among four groups: Elo, win record, box-score stats, schedule (fatigue, travel, rest and b2b columns).
- **Schedule in the middle:** the median rank of the schedule features falls in the middle third of all
  features.
- **Supported** only if both hold. Elo and win % are correlated, so SHAP may split credit between them;
  the shares are read with that in mind.

## Calibration
- **Rows:** `repo`, `pregame`, `pregame_sched_travel`, `pregame_elo`, `stack`, and the market.
- **Splits:** test and validation, from saved models, with nothing refit.
- **Measures:** reliability curves and expected calibration error (ECE) over 10 equal-width probability
  bins, weighted by games, plus calibration intercept and slope from a logistic regression of home win
  on logit(p).
- **CIs:** paired bootstrap of the ECE difference, market − model and model − `repo` (2,000 resamples).

## Reporting
All rows, slices and comparisons above are reported, including nulls. Any new number enters
`docs/findings.md` from a committed result file.

## Deviations
None yet.
