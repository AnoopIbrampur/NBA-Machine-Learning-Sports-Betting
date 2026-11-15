# Recording outline: Modified Implementation (team discussion, about 10 minutes)

These are talking points, not a script. Every number below is in `docs/report/comparison-table.md`
(generated from `results/pregame/enhanced*/`) or `docs/findings.md` §11–§12. Say "a market line",
never "the closing line". Say "partly replicated" for Bowman, never "replicated".

## Before recording
- Use the main working copy. Its game logs, player logs and saved models are already cached, so the
  live commands take seconds. A fresh clone would need the full pipeline first (about 30 minutes).
- `git status` will show `results/pregame/schedule/replication.json` as modified. That's float noise
  from an earlier rerun (about 1e-15) and changes no reported number. Discard it before recording if
  you want a clean status: `git checkout -- results/pregame/schedule/replication.json`.
- Open these, in this order:
  1. The repo on GitHub, branch `ds340w-pregame`
  2. `docs/report/comparison-table.md` (or the HTML version in a browser, which reads more easily)
  3. `docs/enhanced-plan.md` and `docs/enhanced-plan-2.md`
  4. `docs/findings.md` at §11 and §12
  5. A terminal in the repo folder
  6. The paper PDF, at the "Proposed Novel Approach" section

## 1. Where we started (about 1 min). Speaker 1
- The official model is XGBoost on rolling 20-game averages. It tied the repo's season-to-date
  baseline on both held-out periods, and the schedule features didn't help.
- The question for this stage: can a different input block beat the baseline, using information that
  rolling averages miss?

## 2. What we changed and why (about 2 min). Speaker 2
- **Elo ratings.** Alves and Barbosa found Elo was their top pre-game input. Elo carries strength
  across seasons and adjusts for opponents. k = 28 was chosen on train only.
- **Elo variant (round 2).** Adds a home-court term, regression toward 1500 each season and a margin of
  victory term. The setting chosen on train was k = 16, home advantage 100, one-third regression.
- **Player data.** 384,340 player-game rows: rotation strength, who played last game, and roster
  continuity. All of it uses earlier games only.
- **Stacking.** A logistic regression on the base models' out-of-fold predictions, fit on train.
- **Rolling retraining.** A monthly refit, the way a live system would run.
- Also: windows 5 and 7 as ablations, and a calibration check.

## 3. How we kept it honest (about 1.5 min). Speaker 1
- Show `docs/enhanced-plan.md` and `docs/enhanced-plan-2.md`. Both were committed before any of their
  results existed; the git history shows the order.
- Gain rule, fixed in advance: the paired-bootstrap 95% CI has to sit above zero on **both** test and
  validation.
- Round 2 isn't blind. Round 1 results motivated it, so we trust its gains less.
- Run live:

```bash
.venv/bin/python -m unittest discover -s Tests -t .
```

  108 tests, 0 skipped, about 4 seconds. The Elo and player tests change a game's own and later
  results and check that its features don't move.

## 4. The comparison table (about 3 min). Both speakers
Run live, then open the output:

```bash
.venv/bin/python scripts/comparison_table.py
```

- **Speaker 2, Table A:** Elo alone matches the baseline's test AUC, 0.6793, with one feature. The
  round 2 primary row is 0.6918 test and 0.7206 validation AUC, against 0.6793 and 0.7089 for the
  baseline. The market line is still on top at 0.7255 and 0.7420.
- **Speaker 1, Table B (against the baseline):**
  - No row beats the baseline on accuracy on either split.
  - The primary row passes on test AUC but not validation, [−0.0008, +0.0237]. **The primary test was
    not met**, and we report it that way.
  - Three secondary rows pass on AUC: repo + Elo variant + player, stack 2, and rolling stack 2. All
    three include the baseline's own features or predictions.
- **Speaker 2, Table C (against the official pre-game model):** every round 2 static row beats it on
  AUC on both splits, as do round 1's repo + Elo and stack.
- **Speaker 1, Tables D and E (fresh holdout, single look):** 766 games after the split, scored once
  under rules fixed first. Primary vs official: accuracy 0.7076 vs 0.6749, AUC 0.7632 vs 0.7393, both
  CIs above zero. The baseline and market can't be scored there, and the holdout doesn't separate
  Elo from player data.

## 5. What worked and what didn't (about 1.5 min). Speaker 2
- **Worked:** Elo. It's the top SHAP feature (60.4% of importance on test), so hypothesis 2 is half
  supported. Elo is the strongest predictor, but schedule features rank 31 of 43, below the middle.
- **Didn't work:**
  - Rolling retraining: every AUC CI against the static models includes zero.
  - Player features alone: [−0.0027, +0.0081] test, [−0.0033, +0.0109] validation.
  - Windows 5 and 7: significantly worse than the baseline.
  - Calibration: every ECE difference CI includes zero, so the rows can't be ranked.
- A market line is still ahead on AUC on both splits.

## 6. Demo and next steps (about 1 min). Speaker 1
Run live (about 10 seconds):

```bash
.venv/bin/python -m src.Pregame.demo --date 2026-03-15 --team Thunder --reasons 3
```

- It labels the game as never used to build or choose a model, and shows both models' picks, the top
  reasons and the actual winner.
- **Next:** game-day injury and rest information, which a market line likely prices and our
  previous-game player features miss. A wider Elo grid, since the chosen settings sat at its edge.
  Any new round gets scored on games none of our earlier rounds have seen.
