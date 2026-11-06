# Implementation Platform recording: outline

Goal: show that our code gives results similar to the parent paper, Bowman, Harmon & Ashman (2023),
and be honest about where it doesn't. Submit the GitHub link and an unlisted Kaltura Zoom recording.
One member submits.

These are prompts, not a script. Midterm webcam rules likely apply: cameras on, don't read from
notes. Confirm this and any time limit on the assignment page.

## Before recording
- Use a checkout that has already run the pipeline. It takes about 15 minutes, too long to show live.
  Your main working copy works, because its game logs are already cached.
- Open these: the repo on GitHub, a terminal in that checkout, `replication.md` and
  `replication_week_restriction.md`, and `docs/lr-reproducibility.md`.
- If stats.nba.com is down, you can still record: both live commands below use the cached logs and
  features, and a fresh clone would not work.

## 1. The question (about 1 min)
- Do schedule features improve pre-game NBA win prediction beyond team strength and recent form?
- Parent paper: Bowman studies how the schedule affects outcomes. It doesn't test prediction.
- This repo is a fork; our work starts at commit `a38a84a`.

## 2. Run the code (about 2 min)
Show the README's setup and Reproduce sections, then run these live:

```bash
.venv/bin/python -m unittest discover -s Tests -t .
.venv/bin/python -m src.Pregame.schedule_replication
```

- Tests take about a second: 64 tests, 0 skipped.
- The replication takes about 4 seconds on the 11,650 training-split games.
- `git status` then shows `replication.json` as modified. That's float rounding of about 1e-15; the
  table is unchanged. Show `git diff --stat -- results/pregame/schedule/replication.md`, which is empty.
- Point out that everything else was produced by the same commands, and the results are committed.

## 3. Bowman comparison (about 3 min)
Walk through the table on screen.

| Variable | Bowman | Ours (95% CI) | Say |
|---|---|---|---|
| Visitor on a back-to-back | 1.506 | 1.311 [1.180, 1.457] | Same direction, significant, smaller. Our interval doesn't reach their value. |
| Home on a back-to-back | 0.806 | 0.897 [0.780, 1.031] | Same direction, not significant. |
| Home crossed 1+ zone eastward, one day ago | 0.693 | 0.864 [0.605, 1.234] | Same direction, not significant, only 185 games. |
| Visitor long trip | 1.261 | 1.319 [1.035, 1.683] | Close. Not significant with the win % control (1.155). |
| Visitor 3+ games in the last week | 1.153 | 0.995 [0.880, 1.124] | Does not replicate. |
| Home 4+ games in the last week | 0.914 | 1.041 [0.940, 1.152] | Does not replicate. |

## 4. Be honest about the differences (about 2 min)
- **Partial match.** Call it that out loud.
- **Different data:** 2012-22 against their 2000-01 to 2018-19.
- **Different model:** 8 variables adapted from their 20.
- **Different controls:** team-by-season dummies against their home and visiting team dummies.
- **Week variables:** our main table doesn't apply Bowman's "not a back-to-back game" restriction.
  Show the labelled sensitivity check: 0.981 and 1.060, still near 1, so that isn't the explanation.
  Visitor back-to-back there is 1.288 [1.102, 1.5057], with the upper bound just under Bowman's 1.506.

## 5. Corrections to the midterm (about 1 min)
Say both plainly.

- **Baseline.** We said we reproduced 0.6409 accuracy and 0.6324 log loss. That isn't reproducible:
  the unchanged script gives 0.6369 and 0.6303. It isn't a package-version issue
  (`docs/lr-reproducibility.md`). That script also scores a different test set from our XGBoost
  comparisons.
- **The 1.31 vs 1.5 gap.** We said it was traced to season range and controls. We never tested that.
  Those are likely reasons, not established ones.

## 6. Why it matters (about 1 min)
- The effect is real: the home team wins 61.9% when only the visitor is on a back-to-back and 50.0%
  when only the home team is.
- It doesn't help prediction: 0.6412 accuracy / 0.6759 AUC with schedule features, 0.6435 / 0.6746
  for the official pre-game model on test.
- A market line reaches 0.6790 / 0.7255. With its probability controlled for, back-to-backs add
  nothing significant. It's a reference row, not our model.

## 7. What comes next (about 30 s)
For the Modified Implementation: Elo, stacking and calibration, as new labelled results. Existing
results stay as they are.

## Don'ts
- Don't say "replicated" without "partly".
- Don't say "closing line". Say "a market line".
- Don't quote 0.6409 / 0.6324 as our baseline.
- Don't say who did which work.
