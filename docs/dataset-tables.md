# `dataset_2012-24_new` vs `dataset_2012-26`

Both tables live in `Data/dataset.sqlite`. The earlier Logistic Regression number (0.6409 acc)
was computed on `dataset_2012-24_new`, while the fixed train/test/validation split
(`Data/splits/split_keys.csv`) was cut from `dataset_2012-26`.

## They are the same data, one extended by ~1.5 seasons

| | `dataset_2012-24_new` | `dataset_2012-26` |
|---|---|---|
| Rows (games) | 15,115 | 16,969 |
| Date range | 2012-11-04 → 2024-04-28 | 2012-11-04 → 2026-01-07 |
| Seasons | 2012-13 … 2023-24 | 2012-13 … 2025-26 (2025-26 through Jan 7) |
| Extra column | — | `index.1` (row index, dropped by every model) |

Games per season are identical for 2012-13 through 2023-24. `dataset_2012-26` adds 1,314 games from
2024-25 and 540 from 2025-26.

- Every one of the 15,115 games in `dataset_2012-24_new` is in `dataset_2012-26`, in the same row order.
- On those shared games every feature value is identical, except `index` (a row index both scripts
  drop) and 3 rest-day values (`Days-Rest-Home` ×2, `Days-Rest-Away` ×1).

So this is a naming inconsistency, not two different datasets, **up to 2023-24**.

## The part that is not the same: 2024-25 onward leaks the outcome

The rows only `dataset_2012-26` has differ in one important way. From 2024-25 on, each row's team stats
already include the game being predicted (see `results/pregame/leakage_audit.md`). For 2012-13 through
2023-24 they exclude it. `dataset_2012-24_new` stops before the problem, so it is leak-free.

## Is the split still valid?

| Split | Dates | Games | Repo (baseline) features | Pre-game features |
|---|---|---|---|---|
| train | 2012-11-04 → 2022-01-06 | 11,878 | leak-free | leak-free |
| test | 2022-01-07 → 2024-11-13 | 3,394 | leak-free except the last 156 games (2024-25) | leak-free |
| validation | 2024-11-13 → 2026-01-07 | 1,697 | **all leaky** | leak-free |

- **For the pre-game models, yes.** They are built from game logs, not from these tables, and are
  leak-free on every split.
- **For the baseline, only on leak-free games.** Its validation scores are inflated. Compare the
  baseline against the pre-game models on the 3,234 leak-free test games
  (`python -m src.Pregame.leakage_audit`).
- **The earlier LR number is not comparable to the split results either way.** The LR script makes
  its own 90/10 split of whatever table it is given, so on `dataset_2012-24_new` its test set is the
  last 1,512 games of that table (2023-24). That is a different set of games from the split's test
  set. Running the same script on `dataset_2012-26` gives 0.6889 accuracy, because its last 10% falls
  in the leaky 2024-26 period.

## Re-check

```bash
.venv/bin/python -m src.Pregame.leakage_audit
```
