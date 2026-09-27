# Logistic Regression baseline: reproducibility check

**Previously reported:** `src/Train-Models/Logistic_Regression_ML.py --dataset dataset_2012-24_new`
→ **0.6409** test accuracy, **0.6324** test log loss.
**Re-run now:** **0.6369** / **0.6303**, with the script unchanged since the fork was cloned
(last touched upstream in `8e36b0b`, 2026-01-08).

## What the script assumes vs what is installed

`requirements.txt` is the only version spec the script has. It cannot be installed on this machine:
scikit-learn 1.3.1 and tensorflow 2.14 have no Python 3.13 wheels.

| Package | `requirements.txt` | Python 3.13 (`python3`, `.venv`) | Apple system Python 3.9 (`/usr/bin/python3`) |
|---|---|---|---|
| Python | (not pinned) | 3.13.7 | 3.9.6 |
| scikit-learn | 1.3.1 | 1.5.2 | 1.6.1 |
| numpy | (not pinned) | 2.5.2 | 2.0.2 |
| pandas | 2.1.1 | 3.0.5 | 2.3.1 |
| scipy | (not pinned) | 1.18.1 | 1.13.1 |
| xgboost | 2.0.0 (not used by the LR script) | 3.4.1 | — |

The machine has two separate Python environments with scikit-learn installed. The system Python
3.9 one was set up by earlier `/usr/bin/python3 -m pip install scikit-learn` commands; the
python.org 3.13 one had its packages installed on 2026-09-05, with scikit-learn about three minutes
after the rest.

## Tests

Every run uses `--dataset dataset_2012-24_new` unless the row says otherwise.

| Run | Test acc | Test log loss |
|---|---|---|
| Python 3.13, sklearn 1.5.2 (defaults: 50 trials, seed 42, sigmoid) | 0.6369 | 0.6303 |
| Python 3.9, sklearn 1.6.1, numpy 2.0.2, pandas 2.3.1 (same defaults) | 0.6369 | 0.6303 |
| `--calibration isotonic` | 0.6343 | 0.6980 |
| `--calibration none` | 0.6310 | 0.6414 |
| `--trials 25` / `--trials 100` | 0.6369 / 0.6376 | 0.6303 / 0.6307 |
| `--seed 0` / `--seed 1` / `--seed 7` | 0.6349 / 0.6376 / 0.6369 | 0.6310 / 0.6306 / 0.6308 |
| `--dataset dataset_2012-23` (sigmoid / isotonic / none) | 0.6148 / 0.6220 / 0.5811 | 0.6529 / 0.6782 / 0.6772 |
| `--dataset dataset_2012-24` (sigmoid / isotonic / none) | 0.6127 / 0.6213 / 0.5763 | 0.6547 / 0.6827 / 0.6789 |
| `--dataset dataset_2012-26` (sigmoid / isotonic), before the 2024-26 leakage fix | 0.6889 / 0.6718 | 0.5829 / 0.5911 |

The last digit of log loss can wobble between runs (0.6303 vs 0.6304) because the `liblinear`
solver used for L1 trials is not seeded.

## Conclusion

- **Package versions are not the cause.** Two environments that differ in Python, scikit-learn,
  numpy, pandas and scipy give identical results (0.6369 / 0.6303).
- **No combination of dataset table, calibration, trial count or seed reproduces 0.6409 / 0.6324.**
  Accuracy stays in 0.6349–0.6376 across the `dataset_2012-24_new` variants tried.
- **The cause is therefore something about the earlier run itself that is not visible here.** It
  could have been a locally modified copy of the script, a notebook, a different working copy of
  `dataset.sqlite`, or arguments not tried above. The exact command and working directory used for
  0.6409 would settle it.
- **Which number to cite:** 0.6369 / 0.6303 is the reproducible value from the committed script
  and data. Nothing has been re-pinned, and 0.6409 has not been replaced anywhere.
- **Either value is on a different test set from the XGBoost comparisons.** The LR script scores
  the last 10% of `dataset_2012-24_new` (1,512 games, 2023-24), not the fixed test split. See
  `docs/dataset-tables.md`.
