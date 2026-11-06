# Calibration on test and validation

From saved predictions; nothing is refit. ECE uses 10 equal-width bins weighted by games. Intercept and slope come from a logistic regression of home win on logit(p); a calibrated model has 0 and 1. Plot: `reliability.png`.

| Model | Split | n | ECE | Intercept | Slope |
|---|---|---|---|---|---|
| Baseline | test | 3386 | 0.0140 | -0.020 | 1.073 |
| Pre-game w20 | test | 3386 | 0.0218 | -0.045 | 1.038 |
| + fatigue + travel | test | 3386 | 0.0246 | -0.027 | 1.006 |
| + Elo | test | 3386 | 0.0142 | -0.067 | 1.057 |
| Stack | test | 3386 | 0.0143 | +0.015 | 1.037 |
| Market (reference) | test | 3386 | 0.0126 | +0.008 | 0.974 |
| Baseline | validation | 1681 | 0.0294 | -0.130 | 1.258 |
| Pre-game w20 | validation | 1681 | 0.0218 | -0.121 | 1.103 |
| + fatigue + travel | validation | 1681 | 0.0299 | -0.103 | 1.065 |
| + Elo | validation | 1681 | 0.0364 | -0.153 | 1.087 |
| Stack | validation | 1681 | 0.0181 | -0.077 | 1.116 |
| Market (reference) | validation | 1681 | 0.0279 | -0.028 | 0.977 |

## Paired bootstrap 95% CIs of the ECE difference, 2000 resamples

| Difference | Split | CI |
|---|---|---|
| Market − Baseline | test | [-0.0143, +0.0154] |
| Pre-game w20 − Baseline | test | [-0.0106, +0.0185] |
| Market − Pre-game w20 | test | [-0.0206, +0.0117] |
| + fatigue + travel − Baseline | test | [-0.0084, +0.0209] |
| Market − + fatigue + travel | test | [-0.0229, +0.0095] |
| + Elo − Baseline | test | [-0.0148, +0.0135] |
| Market − + Elo | test | [-0.0140, +0.0159] |
| Stack − Baseline | test | [-0.0126, +0.0123] |
| Market − Stack | test | [-0.0140, +0.0156] |
| Market − Baseline | validation | [-0.0231, +0.0236] |
| Pre-game w20 − Baseline | validation | [-0.0199, +0.0142] |
| Market − Pre-game w20 | validation | [-0.0184, +0.0262] |
| + fatigue + travel − Baseline | validation | [-0.0184, +0.0208] |
| Market − + fatigue + travel | validation | [-0.0255, +0.0219] |
| + Elo − Baseline | validation | [-0.0131, +0.0269] |
| Market − + Elo | validation | [-0.0295, +0.0177] |
| Stack − Baseline | validation | [-0.0222, +0.0121] |
| Market − Stack | validation | [-0.0154, +0.0275] |
