# Modified Implementation: Elo, stacking and window ablations (window 20 unless labelled)

Plan fixed in advance: `docs/enhanced-plan.md`. Existing models are the saved ones, unchanged. Positive class = home win, threshold 0.5.

Elo k chosen on train only: **28**. Walk-forward log loss by k: 16: 0.62442, 20: 0.62265, 24: 0.62195, 28: 0.62186, 32: 0.62212.

## Full test and validation sets

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 3386 | 0.6400 | 0.6570 | 0.7577 | 0.7038 | 0.6793 | 0.6350 | 0.2221 |
| Pre-game w20 (official) | test | 3386 | 0.6435 | 0.6513 | 0.7928 | 0.7151 | 0.6746 | 0.6372 | 0.2230 |
| Pre-game w20 + fatigue + travel | test | 3386 | 0.6412 | 0.6532 | 0.7766 | 0.7095 | 0.6759 | 0.6365 | 0.2226 |
| Elo only: logistic on elo_diff | test | 3386 | 0.6364 | 0.6431 | 0.7996 | 0.7129 | 0.6793 | 0.6344 | 0.2220 |
| Pre-game w20 + Elo | test | 3386 | 0.6491 | 0.6531 | 0.8069 | 0.7219 | 0.6875 | 0.6303 | 0.2200 |
| Repo + Elo | test | 3386 | 0.6462 | 0.6608 | 0.7666 | 0.7098 | 0.6883 | 0.6290 | 0.2195 |
| Pre-game w20 + fatigue + travel + Elo | test | 3386 | 0.6453 | 0.6520 | 0.7970 | 0.7172 | 0.6880 | 0.6303 | 0.2200 |
| Stack: repo, pre-game, pre-game + Elo | test | 3386 | 0.6480 | 0.6664 | 0.7535 | 0.7073 | 0.6888 | 0.6292 | 0.2195 |
| Ablation: pre-game window 5 | test | 3386 | 0.6170 | 0.6255 | 0.8006 | 0.7023 | 0.6372 | 0.6550 | 0.2315 |
| Ablation: pre-game window 7 | test | 3386 | 0.6288 | 0.6367 | 0.7970 | 0.7079 | 0.6546 | 0.6479 | 0.2281 |
| Market: devigged moneyline (reference) | test | 3386 | 0.6790 | 0.6914 | 0.7786 | 0.7325 | 0.7255 | 0.6036 | 0.2081 |
| Baseline: repo season-to-date | validation | 1681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 |
| Pre-game w20 (official) | validation | 1681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 |
| Pre-game w20 + fatigue + travel | validation | 1681 | 0.6496 | 0.6538 | 0.7587 | 0.7024 | 0.6998 | 0.6268 | 0.2185 |
| Elo only: logistic on elo_diff | validation | 1681 | 0.6449 | 0.6438 | 0.7795 | 0.7052 | 0.7008 | 0.6254 | 0.2182 |
| Pre-game w20 + Elo | validation | 1681 | 0.6478 | 0.6470 | 0.7784 | 0.7066 | 0.7084 | 0.6213 | 0.2162 |
| Repo + Elo | validation | 1681 | 0.6514 | 0.6583 | 0.7489 | 0.7007 | 0.7123 | 0.6178 | 0.2147 |
| Pre-game w20 + fatigue + travel + Elo | validation | 1681 | 0.6550 | 0.6544 | 0.7773 | 0.7106 | 0.7074 | 0.6219 | 0.2164 |
| Stack: repo, pre-game, pre-game + Elo | validation | 1681 | 0.6562 | 0.6644 | 0.7456 | 0.7027 | 0.7132 | 0.6169 | 0.2143 |
| Ablation: pre-game window 5 | validation | 1681 | 0.6193 | 0.6202 | 0.7773 | 0.6899 | 0.6712 | 0.6446 | 0.2268 |
| Ablation: pre-game window 7 | validation | 1681 | 0.6312 | 0.6303 | 0.7817 | 0.6979 | 0.6749 | 0.6424 | 0.2257 |
| Market: devigged moneyline (reference) | validation | 1681 | 0.6811 | 0.6904 | 0.7522 | 0.7200 | 0.7420 | 0.5925 | 0.2040 |

## Paired bootstrap 95% CIs (row − reference), 2000 resamples

| Row | Reference | Split | Accuracy diff CI | AUC diff CI |
|---|---|---|---|---|
| Elo only: logistic on elo_diff | Baseline: repo season-to-date | test | [-0.0163, +0.0095] | [-0.0102, +0.0104] |
| Pre-game w20 + Elo | Baseline: repo season-to-date | test | [-0.0032, +0.0210] | [-0.0010, +0.0171] |
| Repo + Elo | Baseline: repo season-to-date | test | [-0.0038, +0.0157] | [+0.0022, +0.0159] **excl. 0** |
| Pre-game w20 + fatigue + travel + Elo | Baseline: repo season-to-date | test | [-0.0074, +0.0168] | [-0.0009, +0.0181] |
| Stack: repo, pre-game, pre-game + Elo | Baseline: repo season-to-date | test | [-0.0009, +0.0165] | [+0.0036, +0.0152] **excl. 0** |
| Ablation: pre-game window 5 | Baseline: repo season-to-date | test | [-0.0396, -0.0065] **excl. 0** | [-0.0589, -0.0270] **excl. 0** |
| Ablation: pre-game window 7 | Baseline: repo season-to-date | test | [-0.0269, +0.0041] | [-0.0397, -0.0100] **excl. 0** |
| Elo only: logistic on elo_diff | Pre-game w20 (official) | test | [-0.0204, +0.0065] | [-0.0069, +0.0162] |
| Pre-game w20 + Elo | Pre-game w20 (official) | test | [-0.0050, +0.0168] | [+0.0047, +0.0212] **excl. 0** |
| Repo + Elo | Pre-game w20 (official) | test | [-0.0103, +0.0151] | [+0.0033, +0.0243] **excl. 0** |
| Pre-game w20 + fatigue + travel + Elo | Pre-game w20 (official) | test | [-0.0092, +0.0127] | [+0.0043, +0.0222] **excl. 0** |
| Stack: repo, pre-game, pre-game + Elo | Pre-game w20 (official) | test | [-0.0077, +0.0171] | [+0.0049, +0.0237] **excl. 0** |
| Ablation: pre-game window 5 | Pre-game w20 (official) | test | [-0.0408, -0.0121] **excl. 0** | [-0.0510, -0.0248] **excl. 0** |
| Ablation: pre-game window 7 | Pre-game w20 (official) | test | [-0.0275, -0.0024] **excl. 0** | [-0.0311, -0.0095] **excl. 0** |
| Pre-game w20 + fatigue + travel + Elo | Pre-game w20 + fatigue + travel | test | [-0.0074, +0.0157] | [+0.0030, +0.0205] **excl. 0** |
| Market: devigged moneyline (reference) | Pre-game w20 + Elo | test | [+0.0159, +0.0437] **excl. 0** | [+0.0271, +0.0494] **excl. 0** |
| Market: devigged moneyline (reference) | Stack: repo, pre-game, pre-game + Elo | test | [+0.0174, +0.0455] **excl. 0** | [+0.0263, +0.0479] **excl. 0** |
| Elo only: logistic on elo_diff | Baseline: repo season-to-date | validation | [-0.0297, +0.0077] | [-0.0225, +0.0062] |
| Pre-game w20 + Elo | Baseline: repo season-to-date | validation | [-0.0262, +0.0101] | [-0.0125, +0.0116] |
| Repo + Elo | Baseline: repo season-to-date | validation | [-0.0190, +0.0101] | [-0.0063, +0.0129] |
| Pre-game w20 + fatigue + travel + Elo | Baseline: repo season-to-date | validation | [-0.0184, +0.0167] | [-0.0140, +0.0109] |
| Stack: repo, pre-game, pre-game + Elo | Baseline: repo season-to-date | validation | [-0.0131, +0.0149] | [-0.0037, +0.0125] |
| Ablation: pre-game window 5 | Baseline: repo season-to-date | validation | [-0.0595, -0.0137] **excl. 0** | [-0.0595, -0.0180] **excl. 0** |
| Ablation: pre-game window 7 | Baseline: repo season-to-date | validation | [-0.0482, -0.0024] **excl. 0** | [-0.0528, -0.0144] **excl. 0** |
| Elo only: logistic on elo_diff | Pre-game w20 (official) | validation | [-0.0149, +0.0220] | [-0.0135, +0.0177] |
| Pre-game w20 + Elo | Pre-game w20 (official) | validation | [-0.0095, +0.0208] | [-0.0012, +0.0208] |
| Repo + Elo | Pre-game w20 (official) | validation | [-0.0077, +0.0280] | [+0.0003, +0.0270] **excl. 0** |
| Pre-game w20 + fatigue + travel + Elo | Pre-game w20 (official) | validation | [-0.0030, +0.0303] | [-0.0033, +0.0206] |
| Stack: repo, pre-game, pre-game + Elo | Pre-game w20 (official) | validation | [-0.0024, +0.0327] | [+0.0029, +0.0266] **excl. 0** |
| Ablation: pre-game window 5 | Pre-game w20 (official) | validation | [-0.0422, -0.0012] **excl. 0** | [-0.0453, -0.0104] **excl. 0** |
| Ablation: pre-game window 7 | Pre-game w20 (official) | validation | [-0.0303, +0.0089] | [-0.0392, -0.0095] **excl. 0** |
| Pre-game w20 + fatigue + travel + Elo | Pre-game w20 + fatigue + travel | validation | [-0.0101, +0.0214] | [-0.0046, +0.0195] |
| Market: devigged moneyline (reference) | Pre-game w20 + Elo | validation | [+0.0149, +0.0529] **excl. 0** | [+0.0194, +0.0484] **excl. 0** |
| Market: devigged moneyline (reference) | Stack: repo, pre-game, pre-game + Elo | validation | [+0.0077, +0.0422] **excl. 0** | [+0.0153, +0.0426] **excl. 0** |

## Stack

```json
{
  "members": [
    "repo",
    "pregame",
    "pregame_elo"
  ],
  "n_meta_rows": 9890,
  "intercept": -0.07280257271045104,
  "coefficients": {
    "repo": 0.4853286362428067,
    "pregame": -0.19120857433081426,
    "pregame_elo": 0.7693752876804669
  }
}
```

## Hypothesis 2 (`pregame_sched_travel_elo`, mean |SHAP|)

- **test:** top feature `elo_diff`; best Elo rank 1; schedule median rank 31 of 43; shares: Elo 60.4%, Box-score stats 21.6%, Schedule 10.0%, Win record 8.0%. Elo strongest: True. Schedule in the middle: False. Supported: False.
- **validation:** top feature `elo_diff`; best Elo rank 1; schedule median rank 31 of 43; shares: Elo 62.6%, Box-score stats 20.4%, Schedule 9.2%, Win record 7.8%. Elo strongest: True. Schedule in the middle: False. Supported: False.

## Slices

CIs are row − repo baseline. n < 200: unreliable; n < 30: no CI.

| Split | Slice | Model | n | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | AUC diff vs repo CI | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1. neither team on a b2b | Baseline: repo season-to-date | 2445 | 0.6348 | 0.6708 | 0.6385 | 0.2238 | - | - |  |
| test | 1. neither team on a b2b | Pre-game w20 (official) | 2445 | 0.6401 | 0.6660 | 0.6405 | 0.2245 | - | - |  |
| test | 1. neither team on a b2b | Pre-game w20 + Elo | 2445 | 0.6450 | 0.6797 | 0.6337 | 0.2215 | [-0.0049, +0.0241] | [-0.0026, +0.0201] |  |
| test | 1. neither team on a b2b | Repo + Elo | 2445 | 0.6429 | 0.6796 | 0.6334 | 0.2215 | [-0.0041, +0.0200] | [+0.0004, +0.0171] **excl. 0** |  |
| test | 1. neither team on a b2b | Pre-game w20 + fatigue + travel + Elo | 2445 | 0.6409 | 0.6806 | 0.6335 | 0.2214 | [-0.0086, +0.0201] | [-0.0016, +0.0214] |  |
| test | 1. neither team on a b2b | Stack: repo, pre-game, pre-game + Elo | 2445 | 0.6413 | 0.6809 | 0.6331 | 0.2213 | [-0.0037, +0.0172] | [+0.0033, +0.0170] **excl. 0** |  |
| test | 1. neither team on a b2b | Ablation: pre-game window 5 | 2445 | 0.6188 | 0.6303 | 0.6569 | 0.2323 | [-0.0344, +0.0029] | [-0.0586, -0.0219] **excl. 0** |  |
| test | 1. neither team on a b2b | Ablation: pre-game window 7 | 2445 | 0.6294 | 0.6501 | 0.6493 | 0.2286 | [-0.0237, +0.0127] | [-0.0383, -0.0041] **excl. 0** |  |
| test | 2. away team only on a b2b | Baseline: repo season-to-date | 458 | 0.6616 | 0.6818 | 0.6202 | 0.2150 | - | - |  |
| test | 2. away team only on a b2b | Pre-game w20 (official) | 458 | 0.6659 | 0.6777 | 0.6202 | 0.2149 | - | - |  |
| test | 2. away team only on a b2b | Pre-game w20 + Elo | 458 | 0.6790 | 0.6865 | 0.6158 | 0.2130 | [-0.0197, +0.0546] | [-0.0200, +0.0295] |  |
| test | 2. away team only on a b2b | Repo + Elo | 458 | 0.6572 | 0.6853 | 0.6164 | 0.2134 | [-0.0328, +0.0240] | [-0.0153, +0.0216] |  |
| test | 2. away team only on a b2b | Pre-game w20 + fatigue + travel + Elo | 458 | 0.6703 | 0.6852 | 0.6162 | 0.2133 | [-0.0262, +0.0459] | [-0.0222, +0.0290] |  |
| test | 2. away team only on a b2b | Stack: repo, pre-game, pre-game + Elo | 458 | 0.6681 | 0.6880 | 0.6176 | 0.2139 | [-0.0175, +0.0328] | [-0.0087, +0.0214] |  |
| test | 2. away team only on a b2b | Ablation: pre-game window 5 | 458 | 0.6092 | 0.6171 | 0.6474 | 0.2278 | [-0.0961, -0.0087] **excl. 0** | [-0.1110, -0.0200] **excl. 0** |  |
| test | 2. away team only on a b2b | Ablation: pre-game window 7 | 458 | 0.6397 | 0.6331 | 0.6411 | 0.2249 | [-0.0655, +0.0240] | [-0.0890, -0.0087] **excl. 0** |  |
| test | 3. home team only on a b2b | Baseline: repo season-to-date | 339 | 0.6342 | 0.7161 | 0.6412 | 0.2246 | - | - |  |
| test | 3. home team only on a b2b | Pre-game w20 (official) | 339 | 0.6342 | 0.6943 | 0.6543 | 0.2308 | - | - |  |
| test | 3. home team only on a b2b | Pre-game w20 + Elo | 339 | 0.6283 | 0.7130 | 0.6453 | 0.2266 | [-0.0442, +0.0324] | [-0.0315, +0.0244] |  |
| test | 3. home team only on a b2b | Repo + Elo | 339 | 0.6460 | 0.7244 | 0.6322 | 0.2206 | [-0.0177, +0.0413] | [-0.0126, +0.0294] |  |
| test | 3. home team only on a b2b | Pre-game w20 + fatigue + travel + Elo | 339 | 0.6401 | 0.7129 | 0.6453 | 0.2265 | [-0.0324, +0.0472] | [-0.0323, +0.0233] |  |
| test | 3. home team only on a b2b | Stack: repo, pre-game, pre-game + Elo | 339 | 0.6490 | 0.7191 | 0.6340 | 0.2212 | [-0.0118, +0.0442] | [-0.0161, +0.0215] |  |
| test | 3. home team only on a b2b | Ablation: pre-game window 5 | 339 | 0.5929 | 0.6594 | 0.6699 | 0.2387 | [-0.0944, +0.0088] | [-0.1051, -0.0083] **excl. 0** |  |
| test | 3. home team only on a b2b | Ablation: pre-game window 7 | 339 | 0.5959 | 0.6719 | 0.6644 | 0.2360 | [-0.0914, +0.0118] | [-0.0886, -0.0002] **excl. 0** |  |
| test | 4. both on a b2b | Baseline: repo season-to-date | 144 | 0.6736 | 0.7174 | 0.6084 | 0.2107 | - | - | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 (official) | 144 | 0.6528 | 0.7422 | 0.5940 | 0.2041 | - | - | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + Elo | 144 | 0.6736 | 0.7472 | 0.5829 | 0.1997 | [-0.0486, +0.0486] | [-0.0093, +0.0700] | unreliable (n < 200) |
| test | 4. both on a b2b | Repo + Elo | 144 | 0.6667 | 0.7424 | 0.5877 | 0.2019 | [-0.0556, +0.0347] | [-0.0063, +0.0557] | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + fatigue + travel + Elo | 144 | 0.6528 | 0.7448 | 0.5847 | 0.2005 | [-0.0694, +0.0280] | [-0.0127, +0.0677] | unreliable (n < 200) |
| test | 4. both on a b2b | Stack: repo, pre-game, pre-game + Elo | 144 | 0.6944 | 0.7395 | 0.5892 | 0.2025 | [-0.0208, +0.0625] | [-0.0046, +0.0488] | unreliable (n < 200) |
| test | 4. both on a b2b | Ablation: pre-game window 5 | 144 | 0.6667 | 0.7294 | 0.6135 | 0.2124 | [-0.0833, +0.0694] | [-0.0605, +0.0919] | unreliable (n < 200) |
| test | 4. both on a b2b | Ablation: pre-game window 7 | 144 | 0.6597 | 0.7324 | 0.6082 | 0.2098 | [-0.0833, +0.0556] | [-0.0539, +0.0860] | unreliable (n < 200) |
| test | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 13 | 0.3846 | 0.3500 | 0.7950 | 0.2971 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 (official) | 13 | 0.5385 | 0.4750 | 0.7140 | 0.2596 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + Elo | 13 | 0.6154 | 0.6750 | 0.6597 | 0.2342 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Repo + Elo | 13 | 0.6154 | 0.6500 | 0.7129 | 0.2590 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue + travel + Elo | 13 | 0.6154 | 0.7000 | 0.6547 | 0.2316 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Stack: repo, pre-game, pre-game + Elo | 13 | 0.5385 | 0.6250 | 0.6987 | 0.2527 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Ablation: pre-game window 5 | 13 | 0.4615 | 0.4750 | 0.7276 | 0.2665 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Ablation: pre-game window 7 | 13 | 0.3846 | 0.4000 | 0.7887 | 0.2951 | - | - | no CI (n < 30) |
| test | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 727 | 0.6176 | 0.6767 | 0.6391 | 0.2241 | - | - |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 (official) | 727 | 0.6410 | 0.6631 | 0.6450 | 0.2269 | - | - |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + Elo | 727 | 0.6410 | 0.6709 | 0.6413 | 0.2254 | [-0.0028, +0.0495] | [-0.0257, +0.0142] |  |
| test | 6. away travelled > 1,609 km | Repo + Elo | 727 | 0.6465 | 0.6752 | 0.6369 | 0.2235 | [+0.0069, +0.0523] **excl. 0** | [-0.0174, +0.0136] |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + fatigue + travel + Elo | 727 | 0.6355 | 0.6733 | 0.6400 | 0.2248 | [-0.0083, +0.0426] | [-0.0240, +0.0164] |  |
| test | 6. away travelled > 1,609 km | Stack: repo, pre-game, pre-game + Elo | 727 | 0.6382 | 0.6758 | 0.6377 | 0.2237 | [+0.0014, +0.0413] **excl. 0** | [-0.0144, +0.0117] |  |
| test | 6. away travelled > 1,609 km | Ablation: pre-game window 5 | 727 | 0.6217 | 0.6430 | 0.6535 | 0.2310 | [-0.0316, +0.0385] | [-0.0693, +0.0019] |  |
| test | 6. away travelled > 1,609 km | Ablation: pre-game window 7 | 727 | 0.6217 | 0.6576 | 0.6473 | 0.2281 | [-0.0275, +0.0358] | [-0.0515, +0.0139] |  |
| test | 7. either team under 20 games this season | Baseline: repo season-to-date | 735 | 0.6095 | 0.6338 | 0.6503 | 0.2294 | - | - |  |
| test | 7. either team under 20 games this season | Pre-game w20 (official) | 735 | 0.6503 | 0.6351 | 0.6493 | 0.2283 | - | - |  |
| test | 7. either team under 20 games this season | Pre-game w20 + Elo | 735 | 0.6422 | 0.6660 | 0.6342 | 0.2221 | [+0.0014, +0.0653] **excl. 0** | [+0.0036, +0.0605] **excl. 0** |  |
| test | 7. either team under 20 games this season | Repo + Elo | 735 | 0.6286 | 0.6635 | 0.6368 | 0.2234 | [-0.0095, +0.0490] | [+0.0042, +0.0547] **excl. 0** |  |
| test | 7. either team under 20 games this season | Pre-game w20 + fatigue + travel + Elo | 735 | 0.6408 | 0.6685 | 0.6331 | 0.2216 | [+0.0000, +0.0639] | [+0.0038, +0.0654] **excl. 0** |  |
| test | 7. either team under 20 games this season | Stack: repo, pre-game, pre-game + Elo | 735 | 0.6299 | 0.6635 | 0.6371 | 0.2235 | [-0.0068, +0.0476] | [+0.0084, +0.0514] **excl. 0** |  |
| test | 7. either team under 20 games this season | Ablation: pre-game window 5 | 735 | 0.6408 | 0.6220 | 0.6525 | 0.2301 | [+0.0014, +0.0626] **excl. 0** | [-0.0403, +0.0161] |  |
| test | 7. either team under 20 games this season | Ablation: pre-game window 7 | 735 | 0.6299 | 0.6266 | 0.6506 | 0.2291 | [-0.0109, +0.0517] | [-0.0320, +0.0183] |  |
| validation | 1. neither team on a b2b | Baseline: repo season-to-date | 1193 | 0.6639 | 0.7140 | 0.6168 | 0.2140 | - | - |  |
| validation | 1. neither team on a b2b | Pre-game w20 (official) | 1193 | 0.6513 | 0.7091 | 0.6184 | 0.2148 | - | - |  |
| validation | 1. neither team on a b2b | Pre-game w20 + Elo | 1193 | 0.6563 | 0.7166 | 0.6130 | 0.2125 | [-0.0294, +0.0134] | [-0.0121, +0.0171] |  |
| validation | 1. neither team on a b2b | Repo + Elo | 1193 | 0.6588 | 0.7187 | 0.6114 | 0.2117 | [-0.0243, +0.0134] | [-0.0073, +0.0168] |  |
| validation | 1. neither team on a b2b | Pre-game w20 + fatigue + travel + Elo | 1193 | 0.6647 | 0.7165 | 0.6129 | 0.2124 | [-0.0210, +0.0235] | [-0.0129, +0.0176] |  |
| validation | 1. neither team on a b2b | Stack: repo, pre-game, pre-game + Elo | 1193 | 0.6614 | 0.7202 | 0.6106 | 0.2114 | [-0.0218, +0.0151] | [-0.0036, +0.0160] |  |
| validation | 1. neither team on a b2b | Ablation: pre-game window 5 | 1193 | 0.6194 | 0.6788 | 0.6379 | 0.2238 | [-0.0729, -0.0168] **excl. 0** | [-0.0601, -0.0097] **excl. 0** |  |
| validation | 1. neither team on a b2b | Ablation: pre-game window 7 | 1193 | 0.6354 | 0.6791 | 0.6371 | 0.2233 | [-0.0553, -0.0025] **excl. 0** | [-0.0580, -0.0121] **excl. 0** |  |
| validation | 2. away team only on a b2b | Baseline: repo season-to-date | 205 | 0.6585 | 0.7155 | 0.6091 | 0.2108 | - | - |  |
| validation | 2. away team only on a b2b | Pre-game w20 (official) | 205 | 0.6732 | 0.6850 | 0.6272 | 0.2181 | - | - |  |
| validation | 2. away team only on a b2b | Pre-game w20 + Elo | 205 | 0.6780 | 0.7151 | 0.6060 | 0.2094 | [-0.0244, +0.0634] | [-0.0376, +0.0344] |  |
| validation | 2. away team only on a b2b | Repo + Elo | 205 | 0.6585 | 0.7201 | 0.6021 | 0.2081 | [-0.0341, +0.0341] | [-0.0247, +0.0341] |  |
| validation | 2. away team only on a b2b | Pre-game w20 + fatigue + travel + Elo | 205 | 0.6829 | 0.7109 | 0.6089 | 0.2107 | [-0.0244, +0.0732] | [-0.0437, +0.0322] |  |
| validation | 2. away team only on a b2b | Stack: repo, pre-game, pre-game + Elo | 205 | 0.6732 | 0.7221 | 0.6014 | 0.2077 | [-0.0146, +0.0439] | [-0.0190, +0.0305] |  |
| validation | 2. away team only on a b2b | Ablation: pre-game window 5 | 205 | 0.6488 | 0.6477 | 0.6448 | 0.2264 | [-0.0780, +0.0585] | [-0.1344, -0.0060] **excl. 0** |  |
| validation | 2. away team only on a b2b | Ablation: pre-game window 7 | 205 | 0.6634 | 0.6787 | 0.6316 | 0.2203 | [-0.0635, +0.0732] | [-0.0969, +0.0188] |  |
| validation | 3. home team only on a b2b | Baseline: repo season-to-date | 192 | 0.6302 | 0.7144 | 0.6452 | 0.2268 | - | - | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 (official) | 192 | 0.6042 | 0.6985 | 0.6569 | 0.2316 | - | - | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + Elo | 192 | 0.5885 | 0.6978 | 0.6573 | 0.2320 | [-0.0885, +0.0052] | [-0.0528, +0.0195] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Repo + Elo | 192 | 0.6250 | 0.7073 | 0.6449 | 0.2270 | [-0.0417, +0.0312] | [-0.0389, +0.0215] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + fatigue + travel + Elo | 192 | 0.6042 | 0.6933 | 0.6599 | 0.2331 | [-0.0729, +0.0208] | [-0.0604, +0.0156] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Stack: repo, pre-game, pre-game + Elo | 192 | 0.6354 | 0.7078 | 0.6433 | 0.2259 | [-0.0260, +0.0365] | [-0.0326, +0.0191] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Ablation: pre-game window 5 | 192 | 0.6146 | 0.6918 | 0.6561 | 0.2323 | [-0.0730, +0.0469] | [-0.0845, +0.0365] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Ablation: pre-game window 7 | 192 | 0.6094 | 0.6930 | 0.6573 | 0.2328 | [-0.0833, +0.0417] | [-0.0774, +0.0302] | unreliable (n < 200) |
| validation | 4. both on a b2b | Baseline: repo season-to-date | 91 | 0.5934 | 0.6502 | 0.6725 | 0.2404 | - | - | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 (official) | 91 | 0.5275 | 0.6213 | 0.6860 | 0.2463 | - | - | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + Elo | 91 | 0.5934 | 0.6372 | 0.6873 | 0.2457 | [-0.0659, +0.0549] | [-0.0681, +0.0348] | unreliable (n < 200) |
| validation | 4. both on a b2b | Repo + Elo | 91 | 0.5934 | 0.6483 | 0.6793 | 0.2424 | [-0.0440, +0.0440] | [-0.0430, +0.0368] | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + fatigue + travel + Elo | 91 | 0.5714 | 0.6338 | 0.6885 | 0.2461 | [-0.0769, +0.0223] | [-0.0692, +0.0309] | unreliable (n < 200) |
| validation | 4. both on a b2b | Stack: repo, pre-game, pre-game + Elo | 91 | 0.5934 | 0.6406 | 0.6776 | 0.2419 | [-0.0440, +0.0440] | [-0.0444, +0.0228] | unreliable (n < 200) |
| validation | 4. both on a b2b | Ablation: pre-game window 5 | 91 | 0.5604 | 0.5681 | 0.7084 | 0.2557 | [-0.1319, +0.0659] | [-0.1859, +0.0171] | unreliable (n < 200) |
| validation | 4. both on a b2b | Ablation: pre-game window 7 | 91 | 0.5495 | 0.5710 | 0.7043 | 0.2537 | [-0.1319, +0.0440] | [-0.1885, +0.0222] | unreliable (n < 200) |
| validation | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 8 | 0.8750 | 1.0000 | 0.4648 | 0.1446 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 (official) | 8 | 0.8750 | 0.9333 | 0.4649 | 0.1438 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + Elo | 8 | 0.8750 | 1.0000 | 0.4309 | 0.1321 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Repo + Elo | 8 | 0.8750 | 1.0000 | 0.4325 | 0.1325 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue + travel + Elo | 8 | 0.8750 | 1.0000 | 0.4273 | 0.1305 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Stack: repo, pre-game, pre-game + Elo | 8 | 0.8750 | 1.0000 | 0.4267 | 0.1295 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Ablation: pre-game window 5 | 8 | 0.8750 | 0.9333 | 0.5306 | 0.1723 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Ablation: pre-game window 7 | 8 | 0.7500 | 1.0000 | 0.5287 | 0.1711 | - | - | no CI (n < 30) |
| validation | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 378 | 0.6455 | 0.6982 | 0.6286 | 0.2194 | - | - |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 (official) | 378 | 0.6508 | 0.7075 | 0.6221 | 0.2167 | - | - |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + Elo | 378 | 0.6217 | 0.6886 | 0.6310 | 0.2210 | [-0.0556, +0.0079] | [-0.0341, +0.0130] |  |
| validation | 6. away travelled > 1,609 km | Repo + Elo | 378 | 0.6296 | 0.6897 | 0.6304 | 0.2207 | [-0.0450, +0.0132] | [-0.0263, +0.0088] |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + fatigue + travel + Elo | 378 | 0.6323 | 0.6896 | 0.6303 | 0.2206 | [-0.0450, +0.0159] | [-0.0338, +0.0155] |  |
| validation | 6. away travelled > 1,609 km | Stack: repo, pre-game, pre-game + Elo | 378 | 0.6217 | 0.6913 | 0.6289 | 0.2200 | [-0.0529, +0.0053] | [-0.0225, +0.0081] |  |
| validation | 6. away travelled > 1,609 km | Ablation: pre-game window 5 | 378 | 0.6217 | 0.6839 | 0.6404 | 0.2249 | [-0.0741, +0.0291] | [-0.0593, +0.0352] |  |
| validation | 6. away travelled > 1,609 km | Ablation: pre-game window 7 | 378 | 0.6138 | 0.6710 | 0.6448 | 0.2269 | [-0.0794, +0.0185] | [-0.0682, +0.0154] |  |
| validation | 7. either team under 20 games this season | Baseline: repo season-to-date | 434 | 0.6751 | 0.7223 | 0.6093 | 0.2104 | - | - |  |
| validation | 7. either team under 20 games this season | Pre-game w20 (official) | 434 | 0.6429 | 0.7040 | 0.6176 | 0.2142 | - | - |  |
| validation | 7. either team under 20 games this season | Pre-game w20 + Elo | 434 | 0.6590 | 0.7181 | 0.6059 | 0.2093 | [-0.0599, +0.0230] | [-0.0352, +0.0277] |  |
| validation | 7. either team under 20 games this season | Repo + Elo | 434 | 0.6613 | 0.7294 | 0.6005 | 0.2068 | [-0.0530, +0.0253] | [-0.0210, +0.0370] |  |
| validation | 7. either team under 20 games this season | Pre-game w20 + fatigue + travel + Elo | 434 | 0.6659 | 0.7169 | 0.6062 | 0.2096 | [-0.0530, +0.0323] | [-0.0378, +0.0283] |  |
| validation | 7. either team under 20 games this season | Stack: repo, pre-game, pre-game + Elo | 434 | 0.6682 | 0.7302 | 0.6008 | 0.2069 | [-0.0438, +0.0300] | [-0.0148, +0.0321] |  |
| validation | 7. either team under 20 games this season | Ablation: pre-game window 5 | 434 | 0.6659 | 0.7062 | 0.6196 | 0.2151 | [-0.0461, +0.0300] | [-0.0433, +0.0119] |  |
| validation | 7. either team under 20 games this season | Ablation: pre-game window 7 | 434 | 0.6659 | 0.7147 | 0.6146 | 0.2128 | [-0.0484, +0.0276] | [-0.0327, +0.0178] |  |
