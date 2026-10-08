# Schedule-fatigue and travel features: XGBoost, window=20

Original three models are the saved ones from `train_xgb.py` (unchanged); the three new models use the same split, 40-trial walk-forward search on train only and seed. Positive class = home win, threshold 0.5.

## Full test and validation sets

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 3386 | 0.6400 | 0.6565 | 0.7580 | 0.7036 | 0.6797 | 0.6349 | 0.2221 |
| Pre-game w20 | test | 3386 | 0.6435 | 0.6509 | 0.7931 | 0.7150 | 0.6749 | 0.6371 | 0.2229 |
| Pre-game w20 + margin (ablation) | test | 3386 | 0.6480 | 0.6561 | 0.7894 | 0.7166 | 0.6838 | 0.6326 | 0.2209 |
| Pre-game w20 + fatigue | test | 3386 | 0.6409 | 0.6483 | 0.7936 | 0.7136 | 0.6754 | 0.6370 | 0.2229 |
| Pre-game w20 + fatigue + travel | test | 3386 | 0.6412 | 0.6527 | 0.7768 | 0.7094 | 0.6763 | 0.6364 | 0.2226 |
| Repo + fatigue + travel | test | 3386 | 0.6394 | 0.6571 | 0.7538 | 0.7021 | 0.6809 | 0.6339 | 0.2216 |
| Baseline: repo season-to-date | validation | 1681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 |
| Pre-game w20 | validation | 1681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 |
| Pre-game w20 + margin (ablation) | validation | 1681 | 0.6466 | 0.6507 | 0.7587 | 0.7006 | 0.7107 | 0.6207 | 0.2157 |
| Pre-game w20 + fatigue | validation | 1681 | 0.6472 | 0.6483 | 0.7707 | 0.7042 | 0.6985 | 0.6283 | 0.2192 |
| Pre-game w20 + fatigue + travel | validation | 1681 | 0.6496 | 0.6538 | 0.7587 | 0.7024 | 0.6998 | 0.6268 | 0.2185 |
| Repo + fatigue + travel | validation | 1681 | 0.6526 | 0.6566 | 0.7598 | 0.7045 | 0.7095 | 0.6213 | 0.2161 |

## Paired bootstrap 95% CIs of the difference (model − reference), 2000 resamples

| Model | Reference | Split | Accuracy diff CI | AUC diff CI |
|---|---|---|---|---|
| Pre-game w20 | Baseline: repo season-to-date | test | [-0.0098, +0.0165] | [-0.0152, +0.0057] |
| Pre-game w20 + margin (ablation) | Baseline: repo season-to-date | test | [-0.0047, +0.0210] | [-0.0055, +0.0136] |
| Pre-game w20 + fatigue | Baseline: repo season-to-date | test | [-0.0124, +0.0145] | [-0.0147, +0.0061] |
| Pre-game w20 + fatigue + travel | Baseline: repo season-to-date | test | [-0.0121, +0.0142] | [-0.0141, +0.0072] |
| Repo + fatigue + travel | Baseline: repo season-to-date | test | [-0.0056, +0.0041] | [-0.0003, +0.0027] |
| Pre-game w20 + fatigue | Pre-game w20 | test | [-0.0077, +0.0021] | [-0.0011, +0.0021] |
| Pre-game w20 + fatigue + travel | Pre-game w20 | test | [-0.0086, +0.0036] | [-0.0004, +0.0032] |
| Pre-game w20 + fatigue + travel | Pre-game w20 + fatigue | test | [-0.0053, +0.0059] | [-0.0008, +0.0024] |
| Pre-game w20 | Baseline: repo season-to-date | validation | [-0.0327, +0.0030] | [-0.0232, +0.0021] |
| Pre-game w20 + margin (ablation) | Baseline: repo season-to-date | validation | [-0.0268, +0.0072] | [-0.0097, +0.0136] |
| Pre-game w20 + fatigue | Baseline: repo season-to-date | validation | [-0.0268, +0.0083] | [-0.0232, +0.0017] |
| Pre-game w20 + fatigue + travel | Baseline: repo season-to-date | validation | [-0.0250, +0.0107] | [-0.0219, +0.0034] |
| Repo + fatigue + travel | Baseline: repo season-to-date | validation | [-0.0101, +0.0036] | [-0.0014, +0.0026] |
| Pre-game w20 + fatigue | Pre-game w20 | validation | [-0.0018, +0.0131] | [-0.0022, +0.0020] |
| Pre-game w20 + fatigue + travel | Pre-game w20 | validation | [+0.0006, +0.0149] **excl. 0** | [-0.0013, +0.0034] |
| Pre-game w20 + fatigue + travel | Pre-game w20 + fatigue | validation | [-0.0048, +0.0095] | [-0.0008, +0.0033] |

## Pre-registered slices

Slices were fixed before results were seen and are all reported. b2b = the existing `b2b_home` / `b2b_away` flags (1 day since the previous game). Slice 5 uses net jet lag (sign × max(|time zone shift| − days since previous game, 0)); slice 6 uses great-circle km from the away team's previous venue. Slices 5 and 6 exclude games with unknown travel. CIs are model − repo baseline on the slice's games. n < 200: unreliable; n < 30: no CI.

| Split | Slice | Model | n | Home-win rate | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | AUC diff vs repo CI | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1. neither team on a b2b | Baseline: repo season-to-date | 2445 | 0.5661 | 0.6344 | 0.6707 | 0.6387 | 0.2239 | - | - |  |
| test | 1. neither team on a b2b | Pre-game w20 | 2445 | 0.5661 | 0.6397 | 0.6658 | 0.6408 | 0.2246 | [-0.0106, +0.0213] | [-0.0173, +0.0076] |  |
| test | 1. neither team on a b2b | Pre-game w20 + margin (ablation) | 2445 | 0.5661 | 0.6446 | 0.6752 | 0.6364 | 0.2226 | [-0.0045, +0.0245] | [-0.0073, +0.0163] |  |
| test | 1. neither team on a b2b | Pre-game w20 + fatigue | 2445 | 0.5661 | 0.6397 | 0.6661 | 0.6407 | 0.2246 | [-0.0106, +0.0204] | [-0.0171, +0.0081] |  |
| test | 1. neither team on a b2b | Pre-game w20 + fatigue + travel | 2445 | 0.5661 | 0.6384 | 0.6669 | 0.6405 | 0.2245 | [-0.0119, +0.0200] | [-0.0162, +0.0089] |  |
| test | 1. neither team on a b2b | Repo + fatigue + travel | 2445 | 0.5661 | 0.6303 | 0.6704 | 0.6386 | 0.2239 | [-0.0098, +0.0016] | [-0.0020, +0.0015] |  |
| test | 2. away team only on a b2b | Baseline: repo season-to-date | 458 | 0.6201 | 0.6616 | 0.6818 | 0.6202 | 0.2150 | - | - |  |
| test | 2. away team only on a b2b | Pre-game w20 | 458 | 0.6201 | 0.6659 | 0.6777 | 0.6202 | 0.2149 | [-0.0349, +0.0437] | [-0.0335, +0.0256] |  |
| test | 2. away team only on a b2b | Pre-game w20 + margin (ablation) | 458 | 0.6201 | 0.6507 | 0.6743 | 0.6218 | 0.2159 | [-0.0480, +0.0284] | [-0.0356, +0.0201] |  |
| test | 2. away team only on a b2b | Pre-game w20 + fatigue | 458 | 0.6201 | 0.6550 | 0.6788 | 0.6189 | 0.2144 | [-0.0459, +0.0306] | [-0.0338, +0.0273] |  |
| test | 2. away team only on a b2b | Pre-game w20 + fatigue + travel | 458 | 0.6201 | 0.6550 | 0.6790 | 0.6188 | 0.2143 | [-0.0459, +0.0328] | [-0.0330, +0.0283] |  |
| test | 2. away team only on a b2b | Repo + fatigue + travel | 458 | 0.6201 | 0.6725 | 0.6845 | 0.6177 | 0.2138 | [-0.0044, +0.0262] | [-0.0020, +0.0072] |  |
| test | 3. home team only on a b2b | Baseline: repo season-to-date | 339 | 0.4661 | 0.6372 | 0.7214 | 0.6393 | 0.2237 | - | - |  |
| test | 3. home team only on a b2b | Pre-game w20 | 339 | 0.4661 | 0.6372 | 0.6998 | 0.6521 | 0.2297 | [-0.0413, +0.0442] | [-0.0505, +0.0078] |  |
| test | 3. home team only on a b2b | Pre-game w20 + margin (ablation) | 339 | 0.4661 | 0.6519 | 0.7205 | 0.6384 | 0.2234 | [-0.0236, +0.0531] | [-0.0273, +0.0252] |  |
| test | 3. home team only on a b2b | Pre-game w20 + fatigue | 339 | 0.4661 | 0.6254 | 0.6950 | 0.6536 | 0.2306 | [-0.0501, +0.0295] | [-0.0556, +0.0041] |  |
| test | 3. home team only on a b2b | Pre-game w20 + fatigue + travel | 339 | 0.4661 | 0.6313 | 0.6983 | 0.6500 | 0.2288 | [-0.0442, +0.0383] | [-0.0516, +0.0064] |  |
| test | 3. home team only on a b2b | Repo + fatigue + travel | 339 | 0.4661 | 0.6490 | 0.7266 | 0.6333 | 0.2210 | [-0.0029, +0.0265] | [+0.0004, +0.0103] **excl. 0** |  |
| test | 4. both on a b2b | Baseline: repo season-to-date | 144 | 0.5764 | 0.6736 | 0.7174 | 0.6084 | 0.2107 | - | - | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 | 144 | 0.5764 | 0.6528 | 0.7422 | 0.5940 | 0.2041 | [-0.0833, +0.0417] | [-0.0196, +0.0708] | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + margin (ablation) | 144 | 0.5764 | 0.6875 | 0.7521 | 0.5883 | 0.2013 | [-0.0347, +0.0625] | [-0.0085, +0.0815] | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + fatigue | 144 | 0.5764 | 0.6528 | 0.7501 | 0.5919 | 0.2030 | [-0.0833, +0.0417] | [-0.0113, +0.0791] | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + fatigue + travel | 144 | 0.5764 | 0.6667 | 0.7476 | 0.5909 | 0.2027 | [-0.0625, +0.0486] | [-0.0144, +0.0774] | unreliable (n < 200) |
| test | 4. both on a b2b | Repo + fatigue + travel | 144 | 0.5764 | 0.6667 | 0.7213 | 0.6060 | 0.2097 | [-0.0349, +0.0208] | [-0.0059, +0.0151] | unreliable (n < 200) |
| test | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 13 | 0.3846 | 0.3846 | 0.3500 | 0.7950 | 0.2971 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 | 13 | 0.3846 | 0.5385 | 0.4750 | 0.7140 | 0.2596 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + margin (ablation) | 13 | 0.3846 | 0.5385 | 0.5000 | 0.7310 | 0.2670 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue | 13 | 0.3846 | 0.5385 | 0.4750 | 0.7149 | 0.2601 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue + travel | 13 | 0.3846 | 0.5385 | 0.4750 | 0.7125 | 0.2591 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Repo + fatigue + travel | 13 | 0.3846 | 0.4615 | 0.3250 | 0.7938 | 0.2961 | - | - | no CI (n < 30) |
| test | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 727 | 0.5502 | 0.6176 | 0.6767 | 0.6391 | 0.2241 | - | - |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 | 727 | 0.5502 | 0.6410 | 0.6631 | 0.6450 | 0.2269 | [-0.0028, +0.0496] | [-0.0356, +0.0098] |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + margin (ablation) | 727 | 0.5502 | 0.6396 | 0.6685 | 0.6426 | 0.2259 | [-0.0055, +0.0482] | [-0.0289, +0.0131] |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + fatigue | 727 | 0.5502 | 0.6424 | 0.6637 | 0.6450 | 0.2269 | [-0.0014, +0.0523] | [-0.0353, +0.0102] |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + fatigue + travel | 727 | 0.5502 | 0.6341 | 0.6645 | 0.6447 | 0.2268 | [-0.0110, +0.0454] | [-0.0347, +0.0105] |  |
| test | 6. away travelled > 1,609 km | Repo + fatigue + travel | 727 | 0.5502 | 0.6162 | 0.6761 | 0.6384 | 0.2238 | [-0.0110, +0.0096] | [-0.0042, +0.0029] |  |
| validation | 1. neither team on a b2b | Baseline: repo season-to-date | 1193 | 0.5557 | 0.6639 | 0.7140 | 0.6168 | 0.2140 | - | - |  |
| validation | 1. neither team on a b2b | Pre-game w20 | 1193 | 0.5557 | 0.6513 | 0.7091 | 0.6184 | 0.2148 | [-0.0344, +0.0084] | [-0.0200, +0.0101] |  |
| validation | 1. neither team on a b2b | Pre-game w20 + margin (ablation) | 1193 | 0.5557 | 0.6580 | 0.7197 | 0.6123 | 0.2120 | [-0.0268, +0.0151] | [-0.0087, +0.0201] |  |
| validation | 1. neither team on a b2b | Pre-game w20 + fatigue | 1193 | 0.5557 | 0.6563 | 0.7088 | 0.6196 | 0.2153 | [-0.0285, +0.0134] | [-0.0203, +0.0101] |  |
| validation | 1. neither team on a b2b | Pre-game w20 + fatigue + travel | 1193 | 0.5557 | 0.6597 | 0.7094 | 0.6180 | 0.2146 | [-0.0260, +0.0176] | [-0.0198, +0.0107] |  |
| validation | 1. neither team on a b2b | Repo + fatigue + travel | 1193 | 0.5557 | 0.6605 | 0.7139 | 0.6167 | 0.2139 | [-0.0117, +0.0050] | [-0.0024, +0.0024] |  |
| validation | 2. away team only on a b2b | Baseline: repo season-to-date | 205 | 0.5854 | 0.6585 | 0.7155 | 0.6091 | 0.2108 | - | - |  |
| validation | 2. away team only on a b2b | Pre-game w20 | 205 | 0.5854 | 0.6732 | 0.6850 | 0.6272 | 0.2181 | [-0.0390, +0.0683] | [-0.0695, +0.0065] |  |
| validation | 2. away team only on a b2b | Pre-game w20 + margin (ablation) | 205 | 0.5854 | 0.6585 | 0.6998 | 0.6162 | 0.2132 | [-0.0585, +0.0585] | [-0.0489, +0.0163] |  |
| validation | 2. away team only on a b2b | Pre-game w20 + fatigue | 205 | 0.5854 | 0.6927 | 0.6869 | 0.6263 | 0.2178 | [-0.0195, +0.0927] | [-0.0675, +0.0076] |  |
| validation | 2. away team only on a b2b | Pre-game w20 + fatigue + travel | 205 | 0.5854 | 0.6878 | 0.6932 | 0.6238 | 0.2165 | [-0.0244, +0.0829] | [-0.0610, +0.0132] |  |
| validation | 2. away team only on a b2b | Repo + fatigue + travel | 205 | 0.5854 | 0.6537 | 0.7191 | 0.6071 | 0.2100 | [-0.0146, +0.0000] | [-0.0033, +0.0106] |  |
| validation | 3. home team only on a b2b | Baseline: repo season-to-date | 192 | 0.4531 | 0.6302 | 0.7144 | 0.6452 | 0.2268 | - | - | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 | 192 | 0.4531 | 0.6042 | 0.6985 | 0.6569 | 0.2316 | [-0.0729, +0.0156] | [-0.0539, +0.0236] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + margin (ablation) | 192 | 0.4531 | 0.6094 | 0.7164 | 0.6481 | 0.2278 | [-0.0677, +0.0208] | [-0.0334, +0.0350] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + fatigue | 192 | 0.4531 | 0.5990 | 0.6971 | 0.6589 | 0.2327 | [-0.0781, +0.0104] | [-0.0561, +0.0212] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + fatigue + travel | 192 | 0.4531 | 0.5990 | 0.6974 | 0.6558 | 0.2311 | [-0.0781, +0.0156] | [-0.0552, +0.0226] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Repo + fatigue + travel | 192 | 0.4531 | 0.6302 | 0.7147 | 0.6419 | 0.2253 | [-0.0156, +0.0156] | [-0.0072, +0.0077] | unreliable (n < 200) |
| validation | 4. both on a b2b | Baseline: repo season-to-date | 91 | 0.5055 | 0.5934 | 0.6502 | 0.6725 | 0.2404 | - | - | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 | 91 | 0.5055 | 0.5275 | 0.6213 | 0.6860 | 0.2463 | [-0.1319, -0.0110] **excl. 0** | [-0.0869, +0.0227] | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + margin (ablation) | 91 | 0.5055 | 0.5495 | 0.6536 | 0.6830 | 0.2443 | [-0.0989, +0.0110] | [-0.0467, +0.0478] | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + fatigue | 91 | 0.5055 | 0.5275 | 0.6256 | 0.6829 | 0.2449 | [-0.1319, -0.0110] **excl. 0** | [-0.0817, +0.0262] | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + fatigue + travel | 91 | 0.5055 | 0.5385 | 0.6261 | 0.6880 | 0.2472 | [-0.1099, +0.0000] | [-0.0805, +0.0265] | unreliable (n < 200) |
| validation | 4. both on a b2b | Repo + fatigue + travel | 91 | 0.5055 | 0.5934 | 0.6527 | 0.6698 | 0.2392 | [-0.0330, +0.0330] | [-0.0127, +0.0152] | unreliable (n < 200) |
| validation | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 8 | 0.6250 | 0.8750 | 1.0000 | 0.4648 | 0.1446 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 | 8 | 0.6250 | 0.8750 | 0.9333 | 0.4649 | 0.1438 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + margin (ablation) | 8 | 0.6250 | 0.8750 | 1.0000 | 0.4462 | 0.1370 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue | 8 | 0.6250 | 0.8750 | 0.9333 | 0.4713 | 0.1464 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + fatigue + travel | 8 | 0.6250 | 0.8750 | 1.0000 | 0.4509 | 0.1382 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Repo + fatigue + travel | 8 | 0.6250 | 0.8750 | 1.0000 | 0.4554 | 0.1402 | - | - | no CI (n < 30) |
| validation | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 378 | 0.5423 | 0.6455 | 0.6982 | 0.6286 | 0.2194 | - | - |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 | 378 | 0.5423 | 0.6508 | 0.7075 | 0.6221 | 0.2167 | [-0.0317, +0.0423] | [-0.0188, +0.0349] |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + margin (ablation) | 378 | 0.5423 | 0.6429 | 0.7242 | 0.6128 | 0.2122 | [-0.0370, +0.0317] | [+0.0019, +0.0497] **excl. 0** |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + fatigue | 378 | 0.5423 | 0.6455 | 0.7067 | 0.6235 | 0.2172 | [-0.0397, +0.0370] | [-0.0196, +0.0346] |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + fatigue + travel | 378 | 0.5423 | 0.6508 | 0.7098 | 0.6201 | 0.2156 | [-0.0317, +0.0423] | [-0.0166, +0.0373] |  |
| validation | 6. away travelled > 1,609 km | Repo + fatigue + travel | 378 | 0.5423 | 0.6508 | 0.7025 | 0.6268 | 0.2185 | [-0.0079, +0.0212] | [-0.0011, +0.0099] |  |
