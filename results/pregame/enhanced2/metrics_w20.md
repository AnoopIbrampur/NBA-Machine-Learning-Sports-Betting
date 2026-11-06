# Modified Implementation, round 2: Elo variant, player features and rolling retraining

Plan fixed in advance: `docs/enhanced-plan-2.md`. This round is not blind: round 1 results motivated it. Positive class = home win, threshold 0.5. Rolling rows train on earlier held-out games, as a live system would.

Elo variant chosen on train only: k=16, home advantage=100, regression=0.333, margin=True, walk-forward log loss 0.61935. Replacement level 9.114.

## Full test and validation sets

| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: repo season-to-date | test | 3386 | 0.6400 | 0.6570 | 0.7577 | 0.7038 | 0.6793 | 0.6350 | 0.2221 |
| Pre-game w20 (official) | test | 3386 | 0.6435 | 0.6513 | 0.7928 | 0.7151 | 0.6746 | 0.6372 | 0.2230 |
| Round 1 stack | test | 3386 | 0.6480 | 0.6664 | 0.7535 | 0.7073 | 0.6888 | 0.6292 | 0.2195 |
| Pre-game w20 + Elo variant | test | 3386 | 0.6480 | 0.6541 | 0.7985 | 0.7191 | 0.6892 | 0.6290 | 0.2194 |
| PRIMARY: pre-game w20 + Elo variant + player | test | 3386 | 0.6456 | 0.6549 | 0.7865 | 0.7147 | 0.6918 | 0.6262 | 0.2182 |
| Repo + Elo variant + player | test | 3386 | 0.6509 | 0.6621 | 0.7792 | 0.7159 | 0.6943 | 0.6256 | 0.2178 |
| Stack 2: repo, pre-game, primary | test | 3386 | 0.6474 | 0.6639 | 0.7598 | 0.7086 | 0.6935 | 0.6255 | 0.2179 |
| Rolling baseline | test | 3386 | 0.6415 | 0.6556 | 0.7682 | 0.7075 | 0.6805 | 0.6339 | 0.2216 |
| Rolling primary | test | 3386 | 0.6521 | 0.6589 | 0.7954 | 0.7207 | 0.6914 | 0.6262 | 0.2183 |
| Rolling stack 2 | test | 3386 | 0.6500 | 0.6653 | 0.7645 | 0.7115 | 0.6933 | 0.6254 | 0.2178 |
| Market: devigged moneyline (reference) | test | 3386 | 0.6790 | 0.6914 | 0.7786 | 0.7325 | 0.7255 | 0.6036 | 0.2081 |
| Baseline: repo season-to-date | validation | 1681 | 0.6556 | 0.6576 | 0.7675 | 0.7083 | 0.7089 | 0.6221 | 0.2165 |
| Pre-game w20 (official) | validation | 1681 | 0.6419 | 0.6454 | 0.7609 | 0.6984 | 0.6987 | 0.6275 | 0.2189 |
| Round 1 stack | validation | 1681 | 0.6562 | 0.6644 | 0.7456 | 0.7027 | 0.7132 | 0.6169 | 0.2143 |
| Pre-game w20 + Elo variant | validation | 1681 | 0.6538 | 0.6535 | 0.7762 | 0.7096 | 0.7167 | 0.6154 | 0.2136 |
| PRIMARY: pre-game w20 + Elo variant + player | validation | 1681 | 0.6615 | 0.6617 | 0.7751 | 0.7139 | 0.7206 | 0.6136 | 0.2126 |
| Repo + Elo variant + player | validation | 1681 | 0.6633 | 0.6667 | 0.7642 | 0.7121 | 0.7224 | 0.6122 | 0.2120 |
| Stack 2: repo, pre-game, primary | validation | 1681 | 0.6663 | 0.6725 | 0.7555 | 0.7116 | 0.7220 | 0.6115 | 0.2117 |
| Rolling baseline | validation | 1681 | 0.6532 | 0.6549 | 0.7686 | 0.7072 | 0.7090 | 0.6211 | 0.2162 |
| Rolling primary | validation | 1681 | 0.6585 | 0.6586 | 0.7751 | 0.7121 | 0.7197 | 0.6130 | 0.2124 |
| Rolling stack 2 | validation | 1681 | 0.6609 | 0.6683 | 0.7500 | 0.7068 | 0.7217 | 0.6111 | 0.2116 |
| Market: devigged moneyline (reference) | validation | 1681 | 0.6811 | 0.6904 | 0.7522 | 0.7200 | 0.7420 | 0.5925 | 0.2040 |

## Paired bootstrap 95% CIs (row − reference), 2000 resamples

| Row | Reference | Split | Accuracy diff CI | AUC diff CI |
|---|---|---|---|---|
| Pre-game w20 + Elo variant | Baseline: repo season-to-date | test | [-0.0050, +0.0204] | [+0.0004, +0.0196] **excl. 0** |
| Pre-game w20 + Elo variant | Pre-game w20 (official) | test | [-0.0071, +0.0159] | [+0.0054, +0.0237] **excl. 0** |
| Pre-game w20 + Elo variant | Round 1 stack | test | [-0.0103, +0.0106] | [-0.0055, +0.0061] |
| PRIMARY: pre-game w20 + Elo variant + player | Baseline: repo season-to-date | test | [-0.0086, +0.0195] | [+0.0020, +0.0228] **excl. 0** |
| PRIMARY: pre-game w20 + Elo variant + player | Pre-game w20 (official) | test | [-0.0106, +0.0142] | [+0.0067, +0.0269] **excl. 0** |
| PRIMARY: pre-game w20 + Elo variant + player | Round 1 stack | test | [-0.0154, +0.0103] | [-0.0047, +0.0112] |
| Repo + Elo variant + player | Baseline: repo season-to-date | test | [-0.0015, +0.0230] | [+0.0069, +0.0234] **excl. 0** |
| Repo + Elo variant + player | Pre-game w20 (official) | test | [-0.0047, +0.0195] | [+0.0099, +0.0294] **excl. 0** |
| Repo + Elo variant + player | Round 1 stack | test | [-0.0080, +0.0136] | [-0.0014, +0.0126] |
| Stack 2: repo, pre-game, primary | Baseline: repo season-to-date | test | [-0.0044, +0.0186] | [+0.0067, +0.0217] **excl. 0** |
| Stack 2: repo, pre-game, primary | Pre-game w20 (official) | test | [-0.0092, +0.0165] | [+0.0088, +0.0288] **excl. 0** |
| Stack 2: repo, pre-game, primary | Round 1 stack | test | [-0.0103, +0.0089] | [-0.0009, +0.0106] |
| PRIMARY: pre-game w20 + Elo variant + player | Pre-game w20 + Elo variant | test | [-0.0127, +0.0083] | [-0.0027, +0.0081] |
| Rolling baseline | Baseline: repo season-to-date | test | [-0.0044, +0.0074] | [-0.0007, +0.0032] |
| Rolling primary | PRIMARY: pre-game w20 + Elo variant + player | test | [+0.0009, +0.0121] **excl. 0** | [-0.0020, +0.0013] |
| Rolling stack 2 | Stack 2: repo, pre-game, primary | test | [-0.0018, +0.0074] | [-0.0017, +0.0013] |
| Rolling primary | Rolling baseline | test | [-0.0030, +0.0242] | [+0.0007, +0.0209] **excl. 0** |
| Rolling stack 2 | Rolling baseline | test | [-0.0030, +0.0201] | [+0.0056, +0.0200] **excl. 0** |
| Rolling primary | Baseline: repo season-to-date | test | [-0.0018, +0.0260] | [+0.0018, +0.0221] **excl. 0** |
| Rolling stack 2 | Baseline: repo season-to-date | test | [-0.0015, +0.0213] | [+0.0066, +0.0213] **excl. 0** |
| Market: devigged moneyline (reference) | PRIMARY: pre-game w20 + Elo variant + player | test | [+0.0201, +0.0473] **excl. 0** | [+0.0237, +0.0441] **excl. 0** |
| Market: devigged moneyline (reference) | Stack 2: repo, pre-game, primary | test | [+0.0189, +0.0452] **excl. 0** | [+0.0225, +0.0418] **excl. 0** |
| Market: devigged moneyline (reference) | Rolling stack 2 | test | [+0.0159, +0.0425] **excl. 0** | [+0.0227, +0.0422] **excl. 0** |
| Pre-game w20 + Elo variant | Baseline: repo season-to-date | validation | [-0.0190, +0.0155] | [-0.0041, +0.0196] |
| Pre-game w20 + Elo variant | Pre-game w20 (official) | validation | [-0.0054, +0.0286] | [+0.0066, +0.0296] **excl. 0** |
| Pre-game w20 + Elo variant | Round 1 stack | validation | [-0.0155, +0.0101] | [-0.0035, +0.0098] |
| PRIMARY: pre-game w20 + Elo variant + player | Baseline: repo season-to-date | validation | [-0.0119, +0.0244] | [-0.0008, +0.0237] |
| PRIMARY: pre-game w20 + Elo variant + player | Pre-game w20 (official) | validation | [+0.0024, +0.0381] **excl. 0** | [+0.0096, +0.0347] **excl. 0** |
| PRIMARY: pre-game w20 + Elo variant + player | Round 1 stack | validation | [-0.0101, +0.0208] | [-0.0012, +0.0159] |
| Repo + Elo variant + player | Baseline: repo season-to-date | validation | [-0.0083, +0.0244] | [+0.0033, +0.0235] **excl. 0** |
| Repo + Elo variant + player | Pre-game w20 (official) | validation | [+0.0042, +0.0405] **excl. 0** | [+0.0119, +0.0363] **excl. 0** |
| Repo + Elo variant + player | Round 1 stack | validation | [-0.0065, +0.0214] | [+0.0014, +0.0172] **excl. 0** |
| Stack 2: repo, pre-game, primary | Baseline: repo season-to-date | validation | [-0.0042, +0.0262] | [+0.0039, +0.0223] **excl. 0** |
| Stack 2: repo, pre-game, primary | Pre-game w20 (official) | validation | [+0.0071, +0.0440] **excl. 0** | [+0.0110, +0.0361] **excl. 0** |
| Stack 2: repo, pre-game, primary | Round 1 stack | validation | [-0.0012, +0.0220] | [+0.0024, +0.0154] **excl. 0** |
| PRIMARY: pre-game w20 + Elo variant + player | Pre-game w20 + Elo variant | validation | [-0.0060, +0.0220] | [-0.0033, +0.0109] |
| Rolling baseline | Baseline: repo season-to-date | validation | [-0.0119, +0.0065] | [-0.0030, +0.0033] |
| Rolling primary | PRIMARY: pre-game w20 + Elo variant + player | validation | [-0.0107, +0.0054] | [-0.0034, +0.0017] |
| Rolling stack 2 | Stack 2: repo, pre-game, primary | validation | [-0.0119, +0.0012] | [-0.0026, +0.0019] |
| Rolling primary | Rolling baseline | validation | [-0.0113, +0.0220] | [-0.0013, +0.0230] |
| Rolling stack 2 | Rolling baseline | validation | [-0.0071, +0.0226] | [+0.0036, +0.0218] **excl. 0** |
| Rolling primary | Baseline: repo season-to-date | validation | [-0.0143, +0.0214] | [-0.0015, +0.0228] |
| Rolling stack 2 | Baseline: repo season-to-date | validation | [-0.0089, +0.0196] | [+0.0034, +0.0220] **excl. 0** |
| Market: devigged moneyline (reference) | PRIMARY: pre-game w20 + Elo variant + player | validation | [+0.0018, +0.0393] **excl. 0** | [+0.0080, +0.0358] **excl. 0** |
| Market: devigged moneyline (reference) | Stack 2: repo, pre-game, primary | validation | [-0.0018, +0.0327] | [+0.0070, +0.0334] **excl. 0** |
| Market: devigged moneyline (reference) | Rolling stack 2 | validation | [+0.0042, +0.0375] **excl. 0** | [+0.0072, +0.0336] **excl. 0** |

## Stack 2

```json
{
  "members": [
    "repo",
    "pregame",
    "pregame_elo2_player"
  ],
  "n_meta_rows": 9890,
  "intercept": -0.06171249404170136,
  "coefficients": {
    "repo": 0.38284671701818057,
    "pregame": -0.11246767953717238,
    "pregame_elo2_player": 0.7831154730582323
  }
}
```

## SHAP shares, primary row, test

Elo 57.1%, Player 31.6%, Schedule 4.6%, Box-score stats 4.4%, Win record 2.3%.

Top features: `elo2_diff`, `player_strength_lastgame_diff`, `player_strength_diff`, `elo2_home`, `rest_days_away`, `player_strength_lastgame_away`, `home_win_pct_at_home`, `player_continuity_home`, `2P%_diff`, `player_continuity_away`.

## Slices

CIs are row − repo baseline. n < 200: unreliable; n < 30: no CI.

| Split | Slice | Model | n | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | AUC diff vs repo CI | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1. neither team on a b2b | Baseline: repo season-to-date | 2445 | 0.6348 | 0.6708 | 0.6385 | 0.2238 | - | - |  |
| test | 1. neither team on a b2b | Pre-game w20 + Elo variant | 2445 | 0.6491 | 0.6814 | 0.6330 | 0.2212 | [-0.0012, +0.0294] | [-0.0009, +0.0219] |  |
| test | 1. neither team on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 2445 | 0.6438 | 0.6834 | 0.6301 | 0.2200 | [-0.0070, +0.0246] | [-0.0003, +0.0251] |  |
| test | 1. neither team on a b2b | Repo + Elo variant + player | 2445 | 0.6470 | 0.6855 | 0.6295 | 0.2197 | [-0.0016, +0.0254] | [+0.0044, +0.0246] **excl. 0** |  |
| test | 1. neither team on a b2b | Stack 2: repo, pre-game, primary | 2445 | 0.6462 | 0.6852 | 0.6297 | 0.2198 | [-0.0020, +0.0245] | [+0.0052, +0.0234] **excl. 0** |  |
| test | 1. neither team on a b2b | Rolling baseline | 2445 | 0.6352 | 0.6716 | 0.6376 | 0.2234 | [-0.0065, +0.0074] | [-0.0017, +0.0031] |  |
| test | 1. neither team on a b2b | Rolling primary | 2445 | 0.6515 | 0.6837 | 0.6297 | 0.2198 | [+0.0008, +0.0327] **excl. 0** | [+0.0004, +0.0255] **excl. 0** |  |
| test | 1. neither team on a b2b | Rolling stack 2 | 2445 | 0.6470 | 0.6854 | 0.6292 | 0.2196 | [-0.0012, +0.0249] | [+0.0055, +0.0235] **excl. 0** |  |
| test | 2. away team only on a b2b | Baseline: repo season-to-date | 458 | 0.6616 | 0.6818 | 0.6202 | 0.2150 | - | - |  |
| test | 2. away team only on a b2b | Pre-game w20 + Elo variant | 458 | 0.6507 | 0.6840 | 0.6159 | 0.2135 | [-0.0459, +0.0262] | [-0.0227, +0.0274] |  |
| test | 2. away team only on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 458 | 0.6703 | 0.6841 | 0.6144 | 0.2126 | [-0.0306, +0.0459] | [-0.0241, +0.0281] |  |
| test | 2. away team only on a b2b | Repo + Elo variant + player | 458 | 0.6747 | 0.6913 | 0.6154 | 0.2128 | [-0.0197, +0.0459] | [-0.0120, +0.0304] |  |
| test | 2. away team only on a b2b | Stack 2: repo, pre-game, primary | 458 | 0.6572 | 0.6905 | 0.6156 | 0.2131 | [-0.0328, +0.0240] | [-0.0103, +0.0276] |  |
| test | 2. away team only on a b2b | Rolling baseline | 458 | 0.6638 | 0.6824 | 0.6192 | 0.2145 | [-0.0131, +0.0175] | [-0.0049, +0.0061] |  |
| test | 2. away team only on a b2b | Rolling primary | 458 | 0.6681 | 0.6832 | 0.6148 | 0.2128 | [-0.0284, +0.0415] | [-0.0246, +0.0269] |  |
| test | 2. away team only on a b2b | Rolling stack 2 | 458 | 0.6725 | 0.6889 | 0.6158 | 0.2131 | [-0.0175, +0.0393] | [-0.0113, +0.0259] |  |
| test | 3. home team only on a b2b | Baseline: repo season-to-date | 339 | 0.6342 | 0.7161 | 0.6412 | 0.2246 | - | - |  |
| test | 3. home team only on a b2b | Pre-game w20 + Elo variant | 339 | 0.6165 | 0.7210 | 0.6384 | 0.2233 | [-0.0531, +0.0206] | [-0.0208, +0.0314] |  |
| test | 3. home team only on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 339 | 0.6106 | 0.7251 | 0.6329 | 0.2213 | [-0.0650, +0.0177] | [-0.0212, +0.0385] |  |
| test | 3. home team only on a b2b | Repo + Elo variant + player | 339 | 0.6313 | 0.7331 | 0.6271 | 0.2182 | [-0.0413, +0.0354] | [-0.0078, +0.0402] |  |
| test | 3. home team only on a b2b | Stack 2: repo, pre-game, primary | 339 | 0.6283 | 0.7287 | 0.6260 | 0.2179 | [-0.0413, +0.0295] | [-0.0105, +0.0348] |  |
| test | 3. home team only on a b2b | Rolling baseline | 339 | 0.6431 | 0.7189 | 0.6395 | 0.2239 | [-0.0059, +0.0236] | [-0.0037, +0.0090] |  |
| test | 3. home team only on a b2b | Rolling primary | 339 | 0.6195 | 0.7225 | 0.6340 | 0.2220 | [-0.0590, +0.0265] | [-0.0239, +0.0363] |  |
| test | 3. home team only on a b2b | Rolling stack 2 | 339 | 0.6283 | 0.7268 | 0.6264 | 0.2182 | [-0.0442, +0.0295] | [-0.0125, +0.0336] |  |
| test | 4. both on a b2b | Baseline: repo season-to-date | 144 | 0.6736 | 0.7174 | 0.6084 | 0.2107 | - | - | unreliable (n < 200) |
| test | 4. both on a b2b | Pre-game w20 + Elo variant | 144 | 0.6944 | 0.7537 | 0.5802 | 0.1984 | [-0.0347, +0.0764] | [-0.0061, +0.0766] | unreliable (n < 200) |
| test | 4. both on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 144 | 0.6806 | 0.7571 | 0.5816 | 0.1984 | [-0.0556, +0.0694] | [-0.0085, +0.0881] | unreliable (n < 200) |
| test | 4. both on a b2b | Repo + Elo variant + player | 144 | 0.6875 | 0.7448 | 0.5890 | 0.2016 | [-0.0417, +0.0694] | [-0.0132, +0.0692] | unreliable (n < 200) |
| test | 4. both on a b2b | Stack 2: repo, pre-game, primary | 144 | 0.6806 | 0.7476 | 0.5860 | 0.2005 | [-0.0556, +0.0694] | [-0.0073, +0.0659] | unreliable (n < 200) |
| test | 4. both on a b2b | Rolling baseline | 144 | 0.6736 | 0.7245 | 0.6040 | 0.2087 | [-0.0208, +0.0208] | [-0.0039, +0.0188] | unreliable (n < 200) |
| test | 4. both on a b2b | Rolling primary | 144 | 0.6875 | 0.7490 | 0.5859 | 0.2003 | [-0.0417, +0.0694] | [-0.0162, +0.0793] | unreliable (n < 200) |
| test | 4. both on a b2b | Rolling stack 2 | 144 | 0.6806 | 0.7442 | 0.5880 | 0.2013 | [-0.0486, +0.0625] | [-0.0096, +0.0631] | unreliable (n < 200) |
| test | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 13 | 0.3846 | 0.3500 | 0.7950 | 0.2971 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Pre-game w20 + Elo variant | 13 | 0.6154 | 0.6750 | 0.6920 | 0.2488 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | PRIMARY: pre-game w20 + Elo variant + player | 13 | 0.5385 | 0.6500 | 0.7162 | 0.2597 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Repo + Elo variant + player | 13 | 0.4615 | 0.5750 | 0.7556 | 0.2779 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Stack 2: repo, pre-game, primary | 13 | 0.4615 | 0.6000 | 0.7365 | 0.2696 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Rolling baseline | 13 | 0.4615 | 0.3250 | 0.7996 | 0.2990 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Rolling primary | 13 | 0.6154 | 0.6250 | 0.7072 | 0.2553 | - | - | no CI (n < 30) |
| test | 5. home eastward net jet lag >= 1 h | Rolling stack 2 | 13 | 0.4615 | 0.6250 | 0.7324 | 0.2676 | - | - | no CI (n < 30) |
| test | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 727 | 0.6176 | 0.6767 | 0.6391 | 0.2241 | - | - |  |
| test | 6. away travelled > 1,609 km | Pre-game w20 + Elo variant | 727 | 0.6355 | 0.6672 | 0.6431 | 0.2263 | [-0.0110, +0.0468] | [-0.0299, +0.0111] |  |
| test | 6. away travelled > 1,609 km | PRIMARY: pre-game w20 + Elo variant + player | 727 | 0.6272 | 0.6738 | 0.6370 | 0.2239 | [-0.0179, +0.0385] | [-0.0262, +0.0210] |  |
| test | 6. away travelled > 1,609 km | Repo + Elo variant + player | 727 | 0.6424 | 0.6822 | 0.6324 | 0.2216 | [+0.0000, +0.0495] | [-0.0134, +0.0248] |  |
| test | 6. away travelled > 1,609 km | Stack 2: repo, pre-game, primary | 727 | 0.6437 | 0.6795 | 0.6338 | 0.2223 | [+0.0000, +0.0523] | [-0.0142, +0.0202] |  |
| test | 6. away travelled > 1,609 km | Rolling baseline | 727 | 0.6272 | 0.6769 | 0.6386 | 0.2239 | [-0.0041, +0.0248] | [-0.0042, +0.0046] |  |
| test | 6. away travelled > 1,609 km | Rolling primary | 727 | 0.6424 | 0.6716 | 0.6383 | 0.2244 | [-0.0055, +0.0550] | [-0.0280, +0.0183] |  |
| test | 6. away travelled > 1,609 km | Rolling stack 2 | 727 | 0.6437 | 0.6782 | 0.6347 | 0.2227 | [+0.0014, +0.0523] **excl. 0** | [-0.0152, +0.0188] |  |
| test | 7. either team under 20 games this season | Baseline: repo season-to-date | 735 | 0.6095 | 0.6338 | 0.6503 | 0.2294 | - | - |  |
| test | 7. either team under 20 games this season | Pre-game w20 + Elo variant | 735 | 0.6395 | 0.6692 | 0.6324 | 0.2214 | [+0.0013, +0.0599] **excl. 0** | [+0.0073, +0.0623] **excl. 0** |  |
| test | 7. either team under 20 games this season | PRIMARY: pre-game w20 + Elo variant + player | 735 | 0.6136 | 0.6703 | 0.6314 | 0.2210 | [-0.0272, +0.0367] | [+0.0079, +0.0639] **excl. 0** |  |
| test | 7. either team under 20 games this season | Repo + Elo variant + player | 735 | 0.6054 | 0.6704 | 0.6334 | 0.2217 | [-0.0313, +0.0231] | [+0.0133, +0.0584] **excl. 0** |  |
| test | 7. either team under 20 games this season | Stack 2: repo, pre-game, primary | 735 | 0.6122 | 0.6664 | 0.6350 | 0.2226 | [-0.0245, +0.0313] | [+0.0100, +0.0538] **excl. 0** |  |
| test | 7. either team under 20 games this season | Rolling baseline | 735 | 0.6204 | 0.6347 | 0.6491 | 0.2288 | [-0.0027, +0.0272] | [-0.0038, +0.0055] |  |
| test | 7. either team under 20 games this season | Rolling primary | 735 | 0.6136 | 0.6688 | 0.6319 | 0.2213 | [-0.0259, +0.0354] | [+0.0065, +0.0621] **excl. 0** |  |
| test | 7. either team under 20 games this season | Rolling stack 2 | 735 | 0.6177 | 0.6666 | 0.6350 | 0.2226 | [-0.0190, +0.0367] | [+0.0102, +0.0541] **excl. 0** |  |
| validation | 1. neither team on a b2b | Baseline: repo season-to-date | 1193 | 0.6639 | 0.7140 | 0.6168 | 0.2140 | - | - |  |
| validation | 1. neither team on a b2b | Pre-game w20 + Elo variant | 1193 | 0.6605 | 0.7216 | 0.6086 | 0.2106 | [-0.0251, +0.0176] | [-0.0069, +0.0220] |  |
| validation | 1. neither team on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 1193 | 0.6781 | 0.7306 | 0.6046 | 0.2085 | [-0.0084, +0.0377] | [+0.0018, +0.0321] **excl. 0** |  |
| validation | 1. neither team on a b2b | Repo + Elo variant + player | 1193 | 0.6781 | 0.7311 | 0.6048 | 0.2085 | [-0.0067, +0.0352] | [+0.0048, +0.0300] **excl. 0** |  |
| validation | 1. neither team on a b2b | Stack 2: repo, pre-game, primary | 1193 | 0.6764 | 0.7304 | 0.6042 | 0.2084 | [-0.0075, +0.0327] | [+0.0053, +0.0277] **excl. 0** |  |
| validation | 1. neither team on a b2b | Rolling baseline | 1193 | 0.6605 | 0.7151 | 0.6152 | 0.2134 | [-0.0151, +0.0084] | [-0.0029, +0.0050] |  |
| validation | 1. neither team on a b2b | Rolling primary | 1193 | 0.6756 | 0.7277 | 0.6055 | 0.2090 | [-0.0109, +0.0344] | [-0.0009, +0.0289] |  |
| validation | 1. neither team on a b2b | Rolling stack 2 | 1193 | 0.6681 | 0.7291 | 0.6045 | 0.2086 | [-0.0151, +0.0226] | [+0.0043, +0.0264] **excl. 0** |  |
| validation | 2. away team only on a b2b | Baseline: repo season-to-date | 205 | 0.6585 | 0.7155 | 0.6091 | 0.2108 | - | - |  |
| validation | 2. away team only on a b2b | Pre-game w20 + Elo variant | 205 | 0.6683 | 0.7325 | 0.5923 | 0.2036 | [-0.0341, +0.0537] | [-0.0179, +0.0522] |  |
| validation | 2. away team only on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 205 | 0.6537 | 0.7077 | 0.6103 | 0.2121 | [-0.0439, +0.0341] | [-0.0435, +0.0260] |  |
| validation | 2. away team only on a b2b | Repo + Elo variant + player | 205 | 0.6488 | 0.7118 | 0.6077 | 0.2110 | [-0.0488, +0.0293] | [-0.0333, +0.0247] |  |
| validation | 2. away team only on a b2b | Stack 2: repo, pre-game, primary | 205 | 0.6585 | 0.7146 | 0.6065 | 0.2104 | [-0.0341, +0.0341] | [-0.0275, +0.0256] |  |
| validation | 2. away team only on a b2b | Rolling baseline | 205 | 0.6488 | 0.7118 | 0.6103 | 0.2114 | [-0.0341, +0.0146] | [-0.0138, +0.0061] |  |
| validation | 2. away team only on a b2b | Rolling primary | 205 | 0.6341 | 0.7093 | 0.6086 | 0.2113 | [-0.0683, +0.0244] | [-0.0423, +0.0270] |  |
| validation | 2. away team only on a b2b | Rolling stack 2 | 205 | 0.6488 | 0.7148 | 0.6060 | 0.2102 | [-0.0439, +0.0244] | [-0.0275, +0.0260] |  |
| validation | 3. home team only on a b2b | Baseline: repo season-to-date | 192 | 0.6302 | 0.7144 | 0.6452 | 0.2268 | - | - | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Pre-game w20 + Elo variant | 192 | 0.6198 | 0.7134 | 0.6532 | 0.2302 | [-0.0573, +0.0312] | [-0.0347, +0.0317] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 192 | 0.6146 | 0.7264 | 0.6421 | 0.2251 | [-0.0626, +0.0365] | [-0.0239, +0.0489] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Repo + Elo variant + player | 192 | 0.6250 | 0.7291 | 0.6370 | 0.2227 | [-0.0521, +0.0469] | [-0.0159, +0.0461] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Stack 2: repo, pre-game, primary | 192 | 0.6458 | 0.7293 | 0.6338 | 0.2213 | [-0.0260, +0.0573] | [-0.0119, +0.0423] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Rolling baseline | 192 | 0.6406 | 0.7140 | 0.6423 | 0.2254 | [-0.0104, +0.0312] | [-0.0116, +0.0103] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Rolling primary | 192 | 0.6146 | 0.7265 | 0.6361 | 0.2225 | [-0.0677, +0.0312] | [-0.0216, +0.0485] | unreliable (n < 200) |
| validation | 3. home team only on a b2b | Rolling stack 2 | 192 | 0.6510 | 0.7280 | 0.6296 | 0.2195 | [-0.0208, +0.0677] | [-0.0118, +0.0412] | unreliable (n < 200) |
| validation | 4. both on a b2b | Baseline: repo season-to-date | 91 | 0.5934 | 0.6502 | 0.6725 | 0.2404 | - | - | unreliable (n < 200) |
| validation | 4. both on a b2b | Pre-game w20 + Elo variant | 91 | 0.6044 | 0.6652 | 0.6764 | 0.2403 | [-0.0440, +0.0659] | [-0.0346, +0.0600] | unreliable (n < 200) |
| validation | 4. both on a b2b | PRIMARY: pre-game w20 + Elo variant + player | 91 | 0.5604 | 0.6536 | 0.6782 | 0.2418 | [-0.0879, +0.0220] | [-0.0488, +0.0531] | unreliable (n < 200) |
| validation | 4. both on a b2b | Repo + Elo variant + player | 91 | 0.5824 | 0.6575 | 0.6668 | 0.2373 | [-0.0549, +0.0220] | [-0.0343, +0.0465] | unreliable (n < 200) |
| validation | 4. both on a b2b | Stack 2: repo, pre-game, primary | 91 | 0.5934 | 0.6565 | 0.6714 | 0.2391 | [-0.0330, +0.0330] | [-0.0324, +0.0418] | unreliable (n < 200) |
| validation | 4. both on a b2b | Rolling baseline | 91 | 0.5934 | 0.6420 | 0.6784 | 0.2430 | [-0.0330, +0.0330] | [-0.0238, +0.0063] | unreliable (n < 200) |
| validation | 4. both on a b2b | Rolling primary | 91 | 0.5824 | 0.6512 | 0.6732 | 0.2397 | [-0.0549, +0.0330] | [-0.0481, +0.0470] | unreliable (n < 200) |
| validation | 4. both on a b2b | Rolling stack 2 | 91 | 0.6154 | 0.6570 | 0.6702 | 0.2383 | [+0.0000, +0.0549] | [-0.0301, +0.0413] | unreliable (n < 200) |
| validation | 5. home eastward net jet lag >= 1 h | Baseline: repo season-to-date | 8 | 0.8750 | 1.0000 | 0.4648 | 0.1446 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Pre-game w20 + Elo variant | 8 | 0.8750 | 1.0000 | 0.4232 | 0.1314 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | PRIMARY: pre-game w20 + Elo variant + player | 8 | 0.8750 | 1.0000 | 0.4284 | 0.1321 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Repo + Elo variant + player | 8 | 0.8750 | 1.0000 | 0.4279 | 0.1312 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Stack 2: repo, pre-game, primary | 8 | 0.8750 | 1.0000 | 0.4269 | 0.1306 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Rolling baseline | 8 | 0.7500 | 1.0000 | 0.4693 | 0.1467 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Rolling primary | 8 | 0.7500 | 1.0000 | 0.4387 | 0.1367 | - | - | no CI (n < 30) |
| validation | 5. home eastward net jet lag >= 1 h | Rolling stack 2 | 8 | 0.8750 | 1.0000 | 0.4355 | 0.1343 | - | - | no CI (n < 30) |
| validation | 6. away travelled > 1,609 km | Baseline: repo season-to-date | 378 | 0.6455 | 0.6982 | 0.6286 | 0.2194 | - | - |  |
| validation | 6. away travelled > 1,609 km | Pre-game w20 + Elo variant | 378 | 0.6296 | 0.6939 | 0.6278 | 0.2195 | [-0.0503, +0.0159] | [-0.0267, +0.0176] |  |
| validation | 6. away travelled > 1,609 km | PRIMARY: pre-game w20 + Elo variant + player | 378 | 0.6481 | 0.7095 | 0.6198 | 0.2155 | [-0.0344, +0.0370] | [-0.0129, +0.0344] |  |
| validation | 6. away travelled > 1,609 km | Repo + Elo variant + player | 378 | 0.6481 | 0.7130 | 0.6181 | 0.2145 | [-0.0317, +0.0370] | [-0.0055, +0.0344] |  |
| validation | 6. away travelled > 1,609 km | Stack 2: repo, pre-game, primary | 378 | 0.6270 | 0.7087 | 0.6197 | 0.2155 | [-0.0529, +0.0132] | [-0.0074, +0.0279] |  |
| validation | 6. away travelled > 1,609 km | Rolling baseline | 378 | 0.6349 | 0.6997 | 0.6278 | 0.2190 | [-0.0265, +0.0053] | [-0.0057, +0.0086] |  |
| validation | 6. away travelled > 1,609 km | Rolling primary | 378 | 0.6429 | 0.7067 | 0.6202 | 0.2158 | [-0.0345, +0.0317] | [-0.0149, +0.0324] |  |
| validation | 6. away travelled > 1,609 km | Rolling stack 2 | 378 | 0.6217 | 0.7075 | 0.6199 | 0.2156 | [-0.0556, +0.0053] | [-0.0077, +0.0270] |  |
| validation | 7. either team under 20 games this season | Baseline: repo season-to-date | 434 | 0.6751 | 0.7223 | 0.6093 | 0.2104 | - | - |  |
| validation | 7. either team under 20 games this season | Pre-game w20 + Elo variant | 434 | 0.6659 | 0.7381 | 0.5929 | 0.2036 | [-0.0507, +0.0300] | [-0.0120, +0.0439] |  |
| validation | 7. either team under 20 games this season | PRIMARY: pre-game w20 + Elo variant + player | 434 | 0.6843 | 0.7404 | 0.5945 | 0.2041 | [-0.0300, +0.0461] | [-0.0104, +0.0472] |  |
| validation | 7. either team under 20 games this season | Repo + Elo variant + player | 434 | 0.6843 | 0.7408 | 0.5973 | 0.2052 | [-0.0276, +0.0438] | [-0.0041, +0.0420] |  |
| validation | 7. either team under 20 games this season | Stack 2: repo, pre-game, primary | 434 | 0.6820 | 0.7441 | 0.5938 | 0.2036 | [-0.0276, +0.0415] | [-0.0004, +0.0447] |  |
| validation | 7. either team under 20 games this season | Rolling baseline | 434 | 0.6751 | 0.7284 | 0.6060 | 0.2089 | [-0.0184, +0.0184] | [-0.0003, +0.0132] |  |
| validation | 7. either team under 20 games this season | Rolling primary | 434 | 0.6751 | 0.7412 | 0.5928 | 0.2033 | [-0.0392, +0.0392] | [-0.0086, +0.0468] |  |
| validation | 7. either team under 20 games this season | Rolling stack 2 | 434 | 0.6728 | 0.7449 | 0.5917 | 0.2026 | [-0.0369, +0.0323] | [+0.0011, +0.0455] **excl. 0** |  |
