# Does the betting market price back-to-backs?

## Odds data

Tables used per season and how many rows match a logged game (by date and franchise). `win_margin_agrees` = share of matched rows whose Win_Margin equals the logged home margin.

| season   | table            |   rows |   matched |   win_margin_agrees |
|:---------|:-----------------|-------:|----------:|--------------------:|
| 2012-13  | odds_2012-13_new |   1314 |      1314 |              0.9985 |
| 2013-14  | odds_2013-14_new |   1319 |      1319 |              0.9985 |
| 2014-15  | odds_2014-15_new |   1311 |      1311 |              0.9992 |
| 2015-16  | odds_2015-16_new |   1316 |      1316 |              1      |
| 2016-17  | odds_2016-17_new |   1309 |      1305 |              0.9992 |
| 2017-18  | odds_2017-18_new |   1312 |      1312 |              0.9992 |
| 2018-19  | odds_2018-19_new |   1312 |      1312 |              1      |
| 2019-20  | odds_2019-20_new |   1143 |      1143 |              1      |
| 2020-21  | odds_2020-21_new |   1171 |      1170 |              1      |
| 2021-22  | odds_2021-22_new |   1323 |      1322 |              1      |
| 2022-23  | odds_2022-23_new |   1319 |      1318 |              1      |
| 2023-24  | 2023-24          |   1255 |      1252 |              1      |
| 2023-24  | odds_2023-24_new |   1195 |      1194 |              0.9966 |
| 2024-25  | 2024-25          |   1326 |      1315 |              0.9985 |
| 2025-26  | odds_2025-26     |    554 |       550 |              1      |

Split games with market data (unmatched games are dropped from every market comparison):

| split      |   split_games |   matched |   with_moneylines |   with_line |   match_rate |
|:-----------|--------------:|----------:|------------------:|------------:|-------------:|
| train      |         11873 |     11873 |             11872 |       11758 |            1 |
| test       |          3386 |      3386 |              3386 |        3379 |            1 |
| validation |          1681 |      1681 |              1681 |        1681 |            1 |

Home-win target vs logged home margin on matched games: 99.9882% agree.

## 1. Market benchmark (reference row, not our model)

Same games for every row: split games with both moneylines. Probability = proportional devig of the two moneylines. Saved models only scored.

| Model | Split | n | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|---|---|
| Market: devigged moneyline (reference) | test | 3386 | 0.6796 | 0.7262 | 0.6028 | 0.2078 |
| Baseline: repo season-to-date | test | 3386 | 0.6400 | 0.6797 | 0.6349 | 0.2221 |
| Pre-game w20 | test | 3386 | 0.6435 | 0.6749 | 0.6371 | 0.2229 |
| Pre-game w20 + margin (ablation) | test | 3386 | 0.6480 | 0.6838 | 0.6326 | 0.2209 |
| Pre-game w20 + fatigue | test | 3386 | 0.6409 | 0.6754 | 0.6370 | 0.2229 |
| Pre-game w20 + fatigue + travel | test | 3386 | 0.6412 | 0.6763 | 0.6364 | 0.2226 |
| Repo + fatigue + travel | test | 3386 | 0.6394 | 0.6809 | 0.6339 | 0.2216 |
| Market: devigged moneyline (reference) | validation | 1681 | 0.6811 | 0.7420 | 0.5925 | 0.2040 |
| Baseline: repo season-to-date | validation | 1681 | 0.6556 | 0.7089 | 0.6221 | 0.2165 |
| Pre-game w20 | validation | 1681 | 0.6419 | 0.6987 | 0.6275 | 0.2189 |
| Pre-game w20 + margin (ablation) | validation | 1681 | 0.6466 | 0.7107 | 0.6207 | 0.2157 |
| Pre-game w20 + fatigue | validation | 1681 | 0.6472 | 0.6985 | 0.6283 | 0.2192 |
| Pre-game w20 + fatigue + travel | validation | 1681 | 0.6496 | 0.6998 | 0.6268 | 0.2185 |
| Repo + fatigue + travel | validation | 1681 | 0.6526 | 0.7095 | 0.6213 | 0.2161 |

Paired bootstrap 95% CIs (market − model), 2000 resamples:

| Comparison | Split | n | Accuracy diff CI | AUC diff CI |
|---|---|---|---|---|
| Market − Baseline: repo season-to-date | test | 3386 | [+0.0245, +0.0538] | [+0.0349, +0.0587] |
| Market − Pre-game w20 | test | 3386 | [+0.0213, +0.0511] | [+0.0381, +0.0649] |
| Market − Baseline: repo season-to-date | validation | 1681 | [+0.0059, +0.0452] | [+0.0173, +0.0485] |
| Market − Pre-game w20 | validation | 1681 | [+0.0190, +0.0613] | [+0.0244, +0.0620] |

## 2a. Back-to-back situations controlling for the market

Logistic regression of home win on situation dummies (reference: neither team on a b2b) plus logit(market probability), fitted separately per split. Excluded: 172 restart games and 24 off-site games, and games without both moneylines. Odds multiplier > 1 = the home team wins more often than the market implies. A slope near 1 on logit(market) (multiplier near e = 2.718) means the market is calibrated.

| Split | Term | Games | Odds multiplier | 95% CI | p |
|---|---|---|---|---|---|
| train | away only on a b2b | 2282 | 0.988 | [0.889, 1.097] | 0.820 |
| train | home only on a b2b | 887 | 1.068 | [0.917, 1.244] | 0.397 |
| train | both on a b2b | 651 | 1.124 | [0.942, 1.342] | 0.195 |
| train | logit(market prob.): slope, as odds multiplier per unit | 11682 | 2.676 | [2.550, 2.809] | 0.000 |
| test | away only on a b2b | 458 | 0.967 | [0.774, 1.208] | 0.767 |
| test | home only on a b2b | 339 | 0.821 | [0.640, 1.054] | 0.122 |
| test | both on a b2b | 144 | 1.127 | [0.776, 1.638] | 0.529 |
| test | logit(market prob.): slope, as odds multiplier per unit | 3380 | 2.652 | [2.419, 2.907] | 0.000 |
| validation | away only on a b2b | 205 | 0.892 | [0.639, 1.244] | 0.501 |
| validation | home only on a b2b | 192 | 0.739 | [0.528, 1.036] | 0.079 |
| validation | both on a b2b | 91 | 0.757 | [0.473, 1.212] | 0.247 |
| validation | logit(market prob.): slope, as odds multiplier per unit | 1681 | 2.653 | [2.352, 2.991] | 0.000 |

## 2b. Home cover rate against the spread

Cover = logged home margin − market home line > 0; pushes excluded from the rate and counted. Exact binomial 95% CI and two-sided test against 50%. Context: Ashman, Bowman & Lambrinos (2010): 45.86% home cover, home on a b2b vs rested visitor, 1990-2009.

| Split | Situation | Games | No line | Pushes | Decided | Home cover rate | 95% CI | p vs 50% |
|---|---|---|---|---|---|---|---|---|
| train | home rested / away b2b | 2283 | 22 | 40 | 2221 | 0.5074 | [0.4864, 0.5284] | 0.497 |
| train | both rested | 7862 | 75 | 146 | 7641 | 0.4901 | [0.4789, 0.5014] | 0.086 |
| train | both b2b | 651 | 8 | 9 | 634 | 0.4968 | [0.4572, 0.5365] | 0.905 |
| train | home b2b / away rested | 887 | 8 | 14 | 865 | 0.4821 | [0.4483, 0.5160] | 0.308 |
| train | all | 11683 | 113 | 209 | 11361 | 0.4933 | [0.4840, 0.5025] | 0.154 |
| test | home rested / away b2b | 458 | 2 | 7 | 449 | 0.4989 | [0.4517, 0.5461] | 1.000 |
| test | both rested | 2439 | 5 | 29 | 2405 | 0.5027 | [0.4825, 0.5229] | 0.807 |
| test | both b2b | 144 | 0 | 3 | 141 | 0.5390 | [0.4531, 0.6232] | 0.400 |
| test | home b2b / away rested | 339 | 0 | 6 | 333 | 0.4535 | [0.3991, 0.5086] | 0.100 |
| test | all | 3380 | 7 | 45 | 3328 | 0.4988 | [0.4817, 0.5159] | 0.903 |
| validation | home rested / away b2b | 205 | 0 | 2 | 203 | 0.5074 | [0.4365, 0.5781] | 0.888 |
| validation | both rested | 1193 | 0 | 19 | 1174 | 0.5204 | [0.4914, 0.5494] | 0.170 |
| validation | both b2b | 91 | 0 | 0 | 91 | 0.5275 | [0.4200, 0.6331] | 0.675 |
| validation | home b2b / away rested | 192 | 0 | 1 | 191 | 0.4293 | [0.3581, 0.5028] | 0.060 |
| validation | all | 1681 | 0 | 22 | 1659 | 0.5087 | [0.4844, 0.5331] | 0.492 |

## 3. Exploratory: has the back-to-back effect changed over time?

Labelled exploratory; not used to change any model. The §7f model (schedule variables + team-by-season dummies) on all split seasons (n = 16,685; 59 games with missing rest or travel dropped), plus b2b × (season start year − 2018) interactions. Main effects are at 2018-19. Fit: converged=True, rank 836/836.

| Term | Odds multiplier | 95% CI | p |
|---|---|---|---|
| away_b2b | 1.331 | [1.216, 1.457] | 0.000 |
| home_b2b | 0.824 | [0.735, 0.923] | 0.001 |
| away_b2b_x_season | 1.016 | [0.993, 1.039] | 0.168 |
| home_b2b_x_season | 0.973 | [0.948, 0.999] | 0.044 |

Raw home-win rate by season (split games, restart and off-site excluded):

| SEASON   |   games |   both_rested |   away_only_b2b |   n_away_only |   home_only_b2b |   n_home_only |   gap_away_b2b |   gap_home_b2b |
|:---------|--------:|--------------:|----------------:|--------------:|----------------:|--------------:|---------------:|---------------:|
| 2012-13  |    1278 |        0.619  |          0.6201 |           308 |          0.5714 |            70 |         0.0011 |        -0.0476 |
| 2013-14  |    1301 |        0.5583 |          0.6606 |           274 |          0.4861 |            72 |         0.1023 |        -0.0722 |
| 2014-15  |    1289 |        0.584  |          0.5728 |           309 |          0.5325 |            77 |        -0.0111 |        -0.0515 |
| 2015-16  |    1297 |        0.5914 |          0.6207 |           290 |          0.5556 |            90 |         0.0292 |        -0.0359 |
| 2016-17  |    1285 |        0.579  |          0.6406 |           256 |          0.4725 |            91 |         0.0616 |        -0.1065 |
| 2017-18  |    1293 |        0.57   |          0.641  |           234 |          0.5914 |            93 |         0.071  |         0.0214 |
| 2018-19  |    1293 |        0.5666 |          0.6821 |           195 |          0.5686 |           102 |         0.1155 |         0.002  |
| 2019-20  |     952 |        0.5528 |          0.5541 |           148 |          0.5    |            82 |         0.0012 |        -0.0528 |
| 2020-21  |    1141 |        0.5306 |          0.6    |           170 |          0.5329 |           152 |         0.0694 |         0.0023 |
| 2021-22  |    1306 |        0.5608 |          0.5771 |           201 |          0.3846 |           130 |         0.0163 |        -0.1762 |
| 2022-23  |    1302 |        0.5722 |          0.6557 |           183 |          0.5185 |           108 |         0.0836 |        -0.0537 |
| 2023-24  |    1174 |        0.5541 |          0.6275 |           153 |          0.446  |           139 |         0.0734 |        -0.108  |
| 2024-25  |    1299 |        0.5543 |          0.6139 |           158 |          0.46   |           150 |         0.0596 |        -0.0943 |
| 2025-26  |     534 |        0.5707 |          0.5224 |            67 |          0.4516 |            62 |        -0.0483 |        -0.1191 |
