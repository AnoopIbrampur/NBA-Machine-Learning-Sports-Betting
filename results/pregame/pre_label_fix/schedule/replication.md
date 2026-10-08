# Replicating Bowman, Harmon & Ashman (2023) on our training seasons

Training-split games 2012-11 → 2022-01, n = 11,650. Excluded: 172 2019-20 restart games (one site, no home court), 18 listed off-site games, and 33 games with a missing rest or travel value (season openers, the game after an off-site game, the first restart game).

Primary control: **team season dummies**. Fit diagnostics: team_season_dummies: converged=True, rank 598/598, max SE 1.08; season_to_date_win_pct: converged=True, rank 11/11, max SE 0.17.

Odds multipliers for a home win, 95% Wald CIs (> 1 favours the home team). Reference rest = 2 days. 'Games in the last week' counts the team's games in the 7 days before game day.

| Variable | Games with it | Bowman et al. | Ours: team-by-season dummies | Ours: season-to-date win % control |
|---|---|---|---|---|
| Visitor on a back-to-back | 2930 | 1.506 | 1.311 [1.180, 1.457] (p=0.000) | 1.289 [1.172, 1.419] (p=0.000) |
| Home on a back-to-back | 1535 | 0.806 | 0.897 [0.780, 1.031] (p=0.127) | 0.885 [0.780, 1.004] (p=0.059) |
| Home crossed 1+ time zone west-to-east from a game one day ago | 185 | 0.693 | 0.864 [0.605, 1.234] (p=0.422) | 0.905 [0.655, 1.251] (p=0.546) |
| Visitor travelled >1000 mi on a b2b or >2000 mi otherwise | 396 | 1.261 | 1.319 [1.035, 1.683] (p=0.026) | 1.155 [0.930, 1.433] (p=0.192) |
| Visitor played 3+ games in the last week | 9730 | 1.153 | 0.995 [0.880, 1.124] (p=0.934) | 0.994 [0.888, 1.111] (p=0.909) |
| Home played 4+ games in the last week | 3540 | 0.914 | 1.041 [0.940, 1.152] (p=0.442) | 1.000 [0.912, 1.095] (p=0.995) |
| Visitor rested 3+ days (vs 2) | 2141 | not different from 2 days | 0.916 [0.811, 1.034] (p=0.154) | 0.962 [0.861, 1.074] (p=0.490) |
| Home rested 3+ days (vs 2) | 2652 | not different from 2 days | 1.070 [0.957, 1.196] (p=0.234) | 1.066 [0.963, 1.180] (p=0.218) |

## Raw home-win rate by back-to-back situation

|                                          |     n |   home_win_rate |
|:-----------------------------------------|------:|----------------:|
| ('all', 'home rested / away b2b')        |  2955 |          0.6193 |
| ('all', 'both rested')                   | 11669 |          0.568  |
| ('all', 'both b2b')                      |   887 |          0.584  |
| ('all', 'home b2b / away rested')        |  1429 |          0.499  |
| ('test', 'home rested / away b2b')       |   458 |          0.6201 |
| ('test', 'both rested')                  |  2445 |          0.5661 |
| ('test', 'both b2b')                     |   144 |          0.5764 |
| ('test', 'home b2b / away rested')       |   339 |          0.4661 |
| ('train', 'home rested / away b2b')      |  2292 |          0.6222 |
| ('train', 'both rested')                 |  8031 |          0.5704 |
| ('train', 'both b2b')                    |   652 |          0.5966 |
| ('train', 'home b2b / away rested')      |   898 |          0.5212 |
| ('validation', 'home rested / away b2b') |   205 |          0.5854 |
| ('validation', 'both rested')            |  1193 |          0.5557 |
| ('validation', 'both b2b')               |    91 |          0.5055 |
| ('validation', 'home b2b / away rested') |   192 |          0.4531 |

## Back-to-back frequency by season (split games)

| SEASON   |   games |   home_b2b_share |   away_b2b_share |   any_b2b_share |   b2b_team_games_per_team |
|:---------|--------:|-----------------:|-----------------:|----------------:|--------------------------:|
| 2012-13  |    1279 |           0.1243 |           0.3104 |          0.3651 |                   18.5333 |
| 2013-14  |    1302 |           0.1367 |           0.2919 |          0.3472 |                   18.6    |
| 2014-15  |    1291 |           0.1317 |           0.3114 |          0.371  |                   19.0667 |
| 2015-16  |    1299 |           0.127  |           0.281  |          0.3503 |                   17.6667 |
| 2016-17  |    1288 |           0.1258 |           0.2539 |          0.3245 |                   16.3    |
| 2017-18  |    1296 |           0.1119 |           0.2207 |          0.2924 |                   14.3667 |
| 2018-19  |    1296 |           0.1173 |           0.189  |          0.2677 |                   13.2333 |
| 2019-20  |    1127 |           0.1109 |           0.1677 |          0.2502 |                   10.4667 |
| 2020-21  |    1141 |           0.1919 |           0.2077 |          0.3409 |                   15.2    |
| 2021-22  |    1306 |           0.1348 |           0.1891 |          0.2887 |                   14.1    |
| 2022-23  |    1304 |           0.125  |           0.1825 |          0.2653 |                   13.3667 |
| 2023-24  |    1178 |           0.1604 |           0.1723 |          0.2903 |                   13.0667 |
| 2024-25  |    1299 |           0.1701 |           0.1763 |          0.2918 |                   15      |
| 2025-26  |     534 |           0.1723 |           0.1816 |          0.2978 |                    6.3    |
