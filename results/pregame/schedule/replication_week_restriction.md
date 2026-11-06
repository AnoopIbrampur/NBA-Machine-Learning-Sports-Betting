# Sensitivity check: Bowman's 'not a back-to-back game' restriction

Labelled sensitivity check; `replication.md` keeps the main coefficients unchanged. Bowman's Table 1 defines SFV13 and SFV15 as >=3 / >=4 games in the last week (not a back-to-back game). The main replication does not apply that restriction; this refit does. Same 11,650 games and same controls. Odds multipliers for a home win, 95% Wald CIs.

Fit diagnostics: team_season_dummies: converged=True, rank 598/598, max SE 1.08; season_to_date_win_pct: converged=True, rank 11/11, max SE 0.17.

| Variable | Games, main | Games, restricted | Bowman et al. | Main: team-by-season dummies | Restricted: team-by-season dummies | Restricted: season-to-date win % control |
|---|---|---|---|---|---|---|
| Visitor on a back-to-back | 2930 | 2930 | 1.506 | 1.311 [1.180, 1.457] (p=0.000) | 1.288 [1.102, 1.506] (p=0.001) | 1.264 [1.096, 1.457] (p=0.001) |
| Home on a back-to-back | 1535 | 1535 | 0.806 | 0.897 [0.780, 1.031] (p=0.127) | 0.923 [0.799, 1.066] (p=0.277) | 0.890 [0.781, 1.014] (p=0.079) |
| Home crossed 1+ time zone west-to-east from a game one day ago | 185 | 185 | 0.693 | 0.864 [0.605, 1.234] (p=0.422) | 0.863 [0.604, 1.233] (p=0.418) | 0.905 [0.654, 1.251] (p=0.545) |
| Visitor travelled >1000 mi on a b2b or >2000 mi otherwise | 396 | 396 | 1.261 | 1.319 [1.035, 1.683] (p=0.026) | 1.318 [1.034, 1.681] (p=0.026) | 1.153 [0.929, 1.431] (p=0.196) |
| Visitor played 3+ games in the last week | 9730 | 7026 | 1.153 | 0.995 [0.880, 1.124] (p=0.934) | 0.981 [0.860, 1.118] (p=0.774) | 0.978 [0.868, 1.103] (p=0.720) |
| Home played 4+ games in the last week | 3540 | 2698 | 0.914 | 1.041 [0.940, 1.152] (p=0.442) | 1.060 [0.948, 1.185] (p=0.310) | 1.015 [0.918, 1.122] (p=0.777) |
| Visitor rested 3+ days (vs 2) | 2141 | 2141 | not different from 2 days | 0.916 [0.811, 1.034] (p=0.154) | 0.912 [0.808, 1.030] (p=0.139) | 0.958 [0.857, 1.070] (p=0.447) |
| Home rested 3+ days (vs 2) | 2652 | 2652 | not different from 2 days | 1.070 [0.957, 1.196] (p=0.234) | 1.076 [0.961, 1.204] (p=0.202) | 1.071 [0.966, 1.186] (p=0.192) |
