# SHAP: schedule-feature models (window=20, test split)

Best new pre-game model by training CV log loss: **pregame_sched** (pregame_sched: 0.6304), (pregame_sched_travel: 0.6305). Chosen without looking at test or validation.

## pregame_sched (n=3386)

Share of total mean |SHAP| by group:

| group                             |   share_of_total_abs_shap |
|:----------------------------------|--------------------------:|
| win record                        |                    0.5877 |
| box-score stats                   |                    0.335  |
| fatigue (incl. existing rest/b2b) |                    0.0773 |

Full ranking:

|                      |   rank |   mean_abs_shap | group                             |
|:---------------------|-------:|----------------:|:----------------------------------|
| win_pct_home         |      1 |          0.1999 | win record                        |
| win_pct_away         |      2 |          0.1604 | win record                        |
| home_win_pct_at_home |      3 |          0.1405 | win record                        |
| away_win_pct_on_road |      4 |          0.1086 | win record                        |
| 2P%_diff             |      5 |          0.1059 | box-score stats                   |
| FG%_diff             |      6 |          0.0681 | box-score stats                   |
| DRB_diff             |      7 |          0.0365 | box-score stats                   |
| BLK_diff             |      8 |          0.0301 | box-score stats                   |
| TOV_diff             |      9 |          0.0256 | box-score stats                   |
| STL_diff             |     10 |          0.0218 | box-score stats                   |
| FT%_diff             |     11 |          0.0195 | box-score stats                   |
| 3P%_diff             |     12 |          0.0185 | box-score stats                   |
| rest_diff            |     13 |          0.0172 | fatigue (incl. existing rest/b2b) |
| rest_days_away       |     14 |          0.0148 | fatigue (incl. existing rest/b2b) |
| AST_diff             |     15 |          0.0099 | box-score stats                   |
| b2b_prev_away_away   |     16 |          0.0089 | fatigue (incl. existing rest/b2b) |
| games_last7_away     |     17 |          0.0078 | fatigue (incl. existing rest/b2b) |
| three_in_four_away   |     18 |          0.0076 | fatigue (incl. existing rest/b2b) |
| PF_diff              |     19 |          0.0071 | box-score stats                   |
| b2b_away             |     20 |          0.0048 | fatigue (incl. existing rest/b2b) |
| ORB_diff             |     21 |          0.0043 | box-score stats                   |
| rest_bucket_away     |     22 |          0.004  | fatigue (incl. existing rest/b2b) |
| consec_away_away     |     23 |          0.0033 | fatigue (incl. existing rest/b2b) |
| consec_away_home     |     24 |          0.0031 | fatigue (incl. existing rest/b2b) |
| games_last7_home     |     25 |          0.0031 | fatigue (incl. existing rest/b2b) |
| rest_days_home       |     26 |          0.0019 | fatigue (incl. existing rest/b2b) |
| first_leg_b2b_away   |     27 |          0.0018 | fatigue (incl. existing rest/b2b) |
| rest_bucket_home     |     28 |          0.0009 | fatigue (incl. existing rest/b2b) |
| b2b_home             |     29 |          0.0006 | fatigue (incl. existing rest/b2b) |
| three_in_four_home   |     30 |          0.0003 | fatigue (incl. existing rest/b2b) |
| first_leg_b2b_home   |     31 |          0.0002 | fatigue (incl. existing rest/b2b) |
| b2b_prev_away_home   |     32 |          0.0001 | fatigue (incl. existing rest/b2b) |

Mean SHAP by value of `b2b_away` (log-odds of a home win):

|   x |   size |    mean |
|----:|-------:|--------:|
|   0 |   2784 | -0.0035 |
|   1 |    602 |  0.0106 |

## repo_sched_travel (n=3386)

Share of total mean |SHAP| by group:

| group                             |   share_of_total_abs_shap |
|:----------------------------------|--------------------------:|
| box-score stats                   |                    0.6078 |
| win record                        |                    0.3298 |
| fatigue (incl. existing rest/b2b) |                    0.0447 |
| games played / minutes            |                    0.0134 |
| travel                            |                    0.0043 |

Full ranking:

|                    |   rank |   mean_abs_shap | group                             |
|:-------------------|-------:|----------------:|:----------------------------------|
| PLUS_MINUS_RANK    |      1 |          0.1246 | box-score stats                   |
| PLUS_MINUS         |      2 |          0.104  | box-score stats                   |
| L_RANK.1           |      3 |          0.0875 | win record                        |
| PLUS_MINUS.1       |      4 |          0.0727 | box-score stats                   |
| PLUS_MINUS_RANK.1  |      5 |          0.067  | box-score stats                   |
| W_PCT              |      6 |          0.0561 | win record                        |
| W_RANK             |      7 |          0.0517 | win record                        |
| W_PCT_RANK.1       |      8 |          0.0386 | win record                        |
| L_RANK             |      9 |          0.036  | win record                        |
| W_PCT.1            |     10 |          0.0358 | win record                        |
| FG3A.1             |     11 |          0.0347 | box-score stats                   |
| BLKA_RANK          |     12 |          0.0301 | box-score stats                   |
| FG_PCT             |     13 |          0.0217 | box-score stats                   |
| BLKA.1             |     14 |          0.0203 | box-score stats                   |
| W_PCT_RANK         |     15 |          0.0182 | win record                        |
| W                  |     16 |          0.0178 | win record                        |
| L.1                |     17 |          0.0154 | win record                        |
| FG3M.1             |     18 |          0.0145 | box-score stats                   |
| FG_PCT.1           |     19 |          0.0135 | box-score stats                   |
| W_RANK.1           |     20 |          0.0126 | win record                        |
| FG_PCT_RANK        |     21 |          0.0125 | box-score stats                   |
| rest_diff          |     22 |          0.0118 | fatigue (incl. existing rest/b2b) |
| PTS.1              |     23 |          0.0103 | box-score stats                   |
| AST.1              |     24 |          0.0092 | box-score stats                   |
| rest_bucket_away   |     25 |          0.0092 | fatigue (incl. existing rest/b2b) |
| Days-Rest-Away     |     26 |          0.0089 | fatigue (incl. existing rest/b2b) |
| b2b_prev_away_away |     27 |          0.0081 | fatigue (incl. existing rest/b2b) |
| FG_PCT_RANK.1      |     28 |          0.0077 | box-score stats                   |
| BLKA_RANK.1        |     29 |          0.0069 | box-score stats                   |
| DREB_RANK.1        |     30 |          0.0064 | box-score stats                   |
| L                  |     31 |          0.006  | win record                        |
| BLKA               |     32 |          0.0057 | box-score stats                   |
| FG3A_RANK.1        |     33 |          0.0054 | box-score stats                   |
| AST_RANK           |     34 |          0.0054 | box-score stats                   |
| FGA_RANK.1         |     35 |          0.0051 | box-score stats                   |
| games_last7_away   |     36 |          0.0047 | fatigue (incl. existing rest/b2b) |
| W.1                |     37 |          0.0045 | win record                        |
| DREB.1             |     38 |          0.0043 | box-score stats                   |
| FG3M_RANK          |     39 |          0.0041 | box-score stats                   |
| BLK_RANK           |     40 |          0.004  | box-score stats                   |
| FGM.1              |     41 |          0.004  | box-score stats                   |
| REB_RANK           |     42 |          0.0038 | box-score stats                   |
| three_in_four_away |     43 |          0.0037 | fatigue (incl. existing rest/b2b) |
| AST_RANK.1         |     44 |          0.0035 | box-score stats                   |
| MIN_RANK           |     45 |          0.0035 | games played / minutes            |
| TOV_RANK           |     46 |          0.0035 | box-score stats                   |
| AST                |     47 |          0.0035 | box-score stats                   |
| DREB               |     48 |          0.0035 | box-score stats                   |
| travel_km_away     |     49 |          0.0034 | travel                            |
| BLK                |     50 |          0.0033 | box-score stats                   |
| FT_PCT.1           |     51 |          0.0032 | box-score stats                   |
| PTS_RANK           |     52 |          0.0032 | box-score stats                   |
| FGA                |     53 |          0.0031 | box-score stats                   |
| PTS                |     54 |          0.0031 | box-score stats                   |
| REB                |     55 |          0.003  | box-score stats                   |
| GP                 |     56 |          0.0029 | games played / minutes            |
| FG3A               |     57 |          0.0028 | box-score stats                   |
| TOV                |     58 |          0.0027 | box-score stats                   |
| OREB_RANK.1        |     59 |          0.0026 | box-score stats                   |
| GP_RANK            |     60 |          0.0025 | games played / minutes            |
| PFD.1              |     61 |          0.0024 | box-score stats                   |
| DREB_RANK          |     62 |          0.0024 | box-score stats                   |
| FGM_RANK           |     63 |          0.0023 | box-score stats                   |
| FG3_PCT_RANK       |     64 |          0.0023 | box-score stats                   |
| OREB.1             |     65 |          0.0023 | box-score stats                   |
| FGM_RANK.1         |     66 |          0.0022 | box-score stats                   |
| FTM.1              |     67 |          0.0021 | box-score stats                   |
| FG3M_RANK.1        |     68 |          0.0021 | box-score stats                   |
| FG3A_RANK          |     69 |          0.0019 | box-score stats                   |
| PTS_RANK.1         |     70 |          0.0019 | box-score stats                   |
| PF_RANK            |     71 |          0.0019 | box-score stats                   |
| GP.1               |     72 |          0.0018 | games played / minutes            |
| REB_RANK.1         |     73 |          0.0017 | box-score stats                   |
| PF.1               |     74 |          0.0016 | box-score stats                   |
| consec_away_home   |     75 |          0.0016 | fatigue (incl. existing rest/b2b) |
| FGM                |     76 |          0.0016 | box-score stats                   |
| BLK_RANK.1         |     77 |          0.0016 | box-score stats                   |
| STL_RANK.1         |     78 |          0.0015 | box-score stats                   |
| MIN.1              |     79 |          0.0015 | games played / minutes            |
| FG3_PCT.1          |     80 |          0.0015 | box-score stats                   |
| consec_away_away   |     81 |          0.0015 | fatigue (incl. existing rest/b2b) |
| FG3M               |     82 |          0.0015 | box-score stats                   |
| BLK.1              |     83 |          0.0014 | box-score stats                   |
| FGA_RANK           |     84 |          0.0014 | box-score stats                   |
| FTM_RANK           |     85 |          0.0014 | box-score stats                   |
| FG3_PCT_RANK.1     |     86 |          0.0013 | box-score stats                   |
| MIN                |     87 |          0.0013 | games played / minutes            |
| FTA.1              |     88 |          0.0012 | box-score stats                   |
| FTA                |     89 |          0.0012 | box-score stats                   |
| PF_RANK.1          |     90 |          0.0011 | box-score stats                   |
| FT_PCT_RANK.1      |     91 |          0.0011 | box-score stats                   |
| GP_RANK.1          |     92 |          0.0011 | games played / minutes            |
| STL.1              |     93 |          0.0011 | box-score stats                   |
| FGA.1              |     94 |          0.001  | box-score stats                   |
| PFD_RANK.1         |     95 |          0.001  | box-score stats                   |
| FTM_RANK.1         |     96 |          0.001  | box-score stats                   |
| REB.1              |     97 |          0.001  | box-score stats                   |
| FG3_PCT            |     98 |          0.001  | box-score stats                   |
| STL_RANK           |     99 |          0.001  | box-score stats                   |
| OREB               |    100 |          0.0009 | box-score stats                   |
| FT_PCT             |    101 |          0.0009 | box-score stats                   |
| PFD                |    102 |          0.0009 | box-score stats                   |
| MIN_RANK.1         |    103 |          0.0008 | games played / minutes            |
| STL                |    104 |          0.0008 | box-score stats                   |
| FT_PCT_RANK        |    105 |          0.0008 | box-score stats                   |
| FTA_RANK.1         |    106 |          0.0008 | box-score stats                   |
| OREB_RANK          |    107 |          0.0008 | box-score stats                   |
| games_last7_home   |    108 |          0.0008 | fatigue (incl. existing rest/b2b) |
| PFD_RANK           |    109 |          0.0007 | box-score stats                   |
| FTM                |    110 |          0.0007 | box-score stats                   |
| PF                 |    111 |          0.0007 | box-score stats                   |
| travel_km_home     |    112 |          0.0006 | travel                            |
| TOV_RANK.1         |    113 |          0.0006 | box-score stats                   |
| TOV.1              |    114 |          0.0006 | box-score stats                   |
| tz_shift_away      |    115 |          0.0006 | travel                            |
| b2b_prev_away_home |    116 |          0.0005 | fatigue (incl. existing rest/b2b) |
| first_leg_b2b_home |    117 |          0.0004 | fatigue (incl. existing rest/b2b) |
| Days-Rest-Home     |    118 |          0.0004 | fatigue (incl. existing rest/b2b) |
| FTA_RANK           |    119 |          0.0004 | box-score stats                   |
| visitor_long_trip  |    120 |          0.0002 | travel                            |
| tz_shift_home      |    121 |          0.0002 | travel                            |
| first_leg_b2b_away |    122 |          0      | fatigue (incl. existing rest/b2b) |
| three_in_four_home |    123 |          0      | fatigue (incl. existing rest/b2b) |
| rest_bucket_home   |    124 |          0      | fatigue (incl. existing rest/b2b) |
| jet_lag_home       |    125 |          0      | travel                            |
| jet_lag_away       |    126 |          0      | travel                            |
| home_east_b2b      |    127 |          0      | travel                            |

Mean SHAP by value of `rest_bucket_away` (log-odds of a home win):

|   x |   size |    mean |
|----:|-------:|--------:|
|   1 |    602 |  0.0208 |
|   2 |   2155 | -0.0067 |
|   3 |    629 | -0.0066 |

Mean SHAP by value of `jet_lag_home` (log-odds of a home win):

|   x |   size |   mean |
|----:|-------:|-------:|
|  -1 |     20 |      0 |
|   0 |   3336 |      0 |
|   1 |     13 |      0 |

## pregame_sched_travel (supplementary) (n=3386)

Share of total mean |SHAP| by group:

| group                             |   share_of_total_abs_shap |
|:----------------------------------|--------------------------:|
| win record                        |                    0.5513 |
| box-score stats                   |                    0.3448 |
| fatigue (incl. existing rest/b2b) |                    0.0859 |
| travel                            |                    0.018  |

Full ranking:

|                      |   rank |   mean_abs_shap | group                             |
|:---------------------|-------:|----------------:|:----------------------------------|
| win_pct_home         |      1 |          0.2166 | win record                        |
| win_pct_away         |      2 |          0.1724 | win record                        |
| home_win_pct_at_home |      3 |          0.1424 | win record                        |
| 2P%_diff             |      4 |          0.1271 | box-score stats                   |
| away_win_pct_on_road |      5 |          0.1173 | win record                        |
| FG%_diff             |      6 |          0.0545 | box-score stats                   |
| DRB_diff             |      7 |          0.0438 | box-score stats                   |
| BLK_diff             |      8 |          0.0376 | box-score stats                   |
| TOV_diff             |      9 |          0.0358 | box-score stats                   |
| STL_diff             |     10 |          0.0308 | box-score stats                   |
| FT%_diff             |     11 |          0.0263 | box-score stats                   |
| 3P%_diff             |     12 |          0.0239 | box-score stats                   |
| rest_diff            |     13 |          0.0219 | fatigue (incl. existing rest/b2b) |
| rest_days_away       |     14 |          0.0199 | fatigue (incl. existing rest/b2b) |
| games_last7_away     |     15 |          0.0167 | fatigue (incl. existing rest/b2b) |
| travel_km_away       |     16 |          0.0134 | travel                            |
| PF_diff              |     17 |          0.0122 | box-score stats                   |
| b2b_prev_away_away   |     18 |          0.01   | fatigue (incl. existing rest/b2b) |
| three_in_four_away   |     19 |          0.0079 | fatigue (incl. existing rest/b2b) |
| ORB_diff             |     20 |          0.0068 | box-score stats                   |
| AST_diff             |     21 |          0.0068 | box-score stats                   |
| consec_away_away     |     22 |          0.0056 | fatigue (incl. existing rest/b2b) |
| first_leg_b2b_away   |     23 |          0.0051 | fatigue (incl. existing rest/b2b) |
| b2b_away             |     24 |          0.004  | fatigue (incl. existing rest/b2b) |
| games_last7_home     |     25 |          0.0037 | fatigue (incl. existing rest/b2b) |
| tz_shift_away        |     26 |          0.0036 | travel                            |
| travel_km_home       |     27 |          0.0029 | travel                            |
| consec_away_home     |     28 |          0.0026 | fatigue (incl. existing rest/b2b) |
| rest_days_home       |     29 |          0.0017 | fatigue (incl. existing rest/b2b) |
| tz_shift_home        |     30 |          0.001  | travel                            |
| rest_bucket_away     |     31 |          0.0005 | fatigue (incl. existing rest/b2b) |
| b2b_prev_away_home   |     32 |          0.0005 | fatigue (incl. existing rest/b2b) |
| first_leg_b2b_home   |     33 |          0.0004 | fatigue (incl. existing rest/b2b) |
| rest_bucket_home     |     34 |          0.0003 | fatigue (incl. existing rest/b2b) |
| visitor_long_trip    |     35 |          0.0002 | travel                            |
| three_in_four_home   |     36 |          0.0002 | fatigue (incl. existing rest/b2b) |
| home_east_b2b        |     37 |          0.0001 | travel                            |
| b2b_home             |     38 |          0      | fatigue (incl. existing rest/b2b) |
| jet_lag_home         |     39 |          0      | travel                            |
| jet_lag_away         |     40 |          0      | travel                            |

Mean SHAP by value of `b2b_away` (log-odds of a home win):

|   x |   size |    mean |
|----:|-------:|--------:|
|   0 |   2784 | -0.0029 |
|   1 |    602 |  0.0089 |

Mean SHAP by value of `jet_lag_home` (log-odds of a home win):

|   x |   size |   mean |
|----:|-------:|-------:|
|  -1 |     20 |      0 |
|   0 |   3336 |      0 |
|   1 |     13 |      0 |

