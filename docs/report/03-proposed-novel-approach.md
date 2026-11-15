# III. Proposed Novel Approach

Numbers come from `docs/findings.md` §11–§12 and `docs/report/comparison-table.md`. The Word version is built from the submitted paper with this section added.

## A. What we changed and why

Our official pre-game model is XGBoost trained on rolling 20-game team averages. On the leak-free data it was statistically tied with the repository's season-to-date baseline on both held-out periods, and the schedule features described in the Introduction did not help it. For this stage we changed the input block of the pipeline and tried four techniques. We picked each one because we expected it to carry information that rolling averages miss.

1) Elo ratings. In the SHAP analysis of Alves and Barbosa [2], Elo ratings and the home-game indicator were the most important pre-game inputs, and Elo ratings have also been used to predict match results in football [11]. A rolling average forgets everything older than 20 games, while an Elo rating carries a team's strength across seasons and adjusts for the strength of each opponent. As in [2], every team starts at 1500 and keeps its full rating between seasons, and k was picked from 16 to 32 by walk-forward log loss on the training seasons only (k = 28). Each rating uses only earlier games. In a second round we searched a small grid that adds a home-court term, partial regression toward 1500 at each new season, and a margin-of-victory multiplier. The setting chosen on train was k = 16, a home advantage of 100 points, one-third regression and margin of victory on.

2) Player data. Team averages cannot tell when a team's best players are missing. We downloaded 384,340 player-game rows from the NBA statistics website [9] and built rotation strength from each player's rating over earlier games, availability from who played in the team's previous game, and roster continuity from the previous season. All of these are read before the game is played. Because they use the previous game, they cannot see injury or rest news announced on game day.

3) Stacking. We expected the baseline, the pre-game model and the Elo model to make different mistakes, so we combined them with stacked generalization [10]: a logistic regression on each model's out-of-fold predictions, fit on the training seasons in date order. Round 1 stacked the baseline, the pre-game model and pre-game + Elo. Round 2 (stack 2) replaced the last of these with the round 2 primary model.

4) Rolling retraining. A live system would refit as the season goes on, so we refit each model on the first of every month, using all earlier games and the parameters fixed on train.

We also ran two smaller checks. Rolling windows of 5 and 7 games were tried as ablations, and calibration was measured with the expected calibration error [14] and the calibration slope [13], next to the Brier score [12].

## B. How we evaluated it

Every model was scored on the same 3,386 test games and 1,681 validation games, using the chronological split described in the Introduction. Tuning used a 40-trial search scored by walk-forward cross-validation on the training seasons only. Test and validation games were never used for tuning, selection or stacking fits. Both rounds followed analysis plans that were written and committed before any of their results existed. Round 2 is not blind, because the round 1 results motivated it.

Each comparison is a paired bootstrap with 2,000 resamples, which gives a 95% confidence interval for the difference between two models on the same games. The plans fixed one gain rule in advance: a row counts as better only if the interval for the same metric is entirely above zero on both test and validation. Only accuracy and AUC have intervals. Log loss and Brier score are reported without them. The round 2 primary test was pre-game + Elo variant + player data against the baseline.

The game logs also contain 766 games played after the split, from January 8 to June 13, 2026. Their scoring rules were fixed before any score existed, and they were scored once. The baseline and a market line cannot be scored on these games, because their data stops on January 7, 2026.

## C. Results

Table II lists every row. Table III gives the AUC difference intervals against the baseline and against the official pre-game model.

[Table II — Comparison of every technique on the test and validation games]
[Table III — 95% paired-bootstrap CIs of the AUC difference and the gain rule]
[Table IV — Fresh holdout, single look]

The main result is mixed. Elo was the strongest addition. A logistic regression on the Elo difference alone matched the baseline's test AUC (0.6793). In the SHAP ranking of the pre-game + fatigue + travel + Elo model, the Elo difference was the top feature and the Elo group took 60.4% of the importance on test. The median schedule feature ranked 31st of 43, so our second hypothesis is only half supported. Elo is the strongest predictor, but the schedule features rank below the middle.

No row beat the baseline on accuracy, on either split. On AUC, the round 2 primary model passed on test but its validation interval, [−0.0008, +0.0237], just includes zero, so the primary test was not met. Three secondary rows did pass on AUC against the baseline: repo + Elo variant + player, stack 2, and stack 2 with rolling retraining. All three contain the baseline's own features or predictions. Against the official pre-game model, every new static row in round 2 beat it on AUC on both splits, as did round 1's repo + Elo and stack.

Some things did not work. We could not show any gain from rolling retraining, since its AUC intervals against the static models include zero everywhere. The player features did not show a gain on their own either. Compared with pre-game + Elo variant, the AUC interval was [−0.0027, +0.0081] on test and [−0.0033, +0.0109] on validation. Windows of 5 and 7 games were significantly worse than the baseline on AUC on both splits. On calibration, every ECE difference interval included zero, so the rows cannot be ranked. A market line stayed ahead of all our models. Market − stack 2 was +0.0225 to +0.0418 in AUC on test and +0.0070 to +0.0334 on validation.

The fresh holdout is the strongest evidence, because nothing was chosen after seeing it. On these 766 games, the primary model beat the official pre-game model: accuracy 0.7076 against 0.6749 and AUC 0.7632 against 0.7393. Both intervals exclude zero (accuracy [+0.0065, +0.0587], AUC [+0.0054, +0.0420]). Every row scored higher in this period, so these levels should not be compared with Table II, and the holdout cannot separate the Elo variant from the player data.

## D. What to try next

The biggest gap we see is game-day information. Our player features use the previous game, while a market line likely prices injury and rest news, so adding official injury reports is the next step. The Elo variant's chosen settings sat at the edge of the grid, which suggests a wider search on the training seasons. Any new round should be scored on games that none of our earlier rounds have seen.

## New references (verified on Crossref unless noted)

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016, pp. 785–794, doi: 10.1145/2939672.2939785.
[7] S. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Adv. Neural Inf. Process. Syst. 30 (NIPS 2017), 2017, arXiv:1705.07874. (verified on arXiv)
[8] kyleskom, "NBA-Machine-Learning-Sports-Betting," GitHub repository. [Online]. Available: https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting
[9] National Basketball Association, "NBA Stats." [Online]. Available: https://www.nba.com/stats
[10] D. H. Wolpert, "Stacked generalization," Neural Netw., vol. 5, no. 2, pp. 241–259, 1992, doi: 10.1016/S0893-6080(05)80023-1.
[11] L. M. Hvattum and H. Arntzen, "Using ELO ratings for match result prediction in association football," Int. J. Forecast., vol. 26, no. 3, pp. 460–470, 2010, doi: 10.1016/j.ijforecast.2009.10.002.
[12] G. W. Brier, "Verification of forecasts expressed in terms of probability," Mon. Weather Rev., vol. 78, no. 1, pp. 1–3, 1950, doi: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.
[13] D. R. Cox, "Two further applications of a model for binary regression," Biometrika, vol. 45, no. 3–4, pp. 562–565, 1958, doi: 10.1093/biomet/45.3-4.562.
[14] M. P. Naeini, G. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in Proc. AAAI Conf. Artif. Intell., vol. 29, no. 1, 2015, doi: 10.1609/aaai.v29i1.9602.
