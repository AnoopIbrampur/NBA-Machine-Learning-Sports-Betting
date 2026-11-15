# Does the Schedule Help Predict NBA Games? Fatigue, Travel and the Betting Market

Anoop Ibrampur and Sai Pardeev Reddy Daadi
DS 340W, Penn State, Fall 2026

DRAFT: Abstract, Introduction and Literature Review (due Nov 8). See "Notes for the team" at the end
and delete them before submitting.

## Abstract

Models that report very high accuracy for National Basketball Association (NBA) games often use
statistics from the game being predicted, so they cannot help before tip-off. We study the harder
pre-game task and ask whether schedule context improves prediction beyond team strength and recent
form. Using 16,940 games from 2012-13 to 2025-26 and a fixed chronological split, we trained XGBoost
models on season-to-date statistics and on rolling 20-game form, then added fatigue and travel
features built from published schedule research. The schedule features did not improve prediction:
on test games, accuracy was 0.6412 with them and 0.6435 without, and AUC was 0.6759 and 0.6746. This
held on back-to-back games specifically. The effect itself is real. On training seasons, a visitor
playing a back-to-back raised the home team's odds of winning by a factor of 1.31 (95% CI 1.18 to
1.46). However, a betting-market line outperformed all of our models (test accuracy 0.6790, AUC
0.7255), and once that line was controlled for, back-to-back situations added no significant
information. A leakage-free Elo rating became the strongest input. A stacked model with Elo and player
features reached test accuracy 0.6474 and AUC 0.6935, beating the season-to-date baseline on AUC, but not
accuracy, on both held-out periods. Our pre-specified primary model narrowly missed that test, but it
beat our official pre-game model on 766 later games scored once (accuracy 0.7076 vs 0.6749, AUC 0.7632
vs 0.7393).
We conclude that fatigue affects results, but that its effect is already reflected in
rest-day features and in market prices.

## 1. Introduction

Predicting the winner of an NBA game is easy to do well if the model may look at the game itself.
Ouyang et al. (2024) report 93.3% accuracy from full-game statistics, but by then the result is
already known. A prediction is only useful before tip-off, when the box score does not exist. Alves
and Barbosa (2025), working with pre-game information only, report 65.50% for the NBA. The two
figures come from different data and evaluation designs, so they do not rank the models. They do
show how much of the apparent accuracy depends on what the model is allowed to see.

Pre-game models therefore rely on what is known in advance: team strength, recent form and the
schedule. The schedule is an appealing source of signal because it is published before the season
and because fatigue has a plausible physical cause. Bowman et al. (2023) show that schedule
differences change team outcomes after controlling for which teams played: over 19 seasons, the gap
between the most and least favoured teams averaged 1.87 expected wins per season, and back-to-back
variables made up 44.1% of the measured imbalance. Leota et al. (2022) link eastward jet lag to
poorer performance by home teams. What these studies do not settle is whether schedule information
improves a pre-game prediction once team strength and recent form are already in the model.

This project asks: **do schedule-context features improve pre-game NBA win prediction, and where do
they rank once we look inside the model?** At the midterm we expected that:

1. schedule features would raise accuracy by 1 to 3 percentage points over a rolling-form baseline;
2. the Elo rating would remain the strongest input, with schedule features in the middle;
3. the largest gains would come in games with uneven rest, such as one team on a back-to-back
   against a rested opponent.

After the midterm we built a leakage-free Elo rating, so we test the second expectation directly.

We make five contributions. First, we replace the in-game inputs of Ouyang et al. with leakage-free
rolling pre-game features, and we compare them with season-to-date statistics on the same games and
the same chronological split. Second, we turn back-to-back status from two binary flags into a
fatigue and travel feature group based on Bowman et al. and Leota et al., using arena locations and
time zones computed for the actual game dates. Third, we replicate Bowman et al.'s odds multipliers
on our own seasons. Fourth, we add a test those papers do not run together: whether back-to-backs
carry information beyond the betting market. Fifth, we add a leakage-free Elo rating, player-based
features, out-of-fold stacking, rolling retraining and a calibration comparison, all specified before
their results were computed. Evaluation slices were fixed before results were seen,
and a null result was treated as a valid outcome.

We also found and corrected two data errors in the code base we started from: same-game leakage in
the 2024-25 and 2025-26 baseline features, and game labels stored before games had finished. Both
corrections are documented, tested and included in the results reported here.

## 2. Literature Review

### 2.1 In-game prediction and explainability

Ouyang et al. (2024) build a real-time outcome model from NBA games in the 2021 to 2023 seasons.
Each game is described by box-score indicators, and models are compared at three points: after two
quarters, after three quarters and at full time. XGBoost performs best under ten-fold
cross-validation, with accuracy and AUC of 0.720 and 0.783 after two quarters, 0.798 and 0.876 after
three, and 0.933 and 0.982 at full time. SHAP is then used to explain the model. Field goal
percentage, defensive rebounds and turnovers matter throughout, while blocks rank last.

This paper gives us our modelling and explanation approach, and its 11 statistics are the ones we
adapt. Its limitation for our question is timing: the inputs come from the game being predicted, and
cross-validation does not respect the order of games. Its results are best read as an upper bound
on what in-game information provides, not as a target for a pre-game model.

### 2.2 Pre-game prediction

Alves and Barbosa (2025) address the pre-game task for the NBA and WNBA. They engineer 993 NBA
features, including rolling averages, rest days, opponent information and Elo ratings, and select
smaller sets using a feature selector, with the best results at the top 75. Their evaluation trains
on earlier seasons and predicts later ones, so future games cannot inform past predictions. They
report 65.50% accuracy for the NBA. Deep learning models (LSTM and CNN) did not beat the best
conventional models. In their SHAP analysis, a next-game feature and two Elo features were the three
most influential inputs in all four models examined. We see a similar pattern, with engineered
win-record features ranking above every box-score statistic.

This paper gives us a realistic pre-game framework and a reference accuracy. It includes rest days,
but it does not test a broad schedule and travel group as a separate block, which is the gap we
examine.

### 2.3 Schedule, fatigue and travel

Bowman et al. (2023) study schedule inequity from 2000-01 to 2018-19 using a logistic regression of
home win on 20 schedule variables, with dummy variables for the home and visiting teams as controls,
followed by Monte Carlo season simulations. A visitor on a back-to-back multiplies the home team's
odds by 1.506, and a home team on a back-to-back by 0.806. Crossing at least one time zone eastward
from a game one day earlier costs a home team more (0.693), and long visitor trips help the home
team (1.261). Rest of three or four days does not differ from two. Scheduling differences averaged
1.87 expected wins per season between the most and least favoured teams, and an 8.25% difference in
the probability of making the playoffs. The paper estimates effects on outcomes; it does not test
whether these variables improve a predictive model.

Leota et al. (2022) focus on circadian disruption across 11,481 games from 2011-12 to 2020-21. They
define jet lag from the time zones crossed, less recovery time since the previous game. Eastward
jet lag was associated with a lower home point differential (-1.29 points, p = 0.015) and a lower
home win rate (-6.03%, p = 0.051). Westward travel and away teams showed no significant effect. This
is observational evidence, and the win-rate result is borderline, but it supplies the definition of
our net jet-lag feature.

### 2.4 Betting markets and fatigue

If schedule effects are real and public, a market should price them. Ashman, Bowman and Lambrinos
(2010) test this for point spreads over 19 seasons. Home teams did poorly against the spread on the
second game of a back-to-back when the visitor had one or two days of rest, and more so after
travelling one or two time zones east, which suggests the market underpriced home-team fatigue at
that time. We use this as our reference point and ask whether that gap remains in recent seasons.

### 2.5 The gap

Each strand answers part of our question. Ouyang et al. give an explainable model but use in-game
data. Alves and Barbosa give a pre-game design without a dedicated schedule block. Bowman et al. and
Leota et al. estimate schedule effects but do not test predictive value. Ashman et al. study the
market, using seasons that end in 2009. No single study combines these. We test, on one set of
chronologically held-out games, whether schedule features improve an explainable pre-game model,
whether the published effects replicate, and whether the market already prices them.

## References

Alves, J. M., & Barbosa, R. S. (2025). Machine learning for basketball game outcomes: NBA and WNBA
leagues. *Computation, 13*(10), 230. https://doi.org/10.3390/computation13100230

Ashman, T., Bowman, R. A., & Lambrinos, J. (2010). The role of fatigue in NBA wagering markets: The
surprising "home disadvantage situation." *Journal of Sports Economics, 11*(6), 602-613.

Bowman, R. A., Harmon, O., & Ashman, T. (2023). Schedule inequity in the National Basketball
Association. *Journal of Sports Analytics, 9*(1), 61-76. https://doi.org/10.3233/JSA-220629

Leota, J., et al. (2022). Eastward jet lag is associated with impaired performance and game outcome
in the National Basketball Association. *Frontiers in Physiology, 13*, 892681.
https://doi.org/10.3389/fphys.2022.892681

Ouyang, Y., Li, X., Zhou, W., Hong, W., Zheng, W., Qi, F., & Peng, L. (2024). Integration of machine
learning XGBoost and SHAP models for NBA game outcome prediction and quantitative analysis
methodology. *PLOS ONE, 19*(7), e0307478. https://doi.org/10.1371/journal.pone.0307478

---

## Notes for the team (delete before submitting)

- **Citation style and format:** the assignment page isn't in the repo, so APA and this layout are
  placeholders. Check the page for the required format.
- **Leota full author list:** "et al." needs expanding if the style requires it.
- **Leota figures not checked against the paper:** no PDF is on hand for the 11,481 games, the -1.29
  points (p = 0.015) or the -6.03% (p = 0.051) in section 2.3. Check them before submitting.
- **Ashman et al.:** only the abstract has been read, and no PDF is on hand, so section 2.4 is not
  checked against the full paper. Our analysis uses the 45.86% figure, but it isn't in this draft
  because it couldn't be checked against the paper.
- **Bowman home back-to-back:** Table 1 gives 0.806, but the paper's own text in section 2.3 says
  0.86. This draft uses the table value.
- **Alves & Barbosa Equation (1):** it prints k(S - E + R). Our method follows the standard update and
  should say so in the Methods section.
- **Abstract and contributions:** they now include rounds 1 and 2. See findings sections 11 and 12. Round
  2 wasn't blind, and the Methods section should say so.
