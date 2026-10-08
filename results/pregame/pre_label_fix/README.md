# Results before the game-result label fix (superseded)

These files are the results as they stood before `dataset_2012-26` labels and scores were
checked against the game logs (2026-10-08). They are kept because some of these numbers may
already have been reported. Corrected results are in the parent directory; see
`docs/findings.md` §10.

What was wrong:
- Two **test** games carried the wrong label: 2024-03-28 Hawks vs Celtics and Pelicans vs Bucks
  were home wins stored as `Home-Team-Win = 0`. Their odds rows (`odds_2023-24_new`) were scraped
  before the games finished, with a score of 0 and a win margin of 0.
- 11 more rows had a wrong `Score` with the correct winner (two 2024 playoff games scraped
  mid-game, nine archive or scrape errors). `Score` is not a feature or a label.
- Six rows were games not played that day (postponements), stored with score 0 and label 0. No
  model used them.

What this means for the files here:
- Only **test** numbers moved; training and validation labels were already correct, and no
  model was retrained.
- The test home-win rate was 0.5638 here and is 0.5644 after the fix; test AUC and log loss move
  in the 4th decimal; test accuracy does not change for any model.
- No confidence interval changed sign or significance.
- Files not copied here (SHAP rankings and plots, the replication odds ratios, training CV results,
  market match rates and cover rates) did not change.
