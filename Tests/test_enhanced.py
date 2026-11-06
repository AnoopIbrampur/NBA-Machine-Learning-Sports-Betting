import unittest

import pandas as pd

from src.Pregame.elo import ELO_FEATURES
from src.Pregame.enhanced import add_elo, hypothesis_two, shap_groups


class EnhancedTest(unittest.TestCase):
    def test_add_elo_leaves_existing_feature_lists_alone(self):
        games = pd.DataFrame({"GAME_ID": ["a", "b"], "x": [1.0, 2.0]})
        elo = pd.DataFrame([[1.0, 2.0, -1.0], [2.0, 1.0, 1.0]], index=["a", "b"], columns=ELO_FEATURES)
        columns = {"pregame": ["x"], "repo": ["x"], "pregame_sched_travel": ["x"]}
        out, new = add_elo(games, columns, elo)
        self.assertEqual(columns, {"pregame": ["x"], "repo": ["x"], "pregame_sched_travel": ["x"]})
        self.assertEqual(new["pregame"], ["x"])
        self.assertEqual(new["pregame_elo"], ["x"] + ELO_FEATURES)
        self.assertEqual(len(out), 2)

    def test_add_elo_refuses_a_game_without_a_rating(self):
        games = pd.DataFrame({"GAME_ID": ["a", "b"]})
        elo = pd.DataFrame([[1.0, 2.0, -1.0]], index=["a"], columns=ELO_FEATURES)
        with self.assertRaises(ValueError):
            add_elo(games, {"pregame": [], "repo": [], "pregame_sched_travel": []}, elo)

    def test_every_feature_gets_a_group(self):
        groups = shap_groups(["elo_diff", "win_pct_home", "FG%_diff", "rest_days_home", "b2b_away", "rest_diff"])
        self.assertEqual(groups["rest_diff"], "Schedule")
        self.assertEqual(groups["FG%_diff"], "Box-score stats")
        with self.assertRaises(ValueError):
            shap_groups(["unknown"])

    def test_hypothesis_rule(self):
        groups = ["Elo", "Win record", "Win record", "Schedule", "Schedule", "Schedule",
                  "Box-score stats", "Box-score stats", "Box-score stats"]
        ranking = pd.DataFrame({"mean_abs_shap": [9, 4, 3, 2.5, 2, 1.5, 1, 0.5, 0.2], "group": groups},
                               index=[f"f{i}" for i in range(9)])
        self.assertTrue(hypothesis_two(ranking)["supported"])
        ranking["group"] = ["Win record"] + groups[1:]
        self.assertFalse(hypothesis_two(ranking)["supported"])


if __name__ == "__main__":
    unittest.main()
