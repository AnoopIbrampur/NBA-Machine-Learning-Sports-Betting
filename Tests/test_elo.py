import json
import unittest

import numpy as np
import pandas as pd

from src.Pregame.elo import START_RATING, choose_k, expected_score, game_pairs, pregame_elo
from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.paths import ENHANCED_RESULTS_DIR, GAME_LOGS_DB, features_path
from src.Pregame.train_xgb import TARGET, load_games

ROUND_ONE = ENHANCED_RESULTS_DIR / "training.json"


def schedule(results):
    """Three teams rotating through games; results[i] = 1 means the home team won game i."""
    pairs = [(1, 2), (2, 3), (3, 1)]
    rows = []
    for i, home_won in enumerate(results):
        home, away = pairs[i % 3]
        for team, is_home, won in [(home, True, home_won), (away, False, 1 - home_won)]:
            rows.append({"TEAM_ID": team, "GAME_ID": f"G{i:03d}", "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                         "is_home": is_home, "neutral": False, "WIN": float(won)})
    return pd.DataFrame(rows)


class EloTest(unittest.TestCase):
    RESULTS = [1, 0, 1, 1, 0, 0, 1, 1, 0]

    def elo(self, results, k=20):
        return pregame_elo(schedule(results), k)

    def test_hand_computed_first_two_games(self):
        elo = self.elo([1, 1, 1])
        self.assertEqual(elo.loc["G000", "elo_home"], START_RATING)
        self.assertEqual(elo.loc["G000", "elo_away"], START_RATING)
        # G000: team 1 beats team 2 from level ratings, so 10 points move across.
        self.assertAlmostEqual(elo.loc["G001", "elo_home"], 1490.0)  # team 2
        self.assertAlmostEqual(elo.loc["G001", "elo_away"], 1500.0)  # team 3
        self.assertAlmostEqual(elo.loc["G001", "elo_diff"], -10.0)

    def test_expected_scores_sum_to_one(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)
        self.assertAlmostEqual(expected_score(1600, 1500) + expected_score(1500, 1600), 1.0)
        self.assertAlmostEqual(expected_score(1900, 1500), 10 / 11)

    def test_own_and_later_results_do_not_change_a_games_rating(self):
        base = self.elo(self.RESULTS)
        for game in range(len(self.RESULTS)):
            changed = list(self.RESULTS)
            for later in range(game, len(changed)):
                changed[later] = 1 - changed[later]
            altered = self.elo(changed)
            pd.testing.assert_frame_equal(base.iloc[: game + 1], altered.iloc[: game + 1])

    def test_an_earlier_result_changes_later_ratings(self):
        changed = list(self.RESULTS)
        changed[0] = 1 - changed[0]
        self.assertFalse(np.allclose(self.elo(self.RESULTS).iloc[1:], self.elo(changed).iloc[1:]))

    def test_ratings_carry_over_across_a_season_boundary(self):
        logs = schedule([1, 1, 1])
        logs["SEASON"] = ["2019-20"] * 4 + ["2020-21"] * 2
        self.assertAlmostEqual(pregame_elo(logs, 20).loc["G002", "elo_away"], 1510.0)

    def test_neutral_games_update_ratings_but_give_no_feature_row(self):
        logs = schedule([1, 1, 1])
        logs.loc[logs["GAME_ID"] == "G000", "neutral"] = True
        logs.loc[logs["GAME_ID"] == "G000", "is_home"] = False
        elo = pregame_elo(logs, 20)
        self.assertNotIn("G000", elo.index)
        self.assertAlmostEqual(elo.loc["G001", "elo_home"], 1490.0)

    def test_a_game_with_no_result_does_not_move_ratings(self):
        logs = schedule([1, 1, 1])
        logs.loc[logs["GAME_ID"] == "G000", "WIN"] = 0.0
        elo = pregame_elo(logs, 20)
        self.assertEqual(elo.loc["G001", "elo_home"], START_RATING)

    def test_two_winners_are_rejected(self):
        logs = schedule([1])
        logs["WIN"] = 1.0
        with self.assertRaises(ValueError):
            game_pairs(logs)

    def test_choose_k_ignores_games_outside_train(self):
        rng = np.random.default_rng(0)
        results = rng.integers(0, 2, 60).tolist()
        logs = schedule(results)
        ids = [f"G{i:03d}" for i in range(36)]
        y = np.array(results[:36])
        first, _ = choose_k(logs, ids, y)
        changed = results[:36] + [1 - r for r in results[36:]]
        second, _ = choose_k(schedule(changed), ids, y)
        self.assertEqual(first, second)


class EloVariantTest(unittest.TestCase):
    RESULTS = [1, 0, 1, 1, 0, 0, 1, 1, 0]

    def logs(self):
        logs = schedule(self.RESULTS)
        logs["SEASON"] = ["2019-20"] * 8 + ["2020-21"] * 10
        logs["PLUS_MINUS"] = [12, -12] * 9
        return logs

    def test_defaults_equal_explicit_round_one_settings(self):
        logs = self.logs()
        pd.testing.assert_frame_equal(pregame_elo(logs, 20), pregame_elo(logs, 20, 0.0, 0.0, False, frozenset()))

    def test_home_advantage_changes_the_update_but_not_the_pregame_gap_of_game_one(self):
        logs = self.logs()
        elo = pregame_elo(logs, 20, home_advantage=100)
        self.assertEqual(elo.loc["G000", "elo_diff"], 0.0)
        expected_gain = 20 * (1 - 1 / (1 + 10 ** (-100 / 400)))
        self.assertAlmostEqual(elo.loc["G001", "elo_home"], 1500 - expected_gain)

    def test_no_home_ids_removes_the_advantage(self):
        logs = self.logs()
        elo = pregame_elo(logs, 20, home_advantage=100, no_home_ids=frozenset({"G000"}))
        self.assertAlmostEqual(elo.loc["G001", "elo_home"], 1490.0)

    def test_regression_applies_at_a_new_season(self):
        logs = self.logs()
        plain = pregame_elo(logs, 20)
        regressed = pregame_elo(logs, 20, regression=0.25)
        pd.testing.assert_frame_equal(plain.iloc[:4], regressed.iloc[:4])
        self.assertLess(abs(regressed.loc["G004", "elo_home"] - 1500), abs(plain.loc["G004", "elo_home"] - 1500))

    def test_variant_own_and_later_results_do_not_change_a_games_rating(self):
        logs = self.logs()
        base = pregame_elo(logs, 20, 50, 0.25, True)
        for game in range(9):
            changed = logs.copy()
            later = changed["GAME_ID"].str[1:].astype(int) >= game
            changed.loc[later, "WIN"] = 1 - changed.loc[later, "WIN"]
            changed.loc[later, "PLUS_MINUS"] = -changed.loc[later, "PLUS_MINUS"] * 3
            pd.testing.assert_frame_equal(base.iloc[: game + 1], pregame_elo(changed, 20, 50, 0.25, True).iloc[: game + 1])


@unittest.skipIf(not GAME_LOGS_DB.exists() or not ROUND_ONE.exists() or not features_path(20).exists(),
                 "needs game logs, window 20 features and round 1 results")
class RoundOneUnchangedTest(unittest.TestCase):
    def test_default_elo_reproduces_the_committed_k_search(self):
        games, _ = load_games(20)
        train = games[games["split"] == "train"]
        logs = team_games(load_game_logs())
        _, losses = choose_k(logs, train["GAME_ID"].tolist(), train[TARGET].astype(int).to_numpy())
        saved = json.loads(ROUND_ONE.read_text())["elo_k_log_loss"]
        for k, loss in losses.items():
            self.assertAlmostEqual(loss, saved[str(k)], places=9)


if __name__ == "__main__":
    unittest.main()
