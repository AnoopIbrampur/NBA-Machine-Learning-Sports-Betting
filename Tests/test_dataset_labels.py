import unittest

from src.Pregame.paths import GAME_LOGS_DB

LOGS_MISSING = not GAME_LOGS_DB.exists()


@unittest.skipIf(LOGS_MISSING, "needs Data/pregame/GameLogs.sqlite (python -m src.Pregame.game_logs)")
class TestDatasetLabels(unittest.TestCase):
    """dataset_2012-26 labels and scores must match the game logs, season by season."""

    @classmethod
    def setUpClass(cls):
        from src.Pregame.features import team_games
        from src.Pregame.game_logs import load_game_logs
        from src.Pregame.label_audit import TARGET, label_audit
        cls.target = TARGET
        cls.wrong, cls.matched, cls.unmatched = label_audit(team_games(load_game_logs()))

    def test_labels_match_game_logs_in_every_season(self):
        for season, rows in self.matched.groupby("SEASON"):
            with self.subTest(season=season):
                bad = rows[rows[self.target] != rows["win_logs"]]
                self.assertTrue(bad.empty, bad[["Date", "TEAM_NAME", "TEAM_NAME.1"]].to_string())

    def test_scores_match_game_logs_in_every_season(self):
        for season, rows in self.matched.groupby("SEASON"):
            with self.subTest(season=season):
                bad = rows[rows["Score"] != rows["score_logs"]]
                self.assertTrue(bad.empty, bad[["Date", "TEAM_NAME", "TEAM_NAME.1", "Score"]].to_string())

    def test_no_row_without_a_final_score(self):
        unplayed = self.unmatched[self.unmatched["Score"] == 0]
        self.assertTrue(unplayed.empty, unplayed.to_string())

    def test_almost_every_row_matches_a_logged_game(self):
        # Unmatched rows are NBA Cup finals (not in the logs) and archive rows dated a day off.
        self.assertLessEqual(len(self.unmatched), 10)
        self.assertEqual(self.matched["GAME_ID"].duplicated().sum(), 0)


if __name__ == "__main__":
    unittest.main()
