import unittest

import pandas as pd

from src.Pregame.demo import describe_pick, nearest_dates, select_games, top_reasons


class DemoTest(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({"Date": ["2026-03-01", "2026-03-01", "2026-03-04"],
                                      "TEAM_NAME": ["Boston Celtics", "Utah Jazz", "Utah Jazz"],
                                      "TEAM_NAME.1": ["Miami Heat", "Boston Celtics", "Miami Heat"]})

    def test_select_by_date_and_team_on_either_side(self):
        self.assertEqual(len(select_games(self.features, "2026-03-01")), 2)
        self.assertEqual(len(select_games(self.features, "2026-03-01", "celtics")), 2)
        self.assertEqual(len(select_games(self.features, "2026-03-01", "Heat")), 1)
        self.assertTrue(select_games(self.features, "2026-03-02").empty)

    def test_nearest_dates(self):
        self.assertEqual(nearest_dates(self.features, "2026-03-03", 1), ["2026-03-04"])

    def test_top_reasons_rank_by_size_and_ignore_the_bias_column(self):
        self.assertEqual(top_reasons([0.1, -0.5, 0.2, 9.0], ["a", "b", "c"], 2), [("b", -0.5), ("c", 0.2)])

    def test_pick_shows_the_favoured_team(self):
        self.assertEqual(describe_pick("H", "A", 0.6), "H 60.0%")
        self.assertEqual(describe_pick("H", "A", 0.25), "A 75.0%")


if __name__ == "__main__":
    unittest.main()
