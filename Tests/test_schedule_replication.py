import unittest

import pandas as pd

from src.Pregame.schedule_replication import bowman_week_restriction


class BowmanWeekRestrictionTest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "away_b2b": [1.0, 0.0, 0.0],
            "home_b2b": [0.0, 1.0, 0.0],
            "away_3plus_week": [1.0, 1.0, 0.0],
            "home_4plus_week": [1.0, 1.0, 1.0],
        })

    def test_week_flag_is_cleared_only_for_that_teams_back_to_back(self):
        restricted = bowman_week_restriction(self.X)
        self.assertEqual(restricted["away_3plus_week"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(restricted["home_4plus_week"].tolist(), [1.0, 0.0, 1.0])

    def test_other_columns_and_the_input_are_unchanged(self):
        before = self.X.copy()
        restricted = bowman_week_restriction(self.X)
        pd.testing.assert_frame_equal(restricted[["away_b2b", "home_b2b"]], before[["away_b2b", "home_b2b"]])
        pd.testing.assert_frame_equal(self.X, before)


if __name__ == "__main__":
    unittest.main()
