import unittest

from src.Pregame.leakage_audit import leakage_by_season

# A few rows per season miss the pattern (stat corrections, missing snapshot dates), so allow 1%.
MIN_EXCLUDES_SHARE = 0.99


class TestDatasetLeakage(unittest.TestCase):
    """Each dataset row's team stats must come from games before the one being predicted."""

    @classmethod
    def setUpClass(cls):
        cls.by_season = leakage_by_season()

    def test_every_season_excludes_current_game(self):
        for season, row in self.by_season.iterrows():
            with self.subTest(season=f"{season}-{str(season + 1)[-2:]}"):
                self.assertGreaterEqual(row["excludes_current_game"], MIN_EXCLUDES_SHARE)

    def test_no_season_includes_current_game(self):
        for season, row in self.by_season.iterrows():
            with self.subTest(season=f"{season}-{str(season + 1)[-2:]}"):
                self.assertLess(row["includes_current_game"], 0.9)


if __name__ == "__main__":
    unittest.main()
