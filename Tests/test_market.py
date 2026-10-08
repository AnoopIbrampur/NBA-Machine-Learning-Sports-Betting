import sqlite3
import unittest

import numpy as np

from src.Pregame.market import (ODDS_DB, SEASON_TABLES, american_to_prob, cover_result,
                                devig_home_prob, home_line, load_odds_table)


class TestMoneylineConversion(unittest.TestCase):

    def test_american_to_prob(self):
        np.testing.assert_allclose(american_to_prob([-110, 150, -250, 100, -100]),
                                   [110 / 210, 100 / 250, 250 / 350, 0.5, 0.5])
        self.assertTrue(np.isnan(american_to_prob([50])).all())  # not a valid American price

    def test_devig(self):
        self.assertAlmostEqual(float(devig_home_prob(-110, -110)), 0.5)
        q_home, q_away = 200 / 300, 100 / 270
        self.assertAlmostEqual(float(devig_home_prob(-200, 170)), q_home / (q_home + q_away))
        probs = devig_home_prob(np.array([-300, 250, -120]), np.array([240, -320, 100]))
        np.testing.assert_allclose(probs + devig_home_prob(np.array([240, -320, 100]), np.array([-300, 250, -120])), 1)
        self.assertTrue((probs[[0, 2]] > 0.5).all() and probs[1] < 0.5)


class TestSpreadSign(unittest.TestCase):

    def test_unsigned_lines_take_the_moneyline_favourite(self):
        line = home_line([5.5, 5.0, "PK", 1.5, 0.0], [-250, 190, -110, -110, -110], [210, -230, -110, -110, -110],
                         signed=False)
        np.testing.assert_array_equal(line[[0, 1, 2, 4]], [5.5, -5.0, 0.0, 0.0])
        self.assertTrue(np.isnan(line[3]))  # equal moneylines, non-zero spread: side unknown

    def test_signed_lines_are_kept(self):
        np.testing.assert_array_equal(home_line([-1.5, 7.0], [120, -300], [-140, 240], signed=True), [-1.5, 7.0])

    def test_cover(self):
        np.testing.assert_array_equal(cover_result([7, 5, 3, -2], [5.5, 5, 5.5, -3.5]), [1, 0, -1, 1])

    def test_sign_matches_results_in_every_season(self):
        # The right sign makes the line track the final home margin; flipping it must be much worse.
        with sqlite3.connect(ODDS_DB) as con:
            for season, tables in SEASON_TABLES.items():
                odds = load_odds_table(con, tables[0], season)
                ok = odds["home_line"].notna()
                margin, line = odds.loc[ok, "Win_Margin"], odds.loc[ok, "home_line"]
                mae, flipped = np.mean(np.abs(margin - line)), np.mean(np.abs(margin + line))
                self.assertLess(mae, 11.5, season)
                self.assertLess(mae, flipped - 2.5, season)
                self.assertLess(abs(np.mean(margin - line)), 1.0, season)
                # The devigged moneyline favourite and the line favourite agree on almost every game.
                fav_ml = np.sign(odds.loc[ok, "p_market"] - 0.5)
                agree = np.mean((fav_ml == np.sign(line)) | (line == 0) | (fav_ml == 0))
                self.assertGreater(agree, 0.99, season)


if __name__ == "__main__":
    unittest.main()
