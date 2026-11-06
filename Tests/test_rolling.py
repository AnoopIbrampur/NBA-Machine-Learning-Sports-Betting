import unittest

import numpy as np
import pandas as pd

from src.Pregame.rolling import monthly_segments, rolling_predictions


class RollingTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2021-11-15", "2022-03-20", freq="D").strftime("%Y-%m-%d")
        self.held = np.asarray(pd.to_datetime(self.dates) >= "2022-01-10")

    def test_every_training_date_is_before_every_predicted_date(self):
        stamps = pd.to_datetime(pd.Series(self.dates))
        for cutoff, train, predict in monthly_segments(self.dates, self.held):
            self.assertLess(stamps[train].max(), cutoff)
            self.assertGreaterEqual(stamps[predict].min(), cutoff)
            self.assertLess(stamps[train].max(), stamps[predict].min())

    def test_cutoffs_are_month_starts_after_the_first_held_game(self):
        cutoffs = [str(c.date()) for c, _, _ in monthly_segments(self.dates, self.held)]
        self.assertEqual(cutoffs, ["2022-02-01", "2022-03-01"])

    def test_games_before_the_first_refit_keep_the_static_prediction(self):
        rng = np.random.default_rng(0)
        n = len(self.dates)
        games = pd.DataFrame({"Date": self.dates, "x": rng.normal(size=n)})
        y = rng.integers(0, 2, n)
        static = np.full(n, 0.5)
        tuned = {"params": {"objective": "binary:logistic", "max_depth": 2, "seed": 0}, "num_boost_round": 3}
        out = rolling_predictions(games, ["x"], y, tuned, static, self.held)
        january = np.asarray(pd.to_datetime(self.dates) < "2022-02-01")
        np.testing.assert_array_equal(out[january], static[january])
        self.assertTrue((out[~january] != 0.5).any())

    def test_later_labels_do_not_change_earlier_refits(self):
        rng = np.random.default_rng(1)
        n = len(self.dates)
        games = pd.DataFrame({"Date": self.dates, "x": rng.normal(size=n)})
        y = rng.integers(0, 2, n)
        changed = y.copy()
        march = np.asarray(pd.to_datetime(self.dates) >= "2022-03-01")
        changed[march] = 1 - changed[march]
        tuned = {"params": {"objective": "binary:logistic", "max_depth": 2, "seed": 0}, "num_boost_round": 3}
        first = rolling_predictions(games, ["x"], y, tuned, np.full(n, 0.5), self.held)
        second = rolling_predictions(games, ["x"], changed, tuned, np.full(n, 0.5), self.held)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
