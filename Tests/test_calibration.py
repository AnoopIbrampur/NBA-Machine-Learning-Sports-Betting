import unittest

import numpy as np

from src.Pregame.calibration import expected_calibration_error, intercept_slope, reliability


class CalibrationTest(unittest.TestCase):
    def test_hand_computed_ece(self):
        y = [1, 0, 1, 1]
        p = [0.25, 0.25, 0.75, 0.75]
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.25)

    def test_probability_of_one_lands_in_the_last_bin(self):
        table = reliability([1], [1.0])
        self.assertEqual(table["bin"].tolist(), [9])

    def test_calibrated_data_has_slope_near_one(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.2, 0.8, 50000)
        y = (rng.uniform(size=p.size) < p).astype(int)
        fit = intercept_slope(y, p)
        self.assertAlmostEqual(fit["slope"], 1.0, delta=0.05)
        self.assertLess(expected_calibration_error(y, p), 0.02)


if __name__ == "__main__":
    unittest.main()
