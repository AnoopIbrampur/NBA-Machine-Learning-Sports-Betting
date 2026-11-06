import unittest

import pandas as pd

from src.Pregame.holdout import holdout_games
from src.Pregame.train_xgb import TARGET


class HoldoutSelectionTest(unittest.TestCase):
    def test_keeps_only_played_sited_games_after_the_split(self):
        features = pd.DataFrame({"GAME_ID": ["old", "dup", "new", "off"],
                                 "Date": ["2026-01-07", "2026-02-01", "2026-02-02", "2026-02-03"]})
        rows = []
        for game, wl_home in [("old", "W"), ("dup", "W"), ("new", "L"), ("off", None)]:
            rows.append({"GAME_ID": game, "is_home": True, "neutral": False, "WL": wl_home, "WIN": float(wl_home == "W")})
            rows.append({"GAME_ID": game, "is_home": False, "neutral": False, "WL": "L", "WIN": 0.0})
        out = holdout_games(features, pd.DataFrame(rows), ["dup"])
        self.assertEqual(out["GAME_ID"].tolist(), ["new"])
        self.assertEqual(out[TARGET].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
