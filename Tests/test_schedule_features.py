import unittest

import numpy as np
import pandas as pd

from src.Pregame.features import team_games
from src.Pregame.schedule_features import (MILES_1000_KM, SCHEDULE_FEATURES, add_schedule_features,
                                           build_schedule_features, haversine_km, load_arenas,
                                           net_jet_lag, tz_shift_hours)
from Tests.test_pregame_features import synthetic_logs as box_score_logs

ARENAS = pd.DataFrame({
    "TEAM_ID": [1, 2], "TEAM_ABBREVIATION": ["AAA", "BBB"], "arena": ["A", "B"],
    "latitude": [34.0430, 42.3662], "longitude": [-118.2673, -71.0621],
    "tz": ["America/Los_Angeles", "America/New_York"],
    "valid_from": [pd.Timestamp.min] * 2, "valid_to": [pd.Timestamp.max] * 2,
})


def synthetic_logs(n_games):
    return box_score_logs(n_games=n_games).assign(SEASON_TYPE="Regular Season")


def build(logs):
    return build_schedule_features(logs, ARENAS, set()).set_index("GAME_ID")


class TestScheduleLeakage(unittest.TestCase):

    def setUp(self):
        self.logs = synthetic_logs(n_games=10)
        self.before = build(self.logs)

    def flip_results(self, game_ids):
        altered = self.logs.copy()
        rows = altered["GAME_ID"].isin(game_ids)
        altered.loc[rows, "WL"] = altered.loc[rows, "WL"].map({"W": "L", "L": "W"})
        altered.loc[rows, ["FGM", "DREB", "TOV", "PLUS_MINUS"]] = [0, 0, 99, 99]
        return build(altered)

    def test_current_game_result_does_not_leak(self):
        after = self.flip_results(["G005"])
        pd.testing.assert_series_equal(self.before.loc["G005"], after.loc["G005"])

    def test_later_results_do_not_leak(self):
        after = self.flip_results([f"G{i:03d}" for i in range(6, 10)])
        pd.testing.assert_frame_equal(self.before.loc[:"G005"], after.loc[:"G005"])

    def test_only_next_game_date_is_read(self):
        # Moving every game after G006 (venue and date both) may change G005 only through the date of
        # its next game: with that date fixed, G005's features must not move.
        altered = self.logs.copy()
        later = altered["GAME_ID"] > "G006"
        altered.loc[later, "GAME_DATE"] = (pd.to_datetime(altered.loc[later, "GAME_DATE"])
                                           + pd.Timedelta(days=30)).dt.strftime("%Y-%m-%d")
        swap = {"vs.": "@", "@": "vs."}
        altered.loc[later, "MATCHUP"] = altered.loc[later, "MATCHUP"].str.replace(
            r"vs\.|@", lambda m: swap[m.group(0)], regex=True)
        pd.testing.assert_frame_equal(self.before.loc[:"G005"], build(altered).loc[:"G005"])

    def test_first_leg_uses_next_game_date(self):
        logs = self.logs.copy()
        g5 = pd.Timestamp(logs.loc[logs["GAME_ID"] == "G005", "GAME_DATE"].iloc[0])
        shift = g5 + pd.Timedelta(days=1) - pd.Timestamp(logs.loc[logs["GAME_ID"] == "G006", "GAME_DATE"].iloc[0])
        later = logs["GAME_ID"] >= "G006"
        logs.loc[later, "GAME_DATE"] = (pd.to_datetime(logs.loc[later, "GAME_DATE"]) + shift).dt.strftime("%Y-%m-%d")
        features = build(logs)
        self.assertEqual(features.loc["G005", "first_leg_b2b_home"], 1.0)
        self.assertEqual(features.loc["G005", "first_leg_b2b_away"], 1.0)


class TestHandComputed(unittest.TestCase):

    def test_los_angeles_to_boston(self):
        date = pd.Timestamp("2024-01-10")
        shift = tz_shift_hours("America/Los_Angeles", "America/New_York", date)
        self.assertEqual(shift, 3.0)
        self.assertEqual(float(net_jet_lag(shift, 1)), 2.0)  # Leota et al.'s example: 3 h, 1 day
        self.assertEqual(float(net_jet_lag(-shift, 1)), -2.0)
        self.assertEqual(float(net_jet_lag(shift, 4)), 0.0)

    def test_phoenix_january_vs_july(self):
        self.assertEqual(tz_shift_hours("America/Los_Angeles", "America/Phoenix", "2024-01-10"), 1.0)
        self.assertEqual(tz_shift_hours("America/Los_Angeles", "America/Phoenix", "2024-07-10"), 0.0)
        self.assertEqual(tz_shift_hours("America/Denver", "America/Phoenix", "2024-01-10"), 0.0)
        self.assertEqual(tz_shift_hours("America/Denver", "America/Phoenix", "2024-07-10"), -1.0)

    def test_distance(self):
        # Crypto.com Arena to TD Garden: about 4,180 km great-circle.
        self.assertAlmostEqual(float(haversine_km(34.0430, -118.2673, 42.3662, -71.0621)), 4180, delta=15)
        self.assertEqual(float(haversine_km(40.0, -75.0, 40.0, -75.0)), 0.0)

    def test_two_game_trip(self):
        # AAA (LA) hosts on day 0, then plays at BBB (Boston) on day 1: a cross-country back-to-back.
        logs = synthetic_logs(n_games=2)
        logs["GAME_DATE"] = np.where(logs["GAME_ID"] == "G000", "2020-01-01", "2020-01-02")
        teams = add_schedule_features(team_games(logs), ARENAS, set()).set_index(["GAME_ID", "TEAM_ID"])
        trip = teams.loc[("G001", 1)]
        self.assertEqual(trip["tz_shift"], 3.0)
        self.assertEqual(trip["jet_lag"], 2.0)
        self.assertAlmostEqual(trip["travel_km"], 4180, delta=15)
        self.assertEqual(trip["b2b_prev_away"], 0.0)  # the previous night's game was at home
        self.assertEqual(teams.loc[("G000", 1), "first_leg_b2b"], 1.0)
        game = build(logs).loc["G001"]
        self.assertEqual(game["visitor_long_trip"], 1.0)  # > 1,609 km on a back-to-back
        self.assertGreater(game["travel_km_away"], MILES_1000_KM)


class TestRealSchedule(unittest.TestCase):
    """Checks on the committed arena table (no game logs needed)."""

    def test_toronto_2020_21_in_tampa(self):
        arenas = load_arenas()
        tor = arenas[arenas["TEAM_ABBREVIATION"] == "TOR"]
        on = lambda d: tor[(tor["valid_from"] <= d) & (tor["valid_to"] >= d)]  # noqa: E731
        self.assertEqual(on(pd.Timestamp("2021-02-01"))["tz"].item(), "America/New_York")
        self.assertIn("Tampa", on(pd.Timestamp("2021-02-01"))["arena"].item())
        self.assertIn("Scotiabank", on(pd.Timestamp("2022-02-01"))["arena"].item())
        self.assertIn("Scotiabank", on(pd.Timestamp("2020-02-01"))["arena"].item())

    def test_one_arena_per_team_and_date(self):
        arenas = load_arenas()
        self.assertEqual(arenas["TEAM_ID"].nunique(), 30)
        for date in pd.date_range("2012-10-01", "2026-06-30", freq="MS"):
            active = arenas[(arenas["valid_from"] <= date) & (arenas["valid_to"] >= date)]
            self.assertEqual(len(active), 30, date)

    def test_feature_list(self):
        self.assertEqual(len(SCHEDULE_FEATURES), len(set(SCHEDULE_FEATURES)))


if __name__ == "__main__":
    unittest.main()
