import unittest

import numpy as np
import pandas as pd

from src.Pregame.paths import GAME_LOGS_DB, PLAYER_LOGS_DB
from src.Pregame.player_features import PLAYER_FEATURES, box_value, build_player_features, replacement_level

STATS = ["PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "PF", "TOV"]


def synthetic(n_games=10, seed=0):
    """Teams 1 and 2 play repeatedly, each with players team*10 + 0..3; seasons change halfway."""
    rng = np.random.default_rng(seed)
    teams, players = [], []
    for i in range(n_games):
        game = f"G{i:03d}"
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=2 * i)
        season = "2019-20" if i < n_games // 2 else "2020-21"
        for team, is_home in ((1, True), (2, False)):
            teams.append({"GAME_ID": game, "TEAM_ID": team, "Date": date, "SEASON": season, "is_home": is_home, "neutral": False})
            for slot in range(4):
                stats = {s: int(rng.integers(0, 12)) for s in STATS}
                players.append({"GAME_ID": game, "TEAM_ID": team, "PLAYER_ID": team * 10 + slot, "MIN": int(rng.integers(10, 40)), **stats})
    return pd.DataFrame(players), pd.DataFrame(teams)


class PlayerFeatureTest(unittest.TestCase):
    def setUp(self):
        self.players, self.teams = synthetic()
        self.base = build_player_features(self.players, self.teams, 10.0)

    def alter(self, game_ids):
        changed = self.players.copy()
        mask = changed["GAME_ID"].isin(game_ids)
        changed.loc[mask, "PTS"] += 30
        changed.loc[mask, "MIN"] = 5
        changed.loc[mask & (changed["PLAYER_ID"] % 10 == 0), "MIN"] = 0
        return changed

    def test_first_game_of_a_season_is_missing(self):
        self.assertTrue(self.base.loc["G000"].isna().all())
        self.assertTrue(self.base.loc["G005", "player_strength_home":"player_availability_away"].isna().all())

    def test_a_games_own_and_later_rows_do_not_change_its_features(self):
        ids = list(self.base.index)
        for i, game in enumerate(ids):
            altered = build_player_features(self.alter(ids[i:]), self.teams, 10.0)
            pd.testing.assert_frame_equal(self.base.loc[:game], altered.loc[:game])

    def test_the_previous_game_does_change_features(self):
        altered = build_player_features(self.alter(["G002"]), self.teams, 10.0)
        self.assertFalse(np.allclose(self.base.loc["G003", "player_strength_home"], altered.loc["G003", "player_strength_home"]))

    def test_a_player_missing_last_game_lowers_availability_and_leaves_lastgame_strength(self):
        players = self.players.copy()
        players.loc[(players["GAME_ID"] == "G003") & (players["PLAYER_ID"] == 10), "MIN"] = 0
        altered = build_player_features(players, self.teams, 10.0)
        self.assertLess(altered.loc["G004", "player_availability_home"], 1.0)
        self.assertEqual(self.base.loc["G004", "player_availability_home"], 1.0)

    def test_continuity_uses_last_seasons_roster(self):
        self.assertTrue(np.isnan(self.base.loc["G004", "player_continuity_home"]))
        self.assertEqual(self.base.loc["G006", "player_continuity_home"], 1.0)
        players = self.players.copy()
        late = players["GAME_ID"].isin(["G005", "G006", "G007", "G008", "G009"])
        players.loc[late & (players["TEAM_ID"] == 1), "PLAYER_ID"] += 500
        self.assertEqual(build_player_features(players, self.teams, 10.0).loc["G006", "player_continuity_home"], 0.0)

    def test_new_player_starts_at_replacement_level(self):
        players, teams = synthetic(4)
        players.loc[players["GAME_ID"] == "G000", ["PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "PF", "TOV"]] = 0
        features = build_player_features(players, teams, 12.0)
        minutes = players[(players["GAME_ID"] == "G000") & (players["TEAM_ID"] == 1)]["MIN"]
        expected = float(((200 * 12.0) / (minutes + 200) * minutes).sum() / minutes.sum())
        self.assertAlmostEqual(features.loc["G001", "player_strength_home"], expected)

    def test_box_value_and_replacement_use_named_stats_and_train_games(self):
        row = pd.DataFrame([dict(PTS=10, FGM=4, FGA=10, FTM=2, FTA=2, OREB=1, DREB=3, AST=5, STL=1, BLK=1, PF=2, TOV=2, MIN=36, GAME_ID="A")])
        self.assertAlmostEqual(float(box_value(row).iloc[0]), 10 + 1.6 - 7 + 0.7 + 0.9 + 1 + 3.5 + 0.7 - 0.8 - 2)
        extra = row.assign(GAME_ID="B", PTS=99)
        self.assertEqual(replacement_level(pd.concat([row, extra]), ["A"]), replacement_level(row, ["A"]))


@unittest.skipIf(not PLAYER_LOGS_DB.exists() or not GAME_LOGS_DB.exists(), "needs player and game logs")
class PlayerLogIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.Pregame.game_logs import load_game_logs
        from src.Pregame.player_logs import load_player_logs
        cls.players = load_player_logs()
        cls.logs = load_game_logs()

    def test_player_points_sum_to_team_points(self):
        sums = self.players.groupby(["GAME_ID", "TEAM_ID"])["PTS"].sum().rename("sum").reset_index()
        merged = self.logs.merge(sums, on=["GAME_ID", "TEAM_ID"], how="left")
        played = merged[merged["GAME_ID"] != "0021201214"]
        self.assertFalse(played["sum"].isna().any())
        self.assertEqual(int((played["sum"] != played["PTS"]).sum()), 0)

    def test_player_minutes_are_close_to_team_minutes(self):
        sums = self.players.groupby(["GAME_ID", "TEAM_ID"])["MIN"].sum().rename("sum").reset_index()
        merged = self.logs.merge(sums, on=["GAME_ID", "TEAM_ID"])
        self.assertLessEqual(float((merged["sum"] - merged["MIN"]).abs().max()), 5.0)


if __name__ == "__main__":
    unittest.main()
