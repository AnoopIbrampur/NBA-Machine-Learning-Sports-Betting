import unittest

import numpy as np
import pandas as pd

from src.Pregame.features import add_team_features, build_game_features, team_games

WINDOW = 3


def synthetic_logs(n_games=8, seed=0):
    """Two teams alternating home/away, one game every 1-3 days, random box scores."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(rng.integers(1, 4, n_games)), unit="D")
    rows = []
    for i, date in enumerate(dates):
        home, away = ("AAA", "BBB") if i % 2 == 0 else ("BBB", "AAA")
        home_won = rng.random() < 0.5
        for team, opp, is_home, won in [(home, away, True, home_won), (away, home, False, not home_won)]:
            fga, fg3a, fta = rng.integers(80, 95), rng.integers(25, 40), rng.integers(15, 30)
            rows.append({
                "TEAM_ID": 1 if team == "AAA" else 2, "TEAM_NAME": team, "GAME_ID": f"G{i:03d}",
                "GAME_DATE": date.strftime("%Y-%m-%d"), "SEASON": "2019-20",
                "MATCHUP": f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                "WL": "W" if won else "L",
                "FGA": fga, "FGM": rng.integers(30, fga), "FG3A": fg3a, "FG3M": rng.integers(8, fg3a),
                "FTA": fta, "FTM": rng.integers(8, fta), "OREB": rng.integers(5, 15),
                "DREB": rng.integers(25, 40), "AST": rng.integers(15, 30), "STL": rng.integers(4, 12),
                "BLK": rng.integers(2, 9), "TOV": rng.integers(8, 20), "PF": rng.integers(15, 25),
            })
    return pd.DataFrame(rows)


class TestPregameFeatures(unittest.TestCase):

    def setUp(self):
        self.logs = synthetic_logs()
        self.teams = add_team_features(team_games(self.logs), WINDOW)

    def test_features_match_manual_prior_window(self):
        team = self.teams[self.teams["TEAM_ID"] == 1].reset_index(drop=True)
        for k in range(len(team)):
            prior = team.iloc[max(0, k - WINDOW):k]
            if prior.empty:
                self.assertTrue(np.isnan(team.loc[k, "FG%"]))
                self.assertTrue(np.isnan(team.loc[k, "rest_days"]))
                continue
            self.assertAlmostEqual(team.loc[k, "FG%"], prior["FGM"].sum() / prior["FGA"].sum())
            self.assertAlmostEqual(team.loc[k, "DRB"], prior["DREB"].mean())
            self.assertAlmostEqual(team.loc[k, "win_pct"], prior["WIN"].mean())
            self.assertEqual(team.loc[k, "rest_days"], (team.loc[k, "Date"] - team.loc[k - 1, "Date"]).days)
            same_venue = team.iloc[:k][team.iloc[:k]["is_home"] == team.loc[k, "is_home"]].tail(WINDOW)
            expected = same_venue["WIN"].mean() if len(same_venue) else np.nan
            np.testing.assert_allclose(team.loc[k, "venue_win_pct"], expected)

    def test_current_game_does_not_leak(self):
        target = "G005"
        before = build_game_features(self.logs, WINDOW).set_index("GAME_ID").loc[target]
        altered = self.logs.copy()
        in_game = altered["GAME_ID"] == target
        altered.loc[in_game, ["FGM", "DREB", "TOV", "BLK"]] = [0, 0, 99, 99]
        altered.loc[in_game, "WL"] = altered.loc[in_game, "WL"].map({"W": "L", "L": "W"})
        after = build_game_features(altered, WINDOW).set_index("GAME_ID").loc[target]
        pd.testing.assert_series_equal(before, after)

    def test_rows_are_home_minus_away(self):
        games = build_game_features(self.logs, WINDOW).set_index("GAME_ID")
        teams = self.teams.set_index(["GAME_ID", "is_home"])
        home, away = teams.loc[("G004", True)], teams.loc[("G004", False)]
        self.assertAlmostEqual(games.loc["G004", "FG%_diff"], home["FG%"] - away["FG%"])
        self.assertEqual(games.loc["G004", "TEAM_NAME"], home["TEAM_NAME"])


if __name__ == "__main__":
    unittest.main()
