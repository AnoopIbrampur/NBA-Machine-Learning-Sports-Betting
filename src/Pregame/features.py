"""Build pre-game features: each team's rolling form over its previous N games plus schedule context.

Every feature for a game uses only that team's games strictly before it (shift(1) before
rolling), within the same season. Box-score features follow the 11 categories in the parent
paper's final SHAP model (PLOS ONE 2024, Table 10) and, like the paper, are home minus away.
"""
import argparse

import numpy as np
import pandas as pd

from src.Pregame.game_logs import load_game_logs
from src.Pregame.paths import features_path

# Parent-paper category -> (made, attempted) for percentages, or the per-game count column.
PCT_STATS = {"FG%": ("FGM", "FGA"), "2P%": ("FG2M", "FG2A"), "3P%": ("FG3M", "FG3A"), "FT%": ("FTM", "FTA")}
COUNT_STATS = {"ORB": "OREB", "DRB": "DREB", "AST": "AST", "STL": "STL", "BLK": "BLK", "TOV": "TOV", "PF": "PF"}
PAPER_STATS = list(PCT_STATS) + list(COUNT_STATS)


def team_games(logs):
    """One row per team per game, sorted chronologically within each team-season."""
    games = logs.drop_duplicates(["TEAM_ID", "GAME_ID"]).copy()
    games["Date"] = pd.to_datetime(games["GAME_DATE"])
    games["is_home"] = games["MATCHUP"].str.contains("vs.", regex=False)
    # Neutral-site games (international, NBA Cup knockouts) list both teams as "@".
    games["neutral"] = ~games.groupby("GAME_ID")["is_home"].transform("any")
    games["WIN"] = (games["WL"] == "W").astype(float)
    games["FG2M"] = games["FGM"] - games["FG3M"]
    games["FG2A"] = games["FGA"] - games["FG3A"]
    return games.sort_values(["TEAM_ID", "SEASON", "Date"]).reset_index(drop=True)


def prior_rolling(frame, columns, window, how):
    """Rolling sum/mean of `columns` over each team-season's previous `window` games only."""
    grouped = frame.groupby(["TEAM_ID", "SEASON"], sort=False)[columns]
    return grouped.transform(lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)())


def add_team_features(games, window):
    out = games.copy()

    made_attempted = sorted({col for pair in PCT_STATS.values() for col in pair})
    sums = prior_rolling(out, made_attempted, window, "sum")
    for stat, (made, attempted) in PCT_STATS.items():
        out[stat] = sums[made] / sums[attempted].replace(0, np.nan)

    means = prior_rolling(out, list(COUNT_STATS.values()), window, "mean")
    for stat, column in COUNT_STATS.items():
        out[stat] = means[column]

    out["win_pct"] = prior_rolling(out, ["WIN"], window, "mean")["WIN"]
    out["rest_days"] = out.groupby(["TEAM_ID", "SEASON"], sort=False)["Date"].diff().dt.days
    out["b2b"] = (out["rest_days"] == 1).astype(float).where(out["rest_days"].notna())

    # Form at this game's venue type: last N home games for a home team, last N road games otherwise.
    out["venue_win_pct"] = np.nan
    for is_home in (True, False):
        venue = out[(out["is_home"] == is_home) & ~out["neutral"]]
        out.loc[venue.index, "venue_win_pct"] = prior_rolling(venue, ["WIN"], window, "mean")["WIN"]
    return out


def build_game_features(logs, window):
    teams = add_team_features(team_games(logs), window)
    context = ["win_pct", "venue_win_pct", "rest_days", "b2b"]
    keep = ["GAME_ID", "Date", "TEAM_NAME", "SEASON"] + PAPER_STATS + context

    # Neutral-site games stay in each team's rolling history but have no home side to predict.
    sited = teams[~teams["neutral"]]
    home = sited.loc[sited["is_home"], keep]
    away = sited.loc[~sited["is_home"], keep].drop(columns=["Date", "SEASON"])
    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="one_to_one")

    features = pd.DataFrame({
        "GAME_ID": games["GAME_ID"],
        "SEASON": games["SEASON"],
        "Date": games["Date"].dt.strftime("%Y-%m-%d"),
        "TEAM_NAME": games["TEAM_NAME_home"],
        "TEAM_NAME.1": games["TEAM_NAME_away"],
    })
    for stat in PAPER_STATS:
        features[f"{stat}_diff"] = games[f"{stat}_home"] - games[f"{stat}_away"]
    features["win_pct_home"] = games["win_pct_home"]
    features["win_pct_away"] = games["win_pct_away"]
    features["home_win_pct_at_home"] = games["venue_win_pct_home"]
    features["away_win_pct_on_road"] = games["venue_win_pct_away"]
    for side in ("home", "away"):
        features[f"rest_days_{side}"] = games[f"rest_days_{side}"]
        features[f"b2b_{side}"] = games[f"b2b_{side}"]
    return features.sort_values(["Date", "GAME_ID"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10, help="Number of prior games to average.")
    args = parser.parse_args()

    features = build_game_features(load_game_logs(), args.window)
    path = features_path(args.window)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    print(f"Wrote {len(features)} games x {features.shape[1]} columns to {path}")


if __name__ == "__main__":
    main()
