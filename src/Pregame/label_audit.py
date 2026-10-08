"""Compare dataset_2012-26 labels (Home-Team-Win) and Score with the game logs, for every season.

Dataset rows are matched to logged games by date and franchise TEAM_ID (team names drift by era).
Rows with no logged game on that date are returned separately: neutral-site games, NBA Cup
finals (not in the logs), archive rows dated a day off, and odds rows for games that were not
played that day.
"""
import sqlite3

import pandas as pd

from src.Pregame.paths import DATASET_DB, DATASET_TABLE, SPLIT_KEYS

TARGET = "Home-Team-Win"


def logged_results(team_games_frame):
    """Every logged game (neutral sites included): date, home/away franchise, home win, total points."""
    games = team_games_frame.copy()
    # Neutral-site games list both teams as "@"; treat the first listed team as home only for matching.
    games["side"] = games["is_home"] | (games["neutral"] & ~games.duplicated("GAME_ID"))
    home = games[games["side"]][["GAME_ID", "Date", "TEAM_ID", "PTS", "SEASON", "SEASON_TYPE", "neutral"]]
    away = games[~games["side"]][["GAME_ID", "TEAM_ID", "PTS"]]
    results = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="one_to_one")
    results["Date"] = results["Date"].dt.strftime("%Y-%m-%d")
    results["win_logs"] = (results["PTS_home"] > results["PTS_away"]).astype(float)
    results["score_logs"] = (results["PTS_home"] + results["PTS_away"]).astype(float)
    results["margin_logs"] = (results["PTS_home"] - results["PTS_away"]).astype(float)
    return results


def franchise_ids(team_games_frame):
    ids = team_games_frame.groupby("TEAM_NAME")["TEAM_ID"].unique()
    if (ids.str.len() != 1).any():
        raise ValueError("A team name maps to more than one franchise")
    return ids.str[0]


def match_to_logs(frame, team_games_frame, home="TEAM_NAME", away="TEAM_NAME.1"):
    """Left-join rows with Date/home/away names to logged games, trying both orders for neutral sites."""
    ids = franchise_ids(team_games_frame)
    results = logged_results(team_games_frame)
    frame = frame.assign(TEAM_ID_home=frame[home].map(ids), TEAM_ID_away=frame[away].map(ids))
    if frame[["TEAM_ID_home", "TEAM_ID_away"]].isna().any().any():
        raise ValueError("Team name not found in the game logs")
    # A neutral-site game has no home side in the logs, so it can match in either order.
    flipped = results[results["neutral"]].rename(columns={
        "TEAM_ID_home": "TEAM_ID_away", "TEAM_ID_away": "TEAM_ID_home",
        "PTS_home": "PTS_away", "PTS_away": "PTS_home"})
    flipped = flipped.assign(win_logs=1.0 - flipped["win_logs"], margin_logs=-flipped["margin_logs"])
    both_orders = pd.concat([results, flipped], ignore_index=True)
    matched = frame.merge(both_orders, on=["Date", "TEAM_ID_home", "TEAM_ID_away"], how="left",
                          validate="many_to_one")
    return matched


def label_audit(team_games_frame, db_path=DATASET_DB):
    """(matched rows with any label or score disagreement, all matched rows, unmatched rows)."""
    with sqlite3.connect(db_path) as con:
        data = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    data = data.merge(pd.read_csv(SPLIT_KEYS), on=["Date", "TEAM_NAME", "TEAM_NAME.1"], how="left")
    data["split"] = data["split"].fillna("not in split")
    rows = match_to_logs(data, team_games_frame)
    found = rows["GAME_ID"].notna()
    matched = rows[found]
    wrong = matched[(matched[TARGET] != matched["win_logs"]) | (matched["Score"] != matched["score_logs"])]
    columns = ["Date", "TEAM_NAME", "TEAM_NAME.1", "SEASON", "SEASON_TYPE", "split", TARGET, "win_logs",
               "Score", "score_logs"]
    return wrong[columns], matched, rows.loc[~found, ["Date", "TEAM_NAME", "TEAM_NAME.1", "split", TARGET, "Score"]]
