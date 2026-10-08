"""Betting-market lines from Data/OddsData.sqlite: table choice, sign convention, game matching.

Which table per season (checked against the game logs; see docs/findings.md §9a):
  - 2012-13 .. 2022-23: odds_<season>_new. The legacy odds_<season> tables hold the same values
    with an unparseable season-prefixed date ("2012-13-1030"); the _new copies fix only the date.
  - 2023-24: "2023-24" (1,252 matched games). It agrees with odds_2023-24_new on every line and
    moneyline of the 1,193 games both hold, but _new is missing 59 games and has 4 wrong scores,
    so _new is used only for the one game "2023-24" lacks.
  - 2024-25: "2024-25"; 2025-26: odds_2025-26 (the only tables).

Spread sign: through 2021-22 Spread is the favourite's line, unsigned (never negative); from 2022-23
it is the away team's handicap from SBR (Get_Odds_Data.py stores game["away_spread"]), i.e.
positive when the home team is favoured. Both are converted to `home_line` = the margin the market
expects for the home team (positive = home favoured). Unsigned lines take their side from the
moneyline favourite; when the two moneylines are equal the side is unknown and the line is missing.

Odds team names drift by era ("Charlotte Bobcats" for 2015-16, "Los Angeles Clippers"), so games are
matched to the logs by date and franchise TEAM_ID, never by name.
"""
import sqlite3

import numpy as np
import pandas as pd

from src.Pregame.paths import BASE_DIR

ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
SIGNED_FROM = "2022-23"
SEASON_TABLES = {
    **{f"{y}-{str(y + 1)[-2:]}": [f"odds_{y}-{str(y + 1)[-2:]}_new"] for y in range(2012, 2023)},
    "2023-24": ["2023-24", "odds_2023-24_new"],  # first table wins; the second only fills gaps
    "2024-25": ["2024-25"],
    "2025-26": ["odds_2025-26"],
}


def american_to_prob(ml):
    """Implied probability of an American moneyline (vig included). NaN for |ml| < 100."""
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    return np.where(np.abs(ml) >= 100, prob, np.nan)


def devig_home_prob(ml_home, ml_away):
    """Home-win probability with the bookmaker margin removed proportionally (q_h / (q_h + q_a))."""
    q_home, q_away = american_to_prob(ml_home), american_to_prob(ml_away)
    return q_home / (q_home + q_away)


def parse_spread(spread):
    """Numeric spread; "PK" (pick'em) is 0."""
    return pd.to_numeric(pd.Series(spread).replace({"PK": 0, "pk": 0}), errors="coerce").to_numpy(dtype=float)


def home_line(spread, ml_home, ml_away, signed):
    """Market-expected home margin (positive = home favoured).

    signed=True: the stored spread already is the home team's expected margin.
    signed=False: |spread| goes to the moneyline favourite; equal moneylines leave it unknown
    unless the spread is 0.
    """
    spread = parse_spread(spread)
    if signed:
        return spread
    favourite = np.sign(np.asarray(ml_away, dtype=float) - np.asarray(ml_home, dtype=float))
    line = np.abs(spread) * favourite
    return np.where((favourite == 0) & (np.abs(spread) > 0), np.nan, line)


def cover_result(home_margin, line):
    """+1 home covers, 0 push, -1 home fails to cover, NaN when the line is unknown."""
    diff = np.asarray(home_margin, dtype=float) - np.asarray(line, dtype=float)
    return np.sign(diff)


def load_odds_table(con, table, season):
    odds = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    odds["Date"] = pd.to_datetime(odds["Date"]).dt.strftime("%Y-%m-%d")
    spread = parse_spread(odds["Spread"])
    negative_share = np.mean(spread < 0)
    signed = season >= SIGNED_FROM
    # The convention switch is a property of the data; fail loudly if a table does not follow it.
    if signed != (negative_share > 0.25) or (not signed and (spread < 0).sum() > 1):
        raise ValueError(f"{table}: {negative_share:.1%} negative spreads does not fit signed={signed}")
    odds["home_line"] = home_line(odds["Spread"], odds["ML_Home"], odds["ML_Away"], signed)
    odds["p_market"] = devig_home_prob(odds["ML_Home"], odds["ML_Away"])
    odds["odds_table"], odds["SEASON"] = table, season
    return odds


def logged_games(team_games_frame):
    """One row per sited logged game: GAME_ID, date, home/away TEAM_ID and the home margin."""
    sited = team_games_frame[~team_games_frame["neutral"]]
    home = sited.loc[sited["is_home"], ["GAME_ID", "Date", "TEAM_ID", "PLUS_MINUS"]]
    away = sited.loc[~sited["is_home"], ["GAME_ID", "TEAM_ID"]]
    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="one_to_one")
    games["Date"] = games["Date"].dt.strftime("%Y-%m-%d")
    return games.rename(columns={"PLUS_MINUS": "home_margin_logs"})


def load_market(team_games_frame, db_path=ODDS_DB):
    """Market lines per logged game (one row per GAME_ID), plus a per-table match report."""
    team_ids = team_games_frame.groupby("TEAM_NAME")["TEAM_ID"].unique()
    if (team_ids.str.len() != 1).any():
        raise ValueError("A team name maps to more than one franchise")
    team_ids = team_ids.str[0]
    games = logged_games(team_games_frame)

    frames, report = [], []
    with sqlite3.connect(db_path) as con:
        for season, tables in SEASON_TABLES.items():
            for table in tables:
                odds = load_odds_table(con, table, season)
                odds["TEAM_ID_home"], odds["TEAM_ID_away"] = odds["Home"].map(team_ids), odds["Away"].map(team_ids)
                if odds[["TEAM_ID_home", "TEAM_ID_away"]].isna().any().any():
                    raise ValueError(f"{table}: team name not in the game logs")
                matched = odds.merge(games, on=["Date", "TEAM_ID_home", "TEAM_ID_away"], how="inner")
                report.append({"season": season, "table": table, "rows": len(odds), "matched": len(matched),
                               "win_margin_agrees": float((matched["Win_Margin"] == matched["home_margin_logs"]).mean())})
                frames.append(matched)
    market = pd.concat(frames, ignore_index=True)
    market = market.drop_duplicates("GAME_ID", keep="first")  # earlier table in SEASON_TABLES wins
    keep = ["GAME_ID", "SEASON", "odds_table", "Spread", "ML_Home", "ML_Away", "home_line", "p_market",
            "Win_Margin", "home_margin_logs"]
    return market[keep].reset_index(drop=True), pd.DataFrame(report)
