"""Download per-game team box scores (stats.nba.com leaguegamelog) and cache them in SQLite.

TeamData.sqlite only holds cumulative season averages, rounded to one decimal, so single-game
box scores cannot be recovered from it. One request per season and season type is enough here.
"""
import argparse
import sqlite3
import time

import pandas as pd
import requests

from src.Pregame.paths import GAME_LOGS_DB
from src.Utils.tools import data_headers

URL = "https://stats.nba.com/stats/leaguegamelog"
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2012, 2026)]
SEASON_TYPES = ["Regular Season", "PlayIn", "Playoffs"]
TABLE = "team_game_logs"


def fetch(season, season_type, retries=3):
    params = {
        "Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": "T",
        "Season": season, "SeasonType": season_type, "Sorter": "DATE",
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(URL, params=params, headers=data_headers, timeout=60)
            response.raise_for_status()
            result = response.json()["resultSets"][0]
            return pd.DataFrame(result["rowSet"], columns=result["headers"])
        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"  attempt {attempt} failed: {error}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not fetch {season} {season_type}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    GAME_LOGS_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(GAME_LOGS_DB) as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
        ).fetchone()
        if exists and not args.refresh:
            print(f"Using cached {GAME_LOGS_DB}; pass --refresh to re-download.")
            return

        frames = []
        for season in SEASONS:
            for season_type in SEASON_TYPES:
                df = fetch(season, season_type)
                print(f"{season} {season_type}: {len(df)} team-games")
                if not df.empty:
                    frames.append(df.assign(SEASON=season, SEASON_TYPE=season_type))
                time.sleep(1.5)

        logs = pd.concat(frames, ignore_index=True).drop(columns=["VIDEO_AVAILABLE"], errors="ignore")
        logs.to_sql(TABLE, con, if_exists="replace", index=False)
        print(f"Saved {len(logs)} team-games to {GAME_LOGS_DB}")


def load_game_logs():
    with sqlite3.connect(GAME_LOGS_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{TABLE}"', con)


if __name__ == "__main__":
    main()
