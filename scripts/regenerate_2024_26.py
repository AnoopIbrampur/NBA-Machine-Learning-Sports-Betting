"""Regenerate the 2024-25 and 2025-26 team snapshots and dataset rows after the Get_Data fix.

Those seasons' TeamData tables were fetched after the upstream rewrite that dropped the one-day
offset, so each date's table included that day's games. This re-fetches exactly the dates that
already exist for those seasons with the fixed Get_Data.fetch_data, then rebuilds only those
seasons' rows of dataset_2012-26 with Create_Games' own functions. Rows for 2012-13 to 2023-24
are copied through unchanged.

Run from the repo root:
    python scripts/regenerate_2024_26.py --refetch   # ~15-20 min (rate-limited API)
    python scripts/regenerate_2024_26.py --rebuild
"""
import argparse
import importlib.util
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

SEASONS = ["2024-25", "2025-26"]
TEAMS_DB = BASE_DIR / "Data" / "TeamData.sqlite"
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
DATASET_TABLE = "dataset_2012-26"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_data = load_module("get_data", "src/Process-Data/Get_Data.py")
create_games = load_module("create_games", "src/Process-Data/Create_Games.py")


def season_bounds(config, season):
    value = config["get-data"][season]
    start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    return value, start, end


def refetch(config):
    with sqlite3.connect(TEAMS_DB) as con:
        existing = get_data.get_table_dates(con)
        for season in SEASONS:
            value, start, end = season_bounds(config, season)
            dates = sorted(d for d in existing if start <= d <= end)
            print(f"{season}: re-fetching {len(dates)} snapshots")
            for date_pointer in dates:
                table_name = date_pointer.strftime("%Y-%m-%d")
                df = get_data.fetch_data(config["data_url"], date_pointer, value["start_year"], season)
                if df.empty:
                    # No games before this date yet (opening night), same as earlier seasons.
                    con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                    print(f"  {table_name}: no prior games, table dropped")
                else:
                    df["Date"] = table_name
                    df.to_sql(table_name, con, if_exists="replace", index=False)
                time.sleep(get_data.MIN_DELAY_SECONDS + np.random.random()
                           * (get_data.MAX_DELAY_SECONDS - get_data.MIN_DELAY_SECONDS))


def season_rows(season, odds_con, teams_con):
    """Create_Games.main's per-season loop, returning a frame for one season."""
    odds_table = create_games.select_odds_table(odds_con, season)
    odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
    index_map = create_games.get_team_index_map(season)
    games, extra = [], []
    for row in odds_df.itertuples(index=False):
        team_df = create_games.fetch_team_table(teams_con, create_games.normalize_date(row.Date))
        if team_df is None:
            continue
        game = create_games.build_game_features(team_df, row.Home, row.Away, index_map)
        if game is None:
            continue
        games.append(game)
        ou_cover = 0 if row.Points < row.OU else 1 if row.Points > row.OU else 2
        extra.append({"Score": row.Points, "Home-Team-Win": 1 if row.Win_Margin > 0 else 0, "OU": row.OU,
                      "OU-Cover": ou_cover, "Days-Rest-Home": row.Days_Rest_Home,
                      "Days-Rest-Away": row.Days_Rest_Away})
    frame = pd.concat(games, ignore_index=True, axis=1).T
    frame = frame.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    return pd.concat([frame, pd.DataFrame(extra)], axis=1)


def rebuild(config):
    with sqlite3.connect(DATASET_DB) as con:
        old = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    first_new = min(season_bounds(config, s)[1] for s in SEASONS).isoformat()
    kept = old[old["Date"] < first_new]

    with sqlite3.connect(create_games.ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB) as teams_con:
        new = pd.concat([season_rows(s, odds_con, teams_con) for s in SEASONS], ignore_index=True)
    new = new.reindex(columns=old.columns)
    for column in new.columns:
        if "TEAM_" not in column and "Date" not in column:
            new[column] = new[column].astype(float)

    frame = pd.concat([kept, new], ignore_index=True)
    with sqlite3.connect(DATASET_DB) as con:
        frame.to_sql(DATASET_TABLE, con, if_exists="replace", index=False)
    replaced = (old["Date"] >= first_new).sum()
    print(f"Kept {len(kept)} rows before {first_new}; replaced {replaced} rows with {len(new)} rebuilt rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config = toml.load(BASE_DIR / "config.toml")
    if args.refetch:
        refetch(config)
    if args.rebuild:
        rebuild(config)


if __name__ == "__main__":
    main()
