"""Repair game results in the odds tables Create_Games uses, then rebuild only the affected dataset rows.

Cause (fixed in code): Get_Odds_Data.py's daily mode fetched through today, so games that had not
finished were stored with partial or zero scores, and Create_Games turned a zero win margin into
Home-Team-Win = 0. Get_Odds_Data now fetches through yesterday and skips games without a final
score; Create_Games skips rows without a final result. A few archive rows (2012-13 to 2016-17) and
two 2024-25 rows also carry a wrong total score with the correct winner.

This script repairs the rows already stored:
  --fix-odds  For every season's Create_Games table, set Points and Win_Margin to the game-log values
              wherever a row matches a logged game (date + franchise) and disagrees.
  --rebuild   Recompute Score, Home-Team-Win and OU-Cover of dataset_2012-26 rows from the repaired
              odds rows, and drop rows whose odds row has no final result (games not played that
              day). Team-stat features are untouched; every other row is left as it is.

Run from the repo root (needs Data/pregame/GameLogs.sqlite from src.Pregame.game_logs):
    python scripts/fix_game_results.py --fix-odds --rebuild
"""
import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.Pregame.features import team_games  # noqa: E402
from src.Pregame.game_logs import load_game_logs  # noqa: E402
from src.Pregame.label_audit import franchise_ids, match_to_logs  # noqa: E402

DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
DATASET_TABLE = "dataset_2012-26"
RESULT_COLUMNS = ["Score", "Home-Team-Win", "OU-Cover"]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_games = load_module("create_games", "src/Process-Data/Create_Games.py")


def season_tables(config, odds_con):
    return {season: create_games.select_odds_table(odds_con, season) for season in config["create-games"]}


def read_odds(odds_con, table):
    odds = pd.read_sql_query(f'SELECT rowid AS odds_rowid, * FROM "{table}"', odds_con)
    odds["Date"] = pd.to_datetime(odds["Date"]).dt.strftime("%Y-%m-%d")
    return odds


def fix_odds(config, logs):
    changes = []
    with sqlite3.connect(create_games.ODDS_DB_PATH) as con:
        for season, table in season_tables(config, con).items():
            odds = match_to_logs(read_odds(con, table), logs, home="Home", away="Away")
            found = odds["GAME_ID"].notna()
            wrong = odds[found & ((odds["Points"] != odds["score_logs"]) | (odds["Win_Margin"] != odds["margin_logs"]))]
            for row in wrong.itertuples():
                con.execute(f'UPDATE "{table}" SET Points = ?, Win_Margin = ? WHERE rowid = ?',
                            (int(row.score_logs), int(row.margin_logs), int(row.odds_rowid)))
                changes.append({"season": season, "table": table, "Date": row.Date, "Home": row.Home,
                                "Away": row.Away, "Points": f"{row.Points} -> {int(row.score_logs)}",
                                "Win_Margin": f"{row.Win_Margin} -> {int(row.margin_logs)}"})
    changes = pd.DataFrame(changes)
    print(f"Repaired {len(changes)} odds rows:")
    print(changes.to_string(index=False) if len(changes) else "  none")
    return changes


def result_columns(row):
    """Score, Home-Team-Win and OU-Cover exactly as Create_Games.main derives them."""
    ou_cover = 0 if row.Points < row.OU else 1 if row.Points > row.OU else 2
    return float(row.Points), float(1 if row.Win_Margin > 0 else 0), float(ou_cover)


def rebuild(config, logs):
    ids = franchise_ids(logs)
    with sqlite3.connect(DATASET_DB) as con:
        data = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    with sqlite3.connect(create_games.ODDS_DB_PATH) as odds_con:
        odds = pd.concat([read_odds(odds_con, table) for table in season_tables(config, odds_con).values()],
                         ignore_index=True)
    keys = ["Date", "home_id", "away_id"]
    odds = odds.assign(home_id=odds["Home"].map(ids), away_id=odds["Away"].map(ids))
    odds = odds[keys + ["Points", "Win_Margin", "OU"]]
    source = data.assign(home_id=data["TEAM_NAME"].map(ids), away_id=data["TEAM_NAME.1"].map(ids))[keys].merge(
        odds, on=keys, how="left", validate="many_to_one", indicator=True)
    if (source["_merge"] != "both").any():
        raise ValueError("Every dataset row must come from exactly one odds row")

    final = source.apply(create_games.has_final_result, axis=1).to_numpy()
    rebuilt = data.copy()
    rebuilt.loc[final, RESULT_COLUMNS] = [result_columns(row) for row in source[final].itertuples()]
    changed = final & (rebuilt[RESULT_COLUMNS] != data[RESULT_COLUMNS]).any(axis=1).to_numpy()
    report = data.loc[changed | ~final, ["Date", "TEAM_NAME", "TEAM_NAME.1"] + RESULT_COLUMNS].assign(
        action=["dropped: no final result" if not f else "results rebuilt" for f in final[changed | ~final]])
    report = report.join(rebuilt.loc[changed, RESULT_COLUMNS].add_suffix(" (new)"))
    rebuilt = rebuilt[final].reset_index(drop=True)

    other = [c for c in data.columns if c not in RESULT_COLUMNS]
    pd.testing.assert_frame_equal(rebuilt[other], data.loc[final, other].reset_index(drop=True))
    with sqlite3.connect(DATASET_DB) as con:
        rebuilt.to_sql(DATASET_TABLE, con, if_exists="replace", index=False)
    print(f"{int(changed.sum())} rows rebuilt, {int((~final).sum())} dropped, "
          f"{len(data) - int(changed.sum()) - int((~final).sum())} unchanged:")
    print(report.to_string())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix-odds", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config = toml.load(BASE_DIR / "config.toml")
    logs = team_games(load_game_logs())
    if args.fix_odds:
        fix_odds(config, logs)
    if args.rebuild:
        rebuild(config, logs)


if __name__ == "__main__":
    main()
