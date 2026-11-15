"""Show saved models' pre-game predictions for one date. No model is trained or changed.

    python -m src.Pregame.demo --date 2026-03-15
    python -m src.Pregame.demo --date 2026-03-15 --team Celtics --reasons 3

The official model is pre-game window 20. The other is pre-game + Elo variant + player features.
Run the Reproduce steps first: this reads their saved models and cached logs.
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.Pregame.elo import ELO_FEATURES, pregame_elo
from src.Pregame.enhanced2 import ELO2_FEATURES, PRIMARY
from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.paths import ENHANCED2_RESULTS_DIR, features_path, model_path
from src.Pregame.player_features import PLAYER_FEATURES, build_player_features
from src.Pregame.player_logs import load_player_logs
from src.Pregame.schedule_features import BUBBLE_START, offsite_game_ids
from src.Pregame.train_xgb import load_games

SPLIT_NOTES = {
    "train": "model TRAINED on this game, not a fair test",
    "test": "held-out test game",
    "validation": "held-out validation game",
    "after split": "fresh game, never used to build or choose a model",
}


def select_games(features, date, team=None):
    """The date's games, optionally only those whose home or away name contains team."""
    chosen = features[features["Date"] == date]
    if team:
        hit = chosen["TEAM_NAME"].str.contains(team, case=False, regex=False) \
            | chosen["TEAM_NAME.1"].str.contains(team, case=False, regex=False)
        chosen = chosen[hit]
    return chosen


def nearest_dates(features, date, count=3):
    dates = pd.Series(sorted(features["Date"].unique()))
    gaps = (pd.to_datetime(dates) - pd.Timestamp(date)).abs()
    return sorted(dates[gaps.argsort().to_numpy()[:count]])


def top_reasons(contributions, columns, count):
    """Largest log-odds contributions as (feature, signed value); the bias column is dropped."""
    values = np.asarray(contributions)[: len(columns)]
    order = np.argsort(-np.abs(values))[:count]
    return [(columns[i], float(values[i])) for i in order]


def describe_pick(home, away, probability):
    return f"{home} {probability:.1%}" if probability >= 0.5 else f"{away} {1 - probability:.1%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="Game date, YYYY-MM-DD.")
    parser.add_argument("--team", help="Only games with this text in a team name, such as Celtics.")
    parser.add_argument("--reasons", type=int, default=3, help="Top reasons to show for each game.")
    args = parser.parse_args()

    features = pd.read_csv(features_path(20), dtype={"GAME_ID": str})
    chosen = select_games(features, args.date, args.team)
    if chosen.empty:
        played = select_games(features, args.date)
        if played.empty:
            print(f"No games on {args.date}. Nearest game dates: {', '.join(nearest_dates(features, args.date))}")
        else:
            teams = sorted(set(played["TEAM_NAME"]) | set(played["TEAM_NAME.1"]))
            print(f"No {args.team} game on {args.date}. Teams playing: {', '.join(teams)}")
        return

    training = json.loads((ENHANCED2_RESULTS_DIR / "training.json").read_text())
    variant = training["elo_variant"]
    games, columns = load_games(20, schedule=True)
    logs = team_games(load_game_logs())
    bubble = logs.loc[(logs["SEASON"] == "2019-20") & (logs["Date"] >= BUBBLE_START), "GAME_ID"]
    no_home = frozenset(bubble) | frozenset(offsite_game_ids(logs))
    elo = pregame_elo(logs, variant["k"], variant["home_advantage"], variant["regression"], variant["margin"],
                      no_home).rename(columns=dict(zip(ELO_FEATURES, ELO2_FEATURES)))
    player = build_player_features(load_player_logs(), logs, training["replacement_level"])
    frame = chosen.merge(elo, left_on="GAME_ID", right_index=True, how="left").merge(
        player, left_on="GAME_ID", right_index=True, how="left")

    primary_columns = columns["pregame"] + ELO2_FEATURES + PLAYER_FEATURES
    official = xgb.Booster()
    official.load_model(str(model_path("pregame", 20)))
    primary = xgb.Booster()
    primary.load_model(str(model_path(PRIMARY, 20)))
    primary_matrix = xgb.DMatrix(frame[primary_columns].astype(float))
    official_probability = official.predict(xgb.DMatrix(frame[columns["pregame"]].astype(float)))
    primary_probability = primary.predict(primary_matrix)
    contributions = primary.predict(primary_matrix, pred_contribs=True)

    splits = games.set_index("GAME_ID")["split"]
    home_rows = logs[logs["is_home"]].set_index("GAME_ID")
    print(f"Games on {args.date}. Percentages are each model's pre-game win chance for its pick.\n")
    for i in range(len(frame)):
        row = frame.iloc[i]
        home, away = row["TEAM_NAME"], row["TEAM_NAME.1"]
        print(f"{away} at {home}  [{SPLIT_NOTES[splits.get(row['GAME_ID'], 'after split')]}]")
        print(f"  Official model:          {describe_pick(home, away, float(official_probability[i]))}")
        print(f"  Elo + player model:      {describe_pick(home, away, float(primary_probability[i]))}")
        result = home_rows.loc[row["GAME_ID"], "WL"]
        if isinstance(result, str):
            winner = home if result == "W" else away
            right = (primary_probability[i] >= 0.5) == (result == "W")
            print(f"  Actual winner:           {winner} (Elo + player model {'right' if right else 'wrong'})")
        else:
            print("  Actual winner:           no result logged")
        for name, value in top_reasons(contributions[i], primary_columns, args.reasons):
            print(f"    {name}: {value:+.3f} toward {'home' if value > 0 else 'away'}")
        print()


if __name__ == "__main__":
    main()
