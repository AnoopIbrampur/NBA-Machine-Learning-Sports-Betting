"""Pre-game team features built from player game logs (docs/enhanced-plan-2.md).

Games are processed date by date. All of a date's features are read from the state left by earlier
dates, and only then do that date's player rows update the state, so a game's own rows are never read.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.Pregame.paths import WORK_DIR

RATING_GAMES, PRIOR_MINUTES, REPLACEMENT_SHARE = 40, 200.0, 0.75
STRENGTH_GAMES, AVAILABILITY_GAMES = 5, 10
TEAM_FEATURES = ["player_strength", "player_strength_lastgame", "player_availability", "player_continuity"]
PLAYER_FEATURES = ([f"{f}_{side}" for f in TEAM_FEATURES for side in ("home", "away")]
                   + ["player_strength_diff", "player_strength_lastgame_diff"])


def box_value(p):
    """Our own box-score value for each player-game row."""
    return (p["PTS"] + 0.4 * p["FGM"] - 0.7 * p["FGA"] - 0.4 * (p["FTA"] - p["FTM"]) + 0.7 * p["OREB"]
            + 0.3 * p["DREB"] + p["STL"] + 0.7 * p["AST"] + 0.7 * p["BLK"] - 0.4 * p["PF"] - p["TOV"])


def replacement_level(players, train_game_ids):
    """REPLACEMENT_SHARE of the minutes-weighted league per-36 value, from train-split games only."""
    train = players[players["GAME_ID"].isin(set(train_game_ids))]
    return REPLACEMENT_SHARE * 36.0 * float(box_value(train).sum() / train["MIN"].sum())


def previous_season(season):
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


class PlayerState:
    """Everything known about players and rotations from the dates processed so far."""

    def __init__(self, replacement):
        self.replacement = replacement
        self.history = defaultdict(lambda: deque(maxlen=RATING_GAMES))
        self.recent = defaultdict(lambda: deque(maxlen=AVAILABILITY_GAMES))
        self.roster = defaultdict(set)

    def rating(self, player):
        games = self.history[player]
        value = sum(v for v, _ in games)
        minutes = sum(m for _, m in games)
        return (36.0 * value + PRIOR_MINUTES * self.replacement) / (minutes + PRIOR_MINUTES)

    def team_features(self, team, season):
        games = list(self.recent[(team, season)])
        if not games:
            return dict.fromkeys(TEAM_FEATURES, np.nan)
        strength_minutes, all_minutes = defaultdict(float), defaultdict(float)
        for game in games[-STRENGTH_GAMES:]:
            for player, minutes in game.items():
                strength_minutes[player] += minutes
        for game in games:
            for player, minutes in game.items():
                all_minutes[player] += minutes
        last = set(games[-1])

        def weighted_rating(players):
            total = sum(strength_minutes[p] for p in players)
            return sum(strength_minutes[p] * self.rating(p) for p in players) / total if total else np.nan

        last_season = self.roster.get((team, previous_season(season)))
        continuity = np.nan
        if last_season:
            continuity = sum(m for p, m in strength_minutes.items() if p in last_season) / sum(strength_minutes.values())
        return {"player_strength": weighted_rating(list(strength_minutes)),
                "player_strength_lastgame": weighted_rating([p for p in strength_minutes if p in last]),
                "player_availability": sum(m for p, m in all_minutes.items() if p in last) / sum(all_minutes.values()),
                "player_continuity": continuity}

    def update(self, team, season, rows):
        game = {}
        for player, minutes, value in rows:
            if minutes > 0:
                self.history[player].append((value, minutes))
                game[player] = minutes
                self.roster[(team, season)].add(player)
        if game:
            self.recent[(team, season)].append(game)


def build_player_features(players, team_rows, replacement):
    """One row per sited game, indexed by GAME_ID. team_rows has one row per team per game."""
    players = players.assign(value=box_value(players), GAME_ID=players["GAME_ID"].astype(str))
    rows_by_game = defaultdict(list)
    for game, team, player, minutes, value in players[["GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN", "value"]].itertuples(index=False):
        rows_by_game[(game, team)].append((player, float(minutes), float(value)))

    state = PlayerState(replacement)
    records = {}
    schedule = team_rows.sort_values(["Date", "GAME_ID"], kind="stable")
    for _, day in schedule.groupby("Date", sort=True):
        entries = list(day[["GAME_ID", "TEAM_ID", "SEASON", "is_home", "neutral"]].itertuples(index=False))
        for game, team, season, is_home, neutral in entries:
            if not neutral:
                side = "home" if is_home else "away"
                record = records.setdefault(game, {})
                for name, value in state.team_features(team, season).items():
                    record[f"{name}_{side}"] = value
        for game, team, season, _, _ in entries:
            state.update(team, season, rows_by_game.get((game, team), []))

    out = pd.DataFrame.from_dict(records, orient="index")
    out["player_strength_diff"] = out["player_strength_home"] - out["player_strength_away"]
    out["player_strength_lastgame_diff"] = out["player_strength_lastgame_home"] - out["player_strength_lastgame_away"]
    out.index.name = "GAME_ID"
    return out[PLAYER_FEATURES]


def features_csv():
    return WORK_DIR / "player_features.csv"
