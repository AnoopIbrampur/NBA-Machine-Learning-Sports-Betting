"""Leakage-free Elo ratings from the game logs (Alves & Barbosa 2025, section 3.1.4).

Every franchise starts at 1500 and ratings carry over in full between seasons. The update is the
standard R + k(S - E); the paper's Equation (1) prints k(S - E + R), read here as a typo. The paper
does not define E, so the standard logistic on a 400-point scale is assumed, with no home-court term.

A game's rating columns are read before that game's result is applied, so they depend only on
earlier games. Each team plays at most once per date, so games are processed by date and then GAME_ID.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

START_RATING = 1500.0
K_GRID = (16, 20, 24, 28, 32)
ELO_FEATURES = ["elo_home", "elo_away", "elo_diff"]


def expected_score(rating, opponent):
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def game_pairs(team_rows):
    """One row per game, in date order, holding both franchises and the first listed team's result."""
    rows = team_rows.sort_values(["Date", "GAME_ID", "is_home"], ascending=[True, True, False], kind="stable")
    sizes = rows.groupby("GAME_ID").size()
    if (sizes != 2).any():
        raise ValueError("Every game needs exactly two team rows")
    first = rows.groupby("GAME_ID", sort=False).nth(0).set_index("GAME_ID")
    second = rows.groupby("GAME_ID", sort=False).nth(1).set_index("GAME_ID")
    winners = first["WIN"] + second["WIN"]
    if (winners > 1).any():
        raise ValueError("A game cannot have two winners")
    return pd.DataFrame({"team_a": first["TEAM_ID"], "team_b": second["TEAM_ID"], "a_won": first["WIN"],
                         "a_home": first["is_home"], "neutral": first["neutral"], "played": winners == 1})


def pregame_elo(team_rows, k):
    """Each sited game's ratings before tip-off, indexed by GAME_ID."""
    ratings = {}
    records = []
    for game_id, game in game_pairs(team_rows).iterrows():
        rating_a = ratings.get(game["team_a"], START_RATING)
        rating_b = ratings.get(game["team_b"], START_RATING)
        if game["a_home"] and not game["neutral"]:
            records.append((game_id, rating_a, rating_b, rating_a - rating_b))
        if not game["played"]:  # logged with no result: 2013-04-16 BOS-IND, listed but never played
            continue
        change = k * (game["a_won"] - expected_score(rating_a, rating_b))
        ratings[game["team_a"]] = rating_a + change
        ratings[game["team_b"]] = rating_b - change
    return pd.DataFrame(records, columns=["GAME_ID"] + ELO_FEATURES).set_index("GAME_ID")


def walk_forward_elo_log_loss(elo_diff, y, n_splits=5):
    """Mean log loss of home win on elo_diff under walk-forward CV. Pass train-split rows only."""
    X = np.asarray(elo_diff, dtype=float).reshape(-1, 1) / 400.0
    y = np.asarray(y)
    losses = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = LogisticRegression(C=1e6).fit(X[train_idx], y[train_idx])
        losses.append(log_loss(y[val_idx], model.predict_proba(X[val_idx])[:, 1], labels=[0, 1]))
    return float(np.mean(losses))


def choose_k(team_rows, train_game_ids, train_y, grid=K_GRID, n_splits=5):
    """Pick k on train games only. train_game_ids and train_y must be in date order."""
    losses = {}
    for k in grid:
        elo = pregame_elo(team_rows, k)
        losses[k] = walk_forward_elo_log_loss(elo.loc[train_game_ids, "elo_diff"], train_y, n_splits)
    best = min(grid, key=lambda k: (losses[k], k))
    return best, losses
