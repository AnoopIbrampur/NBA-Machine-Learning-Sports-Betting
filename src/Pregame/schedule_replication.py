"""Replicate Bowman, Harmon & Ashman (JSA 2023) on our training-period seasons, plus raw b2b tables.

Logistic regression of home win on Bowman's schedule variables, controlling for team strength.
The primary control is team-by-season dummies for the home and the visiting team, as in the
paper; a second fit controls for both teams' season-to-date win % instead (from the repo dataset,
pre-game). Training-split games only. Games without a real home court are excluded: the 2019-20
restart and the listed off-site games (neutral-site games are never in the split).
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.paths import SCHEDULE_RESULTS_DIR
from src.Pregame.schedule_features import BUBBLE_START, offsite_game_ids
from src.Pregame.train_xgb import TARGET, load_games

# Bowman et al. (2023), odds multipliers for a home win (> 1 favours the home team).
BOWMAN = {
    "away_b2b": ("Visitor on a back-to-back", 1.506),
    "home_b2b": ("Home on a back-to-back", 0.806),
    "home_east_b2b": ("Home crossed 1+ time zone west-to-east from a game one day ago", 0.693),
    "visitor_long_trip": ("Visitor travelled >1000 mi on a b2b or >2000 mi otherwise", 1.261),
    "away_3plus_week": ("Visitor played 3+ games in the last week", 1.153),
    "home_4plus_week": ("Home played 4+ games in the last week", 0.914),
    "away_rest3plus": ("Visitor rested 3+ days (vs 2)", None),
    "home_rest3plus": ("Home rested 3+ days (vs 2)", None),
}
SITUATIONS = ["home rested / away b2b", "both rested", "both b2b", "home b2b / away rested"]


def schedule_variables(games):
    return pd.DataFrame({
        "away_b2b": games["b2b_away"] == 1,
        "home_b2b": games["b2b_home"] == 1,
        "home_east_b2b": games["home_east_b2b"] == 1,
        "visitor_long_trip": games["visitor_long_trip"] == 1,
        "away_3plus_week": games["games_last7_away"] >= 3,
        "home_4plus_week": games["games_last7_home"] >= 4,
        "away_rest3plus": games["rest_bucket_away"] == 3,
        "home_rest3plus": games["rest_bucket_home"] == 3,
    }, index=games.index).astype(float)


def team_season_dummies(games):
    """Home and visitor team-by-season dummies, minus the columns that are linear combinations of
    the rest and the intercept (one home dummy per season, one visitor dummy overall)."""
    home = pd.get_dummies(games["TEAM_NAME"] + " " + games["SEASON"], prefix="H", dtype=float)
    away = pd.get_dummies(games["TEAM_NAME.1"] + " " + games["SEASON"], prefix="A", dtype=float)
    first_per_season = [c for c in home.columns.to_series().groupby(lambda c: c[-7:]).first()]
    return pd.concat([home.drop(columns=first_per_season), away.drop(columns=away.columns[0])], axis=1)


def fit(y, X):
    X = sm.add_constant(X)
    rank = np.linalg.matrix_rank(X.to_numpy())
    model = sm.Logit(y, X).fit(method="newton", maxiter=200, disp=False)
    stable = bool(model.mle_retvals["converged"]) and rank == X.shape[1] \
        and np.all(np.isfinite(model.bse)) and model.bse.max() < 10
    return model, {"converged": bool(model.mle_retvals["converged"]), "rank": int(rank), "n_columns": X.shape[1],
                   "max_se": float(model.bse.max()), "stable": stable}


def odds_table(model, variables):
    ci = model.conf_int()
    return pd.DataFrame({
        "odds_multiplier": np.exp(model.params[variables]),
        "ci_low": np.exp(ci.loc[variables, 0]),
        "ci_high": np.exp(ci.loc[variables, 1]),
        "p_value": model.pvalues[variables],
    })


def situation(games):
    home, away = games["b2b_home"].fillna(0) == 1, games["b2b_away"].fillna(0) == 1
    return pd.Series(np.select([~home & away, ~home & ~away, home & away, home & ~away], SITUATIONS, default=""),
                     index=games.index)


def raw_tables(games):
    games = games.assign(situation=situation(games), home_win=games[TARGET].astype(float))
    by_split = []
    for split, part in [("all", games)] + list(games.groupby("split")):
        stats = part.groupby("situation")["home_win"].agg(["size", "mean"]).reindex(SITUATIONS)
        by_split.append(stats.rename(columns={"size": "n", "mean": "home_win_rate"}).assign(split=split))
    by_split = pd.concat(by_split).reset_index().set_index(["split", "situation"])

    season_b2b = games.groupby("SEASON").agg(
        games=("home_win", "size"),
        home_b2b_share=("b2b_home", lambda s: (s == 1).mean()),
        away_b2b_share=("b2b_away", lambda s: (s == 1).mean()),
        any_b2b_share=("situation", lambda s: (s != "both rested").mean()),
    )
    team_b2b = (games["b2b_home"] == 1).groupby(games["SEASON"]).sum() + \
        (games["b2b_away"] == 1).groupby(games["SEASON"]).sum()
    season_b2b["b2b_team_games_per_team"] = team_b2b / 30
    return by_split, season_b2b


def main():
    games, _ = load_games(20, schedule=True)
    logs_games = team_games(load_game_logs())
    offsite = offsite_game_ids(logs_games)
    bubble = (games["SEASON"] == "2019-20") & (pd.to_datetime(games["Date"]) >= BUBBLE_START)
    train = games[(games["split"] == "train") & ~bubble & ~games["GAME_ID"].isin(offsite)].copy()
    X = schedule_variables(train)
    complete = train[["rest_bucket_home", "rest_bucket_away", "travel_km_away", "tz_shift_home"]].notna().all(axis=1)
    train, X = train[complete], X[complete]
    y = train[TARGET].astype(int)
    variables = list(BOWMAN)

    fits, tables = {}, {}
    dummies = team_season_dummies(train)
    model, diag = fit(y, pd.concat([X, dummies], axis=1))
    fits["team_season_dummies"], tables["team_season_dummies"] = diag, odds_table(model, variables)
    strength = train[["W_PCT", "W_PCT.1"]].astype(float).rename(columns={"W_PCT": "home_win_pct",
                                                                          "W_PCT.1": "away_win_pct"})
    model, diag = fit(y, pd.concat([X, strength], axis=1))
    fits["season_to_date_win_pct"], tables["season_to_date_win_pct"] = diag, odds_table(model, variables)
    primary = "team_season_dummies" if fits["team_season_dummies"]["stable"] else "season_to_date_win_pct"

    by_split, season_b2b = raw_tables(games)
    SCHEDULE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEDULE_RESULTS_DIR / "replication.json").write_text(json.dumps({
        "n_games": int(len(train)), "excluded_bubble": int(((games["split"] == "train") & bubble).sum()),
        "excluded_offsite": int(((games["split"] == "train") & games["GAME_ID"].isin(offsite)).sum()),
        "excluded_missing": int((~complete).sum()), "primary_control": primary, "fits": fits,
        "odds": {k: t.to_dict(orient="index") for k, t in tables.items()},
    }, indent=2, default=float))
    by_split.to_csv(SCHEDULE_RESULTS_DIR / "home_win_by_b2b_situation.csv")
    season_b2b.to_csv(SCHEDULE_RESULTS_DIR / "b2b_frequency_by_season.csv")

    rows = []
    for var, (label, bowman) in BOWMAN.items():
        cells = [f"{t.loc[var, 'odds_multiplier']:.3f} [{t.loc[var, 'ci_low']:.3f}, {t.loc[var, 'ci_high']:.3f}]"
                 f" (p={t.loc[var, 'p_value']:.3f})" for t in tables.values()]
        rows.append(f"| {label} | {int(X[var].sum())} | {bowman or 'not different from 2 days'} | "
                    f"{cells[0]} | {cells[1]} |")
    report = [
        "# Replicating Bowman, Harmon & Ashman (2023) on our training seasons", "",
        f"Training-split games 2012-11 → 2022-01, n = {len(train):,}. Excluded: "
        f"{int(((games['split'] == 'train') & bubble).sum())} 2019-20 restart games (one site, no home court), "
        f"{int(((games['split'] == 'train') & games['GAME_ID'].isin(offsite)).sum())} listed off-site games, and "
        f"{int((~complete).sum())} games with a missing rest or travel value (season openers, the game after an "
        "off-site game, the first restart game).", "",
        f"Primary control: **{primary.replace('_', ' ')}**. Fit diagnostics: "
        + "; ".join(f"{k}: converged={d['converged']}, rank {d['rank']}/{d['n_columns']}, max SE {d['max_se']:.2f}"
                    for k, d in fits.items()) + ".", "",
        "Odds multipliers for a home win, 95% Wald CIs (> 1 favours the home team). Reference rest = 2 days. "
        "'Games in the last week' counts the team's games in the 7 days before game day.", "",
        "| Variable | Games with it | Bowman et al. | Ours: team-by-season dummies | "
        "Ours: season-to-date win % control |",
        "|---|---|---|---|---|", *rows, "",
        "## Raw home-win rate by back-to-back situation", "",
        by_split.round(4).to_markdown(), "",
        "## Back-to-back frequency by season (split games)", "",
        season_b2b.round(4).to_markdown(),
    ]
    text = "\n".join(report) + "\n"
    (SCHEDULE_RESULTS_DIR / "replication.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
