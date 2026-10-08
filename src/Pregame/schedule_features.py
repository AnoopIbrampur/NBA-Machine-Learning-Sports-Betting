"""Schedule-fatigue and travel features, after Bowman, Harmon & Ashman (JSA 2023) and Leota et al. (2022).

Per team per game, within season, using only the dates and venues of that team's earlier games
plus the published date of its next regular-season game (for the first leg of a back-to-back).
No result or box-score column is read. These features do not depend on the rolling window.

Venues come from Data/arenas.csv (home arena by date, including Toronto's 2020-21 season in
Tampa). Games whose site is unknown from the logs get no venue, so travel is missing for that game
and for each team's next game:
  - neutral-site games the logs mark with "@" on both sides (features.team_games flags these);
  - games the logs list as ordinary home games but that were played elsewhere
    (Data/offsite_games.csv: international games and the 2023 In-Season Tournament semifinals).
Games of the 2019-20 restart (from 2020-07-30) were all played at one site near Orlando, so
travel between them is zero; travel into the first restart game is left missing.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.paths import ARENAS_CSV, OFFSITE_GAMES_CSV, schedule_features_path

# 2019-20 restart: ESPN Wide World of Sports, Lake Buena Vista, Florida.
BUBBLE = {"latitude": 28.3378, "longitude": -81.5560, "tz": "America/New_York"}
BUBBLE_SEASON, BUBBLE_START = "2019-20", pd.Timestamp("2020-07-30")
LOCAL_TIP = time(19, 0)  # offsets are read at a typical tip-off time on the game date
MILES_1000_KM, MILES_2000_KM = 1609.344, 3218.688  # Bowman's visitor long-trip thresholds

FATIGUE_SIDE = ["rest_bucket", "games_last7", "three_in_four", "consec_away", "b2b_prev_away",
                "first_leg_b2b"]
TRAVEL_SIDE = ["travel_km", "tz_shift", "jet_lag"]
FATIGUE_FEATURES = [f"{f}_{s}" for f in FATIGUE_SIDE for s in ("home", "away")] + ["rest_diff"]
TRAVEL_FEATURES = ([f"{f}_{s}" for f in TRAVEL_SIDE for s in ("home", "away")]
                   + ["visitor_long_trip", "home_east_b2b"])
SCHEDULE_FEATURES = FATIGUE_FEATURES + TRAVEL_FEATURES


def load_arenas(path=ARENAS_CSV):
    arenas = pd.read_csv(path)
    arenas["valid_from"] = pd.to_datetime(arenas["valid_from"]).fillna(pd.Timestamp.min)
    arenas["valid_to"] = pd.to_datetime(arenas["valid_to"]).fillna(pd.Timestamp.max)
    return arenas


def tz_offset_hours(tz, date):
    """UTC offset of `tz` at tip-off on `date`, in hours (daylight saving applied for that date)."""
    local = datetime.combine(pd.Timestamp(date).date(), LOCAL_TIP, tzinfo=ZoneInfo(tz))
    return local.utcoffset().total_seconds() / 3600


def tz_shift_hours(from_tz, to_tz, date):
    """Clock change on arrival, east positive. Both zones are read on the same date, so a daylight
    saving switch between two games in one city is not a shift."""
    return tz_offset_hours(to_tz, date) - tz_offset_hours(from_tz, date)


def net_jet_lag(shift, days_since_last):
    """Leota et al.: zones crossed minus one hour of recovery per day since the previous game."""
    shift, days = np.asarray(shift, dtype=float), np.asarray(days_since_last, dtype=float)
    return np.sign(shift) * np.maximum(np.abs(shift) - days, 0) + 0.0  # + 0.0 turns -0.0 into 0.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def offsite_game_ids(games, path=OFFSITE_GAMES_CSV):
    """GAME_IDs of the listed off-site games; every listed game must match exactly one logged game."""
    listed = pd.read_csv(path)
    ids = []
    for row in listed.itertuples():
        on_date = games[games["GAME_DATE"] == row.GAME_DATE]
        teams = on_date.groupby("GAME_ID")["TEAM_ABBREVIATION"].agg(set)
        match = teams[teams.apply(lambda t: t == {row.TEAM_A, row.TEAM_B})].index
        if len(match) != 1:
            raise ValueError(f"Off-site game {row.GAME_DATE} {row.TEAM_A}-{row.TEAM_B} matched {len(match)} games")
        ids.append(match[0])
    return set(ids)


def assign_venues(games, arenas, offsite_ids):
    """Latitude, longitude and time zone of each team-game's venue (NaN when unknown)."""
    out = games.copy()
    out["bubble"] = (out["SEASON"] == BUBBLE_SEASON) & (out["Date"] >= BUBBLE_START)
    out["offsite"] = out["neutral"] | out["GAME_ID"].isin(offsite_ids)

    home = out.loc[out["is_home"] & ~out["offsite"] & ~out["bubble"], ["GAME_ID", "TEAM_ID", "Date"]]
    candidates = home.merge(arenas, on="TEAM_ID", how="left")
    in_range = (candidates["Date"] >= candidates["valid_from"]) & (candidates["Date"] <= candidates["valid_to"])
    sites = candidates[in_range]
    if sites["GAME_ID"].duplicated().any() or len(sites) != len(home):
        raise ValueError("Arena validity ranges must give exactly one arena per home game")
    sites = sites.set_index("GAME_ID")[["latitude", "longitude", "tz"]]

    out = out.join(sites, on="GAME_ID")
    for column, value in BUBBLE.items():
        out.loc[out["bubble"], column] = value
    out["at_home"] = out["is_home"] & ~out["offsite"] & ~out["bubble"]
    return out


def prior_count(dates, lookback_days):
    """Number of the team's earlier games within `lookback_days` days before each game."""
    values = dates.to_numpy()
    starts = np.searchsorted(values, values - np.timedelta64(lookback_days, "D"), side="left")
    return pd.Series(np.arange(len(values)) - starts, index=dates.index, dtype=float)


def add_schedule_features(games, arenas, offsite_ids):
    out = assign_venues(games, arenas, offsite_ids).sort_values(["TEAM_ID", "SEASON", "Date"])
    by_team = out.groupby(["TEAM_ID", "SEASON"], sort=False)
    prev = by_team[["Date", "latitude", "longitude", "tz", "at_home", "bubble"]].shift(1)

    # Fatigue.
    out["days_since_last"] = (out["Date"] - prev["Date"]).dt.days
    out["rest_bucket"] = out["days_since_last"].clip(upper=3)
    out["games_last7"] = by_team["Date"].transform(lambda d: prior_count(d, 7))
    out["three_in_four"] = (by_team["Date"].transform(lambda d: prior_count(d, 3)) >= 2).astype(float)
    away = (~out["at_home"]).astype(int)
    stint = by_team["at_home"].cumsum()  # a new stint starts at each home game
    streak = away.groupby([out["TEAM_ID"], out["SEASON"], stint]).cumsum()
    out["consec_away"] = streak.groupby([out["TEAM_ID"], out["SEASON"]]).shift(1).fillna(0)
    is_b2b = out["days_since_last"] == 1
    out["b2b_prev_away"] = np.where(is_b2b, (~prev["at_home"].astype(bool)).astype(float), np.nan)
    # First leg of a back-to-back: only the date of the next regular-season game is read.
    # Postseason dates depend on results, and no postseason game in the data is a back-to-back.
    regular = out[out["SEASON_TYPE"] == "Regular Season"]
    next_date = regular.groupby(["TEAM_ID", "SEASON"], sort=False)["Date"].shift(-1)
    out["first_leg_b2b"] = 0.0
    out.loc[regular.index, "first_leg_b2b"] = ((next_date - regular["Date"]).dt.days == 1).astype(float)

    # Travel and time zones.
    out["travel_km"] = haversine_km(prev["latitude"], prev["longitude"], out["latitude"], out["longitude"])
    out.loc[out["bubble"] & ~prev["bubble"].astype(bool), "travel_km"] = np.nan  # arrival at the restart
    known = out["tz"].notna() & prev["tz"].notna() & out["travel_km"].notna()
    out["tz_shift"] = np.nan
    out.loc[known, "tz_shift"] = [tz_shift_hours(a, b, d) for a, b, d in
                                  zip(prev.loc[known, "tz"], out.loc[known, "tz"], out.loc[known, "Date"])]
    out["jet_lag"] = net_jet_lag(out["tz_shift"], out["days_since_last"])
    return out.sort_index()


def build_schedule_features(logs, arenas=None, offsite_ids=None):
    games = team_games(logs)
    arenas = load_arenas() if arenas is None else arenas
    offsite_ids = offsite_game_ids(games) if offsite_ids is None else offsite_ids
    teams = add_schedule_features(games, arenas, offsite_ids)

    keep = ["GAME_ID", "Date", "TEAM_NAME", "SEASON", "days_since_last"] + FATIGUE_SIDE + TRAVEL_SIDE
    sited = teams[~teams["neutral"]]
    home = sited.loc[sited["is_home"], keep]
    away = sited.loc[~sited["is_home"], keep].drop(columns=["Date", "SEASON"])
    g = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="one_to_one")

    features = pd.DataFrame({
        "GAME_ID": g["GAME_ID"], "SEASON": g["SEASON"], "Date": g["Date"].dt.strftime("%Y-%m-%d"),
        "TEAM_NAME": g["TEAM_NAME_home"], "TEAM_NAME.1": g["TEAM_NAME_away"],
    })
    for column in FATIGUE_SIDE + TRAVEL_SIDE:
        for side in ("home", "away"):
            features[f"{column}_{side}"] = g[f"{column}_{side}"]
    features["rest_diff"] = g["rest_bucket_home"] - g["rest_bucket_away"]

    away_b2b = g["days_since_last_away"] == 1
    long_trip = np.where(away_b2b, g["travel_km_away"] > MILES_1000_KM, g["travel_km_away"] > MILES_2000_KM)
    features["visitor_long_trip"] = np.where(
        g["travel_km_away"].notna() & g["days_since_last_away"].notna(), long_trip, np.nan)
    home_east = (g["days_since_last_home"] == 1) & (g["tz_shift_home"] >= 1)
    features["home_east_b2b"] = np.where(g["tz_shift_home"].notna(), home_east, np.nan)
    return features[["GAME_ID", "SEASON", "Date", "TEAM_NAME", "TEAM_NAME.1"] + SCHEDULE_FEATURES] \
        .sort_values(["Date", "GAME_ID"]).reset_index(drop=True)


def main():
    features = build_schedule_features(load_game_logs())
    path = schedule_features_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    print(f"Wrote {len(features)} games x {features.shape[1]} columns to {path}")
    print("Missing share:\n" + features[SCHEDULE_FEATURES].isna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
