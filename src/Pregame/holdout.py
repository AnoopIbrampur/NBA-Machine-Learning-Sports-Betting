"""One look at the games after the split, under the rules in docs/enhanced-plan-2.md.

Uses saved models and saved settings. With --check it reports counts only and scores nothing.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.Pregame.elo import ELO_FEATURES, pregame_elo
from src.Pregame.enhanced import predict_saved
from src.Pregame.enhanced2 import ELO2_FEATURES, PRIMARY
from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES, paired_bootstrap
from src.Pregame.paths import ENHANCED2_RESULTS_DIR, features_path
from src.Pregame.player_features import PLAYER_FEATURES, build_player_features
from src.Pregame.player_logs import load_player_logs
from src.Pregame.rolling import rolling_predictions
from src.Pregame.schedule_experiments import fmt_ci
from src.Pregame.schedule_features import BUBBLE_START, offsite_game_ids
from src.Pregame.train_xgb import TARGET, evaluate, load_games

SPLIT_END = "2026-01-07"
ROLL_FROM = "2025-12-01"  # the month before the holdout, so the next refit is 2026-01-01


def holdout_games(features, logs, split_ids):
    """Sited games after the split that have a result, labelled from the logs and in date order."""
    home = logs[logs["is_home"] & ~logs["neutral"]].set_index("GAME_ID")
    away_wl = logs[~logs["is_home"]].drop_duplicates("GAME_ID").set_index("GAME_ID")["WL"]
    out = features[(features["Date"] > SPLIT_END) & ~features["GAME_ID"].isin(set(split_ids))].copy()
    out = out[out["GAME_ID"].map(home["WL"]).notna() & out["GAME_ID"].map(away_wl).notna()]
    out[TARGET] = out["GAME_ID"].map(home["WIN"]).astype(int)
    return out.sort_values(["Date", "GAME_ID"], kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report counts only; score nothing.")
    args = parser.parse_args()

    training = json.loads((ENHANCED2_RESULTS_DIR / "training.json").read_text())
    variant = training["elo_variant"]
    games, columns = load_games(20, schedule=True)
    logs = team_games(load_game_logs())
    bubble = logs.loc[(logs["SEASON"] == "2019-20") & (logs["Date"] >= BUBBLE_START), "GAME_ID"]
    no_home = frozenset(bubble) | frozenset(offsite_game_ids(logs))
    elo = pregame_elo(logs, variant["k"], variant["home_advantage"], variant["regression"], variant["margin"],
                      no_home).rename(columns=dict(zip(ELO_FEATURES, ELO2_FEATURES)))
    player = build_player_features(load_player_logs(), logs, training["replacement_level"])

    primary_columns = columns["pregame"] + ELO2_FEATURES + PLAYER_FEATURES
    add = lambda frame: frame.merge(elo, left_on="GAME_ID", right_index=True, how="left", validate="one_to_one").merge(  # noqa: E731
        player, left_on="GAME_ID", right_index=True, how="left", validate="one_to_one")
    features = pd.read_csv(features_path(20), dtype={"GAME_ID": str})
    fresh = add(holdout_games(features, logs, games["GAME_ID"]))
    if args.check:
        months = pd.to_datetime(fresh["Date"]).dt.strftime("%Y-%m").value_counts().sort_index()
        print(f"Holdout games: {len(fresh)}, {fresh['Date'].min()} to {fresh['Date'].max()}")
        print(months.to_string())
        print("Largest missing share:", round(float(fresh[primary_columns].isna().mean().max()), 4))
        print("Overlap with split:", int(fresh["GAME_ID"].isin(set(games["GAME_ID"])).sum()))
        return

    y = fresh[TARGET].to_numpy()
    probas = {"pregame": predict_saved(fresh, columns["pregame"], "pregame"),
              PRIMARY: predict_saved(fresh, primary_columns, PRIMARY)}
    past = add(games[["GAME_ID", "Date", TARGET] + columns["pregame"]])
    both = pd.concat([past, fresh], ignore_index=True).sort_values("Date", kind="stable").reset_index(drop=True)
    all_y = both[TARGET].astype(int).to_numpy()
    rolled = rolling_predictions(both, primary_columns, all_y, training[PRIMARY], np.full(len(both), np.nan),
                                 (both["Date"] >= ROLL_FROM).to_numpy())
    probas["roll_" + PRIMARY] = pd.Series(rolled, index=both["GAME_ID"]).loc[fresh["GAME_ID"]].to_numpy()

    labels = {"pregame": "Pre-game w20 (official)", PRIMARY: "Pre-game + Elo variant + player",
              "roll_" + PRIMARY: "Rolling primary"}
    pairs = [(PRIMARY, "pregame"), ("roll_" + PRIMARY, "pregame"), ("roll_" + PRIMARY, PRIMARY)]
    results = {"n": len(y), "dates": [fresh["Date"].min(), fresh["Date"].max()], "home_win_rate": float(y.mean()),
               "metrics": {row: evaluate(y, p) for row, p in probas.items()},
               "comparisons": {f"{a} vs {b}": paired_bootstrap(y, probas[a], probas[b]) for a, b in pairs}}
    (ENHANCED2_RESULTS_DIR / "holdout.json").write_text(json.dumps(results, indent=2, default=float))
    lines = [f"| {labels[r]} | {m['accuracy']:.4f} | {m['auc']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} |"
             for r, m in results["metrics"].items()]
    ci = [f"| {labels[a]} | {labels[b]} | {fmt_ci(results['comparisons'][f'{a} vs {b}']['accuracy_diff_ci'])} | "
          f"{fmt_ci(results['comparisons'][f'{a} vs {b}']['auc_diff_ci'])} |" for a, b in pairs]
    text = "\n".join([
        "# Fresh holdout: games after the split", "",
        f"{len(y)} games, {results['dates'][0]} to {results['dates'][1]}. Home-win rate {y.mean():.4f}. Rules fixed in "
        "`docs/enhanced-plan-2.md` before scoring. No baseline or market: their data stops at 2026-01-07.", "",
        "| Row | Accuracy | AUC | Log loss | Brier |", "|---|---|---|---|---|", *lines, "",
        f"## Paired bootstrap 95% CIs (row − reference), {BOOTSTRAP_SAMPLES} resamples", "",
        "| Row | Reference | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|", *ci]) + "\n"
    (ENHANCED2_RESULTS_DIR / "holdout.md").write_text(text)
    print("holdout written")


if __name__ == "__main__":
    main()
