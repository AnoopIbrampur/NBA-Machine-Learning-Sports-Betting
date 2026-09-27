"""Audit the repo's season-to-date features for same-game leakage, and re-score saved models.

For each team's consecutive dataset rows, if the features exclude the game being predicted then
W(next row) - W(this row) equals this game's result; if they include it, W(this row) - W(previous
row) does. The 2024-25 and 2025-26 rows used to include the current game (a lost one-day offset
in Get_Data.py) and were regenerated with the fix by scripts/regenerate_2024_26.py.

The saved models are re-scored on test games from seasons that were never affected (a secondary
check that does not depend on the fix) and on the regenerated seasons separately. Training data
(2012-13 to 2021-22) was never affected, so nothing is retrained.
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.Pregame.paths import DATASET_DB, DATASET_TABLE, RESULTS_DIR, model_path
from src.Pregame.train_xgb import TARGET, evaluate, load_games

FEATURE_SETS = ["repo", "pregame", "pregame_margin"]
REGENERATED_SEASONS = [2024, 2025]  # season start years rebuilt after the Get_Data fix
BOOTSTRAP_SAMPLES = 2000
LABELS = {"repo": "Repo season-to-date (baseline)", "pregame": "Pre-game + context",
          "pregame_margin": "Ablation: pre-game + context + margin"}


def season_start(dates):
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= 8, dates.dt.year - 1)


def leakage_by_season():
    with sqlite3.connect(DATASET_DB) as con:
        data = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    sides = []
    for suffix, won in [("", data[TARGET]), (".1", 1 - data[TARGET])]:
        side = pd.DataFrame({"Date": pd.to_datetime(data["Date"]), "team": data[f"TEAM_NAME{suffix}"],
                             "GP": data[f"GP{suffix}"], "W": data[f"W{suffix}"], "won": won})
        sides.append(side)
    rows = pd.concat(sides).sort_values(["team", "Date"])
    rows["season"] = season_start(rows["Date"])
    by_team = rows.groupby(["team", "season"])
    nxt = rows.assign(nGP=by_team["GP"].shift(-1), nW=by_team["W"].shift(-1)).query("nGP == GP + 1")
    prv = rows.assign(pGP=by_team["GP"].shift(1), pW=by_team["W"].shift(1)).query("GP == pGP + 1")
    return pd.DataFrame({
        "excludes_current_game": ((nxt["nW"] - nxt["W"]) == nxt["won"]).groupby(nxt["season"]).mean(),
        "includes_current_game": ((prv["W"] - prv["pW"]) == prv["won"]).groupby(prv["season"]).mean(),
    })


def paired_bootstrap(y, proba, baseline_proba, seed=0):
    """95% CIs for (model - baseline) accuracy and AUC, resampling the same games for both."""
    rng = np.random.default_rng(seed)
    acc_diffs, auc_diffs = [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        yb, pb, bb = y[idx], proba[idx], baseline_proba[idx]
        acc_diffs.append(np.mean((pb >= 0.5) == yb) - np.mean((bb >= 0.5) == yb))
        auc_diffs.append(roc_auc_score(yb, pb) - roc_auc_score(yb, bb))
    ci = lambda d: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]  # noqa: E731
    return {"accuracy_diff_ci": ci(acc_diffs), "auc_diff_ci": ci(auc_diffs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 20])
    args = parser.parse_args()

    by_season = leakage_by_season()
    leaky_seasons = sorted(int(s) for s in by_season.index[by_season["includes_current_game"] > 0.9])
    print(by_season.round(3).to_string())
    print(f"Seasons whose features include the current game: {leaky_seasons}")

    results, lines, ci_lines = {}, [], []
    for window in args.windows:
        games, feature_columns = load_games(window)
        regenerated = season_start(games["Date"]).isin(REGENERATED_SEASONS)
        subsets = {
            "test, never-affected seasons": (games["split"] == "test") & ~regenerated,
            "test, regenerated seasons": (games["split"] == "test") & regenerated,
            "validation (all regenerated)": (games["split"] == "validation") & regenerated,
        }
        if ((games["split"] == "validation") & ~regenerated).any():
            raise ValueError("Validation has rows outside the regenerated seasons; update the subsets.")
        y = games[TARGET].astype(int).to_numpy()
        ci_subsets = {
            "test, never-affected seasons": subsets["test, never-affected seasons"].to_numpy(),
            "test (all)": (games["split"] == "test").to_numpy(),
            "validation (all)": (games["split"] == "validation").to_numpy(),
        }
        probas = {}
        for feature_set in FEATURE_SETS:
            booster = xgb.Booster()
            booster.load_model(str(model_path(feature_set, window)))
            X = games[feature_columns[feature_set]].astype(float)
            probas[feature_set] = booster.predict(xgb.DMatrix(X))
            for subset, mask in subsets.items():
                mask = mask.to_numpy()
                m = evaluate(y[mask], probas[feature_set][mask])
                results.setdefault(f"w{window}", {}).setdefault(feature_set, {})[subset] = m
                lines.append(f"| {window} | {LABELS[feature_set]} | {subset} | {m['n']} | {m['accuracy']:.4f} | "
                             f"{m['auc']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} |")
            if feature_set == "repo":
                continue
            for subset, mask in ci_subsets.items():
                cis = paired_bootstrap(y[mask], probas[feature_set][mask], probas["repo"][mask])
                results[f"w{window}"][feature_set].setdefault("vs_baseline", {})[subset] = cis
                ci_lines.append(f"| {window} | {LABELS[feature_set]} | {subset} | "
                                f"[{cis['accuracy_diff_ci'][0]:+.4f}, {cis['accuracy_diff_ci'][1]:+.4f}] | "
                                f"[{cis['auc_diff_ci'][0]:+.4f}, {cis['auc_diff_ci'][1]:+.4f}] |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "leakage_audit.json").write_text(json.dumps(
        {"by_season": by_season.to_dict(orient="index"), "leaky_seasons": leaky_seasons, "metrics": results},
        indent=2, default=float))
    report = [
        "# Leakage audit of the repo's season-to-date features", "",
        "Share of team rows consistent with features that exclude / include the game being predicted:", "",
        by_season.round(3).to_markdown(), "",
        "Seasons whose features include the current game: "
        + (", ".join(f"{s}-{str(s + 1)[-2:]}" for s in leaky_seasons) or "none") + ".",
        "Only the baseline uses these features; the pre-game models are built from game logs.", "",
        "## Saved models re-scored on never-affected vs regenerated seasons", "",
        "| Window | Model | Subset | n | Accuracy | AUC | Log loss | Brier |", "|---|---|---|---|---|---|---|---|",
        *lines, "",
        "## Difference vs baseline (paired bootstrap, 95% CI)", "",
        f"{BOOTSTRAP_SAMPLES} resamples of the same games for both models. An interval containing 0 means "
        "the difference is not distinguishable from noise.", "",
        "| Window | Model | Games | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|---|",
        *ci_lines,
    ]
    text = "\n".join(report) + "\n"
    (RESULTS_DIR / "leakage_audit.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
