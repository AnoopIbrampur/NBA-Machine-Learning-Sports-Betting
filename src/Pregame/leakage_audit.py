"""Audit the repo's season-to-date features for same-game leakage, and re-score saved models.

For each team's consecutive dataset rows, if the features exclude the game being predicted then
W(next row) - W(this row) equals this game's result; if they include it, W(this row) - W(previous
row) does. From 2024-25 on, dataset_2012-26 rows include the current game, so the baseline sees the
outcome it predicts. This re-scores the models saved by train_xgb.py on leak-free and affected
games separately; training data (2012-13 to 2021-22) is unaffected, so nothing is retrained.
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
        leaky = season_start(games["Date"]).isin(leaky_seasons)
        subsets = {
            "test, leak-free seasons": (games["split"] == "test") & ~leaky,
            "test, affected seasons": (games["split"] == "test") & leaky,
            "validation (all affected)": (games["split"] == "validation") & leaky,
        }
        if ((games["split"] == "validation") & ~leaky).any():
            raise ValueError("Validation has unaffected rows; update the subset definitions.")
        y = games[TARGET].astype(int).to_numpy()
        clean_test = subsets["test, leak-free seasons"].to_numpy()
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
            if feature_set != "repo":
                cis = paired_bootstrap(y[clean_test], probas[feature_set][clean_test], probas["repo"][clean_test])
                results[f"w{window}"][feature_set]["vs_baseline_leak_free_test"] = cis
                ci_lines.append(f"| {window} | {LABELS[feature_set]} | "
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
        f"Affected seasons (features include the current game): {', '.join(f'{s}-{str(s + 1)[-2:]}' for s in leaky_seasons)}.",
        "Only the baseline uses these features; the pre-game models are built from game logs.", "",
        "## Saved models re-scored on leak-free vs affected games", "",
        "| Window | Model | Subset | n | Accuracy | AUC | Log loss | Brier |", "|---|---|---|---|---|---|---|---|",
        *lines, "",
        "## Difference vs baseline on leak-free test games (paired bootstrap, 95% CI)", "",
        f"{BOOTSTRAP_SAMPLES} resamples of the same games for both models. An interval containing 0 means "
        "the difference is not distinguishable from noise.", "",
        "| Window | Model | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|",
        *ci_lines,
    ]
    text = "\n".join(report) + "\n"
    (RESULTS_DIR / "leakage_audit.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
