"""Train XGBoost on the repo's season-to-date features and on the rolling pre-game features.

Both feature sets use the same games, the same fixed train/test/validation split, and the same
protocol: random search scored by walk-forward CV on train only, then a final model refit on all
of train. Test and validation are only scored, never used for tuning, early stopping or
calibration (validation is chronologically after test, so using it for either would leak).
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

from src.Pregame.features import ABLATION_FEATURES
from src.Pregame.paths import (DATASET_DB, DATASET_TABLE, GAME_KEY, RESULTS_DIR, SPLIT_KEYS,
                               features_path, model_path)

TARGET = "Home-Team-Win"
# Same columns XGBoost_Model_ML.py drops: identifiers, the outcome, and betting-market fields.
REPO_DROP_COLUMNS = ["index", "Score", TARGET, "TEAM_NAME", "Date", "index.1", "TEAM_NAME.1",
                     "Date.1", "OU-Cover", "OU"]
PREGAME_ID_COLUMNS = ["GAME_ID", "SEASON"] + GAME_KEY
EVAL_SPLITS = ["test", "validation"]

# Parent paper (PLOS ONE 2024, Tables 7-9): XGBoost on in-game box scores, 10-fold CV.
PAPER_XGBOOST = {
    "H2 (first two quarters)": {"accuracy": 0.720, "precision": 0.736, "recall": 0.775, "f1": 0.754, "auc": 0.783},
    "H3 (first three quarters)": {"accuracy": 0.798, "precision": 0.834, "recall": 0.807, "f1": 0.820, "auc": 0.876},
    "Full game": {"accuracy": 0.933, "precision": 0.938, "recall": 0.939, "f1": 0.939, "auc": 0.982},
}


def load_games(window):
    """Repo dataset rows that also have pre-game features, labelled with their split."""
    with sqlite3.connect(DATASET_DB) as con:
        repo = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    splits = pd.read_csv(SPLIT_KEYS)
    pregame = pd.read_csv(features_path(window), dtype={"GAME_ID": str})

    repo = repo.merge(splits, on=GAME_KEY, how="inner", validate="one_to_one")
    games = repo.merge(pregame, on=GAME_KEY, how="left", validate="one_to_one", indicator=True)
    matched = games["_merge"] == "both"
    print(f"Split rows: {len(splits)}; matched to game logs: {matched.sum()} "
          f"({(~matched).sum()} unmatched, dropped from both feature sets)")
    games = games[matched].drop(columns="_merge")
    games = games.assign(_date=pd.to_datetime(games["Date"])).sort_values("_date", kind="stable")
    feature_columns = {
        "repo": [c for c in repo.columns if c not in REPO_DROP_COLUMNS + ["split"]],
        "pregame": [c for c in pregame.columns if c not in PREGAME_ID_COLUMNS + ABLATION_FEATURES],
        "pregame_margin": [c for c in pregame.columns if c not in PREGAME_ID_COLUMNS],
    }
    return games.drop(columns="_date").reset_index(drop=True), feature_columns


def sample_params(rng, seed):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": seed,
        "max_depth": int(rng.integers(2, 9)),
        "eta": float(10 ** rng.uniform(np.log10(0.005), np.log10(0.2))),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": int(rng.integers(1, 21)),
        "gamma": float(rng.uniform(0.0, 5.0)),
        "lambda": float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(5.0))),
    }
    return params


def walk_forward_cv(X, y, params, n_splits, max_rounds=3000):
    losses, best_rounds = [], []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx])
        booster = xgb.train(params, dtrain, num_boost_round=max_rounds, evals=[(dval, "val")],
                            early_stopping_rounds=60, verbose_eval=False)
        losses.append(booster.best_score)
        best_rounds.append(booster.best_iteration + 1)
    return float(np.mean(losses)), int(np.median(best_rounds))


def evaluate(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba),
        "brier": brier_score_loss(y, proba),
        "home_win_rate": float(np.mean(y)),  # accuracy of always picking the home team
    }


def train_feature_set(games, feature_set, feature_columns, trials, n_splits, seed, window):
    X = games[feature_columns[feature_set]].astype(float)
    y = games[TARGET].astype(int).to_numpy()
    is_train = (games["split"] == "train").to_numpy()
    X_train, y_train = X[is_train], y[is_train]

    rng = np.random.default_rng(seed)
    best = {"cv_log_loss": float("inf")}
    for trial in range(1, trials + 1):
        params = sample_params(rng, seed)
        cv_loss, rounds = walk_forward_cv(X_train, y_train, params, n_splits)
        if cv_loss < best["cv_log_loss"]:
            best = {"cv_log_loss": cv_loss, "params": params, "num_boost_round": rounds}
        print(f"[{feature_set}] trial {trial}/{trials}: CV log loss {cv_loss:.4f} ({rounds} rounds)")

    booster = xgb.train(best["params"], xgb.DMatrix(X_train, label=y_train),
                        num_boost_round=best["num_boost_round"])
    path = model_path(feature_set, window)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))

    metrics = {}
    for split in EVAL_SPLITS:
        mask = (games["split"] == split).to_numpy()
        metrics[split] = evaluate(y[mask], booster.predict(xgb.DMatrix(X[mask])))
    return {"n_features": X.shape[1], "n_train": int(is_train.sum()), **best, "metrics": metrics}


def markdown_report(results, window):
    lines = [f"# XGBoost: season-to-date (repo) vs rolling pre-game features (window={window})", ""]
    header = "| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier | Home-win rate |"
    lines += [header, "|" + "---|" * 11]
    labels = {
        "repo": "Repo season-to-date features (baseline)",
        "pregame": f"Pre-game rolling-{window} + context",
        "pregame_margin": f"Ablation: pre-game rolling-{window} + context + scoring margin",
    }
    for feature_set, result in results.items():
        for split, m in result["metrics"].items():
            lines.append(f"| {labels[feature_set]} | {split} | {m['n']} | {m['accuracy']:.4f} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | "
                         f"{m['log_loss']:.4f} | {m['brier']:.4f} | {m['home_win_rate']:.4f} |")
    for period, m in PAPER_XGBOOST.items():
        lines.append(f"| Parent paper, in-game {period} | 10-fold CV | - | {m['accuracy']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} | - | - | - |")
    lines += ["", "Positive class = home win; threshold 0.5. Parent-paper rows use box scores from the "
              "game being predicted, so they are an upper bound rather than a like-for-like comparison."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-sets", nargs="+", default=["repo", "pregame", "pregame_margin"],
                        choices=["repo", "pregame", "pregame_margin"])
    args = parser.parse_args()

    games, feature_columns = load_games(args.window)
    results = {fs: train_feature_set(games, fs, feature_columns, args.trials, args.splits, args.seed, args.window)
               for fs in args.feature_sets}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"metrics_w{args.window}.json").write_text(json.dumps(results, indent=2))
    report = markdown_report(results, args.window)
    (RESULTS_DIR / f"metrics_w{args.window}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
