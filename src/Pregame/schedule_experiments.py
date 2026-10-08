"""Train and evaluate the schedule-feature models (window 20), with bootstrap CIs and fixed slices.

New feature sets, trained with exactly the protocol of train_xgb.py (same split, 40-trial
walk-forward search on train only, same seed):
    pregame_sched        = pregame + fatigue features
    pregame_sched_travel = pregame_sched + travel and time zone features
    repo_sched_travel    = repo baseline + all schedule features
The three original models are loaded from disk and never retrained.

The six evaluation slices were fixed before any result was seen; every one is reported.
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES, paired_bootstrap
from src.Pregame.paths import SCHEDULE_RESULTS_DIR, WORK_DIR, model_path
from src.Pregame.schedule_features import MILES_1000_KM
from src.Pregame.train_xgb import EVAL_SPLITS, TARGET, evaluate, load_games, train_feature_set

WINDOW = 20
ORIGINAL_SETS = ["repo", "pregame", "pregame_margin"]
NEW_SETS = ["pregame_sched", "pregame_sched_travel", "repo_sched_travel"]
LABELS = {
    "repo": "Baseline: repo season-to-date",
    "pregame": "Pre-game w20",
    "pregame_margin": "Pre-game w20 + margin (ablation)",
    "pregame_sched": "Pre-game w20 + fatigue",
    "pregame_sched_travel": "Pre-game w20 + fatigue + travel",
    "repo_sched_travel": "Repo + fatigue + travel",
}
# (model, reference) pairs for the bootstrap.
COMPARISONS = ([(fs, "repo") for fs in ["pregame", "pregame_margin"] + NEW_SETS]
               + [("pregame_sched", "pregame"), ("pregame_sched_travel", "pregame"),
                  ("pregame_sched_travel", "pregame_sched")])
UNRELIABLE_N, MIN_CI_N = 200, 30


def slices(games):
    """The six pre-registered slices. A season's first game has no rest value and is not a b2b."""
    b2b_home = games["b2b_home"].fillna(0) == 1
    b2b_away = games["b2b_away"].fillna(0) == 1
    return {
        "1. neither team on a b2b": ~b2b_home & ~b2b_away,
        "2. away team only on a b2b": b2b_away & ~b2b_home,
        "3. home team only on a b2b": b2b_home & ~b2b_away,
        "4. both on a b2b": b2b_home & b2b_away,
        "5. home eastward net jet lag >= 1 h": games["jet_lag_home"] >= 1,
        "6. away travelled > 1,609 km": games["travel_km_away"] > MILES_1000_KM,
    }


def predict(games, feature_columns, feature_set):
    booster = xgb.Booster()
    booster.load_model(str(model_path(feature_set, WINDOW)))
    return booster.predict(xgb.DMatrix(games[feature_columns[feature_set]].astype(float)))


def fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" + (" **excl. 0**" if ci[0] > 0 or ci[1] < 0 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Retrain new models even if saved.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    games, feature_columns = load_games(WINDOW, schedule=True)
    SCHEDULE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    training_path = SCHEDULE_RESULTS_DIR / f"training_w{WINDOW}.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    for feature_set in NEW_SETS:
        if args.retrain or feature_set not in training or not model_path(feature_set, WINDOW).exists():
            result = train_feature_set(games, feature_set, feature_columns, args.trials, args.splits,
                                       args.seed, WINDOW)
            training[feature_set] = {k: v for k, v in result.items() if k != "metrics"}
            training_path.write_text(json.dumps(training, indent=2))

    y = games[TARGET].astype(int).to_numpy()
    probas = {fs: predict(games, feature_columns, fs) for fs in ORIGINAL_SETS + NEW_SETS}
    pd.DataFrame({"GAME_ID": games["GAME_ID"], "Date": games["Date"], "split": games["split"], TARGET: y,
                  **{f"p_{fs}": p for fs, p in probas.items()}}).to_csv(
        WORK_DIR / f"schedule_predictions_w{WINDOW}.csv", index=False)

    results = {"metrics": {}, "comparisons": {}, "slices": {}}
    metric_lines, ci_lines, slice_lines = [], [], []
    for split in EVAL_SPLITS:
        mask = (games["split"] == split).to_numpy()
        for fs in ORIGINAL_SETS + NEW_SETS:
            m = evaluate(y[mask], probas[fs][mask])
            results["metrics"].setdefault(fs, {})[split] = m
            metric_lines.append(f"| {LABELS[fs]} | {split} | {m['n']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                                f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | {m['log_loss']:.4f} | "
                                f"{m['brier']:.4f} |")
        for fs, ref in COMPARISONS:
            cis = paired_bootstrap(y[mask], probas[fs][mask], probas[ref][mask])
            results["comparisons"].setdefault(f"{fs} vs {ref}", {})[split] = cis
            ci_lines.append(f"| {LABELS[fs]} | {LABELS[ref]} | {split} | {fmt_ci(cis['accuracy_diff_ci'])} | "
                            f"{fmt_ci(cis['auc_diff_ci'])} |")

        for name, slice_mask in slices(games).items():
            sm = mask & slice_mask.to_numpy()
            n = int(sm.sum())
            flag = "no CI (n < 30)" if n < MIN_CI_N else ("unreliable (n < 200)" if n < UNRELIABLE_N else "")
            for fs in ORIGINAL_SETS + NEW_SETS:
                p = probas[fs][sm]
                row = {"n": n, "home_win_rate": float(y[sm].mean()) if n else np.nan,
                       "accuracy": float(np.mean((p >= 0.5) == y[sm])) if n else np.nan}
                two_classes = n > 0 and 0 < y[sm].sum() < n
                row.update(evaluate(y[sm], p) if two_classes else {})
                if fs != "repo" and n >= MIN_CI_N and two_classes:
                    row["vs_repo"] = paired_bootstrap(y[sm], p, probas["repo"][sm])
                results["slices"].setdefault(split, {}).setdefault(name, {})[fs] = row
                acc_ci = fmt_ci(row["vs_repo"]["accuracy_diff_ci"]) if "vs_repo" in row else "-"
                auc_ci = fmt_ci(row["vs_repo"]["auc_diff_ci"]) if "vs_repo" in row else "-"
                get = lambda k: f"{row[k]:.4f}" if k in row and not np.isnan(row[k]) else "-"  # noqa: E731
                slice_lines.append(f"| {split} | {name} | {LABELS[fs]} | {n} | {get('home_win_rate')} | "
                                   f"{get('accuracy')} | {get('auc')} | {get('log_loss')} | {get('brier')} | "
                                   f"{acc_ci} | {auc_ci} | {flag} |")

    (SCHEDULE_RESULTS_DIR / f"metrics_w{WINDOW}.json").write_text(json.dumps(results, indent=2, default=float))
    report = [
        f"# Schedule-fatigue and travel features: XGBoost, window={WINDOW}", "",
        "Original three models are the saved ones from `train_xgb.py` (unchanged); the three new models use "
        "the same split, 40-trial walk-forward search on train only and seed. Positive class = home win, "
        "threshold 0.5.", "",
        "## Full test and validation sets", "",
        "| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |",
        "|" + "---|" * 10, *metric_lines, "",
        f"## Paired bootstrap 95% CIs of the difference (model − reference), {BOOTSTRAP_SAMPLES} resamples", "",
        "| Model | Reference | Split | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|---|", *ci_lines, "",
        "## Pre-registered slices", "",
        "Slices were fixed before results were seen and are all reported. b2b = the existing `b2b_home` / "
        "`b2b_away` flags (1 day since the previous game). Slice 5 uses net jet lag "
        "(sign × max(|time zone shift| − days since previous game, 0)); slice 6 uses great-circle km from the "
        "away team's previous venue. Slices 5 and 6 exclude games with unknown travel. CIs are model − repo "
        f"baseline on the slice's games. n < {UNRELIABLE_N}: unreliable; n < {MIN_CI_N}: no CI.", "",
        "| Split | Slice | Model | n | Home-win rate | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | "
        "AUC diff vs repo CI | Note |", "|" + "---|" * 12, *slice_lines,
    ]
    text = "\n".join(report) + "\n"
    (SCHEDULE_RESULTS_DIR / f"metrics_w{WINDOW}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
