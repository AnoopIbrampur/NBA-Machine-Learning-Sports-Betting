"""Modified Implementation rows, fixed in docs/enhanced-plan.md before any result was computed.

Trains only new models under new names, with the protocol of train_xgb.py. It never retrains the
existing models or writes outside results/pregame/enhanced/.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.Pregame.elo import ELO_FEATURES, K_GRID, choose_k, pregame_elo  # noqa: E402
from src.Pregame.features import PAPER_STATS, team_games  # noqa: E402
from src.Pregame.game_logs import load_game_logs  # noqa: E402
from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES, paired_bootstrap  # noqa: E402
from src.Pregame.market import load_market  # noqa: E402
from src.Pregame.paths import ENHANCED_RESULTS_DIR, RESULTS_DIR, WORK_DIR, model_path  # noqa: E402
from src.Pregame.schedule_experiments import MIN_CI_N, UNRELIABLE_N, fmt_ci, slices  # noqa: E402
from src.Pregame.schedule_features import SCHEDULE_FEATURES  # noqa: E402
from src.Pregame.stacking import fit_meta, predict_meta, walk_forward_oof  # noqa: E402
from src.Pregame.train_xgb import EVAL_SPLITS, TARGET, evaluate, load_games, train_feature_set  # noqa: E402

WINDOW = 20
ELO_SETS = ["pregame_elo", "repo_elo", "pregame_sched_travel_elo"]
ABLATION_WINDOWS = [5, 7]
STACK_MEMBERS = ["repo", "pregame", "pregame_elo"]
EXISTING = ["repo", "pregame", "pregame_sched_travel"]
NEW_ROWS = ["elo_logit"] + ELO_SETS + ["stack"] + [f"pregame_w{w}" for w in ABLATION_WINDOWS]
SLICE_ROWS = ELO_SETS + ["stack", "pregame_w5", "pregame_w7"]
LABELS = {
    "repo": "Baseline: repo season-to-date",
    "pregame": "Pre-game w20 (official)",
    "pregame_sched_travel": "Pre-game w20 + fatigue + travel",
    "elo_logit": "Elo only: logistic on elo_diff",
    "pregame_elo": "Pre-game w20 + Elo",
    "repo_elo": "Repo + Elo",
    "pregame_sched_travel_elo": "Pre-game w20 + fatigue + travel + Elo",
    "stack": "Stack: repo, pre-game, pre-game + Elo",
    "pregame_w5": "Ablation: pre-game window 5",
    "pregame_w7": "Ablation: pre-game window 7",
    "market": "Market: devigged moneyline (reference)",
}
COMPARISONS = ([(row, "repo") for row in NEW_ROWS] + [(row, "pregame") for row in NEW_ROWS]
               + [("pregame_sched_travel_elo", "pregame_sched_travel"),
                  ("market", "pregame_elo"), ("market", "stack")])
WIN_RECORD = ["win_pct_home", "win_pct_away", "home_win_pct_at_home", "away_win_pct_on_road"]


def extra_slices(games):
    """The six slices of schedule_experiments plus the one new slice fixed in the plan."""
    return {**slices(games), "7. either team under 20 games this season": (games["GP"] < 20) | (games["GP.1"] < 20)}


def add_elo(games, feature_columns, elo):
    """A copy of games with Elo columns; the existing feature lists are returned untouched."""
    out = games.merge(elo, left_on="GAME_ID", right_index=True, how="left", validate="one_to_one")
    if out[ELO_FEATURES].isna().any().any():
        raise ValueError("Some games have no Elo rating")
    columns = {**feature_columns,
               "pregame_elo": feature_columns["pregame"] + ELO_FEATURES,
               "repo_elo": feature_columns["repo"] + ELO_FEATURES,
               "pregame_sched_travel_elo": feature_columns["pregame_sched_travel"] + ELO_FEATURES}
    return out, columns


def predict_saved(games, columns, feature_set, window=WINDOW):
    booster = xgb.Booster()
    booster.load_model(str(model_path(feature_set, window)))
    return booster.predict(xgb.DMatrix(games[columns].astype(float)))


def stack_rows(games, feature_columns, tuned):
    """Out-of-fold base predictions on train, then the meta-model; scores every game."""
    train = games[games["split"] == "train"]
    y = train[TARGET].astype(int).to_numpy()
    oof = {}
    for member in STACK_MEMBERS:
        X = train[feature_columns[member]].astype(float)

        def fit_predict(train_idx, val_idx):
            booster = xgb.train(tuned[member]["params"], xgb.DMatrix(X.iloc[train_idx], label=y[train_idx]),
                                num_boost_round=tuned[member]["num_boost_round"])
            return booster.predict(xgb.DMatrix(X.iloc[val_idx]))

        oof[member] = walk_forward_oof(len(train), fit_predict)
    meta, usable = fit_meta(np.column_stack([oof[m] for m in STACK_MEMBERS]), y)
    full = np.column_stack([predict_saved(games, feature_columns[m], m) for m in STACK_MEMBERS])
    info = {"members": STACK_MEMBERS, "n_meta_rows": int(usable.sum()), "intercept": float(meta.intercept_[0]),
            "coefficients": dict(zip(STACK_MEMBERS, map(float, meta.coef_[0])))}
    return predict_meta(meta, full), info


def shap_groups(columns):
    groups = {}
    for name in columns:
        if name in ELO_FEATURES:
            groups[name] = "Elo"
        elif name in WIN_RECORD:
            groups[name] = "Win record"
        elif name.removesuffix("_diff") in PAPER_STATS and name.endswith("_diff"):
            groups[name] = "Box-score stats"
        elif name in SCHEDULE_FEATURES or name.startswith(("rest_days_", "b2b_")):
            groups[name] = "Schedule"
        else:
            raise ValueError(f"No SHAP group for {name}")
    return groups


def hypothesis_two(ranking):
    """Apply the plan's H2 rule to a table with mean_abs_shap and group, sorted high to low."""
    shares = ranking.groupby("group")["mean_abs_shap"].sum() / ranking["mean_abs_shap"].sum()
    n = len(ranking)
    ranks = pd.Series(range(1, n + 1), index=ranking.index)
    median_rank = float(ranks[ranking["group"] == "Schedule"].median())
    elo_ranks = ranks[ranking["group"] == "Elo"]
    elo_rank = int(elo_ranks.min()) if len(elo_ranks) else None
    elo_top = ranking["group"].iloc[0] == "Elo" and shares.idxmax() == "Elo"
    middle = n / 3 < median_rank <= 2 * n / 3
    return {"n_features": n, "top_feature": ranking.index[0], "group_shares": shares.sort_values(ascending=False).to_dict(),
            "best_elo_rank": elo_rank, "schedule_median_rank": median_rank,
            "elo_strongest": bool(elo_top), "schedule_in_middle": bool(middle), "supported": bool(elo_top and middle)}


def shap_report(games, feature_columns):
    feature_set = "pregame_sched_travel_elo"
    columns = feature_columns[feature_set]
    booster = xgb.Booster()
    booster.load_model(str(model_path(feature_set, WINDOW)))
    groups, out, tables = shap_groups(columns), {}, {}
    for split in EVAL_SPLITS:
        X = games.loc[games["split"] == split, columns].astype(float)
        values = shap.TreeExplainer(booster)(X).values
        ranking = pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values(ascending=False).to_frame("mean_abs_shap")
        ranking["group"] = ranking.index.map(groups)
        out[split], tables[split] = hypothesis_two(ranking), ranking
        ranking.to_csv(ENHANCED_RESULTS_DIR / f"shap_ranking_{split}.csv", index_label="feature")
        if split == "test":
            shap.summary_plot(values, X, plot_type="bar", show=False, max_display=20)
            plt.title("Pre-game + fatigue + travel + Elo: SHAP, test split")
            plt.tight_layout()
            plt.savefig(ENHANCED_RESULTS_DIR / "shap_bar_test.png", dpi=150)
            plt.close()
    return out, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Retrain the new models even if saved.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ENHANCED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    games, feature_columns = load_games(WINDOW, schedule=True)
    logs = team_games(load_game_logs())
    train = games[games["split"] == "train"]
    best_k, k_losses = choose_k(logs, train["GAME_ID"].tolist(), train[TARGET].astype(int).to_numpy())
    print(f"Chose k={best_k} on train: {k_losses}")
    games, feature_columns = add_elo(games, feature_columns, pregame_elo(logs, best_k))
    market = load_market(logs)[0][["GAME_ID", "p_market"]]
    games = games.merge(market, on="GAME_ID", how="left", validate="one_to_one")

    training_path = ENHANCED_RESULTS_DIR / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    if training.get("elo_k") != best_k:
        training = {}
    training.update({"elo_k": best_k, "elo_k_log_loss": {str(k): k_losses[k] for k in K_GRID}})
    for feature_set in ELO_SETS:
        if args.retrain or feature_set not in training or not model_path(feature_set, WINDOW).exists():
            result = train_feature_set(games, feature_set, feature_columns, args.trials, args.splits, args.seed, WINDOW)
            training[feature_set] = {k: v for k, v in result.items() if k != "metrics"}
            training_path.write_text(json.dumps(training, indent=2))

    probas = {"market": games["p_market"].to_numpy()}
    window_keys = games[["GAME_ID"]].reset_index(drop=True)
    for window in ABLATION_WINDOWS:
        window_games, window_columns = load_games(window)
        if not window_games["GAME_ID"].reset_index(drop=True).equals(window_keys["GAME_ID"]):
            raise ValueError(f"Window {window} games differ from window {WINDOW}")
        key = f"pregame_w{window}"
        if args.retrain or key not in training or not model_path("pregame", window).exists():
            result = train_feature_set(window_games, "pregame", window_columns, args.trials, args.splits, args.seed, window)
            training[key] = {k: v for k, v in result.items() if k != "metrics"}
            training_path.write_text(json.dumps(training, indent=2))
        probas[key] = predict_saved(window_games, window_columns["pregame"], "pregame", window)

    saved = json.loads((RESULTS_DIR / f"metrics_w{WINDOW}.json").read_text())
    tuned = {"repo": saved["repo"], "pregame": saved["pregame"], "pregame_elo": training["pregame_elo"]}
    probas["stack"], training["stack"] = stack_rows(games, feature_columns, tuned)
    train_mask = (games["split"] == "train").to_numpy()
    y = games[TARGET].astype(int).to_numpy()
    logistic = LogisticRegression(C=1e6).fit(games.loc[train_mask, ["elo_diff"]] / 400.0, y[train_mask])
    probas["elo_logit"] = logistic.predict_proba(games[["elo_diff"]] / 400.0)[:, 1]
    training_path.write_text(json.dumps(training, indent=2))
    for feature_set in EXISTING + ELO_SETS:
        probas[feature_set] = predict_saved(games, feature_columns[feature_set], feature_set)

    pd.DataFrame({"GAME_ID": games["GAME_ID"], "split": games["split"], TARGET: y,
                  **{f"p_{name}": p for name, p in probas.items()}}).to_csv(WORK_DIR / "enhanced_predictions_w20.csv", index=False)

    rows = EXISTING + NEW_ROWS + ["market"]
    results = {"elo_k": best_k, "metrics": {}, "comparisons": {}, "slices": {}}
    metric_lines, ci_lines, slice_lines = [], [], []
    for split in EVAL_SPLITS:
        mask = (games["split"] == split).to_numpy()
        for row in rows:
            m = evaluate(y[mask], probas[row][mask])
            results["metrics"].setdefault(row, {})[split] = m
            metric_lines.append(f"| {LABELS[row]} | {split} | {m['n']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                                f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} |")
        for row, ref in COMPARISONS:
            cis = paired_bootstrap(y[mask], probas[row][mask], probas[ref][mask])
            results["comparisons"].setdefault(f"{row} vs {ref}", {})[split] = cis
            ci_lines.append(f"| {LABELS[row]} | {LABELS[ref]} | {split} | {fmt_ci(cis['accuracy_diff_ci'])} | "
                            f"{fmt_ci(cis['auc_diff_ci'])} |")
        for name, slice_mask in extra_slices(games).items():
            sm = mask & slice_mask.to_numpy()
            n = int(sm.sum())
            flag = "no CI (n < 30)" if n < MIN_CI_N else ("unreliable (n < 200)" if n < UNRELIABLE_N else "")
            for row in ["repo", "pregame"] + SLICE_ROWS:
                two_classes = n > 0 and 0 < y[sm].sum() < n
                entry = {"n": n, **(evaluate(y[sm], probas[row][sm]) if two_classes else {})}
                if row in SLICE_ROWS and n >= MIN_CI_N and two_classes:
                    entry["vs_repo"] = paired_bootstrap(y[sm], probas[row][sm], probas["repo"][sm])
                results["slices"].setdefault(split, {}).setdefault(name, {})[row] = entry
                acc = fmt_ci(entry["vs_repo"]["accuracy_diff_ci"]) if "vs_repo" in entry else "-"
                auc = fmt_ci(entry["vs_repo"]["auc_diff_ci"]) if "vs_repo" in entry else "-"
                cells = [f"{entry[k]:.4f}" if k in entry else "-" for k in ("accuracy", "auc", "log_loss", "brier")]
                slice_lines.append(f"| {split} | {name} | {LABELS[row]} | {n} | {' | '.join(cells)} | {acc} | {auc} | {flag} |")

    results["hypothesis_2"], shap_tables = shap_report(games, feature_columns)
    results["training"] = training
    (ENHANCED_RESULTS_DIR / "metrics_w20.json").write_text(json.dumps(results, indent=2, default=float))

    h2_lines = []
    for split, h in results["hypothesis_2"].items():
        shares = ", ".join(f"{g} {v:.1%}" for g, v in h["group_shares"].items())
        h2_lines += [f"- **{split}:** top feature `{h['top_feature']}`; best Elo rank {h['best_elo_rank']}; schedule median "
                     f"rank {h['schedule_median_rank']:g} of {h['n_features']}; shares: {shares}. Elo strongest: "
                     f"{h['elo_strongest']}. Schedule in the middle: {h['schedule_in_middle']}. Supported: {h['supported']}."]
    report = [
        "# Modified Implementation: Elo, stacking and window ablations (window 20 unless labelled)", "",
        "Plan fixed in advance: `docs/enhanced-plan.md`. Existing models are the saved ones, unchanged. Positive "
        "class = home win, threshold 0.5.", "",
        f"Elo k chosen on train only: **{best_k}**. Walk-forward log loss by k: "
        + ", ".join(f"{k}: {v:.5f}" for k, v in k_losses.items()) + ".", "",
        "## Full test and validation sets", "",
        "| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |", "|" + "---|" * 10,
        *metric_lines, "",
        f"## Paired bootstrap 95% CIs (row − reference), {BOOTSTRAP_SAMPLES} resamples", "",
        "| Row | Reference | Split | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|---|", *ci_lines, "",
        "## Stack", "", "```json", json.dumps(training["stack"], indent=2), "```", "",
        "## Hypothesis 2 (`pregame_sched_travel_elo`, mean |SHAP|)", "", *h2_lines, "",
        "## Slices", "",
        f"CIs are row − repo baseline. n < {UNRELIABLE_N}: unreliable; n < {MIN_CI_N}: no CI.", "",
        "| Split | Slice | Model | n | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | AUC diff vs repo CI | Note |",
        "|" + "---|" * 11, *slice_lines,
    ]
    text = "\n".join(report) + "\n"
    (ENHANCED_RESULTS_DIR / "metrics_w20.md").write_text(text)
    print("\n".join(report[:24]))


if __name__ == "__main__":
    main()
