"""Modified Implementation, round 2, fixed in docs/enhanced-plan-2.md before these results existed.

Needs src.Pregame.enhanced to have run. Trains only new models under new names and writes only to
results/pregame/enhanced2/.
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

from src.Pregame.elo import ELO_FEATURES, choose_variant, pregame_elo  # noqa: E402
from src.Pregame.enhanced import WIN_RECORD, extra_slices, predict_saved, stack_rows  # noqa: E402
from src.Pregame.features import PAPER_STATS, team_games  # noqa: E402
from src.Pregame.game_logs import load_game_logs  # noqa: E402
from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES, paired_bootstrap  # noqa: E402
from src.Pregame.market import load_market  # noqa: E402
from src.Pregame.paths import (ENHANCED2_RESULTS_DIR, ENHANCED_RESULTS_DIR, RESULTS_DIR, WORK_DIR,  # noqa: E402
                               model_path)
from src.Pregame.player_features import PLAYER_FEATURES, build_player_features, replacement_level  # noqa: E402
from src.Pregame.player_logs import load_player_logs  # noqa: E402
from src.Pregame.rolling import rolling_predictions  # noqa: E402
from src.Pregame.schedule_experiments import MIN_CI_N, UNRELIABLE_N, fmt_ci  # noqa: E402
from src.Pregame.schedule_features import BUBBLE_START, offsite_game_ids  # noqa: E402
from src.Pregame.stacking import logit  # noqa: E402
from src.Pregame.train_xgb import EVAL_SPLITS, TARGET, evaluate, load_games, train_feature_set  # noqa: E402

WINDOW = 20
ELO2_FEATURES = ["elo2_home", "elo2_away", "elo2_diff"]
NEW_SETS = ["pregame_elo2", "pregame_elo2_player", "repo_elo2_player"]
STACK2_MEMBERS = ["repo", "pregame", "pregame_elo2_player"]
PRIMARY = "pregame_elo2_player"
NEW_ROWS = NEW_SETS + ["stack2", "roll_repo", "roll_pregame_elo2_player", "roll_stack2"]
LABELS = {
    "repo": "Baseline: repo season-to-date",
    "pregame": "Pre-game w20 (official)",
    "stack": "Round 1 stack",
    "pregame_elo2": "Pre-game w20 + Elo variant",
    "pregame_elo2_player": "PRIMARY: pre-game w20 + Elo variant + player",
    "repo_elo2_player": "Repo + Elo variant + player",
    "stack2": "Stack 2: repo, pre-game, primary",
    "roll_repo": "Rolling baseline",
    "roll_pregame_elo2_player": "Rolling primary",
    "roll_stack2": "Rolling stack 2",
    "market": "Market: devigged moneyline (reference)",
}
COMPARISONS = ([(row, ref) for row in NEW_SETS + ["stack2"] for ref in ("repo", "pregame", "stack")]
               + [("pregame_elo2_player", "pregame_elo2"), ("roll_repo", "repo"),
                  ("roll_pregame_elo2_player", "pregame_elo2_player"), ("roll_stack2", "stack2"),
                  ("roll_pregame_elo2_player", "roll_repo"), ("roll_stack2", "roll_repo"),
                  ("roll_pregame_elo2_player", "repo"), ("roll_stack2", "repo"),
                  ("market", "pregame_elo2_player"), ("market", "stack2"), ("market", "roll_stack2")])


def add_round_two(games, feature_columns, elo, players):
    out = games.merge(elo, left_on="GAME_ID", right_index=True, how="left", validate="one_to_one")
    out = out.merge(players, left_on="GAME_ID", right_index=True, how="left", validate="one_to_one")
    if out[ELO2_FEATURES].isna().any().any():
        raise ValueError("Some games have no Elo variant rating")
    columns = {**feature_columns,
               "pregame_elo2": feature_columns["pregame"] + ELO2_FEATURES,
               "pregame_elo2_player": feature_columns["pregame"] + ELO2_FEATURES + PLAYER_FEATURES,
               "repo_elo2_player": feature_columns["repo"] + ELO2_FEATURES + PLAYER_FEATURES}
    return out, columns


def feature_group(name):
    if name in ELO2_FEATURES:
        return "Elo"
    if name in PLAYER_FEATURES:
        return "Player"
    if name in WIN_RECORD:
        return "Win record"
    if name.endswith("_diff") and name.removesuffix("_diff") in PAPER_STATS:
        return "Box-score stats"
    if name.startswith(("rest_days_", "b2b_")):
        return "Schedule"
    raise ValueError(f"No SHAP group for {name}")


def shap_shares(games, columns):
    booster = xgb.Booster()
    booster.load_model(str(model_path(PRIMARY, WINDOW)))
    X = games.loc[games["split"] == "test", columns].astype(float)
    values = shap.TreeExplainer(booster)(X).values
    ranking = pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values(ascending=False).to_frame("mean_abs_shap")
    ranking["group"] = [feature_group(name) for name in ranking.index]
    ranking.to_csv(ENHANCED2_RESULTS_DIR / "shap_ranking_test.csv", index_label="feature")
    shap.summary_plot(values, X, plot_type="bar", show=False, max_display=20)
    plt.title("Pre-game + Elo variant + player: SHAP, test split")
    plt.tight_layout()
    plt.savefig(ENHANCED2_RESULTS_DIR / "shap_bar_test.png", dpi=150)
    plt.close()
    shares = ranking.groupby("group")["mean_abs_shap"].sum() / ranking["mean_abs_shap"].sum()
    return {"top_features": list(ranking.index[:10]), "group_shares": shares.sort_values(ascending=False).to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Retrain the new models even if saved.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ENHANCED2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    games, feature_columns = load_games(WINDOW, schedule=True)
    logs = team_games(load_game_logs())
    train = games[games["split"] == "train"]
    train_ids, train_y = train["GAME_ID"].tolist(), train[TARGET].astype(int).to_numpy()

    bubble = logs.loc[(logs["SEASON"] == "2019-20") & (logs["Date"] >= BUBBLE_START), "GAME_ID"]
    no_home = frozenset(bubble) | frozenset(offsite_game_ids(logs))
    best, losses = choose_variant(logs, train_ids, train_y, no_home)
    k, home, regression, margin = best
    print(f"Elo variant chosen on train: k={k}, home={home}, regression={regression:.3f}, margin={margin}, "
          f"log loss {losses[best]:.5f}")
    elo = pregame_elo(logs, k, home, regression, margin, no_home).rename(columns=dict(zip(ELO_FEATURES, ELO2_FEATURES)))

    player_rows = load_player_logs()
    replacement = replacement_level(player_rows, train_ids)
    player = build_player_features(player_rows, logs, replacement)
    print(f"Replacement level (train only): {replacement:.3f}; missing share by feature:\n"
          + player.reindex(games["GAME_ID"]).isna().mean().round(4).to_string())
    games, feature_columns = add_round_two(games, feature_columns, elo, player)
    market = load_market(logs)[0][["GAME_ID", "p_market"]]
    games = games.merge(market, on="GAME_ID", how="left", validate="one_to_one")

    training_path = ENHANCED2_RESULTS_DIR / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    variant = {"k": k, "home_advantage": home, "regression": regression, "margin": margin, "log_loss": losses[best]}
    if training.get("elo_variant") != variant:
        training = {}
    training.update({"elo_variant": variant, "replacement_level": replacement,
                     "elo_grid_top5": [{"setting": list(key), "log_loss": losses[key]}
                                       for key in sorted(losses, key=lambda key: (losses[key], key))[:5]]})
    for feature_set in NEW_SETS:
        if args.retrain or feature_set not in training or not model_path(feature_set, WINDOW).exists():
            result = train_feature_set(games, feature_set, feature_columns, args.trials, args.splits, args.seed, WINDOW)
            training[feature_set] = {key: v for key, v in result.items() if key != "metrics"}
            training_path.write_text(json.dumps(training, indent=2))

    saved = json.loads((RESULTS_DIR / f"metrics_w{WINDOW}.json").read_text())
    tuned = {"repo": saved["repo"], "pregame": saved["pregame"], PRIMARY: training[PRIMARY]}
    y = games[TARGET].astype(int).to_numpy()
    held = games["split"].isin(EVAL_SPLITS).to_numpy()

    probas = {"market": games["p_market"].to_numpy()}
    for feature_set in ["repo", "pregame"] + NEW_SETS:
        probas[feature_set] = predict_saved(games, feature_columns[feature_set], feature_set)
    round_one = pd.read_csv(WORK_DIR / "enhanced_predictions_w20.csv", dtype={"GAME_ID": str}).set_index("GAME_ID")
    probas["stack"] = round_one.loc[games["GAME_ID"], "p_stack"].to_numpy()
    probas["stack2"], training["stack2"] = stack_rows(games, feature_columns, tuned, STACK2_MEMBERS)
    training_path.write_text(json.dumps(training, indent=2))

    rolled = {m: rolling_predictions(games, feature_columns[m], y, tuned[m], probas[m], held) for m in STACK2_MEMBERS}
    probas["roll_repo"], probas["roll_pregame_elo2_player"] = rolled["repo"], rolled[PRIMARY]
    weights = np.array([training["stack2"]["coefficients"][m] for m in STACK2_MEMBERS])
    meta_logit = training["stack2"]["intercept"] + logit(np.column_stack([rolled[m] for m in STACK2_MEMBERS])) @ weights
    probas["roll_stack2"] = 1.0 / (1.0 + np.exp(-meta_logit))

    pd.DataFrame({"GAME_ID": games["GAME_ID"], "split": games["split"], TARGET: y,
                  **{f"p_{name}": p for name, p in probas.items()}}).to_csv(WORK_DIR / "enhanced2_predictions_w20.csv", index=False)

    rows = ["repo", "pregame", "stack"] + NEW_ROWS + ["market"]
    results = {"elo_variant": variant, "metrics": {}, "comparisons": {}, "slices": {}}
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
            for row in ["repo"] + NEW_ROWS:
                two_classes = n > 0 and 0 < y[sm].sum() < n
                entry = {"n": n, **(evaluate(y[sm], probas[row][sm]) if two_classes else {})}
                if row != "repo" and n >= MIN_CI_N and two_classes:
                    entry["vs_repo"] = paired_bootstrap(y[sm], probas[row][sm], probas["repo"][sm])
                results["slices"].setdefault(split, {}).setdefault(name, {})[row] = entry
                acc = fmt_ci(entry["vs_repo"]["accuracy_diff_ci"]) if "vs_repo" in entry else "-"
                auc = fmt_ci(entry["vs_repo"]["auc_diff_ci"]) if "vs_repo" in entry else "-"
                cells = [f"{entry[key]:.4f}" if key in entry else "-" for key in ("accuracy", "auc", "log_loss", "brier")]
                slice_lines.append(f"| {split} | {name} | {LABELS[row]} | {n} | {' | '.join(cells)} | {acc} | {auc} | {flag} |")

    results["shap_test"] = shap_shares(games, feature_columns[PRIMARY])
    results["training"] = training
    (ENHANCED2_RESULTS_DIR / "metrics_w20.json").write_text(json.dumps(results, indent=2, default=float))
    shares = ", ".join(f"{g} {v:.1%}" for g, v in results["shap_test"]["group_shares"].items())
    report = [
        "# Modified Implementation, round 2: Elo variant, player features and rolling retraining", "",
        "Plan fixed in advance: `docs/enhanced-plan-2.md`. This round is not blind: round 1 results motivated it. "
        "Positive class = home win, threshold 0.5. Rolling rows train on earlier held-out games, as a live system would.", "",
        f"Elo variant chosen on train only: k={k}, home advantage={home:g}, regression={regression:.3f}, margin={margin}, "
        f"walk-forward log loss {losses[best]:.5f}. Replacement level {replacement:.3f}.", "",
        "## Full test and validation sets", "",
        "| Model | Split | n | Accuracy | Precision | Recall | F1 | AUC | Log loss | Brier |", "|" + "---|" * 10,
        *metric_lines, "",
        f"## Paired bootstrap 95% CIs (row − reference), {BOOTSTRAP_SAMPLES} resamples", "",
        "| Row | Reference | Split | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|---|", *ci_lines, "",
        "## Stack 2", "", "```json", json.dumps(training["stack2"], indent=2), "```", "",
        "## SHAP shares, primary row, test", "", shares + ".", "",
        "Top features: " + ", ".join(f"`{f}`" for f in results["shap_test"]["top_features"]) + ".", "",
        "## Slices", "", f"CIs are row − repo baseline. n < {UNRELIABLE_N}: unreliable; n < {MIN_CI_N}: no CI.", "",
        "| Split | Slice | Model | n | Accuracy | AUC | Log loss | Brier | Acc diff vs repo CI | AUC diff vs repo CI | Note |",
        "|" + "---|" * 11, *slice_lines,
    ]
    (ENHANCED2_RESULTS_DIR / "metrics_w20.md").write_text("\n".join(report) + "\n")
    print("finished")


if __name__ == "__main__":
    main()
