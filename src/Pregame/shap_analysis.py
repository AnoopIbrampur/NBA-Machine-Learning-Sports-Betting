"""SHAP feature rankings for the pre-game XGBoost model, compared with the parent paper's.

Uses the model saved by train_xgb.py and explains a held-out split (test by default). Rankings
are by mean |SHAP value|, as in the parent paper's Table 10.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.Pregame.features import PAPER_STATS  # noqa: E402
from src.Pregame.paths import RESULTS_DIR, model_path  # noqa: E402
from src.Pregame.train_xgb import load_games  # noqa: E402

# Parent paper (PLOS ONE 2024), Table 10: SHAP importance order of in-game home-minus-away stats.
PAPER_SHAP_RANKS = {
    "H2": ["FG%", "DRB", "AST", "TOV", "FT%", "PF", "STL", "3P%", "ORB", "2P%", "BLK"],
    "H3": ["FG%", "TOV", "3P%", "DRB", "ORB", "FT%", "PF", "AST", "STL", "2P%", "BLK"],
    "Full game": ["FG%", "3P%", "TOV", "DRB", "ORB", "PF", "FT%", "2P%", "AST", "STL", "BLK"],
}


def shap_values(window, split):
    games, feature_columns = load_games(window)
    columns = feature_columns["pregame"]
    X = games.loc[games["split"] == split, columns].astype(float)
    booster = xgb.Booster()
    booster.load_model(str(model_path("pregame", window)))
    explanation = shap.TreeExplainer(booster)(X)
    return explanation, X


def rank_table(explanation, columns):
    importance = pd.Series(abs(explanation.values).mean(axis=0), index=columns)
    table = importance.sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def paper_comparison(ranking):
    diff_rows = ranking.loc[[f"{stat}_diff" for stat in PAPER_STATS]].sort_values("mean_abs_shap", ascending=False)
    ours = [name.removesuffix("_diff") for name in diff_rows.index]
    table = pd.DataFrame({"Rank": range(1, len(ours) + 1), "Pre-game (ours)": ours})
    for period, order in PAPER_SHAP_RANKS.items():
        table[f"Paper {period}"] = order
    correlations = {
        period: spearmanr([ours.index(s) for s in PAPER_STATS], [order.index(s) for s in PAPER_STATS]).statistic
        for period, order in PAPER_SHAP_RANKS.items()
    }
    return table, correlations


def save_plots(explanation, X, window, split):
    for kind, name in [("bar", "bar"), ("dot", "beeswarm")]:
        shap.summary_plot(explanation.values, X, plot_type=kind, show=False, max_display=X.shape[1])
        plt.title(f"Pre-game XGBoost SHAP (window={window}, {split} split)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_{name}_w{window}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    args = parser.parse_args()

    explanation, X = shap_values(args.window, args.split)
    ranking = rank_table(explanation, X.columns)
    comparison, correlations = paper_comparison(ranking)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(RESULTS_DIR / f"shap_ranking_w{args.window}.csv", index_label="feature")
    save_plots(explanation, X, args.window, args.split)

    report = [
        f"# SHAP rankings: pre-game XGBoost (window={args.window}, {args.split} split, n={len(X)})",
        "", "## All features (mean |SHAP|)", "", ranking.round(4).to_markdown(),
        "", "## The paper's 11 categories only, vs. parent paper Table 10", "",
        comparison.to_markdown(index=False), "",
        "Spearman rank correlation with the paper's order: "
        + ", ".join(f"{period} {rho:.2f}" for period, rho in correlations.items()),
    ]
    text = "\n".join(report) + "\n"
    (RESULTS_DIR / f"shap_w{args.window}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
