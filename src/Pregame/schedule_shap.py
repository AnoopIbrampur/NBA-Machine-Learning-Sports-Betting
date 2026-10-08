"""SHAP for the schedule-feature models (window 20, test split).

Explains the better of the two new pre-game models (chosen by training walk-forward CV log loss,
never by test results) and repo_sched_travel: full ranking, share of total |SHAP| by feature group,
and dependence plots for the away back-to-back flag and the home team's net jet lag. The other new
pre-game model is explained too, as a supplement, because only it carries the jet-lag feature.
Where a model has no b2b_away column, rest_bucket_away == 1 is the same away back-to-back flag.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.Pregame.features import PAPER_STATS  # noqa: E402
from src.Pregame.paths import SCHEDULE_RESULTS_DIR, model_path  # noqa: E402
from src.Pregame.schedule_features import FATIGUE_FEATURES, TRAVEL_FEATURES  # noqa: E402
from src.Pregame.train_xgb import load_games  # noqa: E402

WINDOW, SPLIT = 20, "test"
WIN_RECORD = {"win_pct_home", "win_pct_away", "home_win_pct_at_home", "away_win_pct_on_road"} | {
    f"{c}{s}" for c in ["W", "L", "W_PCT", "W_RANK", "L_RANK", "W_PCT_RANK"] for s in ("", ".1")}
EXISTING_SCHEDULE = {"rest_days_home", "rest_days_away", "b2b_home", "b2b_away", "Days-Rest-Home", "Days-Rest-Away"}
GAMES_PLAYED = {f"{c}{s}" for c in ["GP", "MIN", "GP_RANK", "MIN_RANK"] for s in ("", ".1")}

BOX_SCORE_REPO = {c + r for c in ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
                                  "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"]
                  for r in ("", "_RANK")}


def group(feature):
    if feature in WIN_RECORD:
        return "win record"
    if feature in EXISTING_SCHEDULE or feature in FATIGUE_FEATURES:
        return "fatigue (incl. existing rest/b2b)"
    if feature in TRAVEL_FEATURES:
        return "travel"
    if feature in GAMES_PLAYED:
        return "games played / minutes"
    if feature in {f"{s}_diff" for s in PAPER_STATS} or feature.replace(".1", "") in BOX_SCORE_REPO:
        return "box-score stats"
    raise ValueError(f"No group for {feature}")


def dependence_features(columns):
    away_b2b = "b2b_away" if "b2b_away" in columns else "rest_bucket_away"
    return [f for f in [away_b2b, "jet_lag_home"] if f in columns]


def explain(games, columns, feature_set):
    X = games.loc[games["split"] == SPLIT, columns].astype(float)
    booster = xgb.Booster()
    booster.load_model(str(model_path(feature_set, WINDOW)))
    return shap.TreeExplainer(booster)(X), X


def main():
    games, feature_columns = load_games(WINDOW, schedule=True)
    training = json.loads((SCHEDULE_RESULTS_DIR / f"training_w{WINDOW}.json").read_text())
    best = min(["pregame_sched", "pregame_sched_travel"], key=lambda fs: training[fs]["cv_log_loss"])

    report = [f"# SHAP: schedule-feature models (window={WINDOW}, {SPLIT} split)", "",
              f"Best new pre-game model by training CV log loss: **{best}** "
              + ", ".join(f"({fs}: {training[fs]['cv_log_loss']:.4f})" for fs in ["pregame_sched", "pregame_sched_travel"])
              + ". Chosen without looking at test or validation.", ""]
    summary = {}
    other = "pregame_sched_travel" if best == "pregame_sched" else "pregame_sched"
    for feature_set in [best, "repo_sched_travel", other]:
        explanation, X = explain(games, feature_columns[feature_set], feature_set)
        importance = pd.Series(np.abs(explanation.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
        ranking = importance.rename("mean_abs_shap").to_frame()
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        ranking["group"] = [group(f) for f in ranking.index]
        shares = (ranking.groupby("group")["mean_abs_shap"].sum() / ranking["mean_abs_shap"].sum()) \
            .sort_values(ascending=False).rename("share_of_total_abs_shap")
        ranking.to_csv(SCHEDULE_RESULTS_DIR / f"shap_ranking_{feature_set}_w{WINDOW}.csv", index_label="feature")
        summary[feature_set] = {"group_shares": shares.to_dict(), "n": int(len(X))}

        shap.summary_plot(explanation.values, X, plot_type="bar", show=False, max_display=25)
        plt.title(f"{feature_set} SHAP (window={WINDOW}, {SPLIT})")
        plt.tight_layout()
        plt.savefig(SCHEDULE_RESULTS_DIR / f"shap_bar_{feature_set}_w{WINDOW}.png", dpi=150)
        plt.close()
        for feature in dependence_features(X.columns):
            shap.dependence_plot(feature, explanation.values, X, interaction_index=None, show=False)
            plt.title(f"{feature_set}: SHAP dependence on {feature}", fontsize=9)
            plt.tight_layout()
            plt.savefig(SCHEDULE_RESULTS_DIR / f"shap_dependence_{feature}_{feature_set}_w{WINDOW}.png", dpi=150)
            plt.close()
            values = pd.DataFrame({"x": X[feature], "shap": explanation.values[:, X.columns.get_loc(feature)]})
            summary[feature_set][f"{feature}_mean_shap_by_value"] = (
                values.groupby("x")["shap"].agg(["size", "mean"]).round(4).reset_index().to_dict(orient="records"))

        title = f"{feature_set} (supplementary)" if feature_set == other else feature_set
        report += [f"## {title} (n={len(X)})", "", "Share of total mean |SHAP| by group:", "",
                   shares.round(4).to_frame().to_markdown(), "", "Full ranking:", "",
                   ranking.round(4).to_markdown(), ""]
        for feature in dependence_features(X.columns):
            key = f"{feature}_mean_shap_by_value"
            if key in summary[feature_set]:
                report += [f"Mean SHAP by value of `{feature}` (log-odds of a home win):", "",
                           pd.DataFrame(summary[feature_set][key]).to_markdown(index=False), ""]

    (SCHEDULE_RESULTS_DIR / f"shap_w{WINDOW}.json").write_text(json.dumps({"best": best, **summary}, indent=2,
                                                                         default=float))
    text = "\n".join(report) + "\n"
    (SCHEDULE_RESULTS_DIR / f"shap_w{WINDOW}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
