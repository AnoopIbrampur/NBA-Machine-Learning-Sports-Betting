"""Does the betting market fully price back-to-backs? (window-20 games, results/pregame/market/)

1. Market benchmark: the devigged moneyline probability scored on the same test and validation
   games as our models (games without both moneylines dropped from every row equally), with paired
   bootstrap CIs against the repo baseline and the official pre-game model. A reference row only.
2. Fixed in advance, using the existing b2b_home / b2b_away flags and the four §7 situations:
   a. logistic regression of home win on the situation dummies (reference: neither on a b2b) with
      logit(market probability) as a control, fitted separately on train, test and validation;
   b. home cover rate against the spread by situation and split, exact binomial 95% CI against 50%,
      pushes excluded and counted.
   Restart-bubble and off-site games are excluded, as in schedule_replication.py.
3. Exploratory only: a linear season trend in the visitor-b2b and home-b2b effects (the §7f model
   on all split seasons) and the raw home-win gap versus both-rested by season. Not used for any model.
No model is retrained; the saved models are only scored.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest

from src.Pregame.features import team_games
from src.Pregame.game_logs import load_game_logs
from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES, paired_bootstrap
from src.Pregame.market import cover_result, load_market
from src.Pregame.paths import RESULTS_DIR
from src.Pregame.schedule_experiments import LABELS, NEW_SETS, ORIGINAL_SETS, WINDOW, predict
from src.Pregame.schedule_features import BUBBLE_START, offsite_game_ids
from src.Pregame.schedule_replication import (SITUATIONS, fit, schedule_variables, situation,
                                              team_season_dummies)
from src.Pregame.train_xgb import EVAL_SPLITS, TARGET, evaluate, load_games

MARKET_DIR = RESULTS_DIR / "market"
SPLITS = ["train", "test", "validation"]
SITUATION_DUMMIES = {"away only on a b2b": SITUATIONS[0], "home only on a b2b": SITUATIONS[3],
                     "both on a b2b": SITUATIONS[2]}  # reference: neither (both rested)
TREND_CENTER = 2018  # season start year the trend is centred on (middle of 2012-13 .. 2025-26)
ASHMAN_2010 = "45.86% home cover, home on a b2b vs rested visitor, 1990-2009"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def odds_rows(model, names):
    ci = model.conf_int()
    return {name: {"odds_multiplier": float(np.exp(model.params[name])), "ci_low": float(np.exp(ci.loc[name, 0])),
                   "ci_high": float(np.exp(ci.loc[name, 1])), "p_value": float(model.pvalues[name])}
            for name in names}


def benchmark(games, feature_columns):
    y = games[TARGET].astype(int).to_numpy()
    probas = {fs: predict(games, feature_columns, fs) for fs in ORIGINAL_SETS + NEW_SETS}
    probas["market"] = games["p_market"].to_numpy()
    results, lines, ci_lines = {}, [], []
    for split in EVAL_SPLITS:
        in_split = (games["split"] == split).to_numpy()
        mask = in_split & games["p_market"].notna().to_numpy()
        for name in ["market"] + ORIGINAL_SETS + NEW_SETS:
            m = evaluate(y[mask], probas[name][mask])
            results.setdefault(name, {})[split] = m
            label = "Market: devigged moneyline (reference)" if name == "market" else LABELS[name]
            lines.append(f"| {label} | {split} | {m['n']} | {m['accuracy']:.4f} | {m['auc']:.4f} | "
                         f"{m['log_loss']:.4f} | {m['brier']:.4f} |")
        for ref in ["repo", "pregame"]:
            cis = paired_bootstrap(y[mask], probas["market"][mask], probas[ref][mask])
            results["market"].setdefault("vs", {}).setdefault(ref, {})[split] = cis
            ci_lines.append(f"| Market − {LABELS[ref]} | {split} | {int(mask.sum())} | "
                            f"[{cis['accuracy_diff_ci'][0]:+.4f}, {cis['accuracy_diff_ci'][1]:+.4f}] | "
                            f"[{cis['auc_diff_ci'][0]:+.4f}, {cis['auc_diff_ci'][1]:+.4f}] |")
    return results, lines, ci_lines


def situation_logits(games):
    results, lines = {}, []
    for split in SPLITS:
        part = games[(games["split"] == split) & games["p_market"].notna()]
        sit = situation(part)
        X = pd.DataFrame({name: (sit == value).astype(float) for name, value in SITUATION_DUMMIES.items()},
                         index=part.index)
        X["logit_market"] = logit(part["p_market"])
        model, diag = fit(part[TARGET].astype(int), X)
        rows = odds_rows(model, list(X.columns))
        rows["logit_market"]["coef"] = float(model.params["logit_market"])
        results[split] = {"n": int(len(part)), "fit": diag, "terms": rows,
                          "n_by_situation": sit.value_counts().reindex(SITUATIONS).fillna(0).astype(int).to_dict()}
        for name in X.columns:
            r = rows[name]
            n = int(X[name].sum()) if name != "logit_market" else len(part)
            label = name if name != "logit_market" else "logit(market prob.): slope, as odds multiplier per unit"
            lines.append(f"| {split} | {label} | {n} | {r['odds_multiplier']:.3f} | "
                         f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p_value']:.3f} |")
    return results, lines


def cover_rates(games):
    results, lines = {}, []
    games = games.assign(situation=situation(games),
                         cover=cover_result(games["home_margin_logs"], games["home_line"]))
    for split in SPLITS:
        part = games[games["split"] == split]
        for sit in SITUATIONS + ["all"]:
            cell = part if sit == "all" else part[part["situation"] == sit]
            known = cell["cover"].notna()
            covers, pushes = int((cell["cover"] == 1).sum()), int((cell["cover"] == 0).sum())
            decided = int(known.sum()) - pushes
            test = binomtest(covers, decided, 0.5) if decided else None
            ci = test.proportion_ci(confidence_level=0.95, method="exact") if test else None
            row = {"games": int(len(cell)), "no_line": int((~known).sum()), "pushes": pushes, "decided": decided,
                   "covers": covers, "cover_rate": covers / decided if decided else np.nan,
                   "ci_low": ci.low if ci else np.nan, "ci_high": ci.high if ci else np.nan,
                   "p_value_vs_50": test.pvalue if test else np.nan}
            results.setdefault(split, {})[sit] = row
            flag = " **excl. 50%**" if decided and (row["ci_high"] < 0.5 or row["ci_low"] > 0.5) else ""
            lines.append(f"| {split} | {sit} | {row['games']} | {row['no_line']} | {pushes} | {decided} | "
                         f"{row['cover_rate']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}]{flag} | "
                         f"{row['p_value_vs_50']:.3f} |")
    return results, lines


def season_trend(games):
    """Exploratory: §7f model on all split seasons plus season-trend interactions for the b2b terms."""
    complete = games[["rest_bucket_home", "rest_bucket_away", "travel_km_away", "tz_shift_home"]].notna().all(axis=1)
    part = games[complete]
    X = schedule_variables(part)
    trend = part["SEASON"].str[:4].astype(int) - TREND_CENTER
    X["away_b2b_x_season"] = X["away_b2b"] * trend
    X["home_b2b_x_season"] = X["home_b2b"] * trend
    model, diag = fit(part[TARGET].astype(int), pd.concat([X, team_season_dummies(part)], axis=1))
    terms = odds_rows(model, ["away_b2b", "home_b2b", "away_b2b_x_season", "home_b2b_x_season"])

    sit = situation(games)
    win = games[TARGET].astype(float)
    raw = pd.DataFrame({
        "games": games.groupby("SEASON").size(),
        "both_rested": win[sit == SITUATIONS[1]].groupby(games["SEASON"]).mean(),
        "away_only_b2b": win[sit == SITUATIONS[0]].groupby(games["SEASON"]).mean(),
        "n_away_only": (sit == SITUATIONS[0]).groupby(games["SEASON"]).sum(),
        "home_only_b2b": win[sit == SITUATIONS[3]].groupby(games["SEASON"]).mean(),
        "n_home_only": (sit == SITUATIONS[3]).groupby(games["SEASON"]).sum(),
    })
    raw["gap_away_b2b"] = raw["away_only_b2b"] - raw["both_rested"]
    raw["gap_home_b2b"] = raw["home_only_b2b"] - raw["both_rested"]
    return {"n": int(len(part)), "excluded_missing": int((~complete).sum()), "fit": diag, "terms": terms}, raw



def main():
    games, feature_columns = load_games(WINDOW, schedule=True)
    logs = team_games(load_game_logs())
    market, table_report = load_market(logs)
    games = games.merge(market.drop(columns=["SEASON"]), on="GAME_ID", how="left", validate="one_to_one")
    games["has_market"] = games["p_market"].notna()

    match = games.groupby("split").agg(split_games=("GAME_ID", "size"), matched=("odds_table", lambda s: s.notna().sum()),
                                       with_moneylines=("has_market", "sum"),
                                       with_line=("home_line", lambda s: s.notna().sum())).reindex(SPLITS)
    match["match_rate"] = match["matched"] / match["split_games"]
    agree = games.loc[games["odds_table"].notna()]
    win_check = float(((agree["home_margin_logs"] > 0).astype(int) == agree[TARGET].astype(int)).mean())

    bench, bench_lines, ci_lines = benchmark(games, feature_columns)

    offsite = offsite_game_ids(logs)
    bubble = (games["SEASON"] == "2019-20") & (pd.to_datetime(games["Date"]) >= BUBBLE_START)
    sited = games[~bubble & ~games["GAME_ID"].isin(offsite)]
    excluded = {"bubble": int(bubble.sum()), "offsite": int(games["GAME_ID"].isin(offsite).sum())}
    logits, logit_lines = situation_logits(sited)
    covers, cover_lines = cover_rates(sited)
    trend, raw_gap = season_trend(sited)

    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    table_report.to_csv(MARKET_DIR / "odds_tables.csv", index=False)
    match.to_csv(MARKET_DIR / "match_rates.csv")
    raw_gap.to_csv(MARKET_DIR / "b2b_gap_by_season.csv")
    (MARKET_DIR / "market.json").write_text(json.dumps({
        "match": match.reset_index().to_dict(orient="records"), "target_agrees_with_log_margin": win_check,
        "excluded": excluded, "benchmark": bench, "situation_logit": logits, "cover": covers, "trend": trend,
    }, indent=2, default=float))

    t = trend["terms"]
    report = [
        "# Does the betting market price back-to-backs?", "",
        "## Odds data", "",
        "Tables used per season and how many rows match a logged game (by date and franchise). "
        "`win_margin_agrees` = share of matched rows whose Win_Margin equals the logged home margin.", "",
        table_report.round(4).to_markdown(index=False), "",
        "Split games with market data (unmatched games are dropped from every market comparison):", "",
        match.round(4).to_markdown(), "",
        f"Home-win target vs logged home margin on matched games: {win_check:.4%} agree.", "",
        "## 1. Market benchmark (reference row, not our model)", "",
        "Same games for every row: split games with both moneylines. Probability = proportional devig of "
        "the two moneylines. Saved models only scored.", "",
        "| Model | Split | n | Accuracy | AUC | Log loss | Brier |", "|---|---|---|---|---|---|---|",
        *bench_lines, "",
        f"Paired bootstrap 95% CIs (market − model), {BOOTSTRAP_SAMPLES} resamples:", "",
        "| Comparison | Split | n | Accuracy diff CI | AUC diff CI |", "|---|---|---|---|---|", *ci_lines, "",
        "## 2a. Back-to-back situations controlling for the market", "",
        f"Logistic regression of home win on situation dummies (reference: neither team on a b2b) plus "
        f"logit(market probability), fitted separately per split. Excluded: {excluded['bubble']} restart "
        f"games and {excluded['offsite']} off-site games, and games without both moneylines. Odds "
        "multiplier > 1 = the home team wins more often than the market implies. A slope near 1 on "
        "logit(market) (multiplier near e = 2.718) means the market is calibrated.", "",
        "| Split | Term | Games | Odds multiplier | 95% CI | p |", "|---|---|---|---|---|---|", *logit_lines, "",
        "## 2b. Home cover rate against the spread", "",
        "Cover = logged home margin − market home line > 0; pushes excluded from the rate and counted. "
        f"Exact binomial 95% CI and two-sided test against 50%. Context: Ashman, Bowman & Lambrinos (2010): "
        f"{ASHMAN_2010}.", "",
        "| Split | Situation | Games | No line | Pushes | Decided | Home cover rate | 95% CI | p vs 50% |",
        "|---|---|---|---|---|---|---|---|---|", *cover_lines, "",
        "## 3. Exploratory: has the back-to-back effect changed over time?", "",
        f"Labelled exploratory; not used to change any model. The §7f model (schedule variables + team-by-season "
        f"dummies) on all split seasons (n = {trend['n']:,}; {trend['excluded_missing']} games with missing "
        f"rest or travel dropped), plus b2b × (season start year − {TREND_CENTER}) interactions. Main effects "
        f"are at 2018-19. Fit: converged={trend['fit']['converged']}, rank {trend['fit']['rank']}/"
        f"{trend['fit']['n_columns']}.", "",
        "| Term | Odds multiplier | 95% CI | p |", "|---|---|---|---|",
        *[f"| {k} | {v['odds_multiplier']:.3f} | [{v['ci_low']:.3f}, {v['ci_high']:.3f}] | {v['p_value']:.3f} |"
          for k, v in t.items()], "",
        "Raw home-win rate by season (split games, restart and off-site excluded):", "",
        raw_gap.round(4).to_markdown(),
    ]
    text = "\n".join(report) + "\n"
    (MARKET_DIR / "market.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
