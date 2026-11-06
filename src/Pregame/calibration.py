"""Reliability and calibration error from saved predictions (docs/enhanced-plan.md). Nothing is refit.

Run src.Pregame.enhanced first; it writes the predictions this module reads.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from src.Pregame.leakage_audit import BOOTSTRAP_SAMPLES  # noqa: E402
from src.Pregame.paths import ENHANCED_RESULTS_DIR, WORK_DIR  # noqa: E402
from src.Pregame.stacking import logit  # noqa: E402
from src.Pregame.train_xgb import EVAL_SPLITS, TARGET  # noqa: E402

ROWS = ["repo", "pregame", "pregame_sched_travel", "pregame_elo", "stack", "market"]
LABELS = {"repo": "Baseline", "pregame": "Pre-game w20", "pregame_sched_travel": "+ fatigue + travel",
          "pregame_elo": "+ Elo", "stack": "Stack", "market": "Market (reference)"}
N_BINS = 10


def reliability(y, p, n_bins=N_BINS):
    """Equal-width bins: games, mean predicted and observed home-win rate for each non-empty bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    predicted = np.bincount(bins, p, minlength=n_bins)
    observed = np.bincount(bins, y, minlength=n_bins)
    used = counts > 0
    return pd.DataFrame({"bin": np.arange(n_bins)[used], "n": counts[used], "mean_predicted": predicted[used] / counts[used],
                         "observed": observed[used] / counts[used]})


def expected_calibration_error(y, p, n_bins=N_BINS):
    table = reliability(y, p, n_bins)
    return float((table["n"] / table["n"].sum() * (table["mean_predicted"] - table["observed"]).abs()).sum())


def intercept_slope(y, p):
    model = sm.Logit(np.asarray(y, dtype=float), sm.add_constant(logit(p))).fit(disp=False)
    return {"intercept": float(model.params[0]), "slope": float(model.params[1])}


def ece_diff_ci(y, p, reference, seed=0):
    """95% CI for ECE(p) - ECE(reference), resampling the same games for both."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(expected_calibration_error(y[idx], p[idx]) - expected_calibration_error(y[idx], reference[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def plot(tables):
    fig, axes = plt.subplots(2, len(ROWS), figsize=(13, 4.8), sharex=True, sharey=True)
    for i, split in enumerate(EVAL_SPLITS):
        for j, row in enumerate(ROWS):
            ax, table = axes[i, j], tables[split][row]
            ax.plot([0, 1], [0, 1], color="#9a9a9a", linewidth=1)
            ax.plot(table["mean_predicted"], table["observed"], color="#2a5d9c", linewidth=2, marker="o", markersize=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(LABELS[row], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{split}\nobserved home-win rate", fontsize=9)
            if i == 1:
                ax.set_xlabel("predicted", fontsize=9)
    fig.tight_layout()
    fig.savefig(ENHANCED_RESULTS_DIR / "reliability.png", dpi=150)
    plt.close(fig)


def main():
    frame = pd.read_csv(WORK_DIR / "enhanced_predictions_w20.csv", dtype={"GAME_ID": str})
    results, lines, ci_lines, tables = {}, [], [], {}
    for split in EVAL_SPLITS:
        part = frame[frame["split"] == split]
        y = part[TARGET].to_numpy()
        for row in ROWS:
            p = part[f"p_{row}"].to_numpy()
            table = reliability(y, p)
            tables.setdefault(split, {})[row] = table
            entry = {"n": len(y), "ece": expected_calibration_error(y, p), **intercept_slope(y, p),
                     "bins": table.to_dict(orient="records")}
            results.setdefault(split, {})[row] = entry
            lines.append(f"| {LABELS[row]} | {split} | {len(y)} | {entry['ece']:.4f} | {entry['intercept']:+.3f} | {entry['slope']:.3f} |")
        repo, market = part["p_repo"].to_numpy(), part["p_market"].to_numpy()
        for row in ROWS:
            p = part[f"p_{row}"].to_numpy()
            if row != "repo":
                ci = ece_diff_ci(y, p, repo)
                results[split][row]["ece_minus_repo_ci"] = ci
                ci_lines.append(f"| {LABELS[row]} − Baseline | {split} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
            if row != "market":
                ci = ece_diff_ci(y, market, p)
                results[split][row]["market_minus_row_ece_ci"] = ci
                ci_lines.append(f"| Market − {LABELS[row]} | {split} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    ENHANCED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot(tables)
    (ENHANCED_RESULTS_DIR / "calibration.json").write_text(json.dumps(results, indent=2, default=float))
    report = ["# Calibration on test and validation", "",
              f"From saved predictions; nothing is refit. ECE uses {N_BINS} equal-width bins weighted by games. Intercept and "
              "slope come from a logistic regression of home win on logit(p); a calibrated model has 0 and 1. Plot: "
              "`reliability.png`.", "",
              "| Model | Split | n | ECE | Intercept | Slope |", "|---|---|---|---|---|---|", *lines, "",
              f"## Paired bootstrap 95% CIs of the ECE difference, {BOOTSTRAP_SAMPLES} resamples", "",
              "| Difference | Split | CI |", "|---|---|---|", *ci_lines]
    text = "\n".join(report) + "\n"
    (ENHANCED_RESULTS_DIR / "calibration.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
