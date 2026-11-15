"""Build the Modified Implementation comparison table from committed results. Trains and scores nothing.

    python scripts/comparison_table.py

Reads results/pregame/enhanced/metrics_w20.json, results/pregame/enhanced2/metrics_w20.json and
results/pregame/enhanced2/holdout.json. Writes docs/report/comparison-table.md and
docs/report/comparison-table.html (open the HTML in a browser and copy the tables into Word).

Gain rule (docs/enhanced-plan.md, docs/enhanced-plan-2.md): the paired-bootstrap 95% CI for the same
metric excludes 0 in the positive direction on both test and validation. Only accuracy and AUC have
CIs; log loss and Brier are reported without them, as the plans fixed.
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1 = json.loads((ROOT / "results/pregame/enhanced/metrics_w20.json").read_text())
R2 = json.loads((ROOT / "results/pregame/enhanced2/metrics_w20.json").read_text())
HOLDOUT = json.loads((ROOT / "results/pregame/enhanced2/holdout.json").read_text())
OUT_MD = ROOT / "docs/report/comparison-table.md"
OUT_HTML = ROOT / "docs/report/comparison-table.html"

# (key, label, round, source). Round 1 = findings §11, round 2 = findings §12.
ROWS = [
    ("repo", "Baseline: repo season-to-date", "ref", R1),
    ("pregame", "Pre-game w20 (official)", "ref", R1),
    ("elo_logit", "Elo only (logistic on Elo difference)", "1", R1),
    ("pregame_elo", "Pre-game w20 + Elo", "1", R1),
    ("repo_elo", "Repo + Elo", "1", R1),
    ("pregame_sched_travel_elo", "Pre-game + fatigue + travel + Elo", "1", R1),
    ("stack", "Stack: repo, pre-game, pre-game + Elo", "1", R1),
    ("pregame_w5", "Ablation: pre-game window 5", "1", R1),
    ("pregame_w7", "Ablation: pre-game window 7", "1", R1),
    ("pregame_elo2", "Pre-game + Elo variant", "2", R2),
    ("pregame_elo2_player", "Primary (round 2): pre-game + Elo variant + player", "2", R2),
    ("repo_elo2_player", "Repo + Elo variant + player", "2", R2),
    ("stack2", "Stack 2: repo, pre-game, primary", "2", R2),
    ("roll_repo", "Rolling retraining: baseline", "2", R2),
    ("roll_pregame_elo2_player", "Rolling retraining: primary", "2", R2),
    ("roll_stack2", "Rolling retraining: stack 2", "2", R2),
]
MARKET_LABEL = "Market line, devigged moneyline (reference, not our model)"
HOLDOUT_LABELS = {
    "pregame": "Pre-game w20 (official)",
    "pregame_elo2_player": "Primary: pre-game + Elo variant + player",
    "roll_pregame_elo2_player": "Rolling retraining: primary",
}
METRICS = [("accuracy", "Acc"), ("auc", "AUC"), ("log_loss", "Log loss"), ("brier", "Brier")]
SPLITS = [("test", "Test"), ("validation", "Val")]


def f4(value):
    return f"{value:.4f}"


def fmt_ci(ci):
    return "[" + ", ".join(("+" if v >= 0 else "−") + f"{abs(v):.4f}" for v in ci) + "]"


def comparison(key, ref, source):
    return source["comparisons"].get(f"{key} vs {ref}")


def passes(comp, metric):
    return all(comp[split][f"{metric}_diff_ci"][0] > 0 for split, _ in SPLITS)


def verdict(comp):
    if comp is None:
        return "not computed"
    won = [name for metric, name in (("auc", "AUC"), ("accuracy", "accuracy")) if passes(comp, metric)]
    return "Yes: " + " and ".join(won) if won else "No"


def metrics_table():
    head = ["Row", "Round"] + [f"{s} {m}" for _, s in SPLITS for _, m in METRICS]
    body = []
    for key, label, rnd, source in ROWS:
        values = source["metrics"][key]
        body.append([label, "—" if rnd == "ref" else rnd]
                    + [f4(values[split][metric]) for split, _ in SPLITS for metric, _ in METRICS])
    market = R1["metrics"]["market"]
    body.append([MARKET_LABEL, "ref"] + [f4(market[split][metric]) for split, _ in SPLITS for metric, _ in METRICS])
    return head, body


def ci_table(ref):
    head = ["Row", "Test acc diff", "Test AUC diff", "Val acc diff", "Val AUC diff", "Passes gain rule?"]
    body = []
    for key, label, rnd, source in ROWS:
        if rnd == "ref" or key == ref:
            continue
        comp = comparison(key, ref, source)
        if comp is None:
            body.append([label, "—", "—", "—", "—", "not computed"])
            continue
        body.append([label] + [fmt_ci(comp[split][f"{metric}_diff_ci"])
                               for split, _ in SPLITS for metric in ("accuracy", "auc")] + [verdict(comp)])
    return head, body


def holdout_tables():
    m_head = ["Row"] + [name for _, name in METRICS]
    m_body = [[HOLDOUT_LABELS[k]] + [f4(v[metric]) for metric, _ in METRICS] for k, v in HOLDOUT["metrics"].items()]
    c_head = ["Comparison", "Accuracy diff", "AUC diff"]
    c_body = []
    for name, comp in HOLDOUT["comparisons"].items():
        row, ref = name.split(" vs ")
        c_body.append([f"{HOLDOUT_LABELS[row]} − {HOLDOUT_LABELS[ref]}",
                       fmt_ci(comp["accuracy_diff_ci"]), fmt_ci(comp["auc_diff_ci"])])
    return (m_head, m_body), (c_head, c_body)


def market_notes():
    lines = []
    for source, key in ((R1, "market vs stack"), (R2, "market vs stack2")):
        comp = source["comparisons"][key]
        label = key.replace("market vs ", "market − ").replace("stack2", "stack 2")
        lines.append(f"{label}: AUC {fmt_ci(comp['test']['auc_diff_ci'])} test, "
                     f"{fmt_ci(comp['validation']['auc_diff_ci'])} validation; accuracy "
                     f"{fmt_ci(comp['test']['accuracy_diff_ci'])} test, "
                     f"{fmt_ci(comp['validation']['accuracy_diff_ci'])} validation.")
    return lines


def md_table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def html_table(caption, head, body):
    cells = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in body)
    return f"<table><caption>{html.escape(caption)}</caption><thead><tr>{cells}</tr></thead><tbody>{rows}</tbody></table>"


def main():
    n_test = R1["metrics"]["repo"]["test"]["n"]
    n_val = R1["metrics"]["repo"]["validation"]["n"]
    (hm, hc) = holdout_tables()
    dates = HOLDOUT["dates"]
    sections = [
        ("Table A. Every row on the same test and validation games",
         f"Test n = {n_test:,}, validation n = {n_val:,}. Round 1 is findings §11, round 2 is findings §12. "
         "Rolling rows refit monthly on all earlier games, including earlier test and validation games, "
         "with parameters fixed on train; nothing is tuned on held-out data.",
         metrics_table()),
        ("Table B. Paired-bootstrap 95% CIs of the difference: row − baseline (repo)",
         "2,000 resamples, seed 42. The last column applies the gain rule to accuracy and AUC.",
         ci_table("repo")),
        ("Table C. Paired-bootstrap 95% CIs of the difference: row − official pre-game w20",
         "Rolling rows were not compared with the pre-game model in the plan, so those cells are not computed.",
         ci_table("pregame")),
        (f"Table D. Fresh holdout, single look: {HOLDOUT['n']} games, {dates[0]} to {dates[1]}",
         f"Scored once under rules fixed before scoring. Home-win rate {HOLDOUT['home_win_rate']:.4f}. "
         "The baseline and market can't be scored here (their data stops at 2026-01-07). "
         "Don't compare these levels with Table A; this period scores higher for every row.",
         hm),
        ("Table E. Fresh holdout differences (single look)",
         "The one confirmatory comparison fixed before scoring: primary − official on AUC. The rest are secondary.",
         hc),
    ]
    rule = ("**Gain rule** (fixed in `docs/enhanced-plan.md` and `docs/enhanced-plan-2.md` before any result): "
            "the 95% CI for the same metric excludes 0 in the positive direction on **both** test and validation. "
            "Only accuracy and AUC have CIs; log loss and Brier are reported without CIs, as the plans fixed. "
            "The round 2 primary test was `pregame_elo2_player` against the baseline. Round 2 is not blind: "
            "round 1 results motivated it.")
    md = ["# Modified Implementation: comparison table", "",
          "Generated by `python scripts/comparison_table.py` from committed results. Do not edit by hand.", "",
          rule, ""]
    for title, note, (head, body) in sections:
        md += [f"## {title}", ""] + ([note, ""] if note else []) + [md_table(head, body), ""]
    md += ["## Market line (reference only)", ""] + [f"- {line}" for line in market_notes()] + [
        "", "Whether this market line is a closing line can't be confirmed from this repo.", ""]
    OUT_MD.write_text("\n".join(md))

    style = ("body{font-family:'Times New Roman',serif;font-size:10pt}table{border-collapse:collapse;margin:12pt 0}"
             "caption{font-weight:bold;text-align:left;padding-bottom:4pt}th,td{border:1px solid #000;"
             "padding:2pt 4pt;text-align:center}td:first-child,th:first-child{text-align:left}p{max-width:60em}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Comparison table</title><style>{style}</style>"
             "</head><body>", "<p>Copy a table from this page and paste it into Word.</p>"]
    for title, note, (head, body) in sections:
        parts.append(html_table(title, head, body))
        if note:
            parts.append(f"<p>{html.escape(note)}</p>")
    parts.append("<p>" + html.escape(rule.replace("**", "").replace("`", "")) + "</p>")
    parts += [f"<p>{html.escape(line)}</p>" for line in market_notes()]
    parts.append("</body></html>")
    OUT_HTML.write_text("".join(parts))
    print(f"Wrote {OUT_MD.relative_to(ROOT)} and {OUT_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
