# Fresh holdout: games after the split

766 games, 2026-01-08 to 2026-06-13. Home-win rate 0.5561. Rules fixed in `docs/enhanced-plan-2.md` before scoring. No baseline or market: their data stops at 2026-01-07.

| Row | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|
| Pre-game w20 (official) | 0.6749 | 0.7393 | 0.6083 | 0.2096 |
| Pre-game + Elo variant + player | 0.7076 | 0.7632 | 0.5836 | 0.1988 |
| Rolling primary | 0.6997 | 0.7644 | 0.5819 | 0.1981 |

## Paired bootstrap 95% CIs (row − reference), 2000 resamples

| Row | Reference | Accuracy diff CI | AUC diff CI |
|---|---|---|---|
| Pre-game + Elo variant + player | Pre-game w20 (official) | [+0.0065, +0.0587] **excl. 0** | [+0.0054, +0.0420] **excl. 0** |
| Rolling primary | Pre-game w20 (official) | [-0.0013, +0.0509] | [+0.0075, +0.0429] **excl. 0** |
| Rolling primary | Pre-game + Elo variant + player | [-0.0196, +0.0039] | [-0.0028, +0.0051] |
