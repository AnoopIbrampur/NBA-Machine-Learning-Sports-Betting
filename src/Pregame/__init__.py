"""DS 340W pre-game prediction pipeline.

Run from the repo root, in order:
    python -m src.Pregame.make_split_keys --split-dir <dir with train/test/validation.csv>  (one-off)
    python -m src.Pregame.game_logs
    python -m src.Pregame.features --window 10
    python -m src.Pregame.train_xgb --window 10
    python -m src.Pregame.shap_analysis --window 10
"""
