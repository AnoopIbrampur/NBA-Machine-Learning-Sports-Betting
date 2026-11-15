"""Rolling retraining: monthly refits on all earlier games with fixed, train-only parameters."""
import numpy as np
import pandas as pd
import xgboost as xgb


def monthly_segments(dates, eval_mask):
    """(cutoff, train positions, predict positions) for each month after the first evaluated game."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    eval_mask = np.asarray(eval_mask)
    held = dates[eval_mask]
    cutoff = (held.min() + pd.offsets.MonthBegin(1)).normalize()
    segments = []
    while cutoff <= held.max():
        following = cutoff + pd.offsets.MonthBegin(1)
        predict = np.flatnonzero(eval_mask & (dates >= cutoff).to_numpy() & (dates < following).to_numpy())
        if len(predict):
            segments.append((cutoff, np.flatnonzero((dates < cutoff).to_numpy()), predict))
        cutoff = following
    return segments


def rolling_predictions(games, columns, y, tuned, static_proba, eval_mask):
    """Start from the full-train model's predictions; replace each month with that month's refit."""
    X = games[columns].astype(float)
    proba = np.array(static_proba, dtype=float)
    for _, train_idx, predict_idx in monthly_segments(games["Date"], eval_mask):
        booster = xgb.train(tuned["params"], xgb.DMatrix(X.iloc[train_idx], label=y[train_idx]),
                            num_boost_round=tuned["num_boost_round"])
        proba[predict_idx] = booster.predict(xgb.DMatrix(X.iloc[predict_idx]))
    return proba
