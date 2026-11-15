"""Walk-forward out-of-fold stacking. The meta-model sees train-split predictions only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk_forward_oof(n_rows, fit_predict, n_splits=5):
    """Out-of-fold predictions for rows in date order; the first block stays NaN.

    fit_predict(train_idx, val_idx) must fit on train_idx and return predictions for val_idx.
    """
    oof = np.full(n_rows, np.nan)
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(np.zeros((n_rows, 1))):
        if train_idx.max() >= val_idx.min():
            raise ValueError("Out-of-fold predictions must come from strictly earlier rows")
        oof[val_idx] = fit_predict(train_idx, val_idx)
    return oof


def fit_meta(oof_matrix, y):
    """Logistic regression on logit(p), fit only on rows with every out-of-fold prediction."""
    usable = ~np.isnan(oof_matrix).any(axis=1)
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof_matrix[usable]), np.asarray(y)[usable]), usable


def predict_meta(meta, proba_matrix):
    return meta.predict_proba(logit(proba_matrix))[:, 1]
