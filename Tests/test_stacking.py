import unittest

import numpy as np

from src.Pregame.stacking import fit_meta, predict_meta, walk_forward_oof


class StackingTest(unittest.TestCase):
    def test_each_prediction_uses_only_earlier_rows(self):
        seen = []

        def fit_predict(train_idx, val_idx):
            seen.append((train_idx.copy(), val_idx.copy()))
            return np.full(len(val_idx), 0.5)

        oof = walk_forward_oof(60, fit_predict)
        self.assertEqual(len(seen), 5)
        for train_idx, val_idx in seen:
            self.assertLess(train_idx.max(), val_idx.min())
        self.assertTrue(np.isnan(oof[0]))
        self.assertFalse(np.isnan(oof[-1]))

    def test_later_labels_do_not_change_earlier_predictions(self):
        y = np.random.default_rng(0).integers(0, 2, 60).astype(float)

        def oof_for(labels):
            return walk_forward_oof(60, lambda tr, va: np.full(len(va), labels[tr].mean()))

        base = oof_for(y)
        changed = y.copy()
        changed[40:] = 1 - changed[40:]
        np.testing.assert_array_equal(base[:40], oof_for(changed)[:40])

    def test_a_leaky_splitter_would_be_refused(self):
        def fit_predict(train_idx, val_idx):
            return np.zeros(len(val_idx))

        walk_forward_oof(12, fit_predict, n_splits=2)
        with self.assertRaises(ValueError):
            walk_forward_oof(5, fit_predict, n_splits=10)

    def test_meta_fit_skips_rows_without_predictions(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 100)
        oof = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.1, (2, 100)).T.mean(axis=1), 0.05, 0.95)
        matrix = np.column_stack([oof, oof[::-1]])
        matrix[:20] = np.nan
        meta, usable = fit_meta(matrix, y)
        self.assertEqual(int(usable.sum()), 80)
        self.assertEqual(predict_meta(meta, matrix[20:]).shape, (80,))


if __name__ == "__main__":
    unittest.main()
