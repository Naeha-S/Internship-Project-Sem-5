"""
test_ml_pipeline.py — Unit Tests for ML Feature Engineering & Model Evaluation
--------------------------------------------------------------------------------
Tests out-of-fold (OOF) target encoding, multi-model evaluation metrics,
confusion matrix calculations, and cost-optimal threshold search.
"""

import unittest
import pandas as pd
import numpy as np

from ml.delay_prediction import oof_target_encode, evaluate_model_full

class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        """Synthetic DataFrame setup for ML testing."""
        n_rows = 100
        rng = np.random.default_rng(42)
        self.df = pd.DataFrame({
            "po_id": [f"PO-{i:03d}" for i in range(n_rows)],
            "order_year": rng.choice([2023, 2024, 2025], n_rows, p=[0.4, 0.4, 0.2]),
            "supplier_id": rng.choice(["SUP1001", "SUP1002", "SUP1003"], n_rows),
            "product_id": rng.choice(["PRD2001", "PRD2002"], n_rows),
            "is_late": rng.choice([0, 1], n_rows, p=[0.6, 0.4])
        })

    def test_oof_target_encode_output(self):
        """Test out-of-fold target encoding produces bounded values without NaNs."""
        encoded = oof_target_encode(self.df, "supplier_id", target="is_late", n_splits=3)
        self.assertEqual(len(encoded), len(self.df))
        self.assertFalse(encoded.isna().any(), "OOF target encoding returned NaNs!")
        self.assertTrue((encoded >= 0.0).all() and (encoded <= 1.0).all(), "OOF encoded values out of [0, 1] range!")

    def test_evaluate_model_full(self):
        """Test evaluate_model_full computes threshold search, confusion matrix, and cost metrics."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.7, 0.9, 0.85, 0.6, 0.95])

        metrics = evaluate_model_full("Test Model", probs, y_true)

        self.assertEqual(metrics["model_name"], "Test Model")
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("default_thresh_0.5", metrics)
        self.assertIn("cost_optimal", metrics)

        default_cm = metrics["default_thresh_0.5"]["confusion_matrix"]
        self.assertEqual(len(default_cm), 2)
        self.assertEqual(len(default_cm[0]), 2)

        opt_thresh = metrics["cost_optimal"]["optimal_threshold"]
        self.assertTrue(0.10 <= opt_thresh <= 0.90, f"Optimal threshold out of bounds: {opt_thresh}")

if __name__ == "__main__":
    unittest.main()
