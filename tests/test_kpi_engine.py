"""
test_kpi_engine.py — Unit Test Suite for KPI & Analytics Engine
----------------------------------------------------------------
Tests core KPI calculation functions: min_max_scale, trajectory classification,
HHI spend concentration, health score calculation, and HTML escaping.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analysis.kpi_engine import (
    min_max_scale,
    classify_trajectory,
    compute_spend_hhi,
    compute_health_score,
    generate_executive_narrative
)

class TestKPIEngine(unittest.TestCase):

    def test_min_max_scale(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        scaled = min_max_scale(s)
        self.assertAlmostEqual(scaled.min(), 0.0)
        self.assertAlmostEqual(scaled.max(), 1.0)
        self.assertAlmostEqual(scaled.iloc[2], 0.5)

    def test_min_max_scale_with_fixed_bounds(self):
        s = pd.Series([15.0, 25.0])
        scaled = min_max_scale(s, min_val=10.0, max_val=50.0)
        self.assertAlmostEqual(scaled.iloc[0], (15.0 - 10.0) / 40.0)
        self.assertAlmostEqual(scaled.iloc[1], (25.0 - 10.0) / 40.0)

    def test_classify_trajectory(self):
        self.assertIn("Deteriorating", classify_trajectory(0.05))
        self.assertIn("Improving", classify_trajectory(-0.05))
        self.assertIn("Stable", classify_trajectory(0.01))

    def test_compute_spend_hhi(self):
        df_perf = pd.DataFrame({
            "supplier_id": ["S1", "S2", "S3", "S4"],
            "total_spend": [500.0, 300.0, 150.0, 50.0]
        })
        res = compute_spend_hhi(df_perf)
        self.assertIn("hhi_score", res)
        self.assertGreater(res["hhi_score"], 0)
        self.assertIn("hhi_classification", res)

    def test_compute_health_score(self):
        components = {
            "supplier_reliability": 80.0,
            "inventory_efficiency": 75.0,
            "cost_optimisation": 90.0,
            "delivery_performance": 85.0,
            "risk_score": 70.0
        }
        weights = {
            "reliability": 0.25,
            "inventory": 0.20,
            "cost": 0.20,
            "delivery": 0.20,
            "risk": 0.15
        }
        expected = (80*0.25) + (75*0.20) + (90*0.20) + (85*0.20) + (70*0.15)
        score = compute_health_score(components, weights)
        self.assertAlmostEqual(score, round(expected, 1))

    def test_generate_executive_narrative_escaping(self):
        supplier_perf = pd.DataFrame()
        spend_concentration = {"hhi_score": 1200.0, "hhi_classification": "Moderate Concentration"}
        inventory_exposure = {
            "high_risk_supplier_spend": 15000000.0,
            "high_risk_spend_share_pct": 15.0,
            "understocked_high_risk_skus": 3,
            "understocked_skus": 10,
            "single_source_high_risk_skus": 2
        }
        worst_rel = {"supplier_name": "Acme & Sons <Ltd>", "tier": "Tier 1", "on_time_pct": 45.0}
        worst_trend = {"supplier_name": "Globex > Corporation", "delay_trend_slope": 0.12}

        narrative = generate_executive_narrative(
            supplier_perf, spend_concentration, inventory_exposure, worst_rel, worst_trend
        )

        self.assertIn("Acme &amp; Sons &lt;Ltd&gt;", narrative)
        self.assertIn("Globex &gt; Corporation", narrative)

if __name__ == "__main__":
    unittest.main()
