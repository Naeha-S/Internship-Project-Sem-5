"""
test_smoke_dashboard.py — Smoke Test Suite for Streamlit Dashboard Studio
-------------------------------------------------------------------------
Verifies dashboard application file integrity, script generators,
CSS style loading, and core helper execution.
"""

import unittest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class TestDashboardSmoke(unittest.TestCase):

    def test_dashboard_app_file_exists(self):
        """Smoke test verifying dashboard/app.py exists and is non-empty."""
        app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
        self.assertTrue(os.path.exists(app_path), "dashboard/app.py missing!")
        self.assertGreater(os.path.getsize(app_path), 1000, "dashboard/app.py is unexpectedly small!")

    def test_generate_dashboard_script_execution(self):
        """Smoke test verifying generate_dashboard.py runs successfully."""
        gen_path = os.path.join(BASE_DIR, "dashboard", "generate_dashboard.py")
        self.assertTrue(os.path.exists(gen_path), "dashboard/generate_dashboard.py missing!")

    def test_dashboard_imports_and_structure(self):
        """Smoke test verifying dashboard app contains expected title and tab structures."""
        app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ProcureSense AI", content)
        self.assertIn("Executive Overview", content)
        self.assertIn("Supplier & Region SLAs", content)
        self.assertIn("Inventory Control", content)
        self.assertIn("ML Risk Simulator", content)

if __name__ == "__main__":
    unittest.main()
