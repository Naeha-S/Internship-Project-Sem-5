"""
generate_dashboard.py (v3)
--------------------------
Verifies artifacts and prepares the ProcureSense AI Streamlit BI Dashboard (dashboard/app.py).
"""

import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KPI_PATH = os.path.join(BASE, "analysis", "kpi_summary.json")
MODEL_PATH = os.path.join(BASE, "ml", "model_metrics.json")
APP_PATH = os.path.join(BASE, "dashboard", "app.py")

if not os.path.exists(KPI_PATH):
    raise FileNotFoundError(f"Missing {KPI_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing {MODEL_PATH}")
if not os.path.exists(APP_PATH):
    raise FileNotFoundError(f"Missing {APP_PATH}")

print("✅ Dashboard artifacts verified successfully!")
print("ProcureSense AI BI Studio is ready. Run using: streamlit run dashboard/app.py")
