"""
kpi_engine.py
--------------
Computes all business KPIs and the Procurement Health Score from the
SQLite database. This is the analytical core of the project — pure
pandas/SQL, no ML — meant to demonstrate the "data analyst" skillset
independent of the ML add-on.

Outputs kpi_summary.json, consumed by the dashboard.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from db.build_db import SUPPLIER_PERFORMANCE_SQL, MONTHLY_TREND_SQL, INVENTORY_RISK_SQL, PRICE_TREND_SQL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
OUT_PATH = os.path.join(BASE_DIR, "analysis", "kpi_summary.json")

conn = sqlite3.connect(DB_PATH)

supplier_perf = pd.read_sql(SUPPLIER_PERFORMANCE_SQL, conn)
monthly_trend = pd.read_sql(MONTHLY_TREND_SQL, conn)
inventory_risk = pd.read_sql(INVENTORY_RISK_SQL, conn)
price_trend = pd.read_sql(PRICE_TREND_SQL, conn)

# ---------------------------------------------------------------
# Supplier Risk Classification (rule-based, not ML — kept separate
# from the ML risk model so there's a transparent baseline to compare)
# ---------------------------------------------------------------
def classify_risk(row):
    score = 0
    if row["on_time_pct"] < 75: score += 2
    elif row["on_time_pct"] < 90: score += 1
    if row["defect_rate_pct"] > 5: score += 2
    elif row["defect_rate_pct"] > 2: score += 1
    if row["avg_delay_days"] > 3: score += 1
    if row["tier"] == "Tier 3": score += 1.5 # Tier 3 suppliers are inherently riskier
    if row["region"].startswith("Import"): score += 0.5 # Imports add a slight risk factor

    if score >= 4: return "High Risk"
    if score >= 2: return "Medium Risk"
    return "Low Risk"

supplier_perf["risk_tier"] = supplier_perf.apply(classify_risk, axis=1)

# ---------------------------------------------------------------
# Price trend / inflation flag: compare avg price across years
# ---------------------------------------------------------------
pivot = price_trend.pivot(index="supplier_name", columns="year", values="avg_unit_price")
# Calculate year-over-year percentage change for all available years
price_changes = {}
years = sorted(pivot.columns.dropna().unique())
for i in range(len(years) - 1):
    year1 = str(years[i])
    year2 = str(years[i+1])
    col_name = f"pct_change_{year1}_to_{year2}"
    pivot[col_name] = ((pivot.get(year2, np.nan) - pivot.get(year1, np.nan)) / pivot.get(year1, np.nan) * 100).round(1)

# Use the latest year's change for overall inflation assessment
latest_pct_change_col = f"pct_change_{years[-2]}_to_{years[-1]}" if len(years) >= 2 else None
price_flags = pivot.reset_index()
if latest_pct_change_col:
    price_flags = price_flags[["supplier_name", latest_pct_change_col]].dropna().rename(columns={latest_pct_change_col: "latest_pct_change"}).sort_values("latest_pct_change", ascending=False)
else:
    price_flags = pd.DataFrame(columns=["supplier_name", "latest_pct_change"])

# ---------------------------------------------------------------
# Procurement Health Score (weighted composite, documented formula)
# ---------------------------------------------------------------
supplier_reliability = supplier_perf["on_time_pct"].mean()

inv_counts = inventory_risk["stock_status"].value_counts(normalize=True) * 100
inventory_efficiency = 100 - inv_counts.get("Overstocked", 0) - inv_counts.get("Dead Stock", 0) - inv_counts.get("Understocked", 0) * 0.75 # Increased penalty for understocked

avg_inflation = price_flags["latest_pct_change"].mean() if not price_flags.empty else 0
cost_optimisation = float(np.clip(100 - max(avg_inflation, 0) * 2, 0, 100)) # Adjusted multiplier

delivery_performance = supplier_perf["on_time_pct"].mean() # Using same as reliability for now, can be refined

risk_counts = supplier_perf["risk_tier"].value_counts(normalize=True) * 100
risk_score = 100 - risk_counts.get("High Risk", 0) * 2 - risk_counts.get("Medium Risk", 0) * 0.75 # Increased penalty

weights = {"reliability": 0.25, "inventory": 0.20, "cost": 0.20, "delivery": 0.20, "risk": 0.15}
components = {
    "supplier_reliability": round(float(supplier_reliability), 1),
    "inventory_efficiency": round(float(inventory_efficiency), 1),
    "cost_optimisation": round(float(cost_optimisation), 1),
    "delivery_performance": round(float(delivery_performance), 1),
    "risk_score": round(float(np.clip(risk_score, 0, 100)), 1),
}
overall_health = round(
    components["supplier_reliability"] * weights["reliability"]
    + components["inventory_efficiency"] * weights["inventory"]
    + components["cost_optimisation"] * weights["cost"]
    + components["delivery_performance"] * weights["delivery"]
    + components["risk_score"] * weights["risk"], 1
)

# ---------------------------------------------------------------
# Business narrative candidates: worst supplier vs best comparable alternative
# ---------------------------------------------------------------
worst = supplier_perf.sort_values("on_time_pct").iloc[0]
candidates = supplier_perf[(supplier_perf["on_time_pct"] > 90) & (supplier_perf["tier"] == "Tier 1")].sort_values("avg_order_value", ascending=True)
best_alt = candidates.iloc[0] if not candidates.empty else supplier_perf.sort_values("on_time_pct", ascending=False).iloc[0]

narrative = (
    f"{worst['supplier_name']} (Tier: {worst['tier']}) has an on-time delivery rate of {worst['on_time_pct']}%, "
    f"the lowest among active suppliers, with an average delay of {worst['avg_delay_days']} days "
    f"across {worst['total_orders']} orders (₹{worst['total_spend']:,.0f} total spend). "
    f"Consider shifting high-priority orders to {best_alt['supplier_name']} (Tier: {best_alt['tier']}), which maintains a "
    f"{best_alt['on_time_pct']}% on-time rate with an average order value of ₹{best_alt['avg_order_value']:,.0f}."
)

summary = {
    "overall_health_score": overall_health,
    "components": components,
    "top_suppliers": supplier_perf.sort_values("on_time_pct", ascending=False).head(5).to_dict(orient="records"),
    "bottom_suppliers": supplier_perf.sort_values("on_time_pct").head(5).to_dict(orient="records"),
    "risk_distribution": supplier_perf["risk_tier"].value_counts().to_dict(),
    "monthly_trend": monthly_trend.to_dict(orient="records"),
    "inventory_status": inventory_risk["stock_status"].value_counts().to_dict(),
    "price_inflation_flags": price_flags.to_dict(orient="records"),
    "narrative_example": narrative,
    "total_spend": round(float(supplier_perf["total_spend"].sum()), 2),
    "total_orders": int(supplier_perf["total_orders"].sum()),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Overall Procurement Health Score: {overall_health}/100")
print(json.dumps(components, indent=2))
print(f"\nSaved KPI summary to {OUT_PATH}")
print(f"\nExample narrative:\n{narrative}")

conn.close()
