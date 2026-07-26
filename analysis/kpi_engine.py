"""
kpi_engine.py (v3 — Rigorous Risk Scoring & Inventory-Supplier Linkage)
-----------------------------------------------------------------------
Computes business KPIs, multi-axis risk scoring, inventory stockout linkages,
and Procurement Health Score from SQLite procurement.db.

Outputs kpi_summary.json, consumed by dashboard and executive reports.
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

DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
OUT_PATH = os.path.join(BASE_DIR, "analysis", "kpi_summary.json")

conn = sqlite3.connect(DB_PATH)

supplier_perf = pd.read_sql(SUPPLIER_PERFORMANCE_SQL, conn)
monthly_trend = pd.read_sql(MONTHLY_TREND_SQL, conn)
inventory_risk = pd.read_sql(INVENTORY_RISK_SQL, conn)
price_trend = pd.read_sql(PRICE_TREND_SQL, conn)

# ---------------------------------------------------------------
# 1. MULTI-AXIS SUPPLIER RISK SCORING METHODOLOGY
# ---------------------------------------------------------------
def classify_supplier_risk_axes(row):
    """
    Transparent, weighted point-scoring methodology across 3 risk axes:
    1. Delivery Reliability Axis: On-time % & delay severity
    2. Quality Exposure Axis: Defect rate %
    3. Operational Risk Axis: Tier & Geographic Region
    """
    pts = 0.0
    rel_risk = False
    qual_risk = False

    # Delivery Reliability Axis
    if row["on_time_pct"] < 75.0:
        pts += 2.0
        rel_risk = True
    elif row["on_time_pct"] < 85.0:
        pts += 1.0
        rel_risk = True

    if row["avg_delay_days"] > 3.0:
        pts += 1.0
        rel_risk = True

    # Quality Exposure Axis
    if row["defect_rate_pct"] > 5.0:
        pts += 2.0
        qual_risk = True
    elif row["defect_rate_pct"] > 2.0:
        pts += 1.0
        qual_risk = True

    # Operational & Structural Risk Axis
    if row["tier"] == "Tier 3":
        pts += 1.0
    if row["region"].startswith("Import"):
        pts += 0.5

    # Overall Risk Tier Thresholds
    if pts >= 4.0:
        overall_tier = "High Risk"
    elif pts >= 2.0:
        overall_tier = "Medium Risk"
    else:
        overall_tier = "Low Risk"

    # Primary Risk Driver Axis Classification
    if rel_risk and qual_risk:
        primary_axis = "Dual Risk (Reliability + Quality)"
    elif rel_risk:
        primary_axis = "Reliability Risk (Delivery SLA)"
    elif qual_risk:
        primary_axis = "Quality Risk (Shipment Defect)"
    else:
        primary_axis = "Low Operational Risk"

    return pd.Series({
        "risk_points": round(pts, 1),
        "risk_tier": overall_tier,
        "primary_risk_axis": primary_axis,
        "is_reliability_risk": rel_risk,
        "is_quality_risk": qual_risk
    })

risk_details = supplier_perf.apply(classify_supplier_risk_axes, axis=1)
supplier_perf = pd.concat([supplier_perf, risk_details], axis=1)

# ---------------------------------------------------------------
# 2. PRICE INFLATION / VOLATILITY ASSESSMENT
# ---------------------------------------------------------------
pivot = price_trend.pivot(index="supplier_name", columns="year", values="avg_unit_price")
years = sorted(pivot.columns.dropna().unique())

for i in range(len(years) - 1):
    year1 = str(years[i])
    year2 = str(years[i+1])
    col_name = f"pct_change_{year1}_to_{year2}"
    pivot[col_name] = ((pivot.get(year2, np.nan) - pivot.get(year1, np.nan)) / pivot.get(year1, np.nan) * 100).round(1)

latest_col = f"pct_change_{years[-2]}_to_{years[-1]}" if len(years) >= 2 else None
price_flags = pivot.reset_index()
if latest_col:
    price_flags = price_flags[["supplier_name", latest_col]].dropna().rename(columns={latest_col: "latest_pct_change"}).sort_values("latest_pct_change", ascending=False)
else:
    price_flags = pd.DataFrame(columns=["supplier_name", "latest_pct_change"])

# Merge latest price change back into supplier_perf
supplier_perf = supplier_perf.merge(price_flags, on="supplier_name", how="left")
supplier_perf["latest_pct_change"].fillna(0.0, inplace=True)

# ---------------------------------------------------------------
# 3. EXPLICIT INVENTORY STOCKOUT & HIGH-RISK SUPPLIER LINKAGE
# ---------------------------------------------------------------
inv_linked_sql = """
SELECT 
    i.product_id, p.product_name, p.category, p.primary_supplier_id AS supplier_id,
    s.supplier_name, s.tier, s.region,
    i.current_stock, i.reorder_level, i.avg_monthly_demand, i.months_of_cover,
    CASE
        WHEN i.avg_monthly_demand = 0 THEN 'Dead Stock'
        WHEN i.current_stock < i.reorder_level THEN 'Understocked'
        WHEN i.months_of_cover > 6 THEN 'Overstocked'
        ELSE 'Healthy'
    END AS stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id
JOIN suppliers s ON p.primary_supplier_id = s.supplier_id
"""

inv_linked_df = pd.read_sql(inv_linked_sql, conn)
inv_linked_df = inv_linked_df.merge(
    supplier_perf[["supplier_id", "risk_tier", "primary_risk_axis"]],
    on="supplier_id", how="left"
)

understocked_df = inv_linked_df[inv_linked_df["stock_status"] == "Understocked"]
understocked_high_risk = understocked_df[understocked_df["risk_tier"] == "High Risk"]
understocked_medium_risk = understocked_df[understocked_df["risk_tier"] == "Medium Risk"]

high_risk_spend = supplier_perf[supplier_perf["risk_tier"] == "High Risk"]["total_spend"].sum()
total_spend_all = supplier_perf["total_spend"].sum()

inventory_exposure_summary = {
    "total_products": len(inv_linked_df),
    "healthy_skus": int((inv_linked_df["stock_status"] == "Healthy").sum()),
    "understocked_skus": len(understocked_df),
    "understocked_high_risk_skus": len(understocked_high_risk),
    "understocked_high_risk_pct": round(float(len(understocked_high_risk) / len(understocked_df) * 100), 1) if len(understocked_df) > 0 else 0.0,
    "high_risk_supplier_spend": round(float(high_risk_spend), 2),
    "high_risk_spend_share_pct": round(float(high_risk_spend / total_spend_all * 100), 1) if total_spend_all > 0 else 0.0,
    "reliability_risk_suppliers": int((supplier_perf["primary_risk_axis"].str.contains("Reliability")).sum()),
    "quality_risk_suppliers": int((supplier_perf["primary_risk_axis"].str.contains("Quality")).sum()),
}

# ---------------------------------------------------------------
# 4. PROCUREMENT HEALTH SCORE (Weighted Composite)
# ---------------------------------------------------------------
supplier_reliability = supplier_perf["on_time_pct"].mean()
inv_counts = inventory_risk["stock_status"].value_counts(normalize=True) * 100
inventory_efficiency = 100 - inv_counts.get("Overstocked", 0) - inv_counts.get("Dead Stock", 0) - inv_counts.get("Understocked", 0) * 0.75

avg_inflation = price_flags["latest_pct_change"].mean() if not price_flags.empty else 0
cost_optimisation = float(np.clip(100 - max(avg_inflation, 0) * 2, 0, 100))

delivery_performance = supplier_perf["on_time_pct"].mean()

risk_counts = supplier_perf["risk_tier"].value_counts(normalize=True) * 100
risk_score = 100 - risk_counts.get("High Risk", 0) * 2 - risk_counts.get("Medium Risk", 0) * 0.75

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

# Executive narrative
worst_rel = supplier_perf.sort_values("on_time_pct").iloc[0]
worst_qual = supplier_perf.sort_values("defect_rate_pct", ascending=False).iloc[0]
candidates = supplier_perf[(supplier_perf["on_time_pct"] > 85) & (supplier_perf["risk_tier"] == "Low Risk")].sort_values("avg_order_value")
best_alt = candidates.iloc[0] if not candidates.empty else supplier_perf.sort_values("on_time_pct", ascending=False).iloc[0]

narrative = (
    f"Delivery reliability and quality failures stem from distinct supplier cohorts. "
    f"{worst_rel['supplier_name']} (Tier: {worst_rel['tier']}) is the weakest SLA performer with a {worst_rel['on_time_pct']}% on-time rate and {worst_rel['avg_delay_days']} avg delay days. "
    f"Conversely, {worst_qual['supplier_name']} represents a Quality Risk axis with a {worst_qual['defect_rate_pct']}% defect rate despite decent timelines. "
    f"Crucially, {inventory_exposure_summary['understocked_high_risk_skus']} of the {inventory_exposure_summary['understocked_skus']} understocked SKUs ({inventory_exposure_summary['understocked_high_risk_pct']}%) are primary-sourced from High-Risk suppliers, requiring urgent dual-sourcing."
)

summary = {
    "overall_health_score": overall_health,
    "components": components,
    "risk_scoring_methodology": {
        "formula": "Points: [<75% On-Time: +2, <85% On-Time: +1, Defect >5%: +2, Defect >2%: +1, Delay >3d: +1, Tier 3: +1, Import: +0.5]",
        "thresholds": "High Risk >= 4.0, Medium Risk >= 2.0, Low Risk < 2.0",
        "axis_breakdown": {
            "reliability_risk_count": inventory_exposure_summary["reliability_risk_suppliers"],
            "quality_risk_count": inventory_exposure_summary["quality_risk_suppliers"],
        }
    },
    "inventory_exposure": inventory_exposure_summary,
    "top_suppliers": supplier_perf.sort_values("on_time_pct", ascending=False).head(5).to_dict(orient="records"),
    "bottom_suppliers": supplier_perf.sort_values("on_time_pct").head(5).to_dict(orient="records"),
    "worst_quality_suppliers": supplier_perf.sort_values("defect_rate_pct", ascending=False).head(5).to_dict(orient="records"),
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
print(f"Saved rigorous KPI summary & Inventory Linkage to {OUT_PATH}")
print(f"\nExample Narrative:\n{narrative}")

conn.close()
