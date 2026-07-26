"""
kpi_engine.py (v6 — Dynamic Inventory ROP, Spend Concentration HHI & OLS Price Inflation)
---------------------------------------------------------------------------------------
Computes business KPIs, continuous 4-component composite supplier risk scoring,
3-year linear delay trend slopes (beta), dynamic ROP inventory stockout linkages,
Herfindahl-Hirschman Index (HHI) spend concentration, 3-point OLS price inflation regression,
and Health Score.

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
price_trend   = pd.read_sql(PRICE_TREND_SQL, conn)

# ---------------------------------------------------------------
# 1. 3-YEAR DELAY TREND SLOPE & PRICE VOLATILITY COMPUTATION
# ---------------------------------------------------------------
monthly_sup_sql = """
SELECT 
    s.supplier_id, s.supplier_name,
    strftime('%Y-%m', po.order_date) AS order_month,
    COUNT(po.po_id) AS total_pos,
    SUM(d.is_late) * 1.0 / COUNT(po.po_id) AS monthly_late_rate,
    AVG(d.delay_days) AS monthly_avg_delay,
    AVG(po.unit_price) AS monthly_avg_price
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY s.supplier_id, order_month
ORDER BY s.supplier_id, order_month
"""

monthly_sup_df = pd.read_sql(monthly_sup_sql, conn)

trend_slopes = []
for sup_id, group in monthly_sup_df.groupby("supplier_id"):
    group = group.sort_values("order_month").reset_index(drop=True)
    n = len(group)
    if n >= 6:
        x = np.arange(n)
        y_delay = group["monthly_avg_delay"].values
        slope, _ = np.polyfit(x, y_delay, 1)
        p_mean = group["monthly_avg_price"].mean()
        p_std = group["monthly_avg_price"].std()
        cv = (p_std / p_mean) if p_mean > 0 else 0.0
    else:
        slope = 0.0
        cv = 0.0
        
    trend_slopes.append({
        "supplier_id": sup_id,
        "delay_trend_slope": round(float(slope), 4),
        "price_volatility_cv": round(float(cv), 4)
    })

trend_df = pd.DataFrame(trend_slopes)

supplier_perf = supplier_perf.merge(trend_df, on="supplier_id", how="left")
supplier_perf["delay_trend_slope"].fillna(0.0, inplace=True)
supplier_perf["price_volatility_cv"].fillna(0.0, inplace=True)

# ---------------------------------------------------------------
# 2. CONTINUOUS 4-COMPONENT COMPOSITE RISK SCORE FORMULA
# ---------------------------------------------------------------
def min_max_scale(series):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros(len(series))
    return (series - s_min) / (s_max - s_min)

c1_late = 1.0 - (supplier_perf["on_time_pct"] / 100.0)
c2_defect = supplier_perf["defect_rate_pct"] / 100.0
c3_price_vol = min_max_scale(supplier_perf["price_volatility_cv"])
c4_trend_slope = min_max_scale(supplier_perf["delay_trend_slope"])

supplier_perf["composite_risk_index"] = (
    (0.35 * c1_late) +
    (0.25 * c2_defect) +
    (0.20 * c3_price_vol) +
    (0.20 * c4_trend_slope)
) * 100.0

supplier_perf["composite_risk_index"] = supplier_perf["composite_risk_index"].round(1)

def classify_trajectory(slope):
    if slope > 0.03:
        return "📉 Deteriorating (Delay Escalating)"
    elif slope < -0.03:
        return "📈 Improving (Delay Declining)"
    else:
        return "➡️ Stable Fulfillment"

supplier_perf["trajectory_direction"] = supplier_perf["delay_trend_slope"].apply(classify_trajectory)

q_high_score = supplier_perf["composite_risk_index"].quantile(0.70)
q_low_score  = supplier_perf["composite_risk_index"].quantile(0.25)

def assign_composite_risk_tier(score):
    if score >= q_high_score:
        return "High Risk"
    elif score >= q_low_score:
        return "Medium Risk"
    else:
        return "Low Risk"

supplier_perf["risk_tier"] = supplier_perf["composite_risk_index"].apply(assign_composite_risk_tier)

def assign_primary_driver(row):
    late_contrib = 0.35 * (1.0 - row["on_time_pct"]/100.0)
    qual_contrib = 0.25 * (row["defect_rate_pct"]/100.0)
    vol_contrib  = 0.20 * row["price_volatility_cv"]
    trend_contrib = 0.20 * max(row["delay_trend_slope"], 0)
    
    max_c = max(late_contrib, qual_contrib, vol_contrib, trend_contrib)
    if max_c == late_contrib:
        return "Reliability Risk (Late Delivery)"
    elif max_c == qual_contrib:
        return "Quality Risk (High Defect)"
    elif max_c == trend_contrib:
        return "Trajectory Risk (Deteriorating Trend)"
    else:
        return "Price Volatility Risk"

supplier_perf["primary_risk_axis"] = supplier_perf.apply(assign_primary_driver, axis=1)

# ---------------------------------------------------------------
# 3. SPEND CONCENTRATION ANALYSIS (HERFINDAHL-HIRSCHMAN INDEX - HHI)
# ---------------------------------------------------------------
total_spend = supplier_perf["total_spend"].sum()
supplier_perf["spend_share_pct"] = (supplier_perf["total_spend"] / total_spend) * 100.0
hhi_score = (supplier_perf["spend_share_pct"] ** 2).sum()

top5_spend_share = supplier_perf.sort_values("total_spend", ascending=False).head(5)["spend_share_pct"].sum()
top10_spend_share = supplier_perf.sort_values("total_spend", ascending=False).head(10)["spend_share_pct"].sum()

spend_concentration = {
    "hhi_score": round(float(hhi_score), 1),
    "hhi_classification": "Moderate Market Concentration" if hhi_score >= 1000 else "Unconcentrated (Healthy Competition)",
    "top_5_spend_share_pct": round(float(top5_spend_share), 1),
    "top_10_spend_share_pct": round(float(top10_spend_share), 1),
}

# ---------------------------------------------------------------
# 4. 3-POINT OLS PRICE INFLATION REGRESSION (2023-2025)
# ---------------------------------------------------------------
price_trend_pivot = price_trend.pivot(index="supplier_name", columns="year", values="avg_unit_price")
price_slopes = []

for sup_name, row in price_trend_pivot.iterrows():
    vals = row.dropna().values
    if len(vals) >= 2:
        x = np.arange(len(vals))
        slope, _ = np.polyfit(x, vals, 1)
        mean_p = np.mean(vals)
        annual_pct = (slope / mean_p * 100.0) if mean_p > 0 else 0.0
    else:
        slope = 0.0
        annual_pct = 0.0
    price_slopes.append({
        "supplier_name": sup_name,
        "ols_price_slope_inr_per_yr": round(float(slope), 2),
        "annualized_inflation_trend_pct": round(float(annual_pct), 1)
    })

price_ols_df = pd.DataFrame(price_slopes).sort_values("annualized_inflation_trend_pct", ascending=False)
supplier_perf = supplier_perf.merge(price_ols_df, on="supplier_name", how="left")

# ---------------------------------------------------------------
# 5. DYNAMIC REORDER POINT (ROP) & SINGLE-SOURCE DEPENDENCY RISK
# ---------------------------------------------------------------
inv_enhanced_sql = """
SELECT 
    i.product_id, p.product_name, p.category, p.primary_supplier_id AS supplier_id,
    p.lead_time_days_base, s.supplier_name, s.tier, s.region,
    i.current_stock, i.reorder_level, i.avg_monthly_demand,
    ROUND(i.reorder_level * 0.35, 1) AS safety_stock, i.months_of_cover
FROM inventory i
JOIN products p ON i.product_id = p.product_id
JOIN suppliers s ON p.primary_supplier_id = s.supplier_id
"""

inv_df = pd.read_sql(inv_enhanced_sql, conn)
inv_df = inv_df.merge(
    supplier_perf[["supplier_id", "risk_tier", "avg_delay_days", "composite_risk_index", "trajectory_direction"]],
    on="supplier_id", how="left"
)

# Effective Lead Time = Base Lead Time + Supplier Avg Delay Days
inv_df["avg_daily_demand"] = inv_df["avg_monthly_demand"] / 30.0
inv_df["inflated_lead_time_days"] = inv_df["lead_time_days_base"] + inv_df["avg_delay_days"].fillna(0.0)
inv_df["dynamic_reorder_point"] = (inv_df["inflated_lead_time_days"] * inv_df["avg_daily_demand"] + inv_df["safety_stock"]).round(1)

def compute_dynamic_status(row):
    if row["avg_monthly_demand"] == 0:
        return "Dead Stock"
    elif row["current_stock"] < row["dynamic_reorder_point"]:
        return "Understocked"
    elif row["months_of_cover"] > 6.0:
        return "Overstocked"
    else:
        return "Healthy"

inv_df["stock_status"] = inv_df.apply(compute_dynamic_status, axis=1)

# Single-Source Dependency Risk Flag: SKUs where category has <= 3 suppliers OR primary supplier is High Risk
cat_sup_counts = inv_df.groupby("category")["supplier_id"].nunique().to_dict()
inv_df["category_supplier_count"] = inv_df["category"].map(cat_sup_counts)
inv_df["is_single_source_dependent"] = (
    (inv_df["category_supplier_count"] <= 3) | 
    (inv_df["risk_tier"] == "High Risk")
).astype(int)

understocked_df = inv_df[inv_df["stock_status"] == "Understocked"]
understocked_high_risk = understocked_df[understocked_df["risk_tier"] == "High Risk"]
single_source_high_risk = inv_df[(inv_df["is_single_source_dependent"] == 1) & (inv_df["risk_tier"] == "High Risk")]

high_risk_spend = supplier_perf[supplier_perf["risk_tier"] == "High Risk"]["total_spend"].sum()
total_spend_all = supplier_perf["total_spend"].sum()

inventory_exposure_summary = {
    "total_products": len(inv_df),
    "healthy_skus": int((inv_df["stock_status"] == "Healthy").sum()),
    "understocked_skus": len(understocked_df),
    "understocked_high_risk_skus": len(understocked_high_risk),
    "understocked_high_risk_pct": round(float(len(understocked_high_risk) / len(understocked_df) * 100), 1) if len(understocked_df) > 0 else 0.0,
    "single_source_dependent_skus": int(inv_df["is_single_source_dependent"].sum()),
    "single_source_high_risk_skus": len(single_source_high_risk),
    "high_risk_supplier_spend": round(float(high_risk_spend), 2),
    "high_risk_spend_share_pct": round(float(high_risk_spend / total_spend_all * 100), 1) if total_spend_all > 0 else 0.0,
}

# ---------------------------------------------------------------
# 6. PROCUREMENT HEALTH SCORE (Weighted Composite)
# ---------------------------------------------------------------
supplier_reliability = supplier_perf["on_time_pct"].mean()
inv_counts = inv_df["stock_status"].value_counts(normalize=True) * 100
inventory_efficiency = 100 - inv_counts.get("Overstocked", 0) - inv_counts.get("Dead Stock", 0) - inv_counts.get("Understocked", 0) * 0.75

avg_inflation = price_ols_df["annualized_inflation_trend_pct"].mean() if not price_ols_df.empty else 0
cost_optimisation = float(np.clip(100 - max(avg_inflation, 0) * 2, 0, 100))

delivery_performance = round(
    (1.0 - pd.read_sql("SELECT AVG(is_late) AS r FROM deliveries", conn).iloc[0]["r"]) * 100.0, 1
)

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
worst_trend = supplier_perf.sort_values("delay_trend_slope", ascending=False).iloc[0]

narrative = (
    f"Advanced 4-component risk scoring and spend concentration (HHI: {spend_concentration['hhi_score']}) reveal critical vulnerabilities. "
    f"{worst_rel['supplier_name']} (Tier: {worst_rel['tier']}) represents extreme Reliability Risk ({worst_rel['on_time_pct']}% on-time). "
    f"Conversely, {worst_trend['supplier_name']} exhibits a severe Deteriorating Trajectory (slope: +{worst_trend['delay_trend_slope']} days/mo). "
    f"Crucially, dynamic ROP calculations flag {inventory_exposure_summary['understocked_high_risk_skus']} of {inventory_exposure_summary['understocked_skus']} understocked SKUs as primary-sourced from High-Risk suppliers, with {inventory_exposure_summary['single_source_high_risk_skus']} SKUs exposed to single-source dependency."
)

summary = {
    "overall_health_score": overall_health,
    "components": components,
    "composite_risk_scoring_formula": {
        "equation": "Composite Risk Index = 0.35 * C_late + 0.25 * C_defect + 0.20 * C_price_vol + 0.20 * C_delay_trend_slope",
        "percentile_thresholds": "High Risk (Top 30% riskiest), Medium Risk (Middle 45%), Low Risk (Bottom 25%)"
    },
    "spend_concentration_hhi": spend_concentration,
    "price_ols_inflation_trend": price_ols_df.head(10).to_dict(orient="records"),
    "inventory_exposure": inventory_exposure_summary,
    "top_suppliers": supplier_perf.sort_values("composite_risk_index").head(5).to_dict(orient="records"),
    "bottom_suppliers": supplier_perf.sort_values("composite_risk_index", ascending=False).head(5).to_dict(orient="records"),
    "worst_quality_suppliers": supplier_perf.sort_values("defect_rate_pct", ascending=False).head(5).to_dict(orient="records"),
    "worst_trend_suppliers": supplier_perf.sort_values("delay_trend_slope", ascending=False).head(5).to_dict(orient="records"),
    "risk_distribution": supplier_perf["risk_tier"].value_counts().to_dict(),
    "monthly_trend": monthly_trend.to_dict(orient="records"),
    "inventory_status": inv_df["stock_status"].value_counts().to_dict(),
    "narrative_example": narrative,
    "total_spend": round(float(supplier_perf["total_spend"].sum()), 2),
    "total_orders": int(supplier_perf["total_orders"].sum()),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Overall Procurement Health Score: {overall_health}/100")
print(f"Spend Concentration HHI: {spend_concentration['hhi_score']} ({spend_concentration['hhi_classification']})")
print(f"Saved Dynamic Inventory ROP & HHI Spend Summary to {OUT_PATH}")

conn.close()
