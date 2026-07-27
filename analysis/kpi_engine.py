"""
kpi_engine.py (v10 — Modular Architecture, Timestamped Manifest, Filter-Aware Metrics & HTML Escaping)
---------------------------------------------------------------------------------------------------------
Computes business KPIs, continuous 4-component composite supplier risk scoring,
3-year linear delay trend slopes (beta), dynamic ROP inventory stockout linkages,
Herfindahl-Hirschman Index (HHI) spend concentration, 3-point OLS price inflation regression,
and Procurement Health Score.

Exports pure KPI functions for unit testing and orchestrates end-to-end execution in compute_kpi_summary().
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import sys
import os
import html
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from db.build_db import SUPPLIER_PERFORMANCE_SQL, MONTHLY_TREND_SQL, INVENTORY_RISK_SQL, PRICE_TREND_SQL

DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
OUT_PATH = os.path.join(BASE_DIR, "analysis", "kpi_summary.json")

# Documented Business Weight Rationale for Procurement Health Score
DEFAULT_HEALTH_WEIGHTS = {
    "reliability": 0.25,  # 25% On-time fulfillment & SLA adherence
    "inventory": 0.20,    # 20% Stock cover health & safety stock protection
    "cost": 0.20,         # 20% Inflation containment & price stability
    "delivery": 0.20,     # 20% Low delivery delay rate across purchase orders
    "risk": 0.15          # 15% Minimization of High/Medium risk supplier exposure
}

def min_max_scale(series, min_val=None, max_val=None):
    """
    Scales series between 0.0 and 1.0.
    Supports fixed reference bounds (min_val, max_val) to eliminate global stationarity drift.
    """
    if len(series) == 0:
        return np.array([])
    s_min = series.min() if min_val is None else min_val
    s_max = series.max() if max_val is None else max_val
    if s_max == s_min:
        return np.zeros(len(series))
    scaled = (series - s_min) / (s_max - s_min)
    return np.clip(scaled, 0.0, 1.0)

def classify_trajectory(slope):
    """Classifies fulfillment delay trend direction based on OLS slope (days/month)."""
    if slope > 0.03:
        return "📉 Deteriorating (Delay Escalating)"
    elif slope < -0.03:
        return "📈 Improving (Delay Declining)"
    else:
        return "➡️ Stable Fulfillment"

def compute_delay_and_price_volatility(monthly_sup_df):
    """Computes 3-year delay trend slope (beta) and price volatility CV per supplier."""
    trend_slopes = []
    valid_slopes = []
    valid_cvs = []

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
            
            valid_slopes.append(slope)
            valid_cvs.append(cv)
            trend_slopes.append({
                "supplier_id": sup_id,
                "delay_trend_slope": round(float(slope), 4),
                "price_volatility_cv": round(float(cv), 4)
            })
        else:
            trend_slopes.append({
                "supplier_id": sup_id,
                "delay_trend_slope": np.nan,
                "price_volatility_cv": np.nan
            })

    trend_df = pd.DataFrame(trend_slopes)
    median_slope = float(np.median(valid_slopes)) if valid_slopes else 0.0
    median_cv = float(np.median(valid_cvs)) if valid_cvs else 0.0

    trend_df["delay_trend_slope"].fillna(median_slope, inplace=True)
    trend_df["price_volatility_cv"].fillna(median_cv, inplace=True)
    return trend_df, median_slope, median_cv

def compute_composite_risk(supplier_perf, bounds=None):
    """
    Computes 4-component continuous composite supplier risk score (0 to 100).
    Components:
    - C1 (35%): Late delivery rate (1.0 - on_time_pct/100)
    - C2 (25%): Defect rate (defect_rate_pct/100)
    - C3 (20%): Price volatility CV (scaled)
    - C4 (20%): Delay trend slope (scaled)
    """
    df = supplier_perf.copy()
    c1_late = 1.0 - (df["on_time_pct"] / 100.0)
    c2_defect = df["defect_rate_pct"] / 100.0

    vol_bounds = bounds.get("price_vol") if bounds else (None, None)
    slope_bounds = bounds.get("delay_slope") if bounds else (None, None)

    c3_price_vol = min_max_scale(df["price_volatility_cv"], min_val=vol_bounds[0], max_val=vol_bounds[1])
    c4_trend_slope = min_max_scale(df["delay_trend_slope"], min_val=slope_bounds[0], max_val=slope_bounds[1])

    df["composite_risk_index"] = (
        (0.35 * c1_late) +
        (0.25 * c2_defect) +
        (0.20 * c3_price_vol) +
        (0.20 * c4_trend_slope)
    ) * 100.0

    df["composite_risk_index"] = df["composite_risk_index"].round(1)
    df["trajectory_direction"] = df["delay_trend_slope"].apply(classify_trajectory)

    q_high_score = df["composite_risk_index"].quantile(0.70)
    q_low_score  = df["composite_risk_index"].quantile(0.25)

    def assign_risk_tier(score):
        if score >= q_high_score:
            return "High Risk"
        elif score >= q_low_score:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_tier"] = df["composite_risk_index"].apply(assign_risk_tier)

    def assign_primary_driver(row):
        late_c = 0.35 * (1.0 - row["on_time_pct"]/100.0)
        qual_c = 0.25 * (row["defect_rate_pct"]/100.0)
        vol_c  = 0.20 * row["price_volatility_cv"]
        trend_c = 0.20 * max(row["delay_trend_slope"], 0)
        
        max_c = max(late_c, qual_c, vol_c, trend_c)
        if max_c == late_c:
            return "Reliability Risk (Late Delivery)"
        elif max_c == qual_c:
            return "Quality Risk (High Defect)"
        elif max_c == trend_c:
            return "Trajectory Risk (Deteriorating Trend)"
        else:
            return "Price Volatility Risk"

    df["primary_risk_axis"] = df.apply(assign_primary_driver, axis=1)
    return df

def compute_spend_hhi(supplier_perf):
    """Computes Herfindahl-Hirschman Index (HHI) spend concentration metric."""
    df = supplier_perf.copy()
    total_spend = df["total_spend"].sum()
    df["spend_share_pct"] = (df["total_spend"] / total_spend) * 100.0 if total_spend > 0 else 0.0
    hhi_score = (df["spend_share_pct"] ** 2).sum()

    top5_share = df.sort_values("total_spend", ascending=False).head(5)["spend_share_pct"].sum()
    top10_share = df.sort_values("total_spend", ascending=False).head(10)["spend_share_pct"].sum()

    return {
        "hhi_score": round(float(hhi_score), 1),
        "hhi_classification": "Moderate Market Concentration" if hhi_score >= 1000 else "Unconcentrated (Healthy Competition)",
        "top_5_spend_share_pct": round(float(top5_share), 1),
        "top_10_spend_share_pct": round(float(top10_share), 1),
    }

def compute_price_ols_inflation(price_trend):
    """Computes 3-point OLS price inflation regression across all suppliers."""
    price_trend_pivot = price_trend.pivot(index=["supplier_id", "supplier_name"], columns="year", values="avg_unit_price")
    price_slopes = []

    for (sup_id, sup_name), row in price_trend_pivot.iterrows():
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
            "supplier_id": sup_id,
            "supplier_name": sup_name,
            "ols_price_slope_inr_per_yr": round(float(slope), 2),
            "annualized_inflation_trend_pct": round(float(annual_pct), 1)
        })

    return pd.DataFrame(price_slopes).sort_values("annualized_inflation_trend_pct", ascending=False)

def compute_dynamic_rop_inventory(inv_df, supplier_perf):
    """Computes dynamic lead-time inflated Reorder Point (ROP) and single-source dependency risk."""
    df = inv_df.merge(
        supplier_perf[["supplier_id", "risk_tier", "avg_delay_days", "composite_risk_index", "trajectory_direction"]],
        on="supplier_id", how="left"
    )

    df["avg_daily_demand"] = df["avg_monthly_demand"] / 30.0
    df["inflated_lead_time_days"] = df["lead_time_days_base"] + df["avg_delay_days"].fillna(0.0)
    df["dynamic_reorder_point"] = (df["inflated_lead_time_days"] * df["avg_daily_demand"] + df["safety_stock"]).round(1)

    def compute_dynamic_status(row):
        if row["avg_monthly_demand"] == 0:
            return "Dead Stock"
        elif row["current_stock"] < row["dynamic_reorder_point"]:
            return "Understocked"
        elif row["months_of_cover"] > 6.0:
            return "Overstocked"
        else:
            return "Healthy"

    df["stock_status"] = df.apply(compute_dynamic_status, axis=1)

    cat_sup_counts = df.groupby("category")["supplier_id"].nunique().to_dict()
    df["category_supplier_count"] = df["category"].map(cat_sup_counts)
    df["is_single_source_dependent"] = (
        (df["category_supplier_count"] <= 3) | 
        (df["risk_tier"] == "High Risk")
    ).astype(int)

    understocked_df = df[df["stock_status"] == "Understocked"]
    understocked_high_risk = understocked_df[understocked_df["risk_tier"] == "High Risk"]
    single_source_high_risk = df[(df["is_single_source_dependent"] == 1) & (df["risk_tier"] == "High Risk")]

    high_risk_spend = supplier_perf[supplier_perf["risk_tier"] == "High Risk"]["total_spend"].sum()
    total_spend_all = supplier_perf["total_spend"].sum()

    inventory_summary = {
        "total_products": len(df),
        "healthy_skus": int((df["stock_status"] == "Healthy").sum()),
        "understocked_skus": len(understocked_df),
        "understocked_high_risk_skus": len(understocked_high_risk),
        "understocked_high_risk_pct": round(float(len(understocked_high_risk) / len(understocked_df) * 100), 1) if len(understocked_df) > 0 else 0.0,
        "single_source_dependent_skus": int(df["is_single_source_dependent"].sum()),
        "single_source_high_risk_skus": len(single_source_high_risk),
        "high_risk_supplier_spend": round(float(high_risk_spend), 2),
        "high_risk_spend_share_pct": round(float(high_risk_spend / total_spend_all * 100), 1) if total_spend_all > 0 else 0.0,
    }

    return df, inventory_summary

def compute_health_score(components, weights=None):
    """Computes overall weighted Procurement Health Score using configurable weights dictionary."""
    if weights is None:
        weights = DEFAULT_HEALTH_WEIGHTS

    score = (
        components.get("supplier_reliability", 0.0) * weights.get("reliability", 0.25)
        + components.get("inventory_efficiency", 0.0) * weights.get("inventory", 0.20)
        + components.get("cost_optimisation", 0.0) * weights.get("cost", 0.20)
        + components.get("delivery_performance", 0.0) * weights.get("delivery", 0.20)
        + components.get("risk_score", 0.0) * weights.get("risk", 0.15)
    )
    return round(float(score), 1)

def generate_executive_narrative(supplier_perf, spend_concentration, inventory_exposure_summary, worst_rel, worst_trend):
    """Generates HTML executive summary narrative with strict character escaping."""
    safe_rel_name = html.escape(str(worst_rel['supplier_name']))
    safe_rel_tier = html.escape(str(worst_rel['tier']))
    safe_trend_name = html.escape(str(worst_trend['supplier_name']))

    hhi_score = html.escape(str(spend_concentration['hhi_score']))
    hhi_class = html.escape(str(spend_concentration['hhi_classification']))

    narrative = (
        f"<b>Executive Operational Summary &amp; Supply Chain Risk Audit</b><br><br>"
        f"• <b>Market Concentration &amp; Spend Exposure</b>: Multi-factor risk scoring evaluates overall market concentration at HHI <b>{hhi_score}</b> ({hhi_class}). However, <b>&#8377;{inventory_exposure_summary['high_risk_supplier_spend']/1e7:.1f} Cr ({inventory_exposure_summary['high_risk_spend_share_pct']}%)</b> of total spend remains tied to High-Risk suppliers.<br>"
        f"• <b>Reliability Bottleneck</b>: <b>{safe_rel_name}</b> ({safe_rel_tier}) represents critical SLA exposure with an on-time fulfillment rate of only <b>{worst_rel['on_time_pct']:.1f}%</b>.<br>"
        f"• <b>Performance Decay Trajectory</b>: <b>{safe_trend_name}</b> exhibits severe monthly SLA deterioration with a delay trend slope of <b>+{worst_trend['delay_trend_slope']:.4f} days/month</b>.<br>"
        f"• <b>Dynamic ROP &amp; Single-Source Vulnerability</b>: Dynamic ROP calculations flag <b>{inventory_exposure_summary['understocked_high_risk_skus']} of {inventory_exposure_summary['understocked_skus']} understocked SKUs ({(inventory_exposure_summary['understocked_high_risk_skus']/max(inventory_exposure_summary['understocked_skus'],1))*100:.1f}%)</b> as primary-sourced from High-Risk suppliers, with <b>{inventory_exposure_summary['single_source_high_risk_skus']} SKUs</b> exposed to single-source dependency risk."
    )
    return narrative

def compute_kpi_summary(db_path=DB_PATH, out_path=OUT_PATH, weights=None, verbose=True):
    """
    Computes procurement KPIs, supplier risk scores, dynamic ROP stockouts,
    spend concentration HHI, price inflation OLS, and exports kpi_summary.json.
    """
    if weights is None:
        weights = DEFAULT_HEALTH_WEIGHTS

    conn = sqlite3.connect(db_path)

    supplier_perf = pd.read_sql(SUPPLIER_PERFORMANCE_SQL, conn)
    monthly_trend = pd.read_sql(MONTHLY_TREND_SQL, conn)
    price_trend   = pd.read_sql(PRICE_TREND_SQL, conn)

    # 1. 3-YEAR DELAY TREND SLOPE & PRICE VOLATILITY COMPUTATION
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
    trend_df, median_slope, median_cv = compute_delay_and_price_volatility(monthly_sup_df)

    supplier_perf = supplier_perf.merge(trend_df, on="supplier_id", how="left")
    supplier_perf["delay_trend_slope"].fillna(median_slope, inplace=True)
    supplier_perf["price_volatility_cv"].fillna(median_cv, inplace=True)

    # 2. CONTINUOUS 4-COMPONENT COMPOSITE RISK SCORE FORMULA
    supplier_perf = compute_composite_risk(supplier_perf)

    # 3. SPEND CONCENTRATION ANALYSIS (HERFINDAHL-HIRSCHMAN INDEX - HHI)
    spend_concentration = compute_spend_hhi(supplier_perf)

    # 4. 3-POINT OLS PRICE INFLATION REGRESSION (2023-2025) across ALL suppliers
    price_ols_df = compute_price_ols_inflation(price_trend)
    supplier_perf = supplier_perf.merge(
        price_ols_df[["supplier_id", "ols_price_slope_inr_per_yr", "annualized_inflation_trend_pct"]],
        on="supplier_id", how="left"
    )

    # 5. DYNAMIC REORDER POINT (ROP) & SINGLE-SOURCE DEPENDENCY RISK
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
    inv_raw = pd.read_sql(inv_enhanced_sql, conn)
    inv_df, inventory_exposure_summary = compute_dynamic_rop_inventory(inv_raw, supplier_perf)

    # 6. PROCUREMENT HEALTH SCORE
    supplier_reliability = supplier_perf["on_time_pct"].mean()
    inv_counts = inv_df["stock_status"].value_counts(normalize=True) * 100
    inventory_efficiency = 100 - inv_counts.get("Overstocked", 0) - inv_counts.get("Dead Stock", 0) - inv_counts.get("Understocked", 0) * 0.75

    avg_inflation = price_ols_df["annualized_inflation_trend_pct"].mean() if not price_ols_df.empty else 0
    cost_optimisation = float(np.clip(100 - max(avg_inflation, 0) * 2, 0, 100))

    delivery_performance = round(
        (1.0 - pd.read_sql("SELECT AVG(is_late) AS r FROM deliveries", conn).iloc[0]["r"]) * 100.0, 1
    )

    risk_counts = supplier_perf["risk_tier"].value_counts(normalize=True) * 100
    risk_score = 100 - risk_counts.get("High Risk", 0.0) * 2.0 - risk_counts.get("Medium Risk", 0.0) * 0.75

    components = {
        "supplier_reliability": round(float(supplier_reliability), 1),
        "inventory_efficiency": round(float(inventory_efficiency), 1),
        "cost_optimisation": round(float(cost_optimisation), 1),
        "delivery_performance": round(float(delivery_performance), 1),
        "risk_score": round(float(np.clip(risk_score, 0, 100)), 1),
    }

    overall_health = compute_health_score(components, weights=weights)

    # Executive narrative with HTML escaping
    worst_rel = supplier_perf.sort_values("on_time_pct").iloc[0]
    worst_trend = supplier_perf.sort_values("delay_trend_slope", ascending=False).iloc[0]

    narrative = generate_executive_narrative(
        supplier_perf=supplier_perf,
        spend_concentration=spend_concentration,
        inventory_exposure_summary=inventory_exposure_summary,
        worst_rel=worst_rel,
        worst_trend=worst_trend
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "overall_health_score": overall_health,
        "health_score_weights": weights,
        "components": components,
        "composite_risk_scoring_formula": {
            "equation": "Composite Risk Index = 0.35 * C_late + 0.25 * C_defect + 0.20 * C_price_vol + 0.20 * C_delay_trend_slope",
            "percentile_thresholds": "High Risk (Top 30% riskiest), Medium Risk (Middle 45%), Low Risk (Bottom 25%)"
        },
        "spend_concentration_hhi": spend_concentration,
        "price_ols_inflation_trend": price_ols_df.to_dict(orient="records"),
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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"Overall Procurement Health Score: {overall_health}/100")
        print(f"Spend Concentration HHI: {spend_concentration['hhi_score']} ({spend_concentration['hhi_classification']})")
        print(f"Saved Dynamic Inventory ROP & HHI Spend Summary to {out_path}")

    conn.close()

if __name__ == "__main__":
    compute_kpi_summary()
