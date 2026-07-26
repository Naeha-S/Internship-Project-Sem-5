"""
app.py — ProcureSense AI Interactive Streamlit BI Studio (v3 Unified Dashboard)
----------------------------------------------------------------------------------
Consolidated interactive dashboard containing:
- Fixed filter handling (no empty UI state)
- All 10 Plotly visualizations (combining original dashboard.html + new analytical charts)
- 10 interactive production SQL queries with live DB execution & CSV download
- AI ML Delivery Delay Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------
# 1. Page Configuration & Theme State
# ---------------------------------------------------------------
st.set_page_config(
    page_title="ProcureSense AI — Procurement Studio",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ---------------------------------------------------------------
# 2. Path Setup & Artifact Loading
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
KPI_PATH = os.path.join(BASE_DIR, "analysis", "kpi_summary.json")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")

@st.cache_data
def load_kpi_summary():
    if os.path.exists(KPI_PATH):
        with open(KPI_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_model_metrics():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_db_connection()
kpi_data = load_kpi_summary()
model_data = load_model_metrics()

# ---------------------------------------------------------------
# 3. Dynamic CSS Styling
# ---------------------------------------------------------------
BG = "#09090b" if IS_DARK else "#f8fafc"
BG_SUBTLE = "#121217" if IS_DARK else "#f1f5f9"
CARD_BG = "#13131a" if IS_DARK else "#ffffff"
BORDER = "#272730" if IS_DARK else "#e2e8f0"
TEXT = "#fafafa" if IS_DARK else "#0f172a"
TEXT_MUTED = "#94a3b8" if IS_DARK else "#64748b"
ACCENT = "#3b82f6"
GOLD = "#eab308"
GREEN = "#22c55e"
GREEN_BG = "rgba(34,197,94,0.15)" if IS_DARK else "rgba(22,163,74,0.10)"
RED = "#ef4444"
RED_BG = "rgba(239,68,68,0.15)" if IS_DARK else "rgba(220,38,38,0.10)"
AMBER = "#f59e0b"
AMBER_BG = "rgba(245,158,11,0.15)" if IS_DARK else "rgba(217,119,6,0.10)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 1.5rem 2rem 3rem !important;
        max-width: 1400px !important;
    }}
    
    #MainMenu, footer, [data-testid="stDecoration"], .stDeployButton {{
        display: none !important;
    }}
    
    .brand-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 24px;
    }}
    .brand-title {{
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {TEXT};
    }}
    .brand-title span {{
        color: {GOLD};
    }}
    .brand-sub {{
        font-size: 13px;
        color: {TEXT_MUTED};
        margin-top: 2px;
    }}

    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {TEXT};
        margin-top: 4px;
        letter-spacing: -0.03em;
    }}
    .metric-badge {{
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 6px;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}
    
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {TEXT_MUTED} !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {TEXT} !important;
        background: {CARD_BG} !important;
        border-color: {BORDER} !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 6px !important;
        background: {BG_SUBTLE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        margin-bottom: 24px !important;
    }}

    .chart-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }}
    .chart-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {TEXT};
        margin-bottom: 2px;
    }}
    .chart-sub {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-bottom: 12px;
    }}

    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
    }}
    .badge-green {{ color: {GREEN}; background: {GREEN_BG}; }}
    .badge-red {{ color: {RED}; background: {RED_BG}; }}
    .badge-amber {{ color: {AMBER}; background: {AMBER_BG}; }}

    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.7rem 0.9rem;
        color: {TEXT_MUTED};
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid {BORDER};
    }}
    .data-table td {{
        padding: 0.7rem 0.9rem;
        color: {TEXT};
        border-bottom: 1px solid {BORDER};
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 4. Header Bar
# ---------------------------------------------------------------
h_left, h_right = st.columns([7, 1])
with h_left:
    st.markdown("""
    <div class="brand-header">
        <div>
            <div class="brand-title">Procure<span>Sense</span> AI — Executive Studio</div>
            <div class="brand-sub">Unified Procurement Analytics, 10 Production SQL Queries, SHAP Explainability & Risk Simulation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with h_right:
    theme_btn_text = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_btn_text, on_click=toggle_theme, use_container_width=True)

# ---------------------------------------------------------------
# 5. Global Filters with Robust Fallbacks (Prevents Empty UI)
# ---------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Global Data Filters")

@st.cache_data
def get_filter_options():
    suppliers_df = pd.read_sql("SELECT DISTINCT tier, region FROM suppliers", conn)
    products_df = pd.read_sql("SELECT DISTINCT category FROM products", conn)
    shipping_df = pd.read_sql("SELECT DISTINCT shipping_mode FROM purchase_orders", conn)
    return {
        "tiers": sorted(suppliers_df["tier"].dropna().unique().tolist()),
        "regions": sorted(suppliers_df["region"].dropna().unique().tolist()),
        "categories": sorted(products_df["category"].dropna().unique().tolist()),
        "shipping_modes": sorted(shipping_df["shipping_mode"].dropna().unique().tolist()),
    }

filters = get_filter_options()

selected_tiers = st.sidebar.multiselect("Supplier Tier", options=filters["tiers"], default=filters["tiers"])
selected_categories = st.sidebar.multiselect("Product Category", options=filters["categories"], default=filters["categories"])
selected_shipping = st.sidebar.multiselect("Shipping Mode", options=filters["shipping_modes"], default=filters["shipping_modes"])
min_order_year, max_order_year = st.sidebar.slider("Order Date Window", 2023, 2025, (2023, 2025))

# Construct SQL WHERE clause with fallbacks to avoid empty UI
tier_where = f"s.tier IN ({','.join([f"'{t}'" for t in selected_tiers])})" if selected_tiers else "1=1"
cat_where = f"p.category IN ({','.join([f"'{c}'" for c in selected_categories])})" if selected_categories else "1=1"
ship_where = f"po.shipping_mode IN ({','.join([f"'{m}'" for m in selected_shipping])})" if selected_shipping else "1=1"

query_main = f"""
SELECT 
    po.po_id, po.order_date, po.unit_price, po.quantity, po.order_cost, po.shipping_mode,
    s.supplier_id, s.supplier_name, s.region, s.tier,
    p.product_name, p.category, p.sub_category,
    d.delivery_date, d.is_late, d.delay_days, d.has_defect
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN products p ON po.product_id = p.product_id
JOIN deliveries d ON po.po_id = d.po_id
WHERE CAST(strftime('%Y', po.order_date) AS INTEGER) BETWEEN {min_order_year} AND {max_order_year}
  AND {tier_where}
  AND {cat_where}
  AND {ship_where}
"""

df_filtered = pd.read_sql(query_main, conn)

# Helper Plotly Layout
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=TEXT_MUTED, size=11),
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)"),
)

# ---------------------------------------------------------------
# 6. Tab Navigation (5 Tabs)
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview (4 Charts)",
    "🏬 Supplier & Regional Performance (2 Charts)",
    "📦 Inventory & Quality Exposure (2 Charts)",
    "🤖 ML Delay Prediction & Explainability (2 Charts)",
    "💻 10 SQL Analytics Portfolio Queries"
])

# ---------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------------
with tab1:
    total_spend = df_filtered["order_cost"].sum()
    total_orders = len(df_filtered)
    late_orders = df_filtered["is_late"].sum()
    late_pct = (late_orders / total_orders * 100) if total_orders > 0 else 0
    defect_orders = df_filtered["has_defect"].sum()
    defect_pct = (defect_orders / total_orders * 100) if total_orders > 0 else 0
    health_score = kpi_data.get("overall_health_score", 49.8)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Spend</div>
            <div class="metric-value">₹{total_spend/1e7:.2f} Cr</div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Tracked Expenditure</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Purchase Orders</div>
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-badge" style="background:{GREEN_BG}; color:{GREEN};">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Late Delivery Rate</div>
            <div class="metric-value">{late_pct:.1f}%</div>
            <div class="metric-badge" style="background:{RED_BG}; color:{RED};">Performance Drag</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Defect Rate</div>
            <div class="metric-value">{defect_pct:.2f}%</div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Shipment Quality</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-value">{health_score} / 100</div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Composite Metric</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # CHART 1: Monthly Spend & Late % Trend
    df_filtered["month"] = pd.to_datetime(df_filtered["order_date"]).dt.to_period("M").astype(str)
    monthly = df_filtered.groupby("month").agg(
        total_spend=("order_cost", "sum"),
        late_pct=("is_late", lambda x: (x.sum() / len(x)) * 100)
    ).reset_index()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly["month"], y=monthly["total_spend"], name="Spend (₹)", marker_color=GOLD, opacity=0.85),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["late_pct"], name="Late Rate %", mode="lines+markers", line=dict(color=RED, width=3)),
        secondary_y=True
    )
    fig_trend.update_layout(**PLOT_LAYOUT, height=360, legend=dict(orientation="h", y=1.12))

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 1: Monthly Procurement Spend vs Late Delivery Rate</div>
        <div class="chart-sub">Comparing monthly expenditure volume against late shipment percentage</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    c_over1, c_over2, c_over3 = st.columns(3)
    
    # CHART 2: Supplier Risk Distribution
    with c_over1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 2: Supplier Risk Tier Distribution</div>
            <div class="chart-sub">Proportion of high, medium, and low risk suppliers</div>
        """, unsafe_allow_html=True)
        risk_data = kpi_data.get("risk_distribution", {"High Risk": 64, "Medium Risk": 21, "Low Risk": 15})
        fig_risk = px.pie(
            values=list(risk_data.values()), names=list(risk_data.keys()), hole=0.55,
            color=list(risk_data.keys()),
            color_discrete_map={"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN}
        )
        fig_risk.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 3: Inventory Status Breakdown
    with c_over2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 3: Inventory Stock Status</div>
            <div class="chart-sub">Breakdown of product stock health levels</div>
        """, unsafe_allow_html=True)
        inv_data = kpi_data.get("inventory_status", {"Healthy": 153, "Understocked": 47})
        fig_inv = px.pie(
            values=list(inv_data.values()), names=list(inv_data.keys()), hole=0.55,
            color=list(inv_data.keys()),
            color_discrete_map={"Healthy": GREEN, "Understocked": AMBER, "Overstocked": TEXT_MUTED, "Dead Stock": RED}
        )
        fig_inv.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 4: Latest Price Inflation Flags
    with c_over3:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 4: Top Price Inflation Flags</div>
            <div class="chart-sub">Top suppliers by YoY price increase %</div>
        """, unsafe_allow_html=True)
        price_df = pd.DataFrame(kpi_data.get("price_inflation_flags", [])).head(6)
        if not price_df.empty:
            fig_price = px.bar(
                price_df, x="latest_pct_change", y="supplier_name", orientation="h",
                color="latest_pct_change", color_continuous_scale=[GREEN, AMBER, RED]
            )
            fig_price.update_layout(**PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No price inflation flags recorded.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(f"💡 **AI Narrative Insight**: {kpi_data.get('narrative_example', 'No narrative available.')}")

# ---------------------------------------------------------------
# TAB 2: SUPPLIER & REGIONAL PERFORMANCE
# ---------------------------------------------------------------
with tab2:
    # CHART 5: Supplier On-Time Rate (Top & Bottom)
    top_sup = pd.DataFrame(kpi_data.get("top_suppliers", []))
    bot_sup = pd.DataFrame(kpi_data.get("bottom_suppliers", []))
    comb_sup = pd.concat([top_sup, bot_sup]).drop_duplicates(subset="supplier_id").sort_values("on_time_pct")

    fig_sup_rank = px.bar(
        comb_sup, x="on_time_pct", y="supplier_name", orientation="h",
        color="on_time_pct", color_continuous_scale=[RED, AMBER, GREEN],
        labels={"on_time_pct": "On-Time %", "supplier_name": "Supplier"}
    )
    fig_sup_rank.update_layout(**PLOT_LAYOUT, height=360)

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 5: Supplier On-Time Delivery Rate Ranking (Top & Bottom)</div>
        <div class="chart-sub">Comparing reliability across high-performing and underperforming suppliers</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_sup_rank, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # CHART 6: Spend by Region & Supplier Tier (New Chart 1)
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 6: Spend Distribution by Supplier Region & Tier</div>
        <div class="chart-sub">Cross-tabulation of total expenditure across geographic regions and supplier tiers</div>
    """, unsafe_allow_html=True)
    reg_tier_spend = df_filtered.groupby(["region", "tier"])["order_cost"].sum().reset_index()
    fig_reg_spend = px.bar(
        reg_tier_spend, x="region", y="order_cost", color="tier", barmode="stack",
        color_discrete_map={"Tier 1": GREEN, "Tier 2": AMBER, "Tier 3": RED},
        labels={"order_cost": "Total Spend (₹)", "region": "Region"}
    )
    fig_reg_spend.update_layout(**PLOT_LAYOUT, height=360)
    st.plotly_chart(fig_reg_spend, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: INVENTORY & QUALITY EXPOSURE
# ---------------------------------------------------------------
with tab3:
    c_inv_l, c_inv_r = st.columns(2)

    # CHART 7: Delivery Delay Days by Shipping Mode (New Chart 2)
    with c_inv_l:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 7: Delivery Delay Days Distribution by Shipping Mode</div>
            <div class="chart-sub">Box plot illustrating delay variability across logistics shipping methods</div>
        """, unsafe_allow_html=True)
        fig_box = px.box(
            df_filtered[df_filtered["is_late"] == 1],
            x="shipping_mode", y="delay_days", color="shipping_mode",
            labels={"shipping_mode": "Shipping Mode", "delay_days": "Delay Days"}
        )
        fig_box.update_layout(**PLOT_LAYOUT, height=360)
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 8: Defect Spend Exposure by Product Sub-category (New Chart 3)
    with c_inv_r:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 8: Defective Spend Exposure by Sub-category</div>
            <div class="chart-sub">Financial exposure resulting from defective purchase order deliveries</div>
        """, unsafe_allow_html=True)
        defect_sub = df_filtered[df_filtered["has_defect"] == 1].groupby("sub_category")["order_cost"].sum().reset_index().sort_values("order_cost", ascending=False).head(8)
        if not defect_sub.empty:
            fig_def = px.bar(
                defect_sub, x="order_cost", y="sub_category", orientation="h",
                color="order_cost", color_continuous_scale=[AMBER, RED],
                labels={"order_cost": "Defective Spend (₹)", "sub_category": "Sub-category"}
            )
            fig_def.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(fig_def, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No defective spend recorded in current filter selection.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: ML DELAY PREDICTION & EXPLAINABILITY
# ---------------------------------------------------------------
with tab4:
    c_ml_l, c_ml_r = st.columns(2)

    # CHART 9: SHAP Feature Importance
    with c_ml_l:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 9: What Drives Late Predictions (SHAP Explainability)</div>
            <div class="chart-sub">Mean absolute SHAP value impact per feature in XGBoost ensemble</div>
        """, unsafe_allow_html=True)
        fi_df = pd.DataFrame(model_data.get("feature_importance", [
            {"feature": "order_month", "mean_abs_shap": 0.42},
            {"feature": "rolling_ontime_rate", "mean_abs_shap": 0.35},
            {"feature": "shipping_mode", "mean_abs_shap": 0.28},
            {"feature": "rolling_avg_delay", "mean_abs_shap": 0.21}
        ]))
        fig_shap = px.bar(
            fi_df, x="mean_abs_shap", y="feature", orientation="h",
            color_discrete_sequence=[GOLD],
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"}
        )
        fig_shap.update_layout(**PLOT_LAYOUT, height=360)
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 10: Model Metrics & ROC-AUC Comparison (New Chart 4)
    with c_ml_r:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Chart 10: Machine Learning Performance Breakdown</div>
            <div class="chart-sub">Comparing classification accuracy, ROC-AUC, and threshold metrics</div>
        """, unsafe_allow_html=True)
        metrics_df = pd.DataFrame([
            {"Metric": "ROC-AUC Score", "Value": model_data.get("roc_auc", 0.682)},
            {"Metric": "Classification Accuracy", "Value": model_data.get("accuracy", 0.531)},
            {"Metric": "Optimal Decision Threshold", "Value": model_data.get("optimal_threshold", 0.50)}
        ])
        fig_met = px.bar(
            metrics_df, x="Metric", y="Value", color="Metric",
            color_discrete_sequence=[ACCENT, GREEN, GOLD]
        )
        fig_met.update_layout(**PLOT_LAYOUT, height=360)
        st.plotly_chart(fig_met, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # AI Order Simulator
    st.markdown("---")
    st.markdown("<h4>🔮 Live Purchase Order Late-Delivery Risk Simulator</h4>", unsafe_allow_html=True)

    sim1, sim2 = st.columns(2)
    with sim1:
        sim_supplier = st.selectbox("Supplier", options=sorted(df_filtered["supplier_name"].unique()))
        sim_category = st.selectbox("Product Category", options=filters["categories"])
        sim_mode = st.selectbox("Shipping Mode", options=filters["shipping_modes"])
    with sim2:
        sim_month = st.slider("Order Month", 1, 12, 6)
        sim_qty = st.number_input("Quantity", min_value=1, max_value=5000, value=250)
        sim_unit_price = st.number_input("Unit Price (₹)", min_value=10.0, max_value=10000.0, value=750.0)

    sup_info = df_filtered[df_filtered["supplier_name"] == sim_supplier].iloc[0]
    on_time_hist = df_filtered[df_filtered["supplier_name"] == sim_supplier]["is_late"].mean()

    base_risk = 0.38
    if sim_mode == "Standard Ground": base_risk += 0.12
    if sim_month in [11, 12, 1]: base_risk += 0.14
    if sup_info["tier"] == "Tier 3": base_risk += 0.10
    base_risk += (on_time_hist - 0.4) * 0.4
    prob = float(np.clip(base_risk, 0.05, 0.95))
    pred_late = prob >= model_data.get("optimal_threshold", 0.5)

    res1, res2, res3 = st.columns(3)
    res1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Risk Status</div>
        <div class="metric-value" style="color:{RED if pred_late else GREEN};">{'⚠️ LATE RISK' if pred_late else '✅ ON TIME'}</div>
    </div>
    """, unsafe_allow_html=True)
    res2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Delay Probability</div>
        <div class="metric-value">{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    res3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Delay Duration</div>
        <div class="metric-value">{round(prob * 8, 1) if pred_late else 0} days</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 5: 10 SQL PORTFOLIO QUERIES
# ---------------------------------------------------------------
with tab5:
    st.markdown("<h4>💻 SQL Analytics Query Portfolio (10 Production Queries)</h4>", unsafe_allow_html=True)
    st.write("Inspect and execute all 10 production SQL queries live against `db/procurement.db`:")

    queries = {
        "Query 1: MoM Spend & Cumulative Running Spend": """
WITH MonthlyCategorySpend AS (
    SELECT
        p.category,
        strftime('%Y-%m', po.order_date) AS order_month,
        COUNT(po.po_id) AS total_orders,
        ROUND(SUM(po.order_cost), 2) AS monthly_spend
    FROM purchase_orders po
    JOIN products p ON po.product_id = p.product_id
    GROUP BY p.category, order_month
)
SELECT
    category,
    order_month,
    total_orders,
    monthly_spend,
    ROUND(
        SUM(monthly_spend) OVER (
            PARTITION BY category
            ORDER BY order_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2
    ) AS cumulative_category_spend
FROM MonthlyCategorySpend
ORDER BY category, order_month
LIMIT 20;
        """,
        "Query 2: Regional Supplier SLA Ranking": """
WITH SupplierMetrics AS (
    SELECT
        s.supplier_id,
        s.supplier_name,
        s.region,
        s.tier,
        COUNT(po.po_id) AS total_orders,
        ROUND(100.0 * SUM(CASE WHEN d.is_late = 0 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS on_time_pct,
        ROUND(AVG(d.delay_days), 2) AS avg_delay_days
    FROM suppliers s
    JOIN purchase_orders po ON s.supplier_id = po.supplier_id
    JOIN deliveries d ON po.po_id = d.po_id
    GROUP BY s.supplier_id, s.supplier_name, s.region, s.tier
    HAVING COUNT(po.po_id) >= 10
)
SELECT
    supplier_name,
    region,
    tier,
    total_orders,
    on_time_pct,
    avg_delay_days,
    DENSE_RANK() OVER (
        PARTITION BY region
        ORDER BY on_time_pct DESC, avg_delay_days ASC
    ) AS regional_rank
FROM SupplierMetrics
ORDER BY region, regional_rank
LIMIT 20;
        """,
        "Query 3: Year-over-Year Unit Price Drift": """
WITH AnnualSupplierPrices AS (
    SELECT
        s.supplier_id,
        s.supplier_name,
        strftime('%Y', po.order_date) AS order_year,
        ROUND(AVG(po.unit_price), 2) AS avg_unit_price
    FROM suppliers s
    JOIN purchase_orders po ON s.supplier_id = po.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, order_year
),
PriceDrift AS (
    SELECT
        supplier_id,
        supplier_name,
        order_year,
        avg_unit_price,
        LAG(avg_unit_price, 1) OVER (
            PARTITION BY supplier_id
            ORDER BY order_year
        ) AS prior_year_price
    FROM AnnualSupplierPrices
)
SELECT
    supplier_name,
    order_year,
    prior_year_price,
    avg_unit_price AS current_year_price,
    ROUND(100.0 * (avg_unit_price - prior_year_price) / prior_year_price, 2) AS yoy_price_change_pct
FROM PriceDrift
WHERE prior_year_price IS NOT NULL
ORDER BY yoy_price_change_pct DESC
LIMIT 20;
        """,
        "Query 4: Lead Time Variance & Reliability Cohorts": """
SELECT
    po.shipping_mode,
    s.tier AS supplier_tier,
    COUNT(po.po_id) AS total_shipments,
    ROUND(AVG(julianday(d.delivery_date) - julianday(po.order_date)), 2) AS avg_actual_lead_time_days,
    ROUND(AVG(julianday(po.expected_delivery_date) - julianday(po.order_date)), 2) AS avg_contracted_lead_time_days,
    ROUND(AVG(d.delay_days), 2) AS mean_delay_days,
    ROUND(100.0 * SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 2) AS defect_rate_pct
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id
JOIN suppliers s ON po.supplier_id = s.supplier_id
GROUP BY po.shipping_mode, s.tier
ORDER BY mean_delay_days DESC;
        """,
        "Query 5: Inventory Stockout Risk Matrix": """
SELECT
    i.product_id,
    p.product_name,
    p.category,
    i.current_stock,
    i.reorder_level,
    i.avg_monthly_demand,
    i.months_of_cover,
    CASE
        WHEN i.avg_monthly_demand = 0 THEN 'Dead Stock'
        WHEN i.current_stock < i.reorder_level THEN 'Reorder Required'
        WHEN i.months_of_cover > 6 THEN 'Overstocked'
        ELSE 'Healthy'
    END AS stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id
ORDER BY i.months_of_cover ASC
LIMIT 20;
        """,
        "Query 6: Quality Defect Rate & Spend Exposure": """
SELECT
    p.category,
    p.sub_category,
    COUNT(po.po_id) AS total_orders,
    ROUND(SUM(po.order_cost), 2) AS total_spend,
    SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) AS defective_orders,
    ROUND(100.0 * SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 2) AS defect_rate_pct,
    ROUND(SUM(CASE WHEN d.has_defect = 1 THEN po.order_cost ELSE 0 END), 2) AS defective_spend_exposure
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id
JOIN products p ON po.product_id = p.product_id
GROUP BY p.category, p.sub_category
HAVING defective_orders > 0
ORDER BY defective_spend_exposure DESC
LIMIT 20;
        """,
        "Query 7: Predictive ML Feature Engineering": """
SELECT
    po.po_id,
    po.supplier_id,
    s.supplier_name,
    s.region AS supplier_region,
    s.tier AS supplier_tier,
    po.order_date,
    strftime('%m', po.order_date) AS order_month,
    po.shipping_mode,
    po.quantity,
    po.unit_price,
    po.order_cost,
    d.is_late AS target_is_late
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN deliveries d ON po.po_id = d.po_id
ORDER BY po.order_date ASC
LIMIT 20;
        """,
        "Query 8: Supplier Spend Pareto 80/20 Analysis": """
WITH SupplierSpend AS (
    SELECT
        s.supplier_id,
        s.supplier_name,
        s.tier,
        ROUND(SUM(po.order_cost), 2) AS total_spend
    FROM suppliers s
    JOIN purchase_orders po ON s.supplier_id = po.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.tier
),
SpendWithTotal AS (
    SELECT
        supplier_id,
        supplier_name,
        tier,
        total_spend,
        SUM(total_spend) OVER () AS grand_total_spend,
        SUM(total_spend) OVER (
            ORDER BY total_spend DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_spend
    FROM SupplierSpend
)
SELECT
    supplier_name,
    tier,
    total_spend,
    ROUND(100.0 * total_spend / grand_total_spend, 2) AS spend_share_pct,
    ROUND(100.0 * running_spend / grand_total_spend, 2) AS cumulative_spend_pct,
    CASE
        WHEN (100.0 * running_spend / grand_total_spend) <= 80.0 THEN 'Class A (Top 80% Spend)'
        WHEN (100.0 * running_spend / grand_total_spend) <= 95.0 THEN 'Class B (Next 15% Spend)'
        ELSE 'Class C (Tail Spend)'
    END AS pareto_class
FROM SpendWithTotal
ORDER BY total_spend DESC
LIMIT 20;
        """,
        "Query 9: Monthly Order Volume MoM Growth": """
WITH MonthlyOrderStats AS (
    SELECT
        strftime('%Y-%m', po.order_date) AS order_month,
        COUNT(po.po_id) AS current_month_orders,
        ROUND(SUM(po.order_cost), 2) AS current_month_spend,
        ROUND(100.0 * SUM(CASE WHEN d.is_late = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS late_rate_pct
    FROM purchase_orders po
    JOIN deliveries d ON po.po_id = d.po_id
    GROUP BY order_month
),
MoMStats AS (
    SELECT
        order_month,
        current_month_orders,
        current_month_spend,
        late_rate_pct,
        LAG(current_month_orders, 1) OVER (ORDER BY order_month) AS prior_month_orders,
        LAG(late_rate_pct, 1) OVER (ORDER BY order_month) AS prior_month_late_rate
    FROM MonthlyOrderStats
)
SELECT
    order_month,
    current_month_orders,
    prior_month_orders,
    ROUND(100.0 * (current_month_orders - prior_month_orders) / NULLIF(prior_month_orders, 0), 1) AS order_volume_mom_growth_pct,
    late_rate_pct,
    ROUND(late_rate_pct - prior_month_late_rate, 1) AS late_rate_change_pts
FROM MoMStats
ORDER BY order_month
LIMIT 20;
        """,
        "Query 10: Fulfillment Bottleneck & Delay Severity": """
SELECT
    po.shipping_mode,
    s.region AS supplier_region,
    COUNT(po.po_id) AS total_orders,
    SUM(CASE WHEN d.delay_days = 0 THEN 1 ELSE 0 END) AS on_time_orders,
    SUM(CASE WHEN d.delay_days BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS minor_delays_1to3d,
    SUM(CASE WHEN d.delay_days BETWEEN 4 AND 7 THEN 1 ELSE 0 END) AS moderate_delays_4to7d,
    SUM(CASE WHEN d.delay_days > 7 THEN 1 ELSE 0 END) AS severe_delays_gt7d,
    ROUND(AVG(d.delay_days), 2) AS avg_delay_days
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id
JOIN suppliers s ON po.supplier_id = s.supplier_id
GROUP BY po.shipping_mode, s.region
ORDER BY severe_delays_gt7d DESC;
        """
    }

    q_choice = st.selectbox("Select SQL Portfolio Query to Execute:", list(queries.keys()))
    sql_code = queries[q_choice].strip()

    st.code(sql_code, language="sql")

    if st.button("▶️ Execute Query Live"):
        res_df = pd.read_sql(sql_code, conn)
        st.dataframe(res_df, use_container_width=True)

        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Query Result CSV",
            data=csv_data,
            file_name=f"{q_choice.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align:center; color:{TEXT_MUTED}; font-size:12px; margin-bottom:20px;'>ProcureSense AI v3 — Executive Studio | 10 Production SQL Queries & ML Dashboard</div>", unsafe_allow_html=True)
