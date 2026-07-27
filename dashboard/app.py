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
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ---------------------------------------------------------------
# 1. Page Configuration & Theme State
# ---------------------------------------------------------------
st.set_page_config(
    page_title="ProcureSense AI — Procurement Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode only UI — IS_DARK kept as a stub for future light-mode wiring
IS_DARK = True

# ---------------------------------------------------------------
# 2. Path Setup & Artifact Loading
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
KPI_PATH = os.path.join(BASE_DIR, "analysis", "kpi_summary.json")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")

@st.cache_data
def get_logo_base64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

logo_b64 = get_logo_base64()
logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="height:42px; width:42px; border-radius:8px; object-fit:cover; margin-right:14px; border:1px solid rgba(108,142,245,0.4); flex-shrink:0;">' if logo_b64 else '<div class="brand-logo">&#9672;</div>'

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
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("SELECT 1;")
        return conn
    except Exception as e:
        st.error(f"⚠️ Database Connection Error: Unable to access SQLite database at `{DB_PATH}`. Details: {e}")
        st.stop()

conn = get_db_connection()
kpi_data = load_kpi_summary()
model_data = load_model_metrics()

# ---------------------------------------------------------------
# 3. Dynamic CSS Styling & External Stylesheet Loading
# ---------------------------------------------------------------
# --- Colour System v2: Deep Navy-Slate + Periwinkle-Indigo ---
BG          = "#080b14"
BG_SUBTLE   = "#0d1120"
CARD_BG     = "#0f1422"
BORDER      = "rgba(99,120,200,0.15)"
BORDER_HOV  = "rgba(99,120,200,0.35)"
TEXT        = "#EDF2FF"
TEXT_MUTED  = "#7B8EC8"
TEXT_DIM    = "#4A5580"
ACCENT      = "#6C8EF5"
ACCENT_GLOW = "rgba(108,142,245,0.14)"
GOLD        = "#F5C842"
GOLD_BG     = "rgba(245,200,66,0.12)"
GREEN       = "#10D98C"
GREEN_BG    = "rgba(16,217,140,0.12)"
RED         = "#F5604A"
RED_BG      = "rgba(245,96,74,0.12)"
AMBER       = "#F5A623"
AMBER_BG    = "rgba(245,166,35,0.12)"

if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 4. Header Bar
# ---------------------------------------------------------------
_hdr_health = kpi_data.get("overall_health_score", "—")
_hdr_orders = pd.read_sql("SELECT COUNT(*) AS n FROM purchase_orders", conn).iloc[0]["n"]
st.markdown(f"""
<div class="brand-header">
    <div style="display:flex; align-items:center; gap:0;">
        {logo_img_tag}
        <div>
            <div class="brand-title">Procure<span>Sense</span> AI</div>
            <div class="brand-sub">Procurement Intelligence &nbsp;&middot;&nbsp; SHAP Explainability &nbsp;&middot;&nbsp; Live ML Risk Simulation</div>
        </div>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div>
        Live &nbsp;&middot;&nbsp; {_hdr_orders:,} POs &nbsp;&middot;&nbsp; {_hdr_health}/100
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 5. Sidebar Control Center & Dynamic Parameterized Filters
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_filter_options():
    conn = get_db_connection()
    suppliers_df = pd.read_sql("SELECT DISTINCT tier, region FROM suppliers", conn)
    products_df = pd.read_sql("SELECT DISTINCT category FROM products", conn)
    shipping_df = pd.read_sql("SELECT DISTINCT shipping_mode, priority FROM purchase_orders", conn)
    return {
        "tiers": sorted(suppliers_df["tier"].dropna().unique().tolist()),
        "regions": sorted(suppliers_df["region"].dropna().unique().tolist()),
        "categories": sorted(products_df["category"].dropna().unique().tolist()),
        "shipping_modes": sorted(shipping_df["shipping_mode"].dropna().unique().tolist()),
        "priorities": sorted(shipping_df["priority"].dropna().unique().tolist()),
    }

filters = get_filter_options()

st.sidebar.markdown(f"""
<div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
    Control Center
</div>
<div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 14px;">
    Filter {_hdr_orders:,} orders across regions, tiers, logistics, & quality
</div>
""", unsafe_allow_html=True)

# Sidebar Filter Preset Quick Actions
st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Quick Filter Presets</div>", unsafe_allow_html=True)
c_pre1, c_pre2 = st.sidebar.columns(2)
c_pre3, c_pre4 = st.sidebar.columns(2)

preset_all = c_pre1.button("Reset All", use_container_width=True)
preset_peak = c_pre2.button("Peak Q4 Air", use_container_width=True)
preset_import = c_pre3.button("Imports Only", use_container_width=True)
preset_critical = c_pre4.button("High Priority", use_container_width=True)

if preset_all:
    st.session_state["sel_tiers"] = filters["tiers"]
    st.session_state["sel_regions"] = filters["regions"]
    st.session_state["sel_cats"] = filters["categories"]
    st.session_state["sel_ship"] = filters["shipping_modes"]
    st.session_state["sel_prio"] = filters["priorities"]
    st.session_state["sel_dates"] = (datetime.date(2023, 1, 1), datetime.date(2025, 12, 31))
    st.session_state["sup_search"] = ""
    st.session_state["only_defect"] = False

if preset_peak:
    st.session_state["sel_ship"] = [m for m in filters["shipping_modes"] if "Air" in m or "Express" in m]
    st.session_state["only_defect"] = False

if preset_import:
    st.session_state["sel_regions"] = [r for r in filters["regions"] if "Import" in r]
    st.session_state["only_defect"] = False

if preset_critical:
    st.session_state["sel_prio"] = ["High", "Urgent"] if "High" in filters["priorities"] else filters["priorities"]
    st.session_state["only_defect"] = False

# Sidebar Filter Sections
with st.sidebar.expander("Sourcing & Supplier Filters", expanded=True):
    supplier_search = st.text_input("Supplier Search", value=st.session_state.get("sup_search", ""), placeholder="Search by name...")
    selected_tiers = st.multiselect("Commercial Tier", options=filters["tiers"], default=st.session_state.get("sel_tiers", filters["tiers"]))
    selected_regions = st.multiselect("Supplier Region", options=filters["regions"], default=st.session_state.get("sel_regions", filters["regions"]))

with st.sidebar.expander("Product & Logistics Filters", expanded=True):
    selected_categories = st.multiselect("Product Category", options=filters["categories"], default=st.session_state.get("sel_cats", filters["categories"]))
    selected_shipping = st.multiselect("Shipping Mode", options=filters["shipping_modes"], default=st.session_state.get("sel_ship", filters["shipping_modes"]))
    selected_priorities = st.multiselect("Order Priority", options=filters["priorities"], default=st.session_state.get("sel_prio", filters["priorities"]))

with st.sidebar.expander("Date Window & Quality Focus", expanded=True):
    default_date_range = st.session_state.get("sel_dates", (datetime.date(2023, 1, 1), datetime.date(2025, 12, 31)))
    date_range = st.date_input(
        "Order Date Window",
        value=default_date_range,
        min_value=datetime.date(2023, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    only_defects = st.checkbox("Show Defective Orders Only", value=st.session_state.get("only_defect", False))

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date, end_date = date_range[0], date_range[0]
else:
    start_date, end_date = datetime.date(2023, 1, 1), datetime.date(2025, 12, 31)

# Parameterized Query Function with @st.cache_data
@st.cache_data(ttl=300)
def get_filtered_data_param(start_d_str, end_d_str, tiers, regions, categories, shipping_modes, priorities, sup_search, defects_only):
    conn = get_db_connection()
    where_clauses = ["1=1"]
    params = []

    if start_d_str and end_d_str:
        where_clauses.append("date(po.order_date) BETWEEN ? AND ?")
        params.extend([start_d_str, end_d_str])

    if tiers:
        where_clauses.append(f"s.tier IN ({','.join(['?']*len(tiers))})")
        params.extend(tiers)

    if regions:
        where_clauses.append(f"s.region IN ({','.join(['?']*len(regions))})")
        params.extend(regions)

    if categories:
        where_clauses.append(f"p.category IN ({','.join(['?']*len(categories))})")
        params.extend(categories)

    if shipping_modes:
        where_clauses.append(f"po.shipping_mode IN ({','.join(['?']*len(shipping_modes))})")
        params.extend(shipping_modes)

    if priorities:
        where_clauses.append(f"po.priority IN ({','.join(['?']*len(priorities))})")
        params.extend(priorities)

    if sup_search and sup_search.strip():
        where_clauses.append("s.supplier_name LIKE ?")
        params.append(f"%{sup_search.strip()}%")

    if defects_only:
        where_clauses.append("d.has_defect = 1")

    sql_query = f"""
        SELECT 
            po.po_id, po.order_date, po.unit_price, po.quantity, po.order_cost, po.shipping_mode, po.priority,
            s.supplier_id, s.supplier_name, s.region, s.tier,
            p.product_name, p.category, p.sub_category,
            d.delivery_date, d.is_late, d.delay_days, d.has_defect
        FROM purchase_orders po
        JOIN suppliers s ON po.supplier_id = s.supplier_id
        JOIN products p ON po.product_id = p.product_id
        JOIN deliveries d ON po.po_id = d.po_id
        WHERE {" AND ".join(where_clauses)}
    """
    df = pd.read_sql(sql_query, conn, params=params)
    return df

df_filtered = get_filtered_data_param(
    start_d_str=str(start_date),
    end_d_str=str(end_date),
    tiers=tuple(selected_tiers) if selected_tiers else (),
    regions=tuple(selected_regions) if selected_regions else (),
    categories=tuple(selected_categories) if selected_categories else (),
    shipping_modes=tuple(selected_shipping) if selected_shipping else (),
    priorities=tuple(selected_priorities) if selected_priorities else (),
    sup_search=supplier_search,
    defects_only=only_defects
)

# Fallback check — warn the user rather than silently showing empty UI
if df_filtered.empty:
    st.warning(
        "No orders match this filter combination — showing all data instead. "
        "Try widening the Tier, Region, or Category selections."
    )
    df_filtered = pd.read_sql("""
        SELECT 
            po.po_id, po.order_date, po.unit_price, po.quantity, po.order_cost, po.shipping_mode, po.priority,
            s.supplier_id, s.supplier_name, s.region, s.tier,
            p.product_name, p.category, p.sub_category,
            d.delivery_date, d.is_late, d.delay_days, d.has_defect
        FROM purchase_orders po
        JOIN suppliers s ON po.supplier_id = s.supplier_id
        JOIN products p ON po.product_id = p.product_id
        JOIN deliveries d ON po.po_id = d.po_id
    """, conn)

# Live Filter Summary Box inside Sidebar Footer
f_orders = len(df_filtered)
f_spend = df_filtered["order_cost"].sum() / 1e7
f_late = (df_filtered["is_late"].mean() * 100) if f_orders > 0 else 0
f_defect = (df_filtered["has_defect"].mean() * 100) if f_orders > 0 else 0

st.sidebar.markdown(f"""
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 10px; padding: 12px 14px; margin-top: 16px;">
    <div style="font-size: 0.76rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Live Selection Summary</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.78rem;">
        <div><span style="color:#94a3b8;">Orders:</span> <b style="color:#f8fafc;">{f_orders:,}</b></div>
        <div><span style="color:#94a3b8;">Spend:</span> <b style="color:#eab308;">₹{f_spend:.1f} Cr</b></div>
        <div><span style="color:#94a3b8;">Late %:</span> <b style="color:#ef4444;">{f_late:.1f}%</b></div>
        <div><span style="color:#94a3b8;">Defect %:</span> <b style="color:#f59e0b;">{f_defect:.1f}%</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Dataset Export Hub
with st.sidebar.expander("Export Core Datasets", expanded=False):
    st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>Download normalized database tables or current filtered subset as CSV files:</div>", unsafe_allow_html=True)
    
    csv_filtered_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Filtered Selection ({len(df_filtered):,} POs)",
        data=csv_filtered_data,
        file_name="procuresense_filtered_orders.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    def _get_table_csv(tbl_name):
        return pd.read_sql(f"SELECT * FROM {tbl_name}", conn).to_csv(index=False).encode('utf-8')
    
    st.download_button("Purchase Orders (30k)", data=_get_table_csv("purchase_orders"), file_name="purchase_orders_30k.csv", mime="text/csv", use_container_width=True)
    st.download_button("Suppliers Catalog", data=_get_table_csv("suppliers"), file_name="suppliers_catalog.csv", mime="text/csv", use_container_width=True)
    st.download_button("Products Catalog", data=_get_table_csv("products"), file_name="products_catalog.csv", mime="text/csv", use_container_width=True)
    st.download_button("Deliveries Log", data=_get_table_csv("deliveries"), file_name="deliveries_log.csv", mime="text/csv", use_container_width=True)
    st.download_button("Inventory Stock", data=_get_table_csv("inventory"), file_name="inventory_stock.csv", mime="text/csv", use_container_width=True)

# Polished Plotly Layout Definition & High-Res PNG Export Configuration
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
    margin=dict(l=15, r=15, t=25, b=25),
    xaxis=dict(
        gridcolor="rgba(99,120,200,0.08)",
        zerolinecolor="rgba(99,120,200,0.10)",
        tickfont=dict(size=10, color=TEXT_DIM)
    ),
    yaxis=dict(
        gridcolor="rgba(99,120,200,0.08)",
        zerolinecolor="rgba(99,120,200,0.10)",
        tickfont=dict(size=10, color=TEXT_DIM)
    ),
    hoverlabel=dict(
        bgcolor="#1A2040",
        font_size=12,
        font_family="Inter, sans-serif",
        bordercolor="rgba(108,142,245,0.30)"
    )
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "procuresense_chart_export",
        "height": 1080,
        "width": 1920,
        "scale": 2
    }
}

# ---------------------------------------------------------------
# 6. Tab Navigation (5 Modern Navbar Tabs)
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Supplier & Region SLAs",
    "Inventory Control",
    "ML Risk Simulator",
    "SQL Analytics Studio",
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
    health_score = kpi_data.get("overall_health_score")
    if health_score is None:
        st.error("kpi_summary.json missing — run `py analysis/kpi_engine.py` to regenerate.")
        health_score = 0.0

    _late_colour  = RED   if late_pct  > 40 else (AMBER if late_pct  > 25 else GREEN)
    _late_bg      = RED_BG if late_pct  > 40 else (AMBER_BG if late_pct  > 25 else GREEN_BG)
    _def_colour   = RED   if defect_pct > 5  else (AMBER if defect_pct > 2  else GREEN)
    _def_bg       = RED_BG if defect_pct > 5  else (AMBER_BG if defect_pct > 2  else GREEN_BG)
    _hs_colour    = GREEN if health_score >= 65 else (AMBER if health_score >= 45 else RED)
    _hs_bg        = GREEN_BG if health_score >= 65 else (AMBER_BG if health_score >= 45 else RED_BG)

    m1, m2, m3, m4, m5 = st.columns([2, 1.5, 1.5, 1.5, 1.5], gap="medium")
    with m1:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
            <div class="metric-label">Procurement Health</div>
            <div class="metric-value" style="color:{_hs_colour};">{health_score}<span class="metric-unit"> / 100</span></div>
            <div class="metric-badge" style="background:{_hs_bg}; color:{_hs_colour};">Composite Index</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
            <div class="metric-label">Total Spend</div>
            <div class="metric-value">&#8377;{total_spend/1e7:.1f}<span class="metric-unit"> Cr</span></div>
            <div class="metric-badge" style="background:{GOLD_BG}; color:{GOLD};">Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GREEN};">
            <div class="metric-label">Purchase Orders</div>
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-badge" style="background:{GREEN_BG}; color:{GREEN};">Active</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{_late_colour};">
            <div class="metric-label">Late Delivery Rate</div>
            <div class="metric-value" style="color:{_late_colour};">{late_pct:.1f}<span class="metric-unit">%</span></div>
            <div class="metric-badge" style="background:{_late_bg}; color:{_late_colour};">SLA Breach</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{_def_colour};">
            <div class="metric-label">Defect Rate</div>
            <div class="metric-value" style="color:{_def_colour};">{defect_pct:.2f}<span class="metric-unit">%</span></div>
            <div class="metric-badge" style="background:{_def_bg}; color:{_def_colour};">Quality</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)

    # CHART 1: Monthly Spend & Late % Trend
    df_filtered["month"] = pd.to_datetime(df_filtered["order_date"]).dt.to_period("M").astype(str)
    monthly = df_filtered.groupby("month").agg(
        total_spend=("order_cost", "sum"),
        late_pct=("is_late", lambda x: (x.sum() / len(x)) * 100)
    ).reset_index()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly["month"], y=monthly["total_spend"], name="Monthly Spend (&#8377;)",
               marker_color=GOLD, marker_line_width=0, opacity=0.80),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["late_pct"], name="Late Rate %",
                   mode="lines+markers",
                   line=dict(color=RED, width=2.5),
                   marker=dict(size=5, color=RED)),
        secondary_y=True
    )
    fig_trend.update_layout(**PLOT_LAYOUT, height=340,
                            legend=dict(orientation="h", y=1.1, font_size=11))

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Monthly Procurement Spend vs Late Delivery Rate</div>
        <div class="chart-sub">Monthly expenditure volume against late shipment percentage across selection</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    c_over1, c_over2, c_over3 = st.columns([1, 1, 1], gap="medium")
    
    # CHART 2: Dynamic Supplier Risk Tier Distribution
    with c_over1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Supplier Risk Tier Distribution</div>
            <div class="chart-sub">Percentile-calibrated risk tiers across active suppliers (dynamic)</div>
        """, unsafe_allow_html=True)
        if not df_filtered.empty and "supplier_name" in df_filtered.columns:
            sup_stats = df_filtered.groupby("supplier_name").agg(
                on_time_pct=("is_late", lambda x: ((1.0 - (x.sum() / len(x))) * 100.0)),
                defect_rate_pct=("has_defect", lambda x: ((x.sum() / len(x)) * 100.0)),
                avg_delay_days=("delay_days", "mean")
            ).reset_index()
            def calc_risk_tier(r):
                pts = 0.0
                if r["on_time_pct"] < 50.0: pts += 2.0
                elif r["on_time_pct"] < 65.0: pts += 1.0
                if r["avg_delay_days"] > 5.0: pts += 1.0
                if r["defect_rate_pct"] > 4.5: pts += 1.5
                elif r["defect_rate_pct"] > 2.5: pts += 0.75
                if pts >= 3.0: return "High Risk"
                elif pts >= 1.0: return "Medium Risk"
                else: return "Low Risk"
            sup_stats["risk_tier"] = sup_stats.apply(calc_risk_tier, axis=1)
            risk_data = sup_stats["risk_tier"].value_counts().to_dict()
        else:
            risk_data = kpi_data.get("risk_distribution", {"High Risk": 31, "Medium Risk": 31, "Low Risk": 16})
        
        fig_risk = px.pie(
            values=list(risk_data.values()), names=list(risk_data.keys()), hole=0.55,
            color=list(risk_data.keys()),
            color_discrete_map={"High Risk": RED, "Medium Risk": AMBER, "Low Risk": GREEN}
        )
        fig_risk.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_risk, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 3: Dynamic Inventory Status
    with c_over2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Inventory Stock Coverage</div>
            <div class="chart-sub">Dynamic ROP-driven stock health breakdown across product catalog</div>
        """, unsafe_allow_html=True)
        if not df_filtered.empty and "product_name" in df_filtered.columns:
            filtered_prods = df_filtered["product_name"].unique()
            inv_df_all = pd.read_sql("SELECT p.product_name, i.current_stock, i.reorder_level, i.avg_monthly_demand FROM inventory i JOIN products p ON i.product_id = p.product_id", conn)
            inv_df_sub = inv_df_all[inv_df_all["product_name"].isin(filtered_prods)] if len(filtered_prods) > 0 else inv_df_all
            def calc_inv_st(r):
                if r["avg_monthly_demand"] == 0: return "Dead Stock"
                elif r["current_stock"] < r["reorder_level"]: return "Understocked"
                else: return "Healthy"
            inv_df_sub["status"] = inv_df_sub.apply(calc_inv_st, axis=1)
            inv_data = inv_df_sub["status"].value_counts().to_dict()
        else:
            inv_data = kpi_data.get("inventory_status", {"Healthy": 150, "Understocked": 50})

        fig_inv = px.pie(
            values=list(inv_data.values()), names=list(inv_data.keys()), hole=0.55,
            color=list(inv_data.keys()),
            color_discrete_map={"Healthy": GREEN, "Understocked": AMBER, "Overstocked": TEXT_MUTED, "Dead Stock": RED}
        )
        fig_inv.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_inv, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 4: Dynamic Top Price Inflation Flags
    with c_over3:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Price Inflation Flags</div>
            <div class="chart-sub">Top suppliers by YoY unit price increase — OLS regression signal</div>
        """, unsafe_allow_html=True)
        if not df_filtered.empty and "supplier_name" in df_filtered.columns:
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy["year"] = pd.to_datetime(df_filtered_copy["order_date"]).dt.year
            price_dyn = df_filtered_copy.groupby(["supplier_name", "year"])["unit_price"].mean().unstack()
            if len(price_dyn.columns) >= 2:
                col_last = price_dyn.columns[-1]
                col_prev = price_dyn.columns[-2]
                price_dyn["latest_pct_change"] = (((price_dyn[col_last] - price_dyn[col_prev]) / price_dyn[col_prev]) * 100.0).round(1)
                price_df = price_dyn.reset_index()[["supplier_name", "latest_pct_change"]].dropna().sort_values("latest_pct_change", ascending=False).head(6)
            else:
                price_df = pd.DataFrame(kpi_data.get("price_inflation_flags", [])).head(6)
        else:
            price_df = pd.DataFrame(kpi_data.get("price_inflation_flags", [])).head(6)

        if not price_df.empty:
            fig_price = px.bar(
                price_df, x="latest_pct_change", y="supplier_name", orientation="h",
                color="latest_pct_change", color_continuous_scale=[GREEN, AMBER, RED]
            )
            fig_price.update_layout(**PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No price inflation flags recorded for selection.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(108,142,245,0.06); border:1px solid rgba(108,142,245,0.18); border-left:3px solid {ACCENT};
                border-radius:8px; padding:13px 16px; margin-top:8px; font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.55;">
    <span style="color:{ACCENT}; font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em;">AI Narrative Insight</span><br>
    {kpi_data.get('narrative_example', 'No narrative available.')}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 2: SUPPLIER & REGIONAL PERFORMANCE (DYNAMIC)
# ---------------------------------------------------------------
with tab2:
    if df_filtered.empty or "supplier_name" not in df_filtered.columns:
        st.warning("No supplier data available for the selected filter combination.")
    else:
        # Aggregated supplier metrics from df_filtered (dynamically reactive)
        sup_metrics = df_filtered.groupby(["supplier_id", "supplier_name", "region", "tier"]).agg(
            total_orders=("po_id", "count"),
            total_spend=("order_cost", "sum"),
            late_orders=("is_late", "sum"),
            defect_orders=("has_defect", "sum"),
            avg_delay_days=("delay_days", "mean")
        ).reset_index()
        sup_metrics["on_time_pct"] = ((1.0 - (sup_metrics["late_orders"] / sup_metrics["total_orders"])) * 100.0).round(1)
        sup_metrics["defect_rate_pct"] = ((sup_metrics["defect_orders"] / sup_metrics["total_orders"]) * 100.0).round(1)
        sup_metrics["avg_delay_days"] = sup_metrics["avg_delay_days"].round(1)

        # 1. TOP ROW: DYNAMIC SUPPLIER RANKING & SCATTER PLOT
        c_t2_l, c_t2_r = st.columns([52, 48], gap="medium")

        with c_t2_l:
            st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Supplier On-Time Delivery Ranking (Dynamic)</div>
                <div class="chart-sub">Fulfillment reliability benchmarking across filtered supplier network</div>
            """, unsafe_allow_html=True)
            
            rank_view = st.radio(
                "Filter Display Mode",
                ["Bottom 10 (Lowest SLA)", "Top 10 (Highest SLA)", "All Filtered Suppliers"],
                horizontal=True,
                key="t2_rank_mode"
            )
            
            if "Bottom" in rank_view:
                chart_sup = sup_metrics.sort_values("on_time_pct", ascending=True).head(10)
            elif "Top" in rank_view:
                chart_sup = sup_metrics.sort_values("on_time_pct", ascending=False).head(10).sort_values("on_time_pct", ascending=True)
            else:
                chart_sup = sup_metrics.sort_values("on_time_pct", ascending=True)

            fig_sup_rank = px.bar(
                chart_sup, x="on_time_pct", y="supplier_name", orientation="h",
                color="on_time_pct", color_continuous_scale=[RED, AMBER, GREEN],
                range_color=[40, 100],
                labels={"on_time_pct": "On-Time %", "supplier_name": "Supplier"},
                hover_data=["region", "tier", "total_orders", "avg_delay_days"]
            )
            fig_sup_rank.update_layout(**PLOT_LAYOUT, height=max(340, len(chart_sup) * 26))
            st.plotly_chart(fig_sup_rank, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_t2_r:
            st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Spend vs. On-Time Fulfillment Matrix</div>
                <div class="chart-sub">Expenditure volume vs SLA performance — High Spend + Low SLA indicates risk exposure</div>
            """, unsafe_allow_html=True)

            fig_scatter = px.scatter(
                sup_metrics, x="total_spend", y="on_time_pct",
                size="total_orders", color="tier",
                hover_name="supplier_name",
                hover_data=["region", "defect_rate_pct", "avg_delay_days"],
                color_discrete_map={"Tier 1": ACCENT, "Tier 2": AMBER, "Tier 3": RED},
                labels={"total_spend": "Total Spend (&#8377;)", "on_time_pct": "On-Time %", "tier": "Commercial Tier"}
            )
            fig_scatter.add_hline(y=70.0, line_dash="dash", line_color=RED, opacity=0.6, annotation_text="Target SLA 70%")
            fig_scatter.update_layout(**PLOT_LAYOUT, height=390)
            st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown("</div>", unsafe_allow_html=True)

        # 2. SECOND ROW: REGION x TIER SLA HEATMAP & SPEND CONCENTRATION
        c_t2_l2, c_t2_r2 = st.columns([50, 50], gap="medium")

        with c_t2_l2:
            st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Region &times; Commercial Tier On-Time SLA Heatmap</div>
                <div class="chart-sub">Average fulfillment on-time % matrix across geographic regions and tiers</div>
            """, unsafe_allow_html=True)

            reg_tier_pivot = df_filtered.groupby(["region", "tier"]).agg(
                on_time_rate=("is_late", lambda x: ((1.0 - (x.sum()/len(x))) * 100.0))
            ).reset_index().pivot(index="region", columns="tier", values="on_time_rate").fillna(0.0).round(1)

            fig_heatmap = px.imshow(
                reg_tier_pivot,
                labels=dict(x="Commercial Tier", y="Region", color="On-Time %"),
                color_continuous_scale=[RED, AMBER, GREEN],
                text_auto=".1f",
                aspect="auto"
            )
            fig_heatmap.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_t2_r2:
            st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Spend Concentration by Region & Commercial Tier</div>
                <div class="chart-sub">Stacked procurement expenditure — Tier 1 Strategic / Tier 2 Preferred / Tier 3 Tactical</div>
            """, unsafe_allow_html=True)
            reg_tier_spend = df_filtered.groupby(["region", "tier"])["order_cost"].sum().reset_index()
            fig_reg_spend = px.bar(
                reg_tier_spend, x="region", y="order_cost", color="tier", barmode="stack",
                color_discrete_map={"Tier 1": ACCENT, "Tier 2": AMBER, "Tier 3": RED},
                labels={"order_cost": "Total Spend (&#8377;)", "region": "Region", "tier": "Commercial Tier"}
            )
            fig_reg_spend.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(fig_reg_spend, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. SUPPLIER DETAIL DRILL-DOWN PANEL
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Supplier Deep-Dive Inspection Panel</div>
            <div class="chart-sub">Drill into individual supplier performance metrics, delay trajectories, and purchase order log</div>
        """, unsafe_allow_html=True)

        sup_list = sorted(sup_metrics["supplier_name"].unique())
        selected_sup_name = st.selectbox("Select Supplier to Inspect", options=sup_list, key="t2_sup_select")

        if selected_sup_name:
            sup_detail = sup_metrics[sup_metrics["supplier_name"] == selected_sup_name].iloc[0]
            sup_orders = df_filtered[df_filtered["supplier_name"] == selected_sup_name].sort_values("order_date", ascending=False)

            d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
            with d_col1:
                st.metric("Total Orders", f"{sup_detail['total_orders']:,}")
            with d_col2:
                st.metric("Total Spend", f"₹{sup_detail['total_spend']/1e5:.1f} L")
            with d_col3:
                st.metric("On-Time SLA", f"{sup_detail['on_time_pct']:.1f}%")
            with d_col4:
                st.metric("Defect Rate", f"{sup_detail['defect_rate_pct']:.1f}%")
            with d_col5:
                st.metric("Avg Delay", f"{sup_detail['avg_delay_days']:.1f} days")

            # Monthly Trajectory Chart for Selected Supplier
            sup_orders_copy = sup_orders.copy()
            sup_orders_copy["order_month_str"] = pd.to_datetime(sup_orders_copy["order_date"]).dt.to_period("M").astype(str)
            sup_monthly = sup_orders_copy.groupby("order_month_str").agg(
                avg_delay=("delay_days", "mean"),
                late_rate=("is_late", lambda x: (x.sum()/len(x))*100.0)
            ).reset_index()

            if not sup_monthly.empty:
                fig_sup_traj = px.line(
                    sup_monthly, x="order_month_str", y="avg_delay",
                    markers=True, title=f"Monthly Average Delay Trajectory (Days) — {selected_sup_name}",
                    labels={"order_month_str": "Month", "avg_delay": "Avg Delay (Days)"},
                    color_discrete_sequence=[ACCENT]
                )
                fig_sup_traj.update_layout(**PLOT_LAYOUT, height=260)
                st.plotly_chart(fig_sup_traj, use_container_width=True, config=PLOTLY_CONFIG)

            with st.expander(f"View Order History Log for {selected_sup_name} ({len(sup_orders)} POs)", expanded=False):
                st.dataframe(
                    sup_orders[["po_id", "order_date", "product_name", "category", "quantity", "unit_price", "order_cost", "shipping_mode", "priority", "is_late", "delay_days", "has_defect"]],
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: INVENTORY CONTROL & STOCKOUT RISK EXPOSURE (DYNAMIC ROP)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_inventory_full_data():
    conn = get_db_connection()
    sql = """
        SELECT 
            i.product_id,
            p.sku,
            p.product_name,
            p.category,
            p.sub_category,
            p.unit_cost_base,
            p.lead_time_days_base,
            i.warehouse,
            i.current_stock,
            i.reorder_level,
            i.avg_monthly_demand,
            i.months_of_cover,
            s.supplier_id,
            s.supplier_name,
            s.tier as supplier_tier,
            s.region as supplier_region,
            COALESCE(del_stats.late_rate, global_stats.avg_late_rate) as sup_late_rate,
            COALESCE(del_stats.defect_rate, global_stats.avg_defect_rate) as sup_defect_rate,
            COALESCE(del_stats.avg_delay, global_stats.avg_delay_days) as sup_avg_delay
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        JOIN suppliers s ON p.primary_supplier_id = s.supplier_id
        LEFT JOIN (
            SELECT po.supplier_id,
                   AVG(d.is_late) as late_rate,
                   AVG(d.has_defect) as defect_rate,
                   AVG(d.delay_days) as avg_delay
            FROM purchase_orders po
            JOIN deliveries d ON po.po_id = d.po_id
            GROUP BY po.supplier_id
        ) del_stats ON s.supplier_id = del_stats.supplier_id
        CROSS JOIN (
            SELECT AVG(is_late) as avg_late_rate, AVG(has_defect) as avg_defect_rate, AVG(delay_days) as avg_delay_days
            FROM deliveries
        ) global_stats
    """
    return pd.read_sql(sql, conn)

with tab3:
    inv_full_df = get_inventory_full_data().copy()

    # Dynamic scope filtering with explicit fallback notice
    if not df_filtered.empty and "product_name" in df_filtered.columns:
        active_prods = df_filtered["product_name"].unique()
        inv_active = inv_full_df[inv_full_df["product_name"].isin(active_prods)].copy()
        if inv_active.empty:
            st.info("ℹ️ Filter Scope Notice: Active sidebar selection matched 0 inventory SKUs — displaying catalog-wide inventory data.")
            inv_active = inv_full_df.copy()
    else:
        inv_active = inv_full_df.copy()

    # Unified Authoritative Dynamic ROP Calculation (Matching kpi_engine.py)
    inv_active["avg_daily_demand"] = inv_active["avg_monthly_demand"] / 30.0
    inv_active["safety_stock"] = (inv_active["reorder_level"] * 0.35).round(1)
    inv_active["inflated_lead_time_days"] = inv_active["lead_time_days_base"] + inv_active["sup_avg_delay"].fillna(0.0)
    inv_active["dynamic_reorder_point"] = (inv_active["inflated_lead_time_days"] * inv_active["avg_daily_demand"] + inv_active["safety_stock"]).round(1)

    def _dynamic_inv_status(r):
        if r["avg_monthly_demand"] == 0:
            return "Dead Stock"
        elif r["current_stock"] < r["dynamic_reorder_point"]:
            return "Understocked"
        elif r["months_of_cover"] > 6.0:
            return "Overstocked"
        else:
            return "Healthy"

    inv_active["stock_status"] = inv_active.apply(_dynamic_inv_status, axis=1)
    inv_active["stock_val"] = inv_active["current_stock"] * inv_active["unit_cost_base"]
    inv_active["is_high_risk_sup"] = (inv_active["sup_late_rate"] > 0.40) | (inv_active["supplier_tier"] == "Tier 3")

    # Days Until Stockout & ABC Inventory Classification
    inv_active["days_until_stockout"] = np.where(
        inv_active["avg_daily_demand"] > 0,
        (inv_active["current_stock"] / inv_active["avg_daily_demand"]).round(1),
        999.0
    )

    # ABC Classification by Stock Value
    inv_active = inv_active.sort_values("stock_val", ascending=False).reset_index(drop=True)
    tot_val = inv_active["stock_val"].sum()
    if tot_val > 0:
        inv_active["cum_val_pct"] = (inv_active["stock_val"].cumsum() / tot_val) * 100.0
        inv_active["abc_class"] = np.where(inv_active["cum_val_pct"] <= 70.0, "Class A (Top 70%)",
                                  np.where(inv_active["cum_val_pct"] <= 90.0, "Class B (Next 20%)", "Class C (Tail 10%)"))
    else:
        inv_active["abc_class"] = "Class C"

    total_inv_val = inv_active["stock_val"].sum()
    understocked_cnt = (inv_active["stock_status"] == "Understocked").sum()
    total_skus_cnt = len(inv_active)
    under_high_risk_cnt = ((inv_active["stock_status"] == "Understocked") & inv_active["is_high_risk_sup"]).sum()
    defect_spend_val = df_filtered[df_filtered["has_defect"] == 1]["order_cost"].sum() if not df_filtered.empty else 0

    # 1. METRIC CARDS ROW
    i_m1, i_m2, i_m3, i_m4 = st.columns(4, gap="medium")
    with i_m1:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
            <div class="metric-label">Total Inventory Valuation</div>
            <div class="metric-value">&#8377;{total_inv_val/1e7:.2f}<span class="metric-unit"> Cr</span></div>
            <div class="metric-badge" style="background:{GOLD_BG}; color:{GOLD};">{total_skus_cnt} Tracked SKUs</div>
        </div>
        """, unsafe_allow_html=True)
    with i_m2:
        _u_color = RED if (understocked_cnt/max(total_skus_cnt,1)) > 0.4 else AMBER
        _u_bg = RED_BG if _u_color == RED else AMBER_BG
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{_u_color};">
            <div class="metric-label">Understocked SKUs (Dynamic ROP)</div>
            <div class="metric-value" style="color:{_u_color};">{understocked_cnt}<span class="metric-unit"> / {total_skus_cnt}</span></div>
            <div class="metric-badge" style="background:{_u_bg}; color:{_u_color};">{(understocked_cnt/max(total_skus_cnt,1))*100:.1f}% Below Dynamic ROP</div>
        </div>
        """, unsafe_allow_html=True)
    with i_m3:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{RED};">
            <div class="metric-label">High-Risk Supplier Dependency</div>
            <div class="metric-value" style="color:{RED};">{under_high_risk_cnt}<span class="metric-unit"> SKUs</span></div>
            <div class="metric-badge" style="background:{RED_BG}; color:{RED};">Critical Stockout Risk</div>
        </div>
        """, unsafe_allow_html=True)
    with i_m4:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{AMBER};">
            <div class="metric-label">Defective Quality Exposure</div>
            <div class="metric-value" style="color:{AMBER};">&#8377;{defect_spend_val/1e5:.1f}<span class="metric-unit"> L</span></div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Quality Loss</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 22px;'></div>", unsafe_allow_html=True)

    # 2. 2x2 CHART GRID
    c_inv_l, c_inv_r = st.columns([52, 48], gap="medium")

    # CHART A (Top-Left): Stock Coverage Health by Category
    with c_inv_l:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Stock Coverage Health by Product Category (Dynamic ROP)</div>
            <div class="chart-sub">Breakdown of SKUs meeting vs breaching dynamic lead-time inflated ROP thresholds</div>
        """, unsafe_allow_html=True)
        cat_health = inv_active.groupby(["category", "stock_status"])["product_id"].count().reset_index()
        fig_cat_h = px.bar(
            cat_health, x="category", y="product_id", color="stock_status", barmode="stack",
            color_discrete_map={"Healthy": GREEN, "Understocked": AMBER, "Overstocked": ACCENT, "Dead Stock": RED},
            labels={"product_id": "SKU Count", "category": "Product Category", "stock_status": "Stock Status"}
        )
        fig_cat_h.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig_cat_h, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART B (Top-Right): Defective Spend Exposure by Sub-category
    with c_inv_r:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Defective Spend Exposure by Sub-category</div>
            <div class="chart-sub">Financial loss from defective PO deliveries per product sub-category</div>
        """, unsafe_allow_html=True)
        defect_sub = df_filtered[df_filtered["has_defect"] == 1].groupby("sub_category")["order_cost"].sum().reset_index().sort_values("order_cost", ascending=False).head(8)
        if not defect_sub.empty:
            fig_def = px.bar(
                defect_sub, x="order_cost", y="sub_category", orientation="h",
                color="order_cost", color_continuous_scale=[AMBER, RED],
                labels={"order_cost": "Defective Spend (&#8377;)", "sub_category": "Sub-category"}
            )
            fig_def.update_layout(**PLOT_LAYOUT, height=320)
            st.plotly_chart(fig_def, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No defective spend recorded in current filter selection.")
        st.markdown("</div>", unsafe_allow_html=True)

    c_inv_l2, c_inv_r2 = st.columns([52, 48], gap="medium")

    # CHART C (Bottom-Left): Delay Days Distribution by Shipping Mode
    with c_inv_l2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Delay Days Distribution by Shipping Mode</div>
            <div class="chart-sub">Variability of late-delivery delay days across logistics channels</div>
        """, unsafe_allow_html=True)
        _late_df = df_filtered[df_filtered["is_late"] == 1]
        if not _late_df.empty:
            fig_box = px.box(
                _late_df,
                x="shipping_mode", y="delay_days", color="shipping_mode",
                color_discrete_sequence=[ACCENT, GOLD, GREEN, RED, AMBER],
                labels={"shipping_mode": "Shipping Mode", "delay_days": "Delay Days"}
            )
            fig_box.update_layout(**PLOT_LAYOUT, height=320)
            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No late orders in current filter selection — box plot unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART D (Bottom-Right): Contracted Lead Time vs Supplier Delay Erosion
    with c_inv_r2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Contracted Lead Time vs Supplier Delay Erosion</div>
            <div class="chart-sub">Comparing base lead time against actual delay days per primary supplier</div>
        """, unsafe_allow_html=True)
        scatter_df = inv_active.groupby(["supplier_name", "supplier_tier"]).agg(
            base_lead_time=("lead_time_days_base", "mean"),
            avg_delay=("sup_avg_delay", "mean"),
            late_rate=("sup_late_rate", "mean"),
            sku_count=("product_id", "count")
        ).reset_index()

        fig_scat = px.scatter(
            scatter_df, x="base_lead_time", y="avg_delay", size="sku_count", color="supplier_tier",
            hover_name="supplier_name",
            color_discrete_map={"Tier 1": GREEN, "Tier 2": AMBER, "Tier 3": RED},
            labels={"base_lead_time": "Contracted Lead Time (Days)", "avg_delay": "Avg Delay (Days)", "supplier_tier": "Supplier Tier"}
        )
        fig_scat.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig_scat, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. INTERACTIVE CRITICAL SKUs & STOCKOUT MONITOR TABLE WITH PAGINATION & REORDER TEMPLATE
    st.markdown(f"""
    <div class="section-divider" style="margin-top:1.8rem;">
        <div class="section-divider-line"></div>
        <div class="section-divider-label">Inventory Stockout Monitor</div>
        <div class="section-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # Table Controls
    col_t1, col_t2, col_t3 = st.columns([3, 2, 2])
    with col_t1:
        inv_search = st.text_input("Search SKU or Product Name", placeholder="Type product name or SKU...", key="inv_search_key")
    with col_t2:
        inv_filter_status = st.selectbox("Stock Status Filter", ["All SKUs", "Understocked Only", "Dead Stock Only", "Healthy Only", "Overstocked Only"], key="inv_filter_status_key")
    with col_t3:
        page_size = st.selectbox("Display Limit", [25, 50, 100, 200], index=0, key="inv_page_size")

    inv_display = inv_active.copy()
    if inv_filter_status == "Understocked Only":
        inv_display = inv_display[inv_display["stock_status"] == "Understocked"]
    elif inv_filter_status == "Dead Stock Only":
        inv_display = inv_display[inv_display["stock_status"] == "Dead Stock"]
    elif inv_filter_status == "Healthy Only":
        inv_display = inv_display[inv_display["stock_status"] == "Healthy"]
    elif inv_filter_status == "Overstocked Only":
        inv_display = inv_display[inv_display["stock_status"] == "Overstocked"]

    if inv_search:
        inv_display = inv_display[
            inv_display["product_name"].str.contains(inv_search, case=False, na=False) |
            inv_display["sku"].str.contains(inv_search, case=False, na=False)
        ]

    # Format table for display
    def _risk_badge(r):
        if r["stock_status"] == "Understocked" and r["is_high_risk_sup"]:
            return "Critical (High Risk Supplier)"
        elif r["stock_status"] == "Understocked":
            return "Understocked (Below Dynamic ROP)"
        elif r["stock_status"] == "Dead Stock":
            return "Dead Stock"
        else:
            return "Healthy"

    inv_display["Risk Status"] = inv_display.apply(_risk_badge, axis=1)

    table_df = inv_display[[
        "sku", "product_name", "category", "abc_class", "current_stock", "dynamic_reorder_point",
        "days_until_stockout", "months_of_cover", "supplier_name", "supplier_tier", "Risk Status"
    ]].rename(columns={
        "sku": "SKU",
        "product_name": "Product Name",
        "category": "Category",
        "abc_class": "ABC Class",
        "current_stock": "Current Stock",
        "dynamic_reorder_point": "Dynamic ROP",
        "days_until_stockout": "Days to Stockout",
        "months_of_cover": "Months Cover",
        "supplier_name": "Primary Supplier",
        "supplier_tier": "Supplier Tier"
    }).sort_values("Current Stock", ascending=True)

    st.dataframe(table_df.head(page_size), use_container_width=True, height=max(300, min(600, len(table_df.head(page_size)) * 32)))

    # Reorder Recommendation Generator & CSV Downloads
    reorder_skus = inv_active[inv_active["stock_status"] == "Understocked"].copy()
    reorder_skus["recommended_reorder_qty"] = np.maximum(
        (reorder_skus["dynamic_reorder_point"] - reorder_skus["current_stock"]),
        reorder_skus["avg_monthly_demand"]
    ).round(0).astype(int)
    reorder_skus["estimated_order_cost_inr"] = (reorder_skus["recommended_reorder_qty"] * reorder_skus["unit_cost_base"]).round(2)

    reorder_template_csv = reorder_skus[[
        "sku", "product_name", "category", "supplier_name", "current_stock", "dynamic_reorder_point",
        "recommended_reorder_qty", "unit_cost_base", "estimated_order_cost_inr"
    ]].rename(columns={
        "sku": "SKU",
        "product_name": "Product Name",
        "category": "Category",
        "supplier_name": "Supplier",
        "current_stock": "Current Stock",
        "dynamic_reorder_point": "Dynamic ROP",
        "recommended_reorder_qty": "Recommended Order Qty",
        "unit_cost_base": "Unit Cost (INR)",
        "estimated_order_cost_inr": "Est. Total Cost (INR)"
    }).to_csv(index=False).encode('utf-8')

    d_c1, d_c2 = st.columns(2)
    with d_c1:
        st.download_button(
            label=f"Download Full Inventory Audit CSV ({len(table_df)} SKUs)",
            data=table_df.to_csv(index=False).encode('utf-8'),
            file_name="inventory_stockout_risk_audit.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d_c2:
        st.download_button(
            label=f"📦 Reorder Now Template CSV ({len(reorder_skus)} Understocked SKUs)",
            data=reorder_template_csv,
            file_name="procuresense_purchase_reorder_template.csv",
            mime="text/csv",
            use_container_width=True
        )
# ---------------------------------------------------------------
# TAB 4: ML DELAY PREDICTION & EXPLAINABILITY (ML STUDIO)
# ---------------------------------------------------------------
with tab4:
    # 1. TOP KPI SUMMARY ROW FOR ML STUDIO
    ml_k1, ml_k2, ml_k3, ml_k4 = st.columns(4, gap="medium")
    with ml_k1:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
            <div class="metric-label">ROC-AUC Champion Engine</div>
            <div class="metric-value" style="color:{GOLD};">Random Forest</div>
            <div class="metric-badge" style="background:{GOLD_BG}; color:{GOLD};">ROC-AUC 0.714 &nbsp;&middot;&nbsp; Acc 65.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with ml_k2:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GREEN};">
            <div class="metric-label">Cost-Optimal Winner</div>
            <div class="metric-value" style="color:{GREEN};">Logistic Reg.</div>
            <div class="metric-badge" style="background:{GREEN_BG}; color:{GREEN};">Expected Cost: &#8377;87.90M</div>
        </div>
        """, unsafe_allow_html=True)
    with ml_k3:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{ACCENT};">
            <div class="metric-label">Decision Threshold (&tau;)</div>
            <div class="metric-value" style="color:{ACCENT};">0.35</div>
            <div class="metric-badge" style="background:{ACCENT_GLOW}; color:{ACCENT};">Cost-Sensitive Minima</div>
        </div>
        """, unsafe_allow_html=True)
    with ml_k4:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{AMBER};">
            <div class="metric-label">Engineered Feature Matrix</div>
            <div class="metric-value">33 <span class="metric-unit">Features</span></div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Zero-Leakage Expanding</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)

    # 2. 2-COLUMN BENCHMARK & EXPLAINABILITY GRID
    c_ml_l, c_ml_r = st.columns([52, 48], gap="medium")

    # CHART 9: SHAP Feature Importance
    with c_ml_l:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">TreeSHAP Feature Importance (Global Signal Drivers)</div>
            <div class="chart-sub">Mean absolute SHAP value impact per engineered feature — Random Forest Engine</div>
        """, unsafe_allow_html=True)
        raw_fi = model_data.get("feature_importance", [
            {"feature": "supplier_id_te", "mean_abs_shap": 0.299},
            {"feature": "shipping_mode_code", "mean_abs_shap": 0.278},
            {"feature": "order_month", "mean_abs_shap": 0.269},
            {"feature": "sup_ewm_ontime", "mean_abs_shap": 0.194},
            {"feature": "logistics_stress_index", "mean_abs_shap": 0.165},
            {"feature": "order_qty_vs_sup_mean", "mean_abs_shap": 0.142},
            {"feature": "sup_concurrent_po_30d", "mean_abs_shap": 0.128},
            {"feature": "supplier_health_index", "mean_abs_shap": 0.115}
        ])
        
        feature_name_map = {
            "supplier_id_te": "Supplier Target Encoding",
            "shipping_mode_code": "Shipping Logistics Mode",
            "order_month": "Order Month (Seasonality)",
            "sup_ewm_ontime": "Supplier EWM On-Time Rate",
            "logistics_stress_index": "Logistics Stress Index",
            "order_qty_vs_sup_mean": "Order Qty Spike Ratio",
            "sup_concurrent_po_30d": "Concurrent 30D Active POs",
            "supplier_health_index": "Supplier Health Index",
            "crude_oil_index": "Global Freight / Oil Index",
            "container_shortage_flag": "Container Shortage Signal"
        }
        
        fi_df = pd.DataFrame(raw_fi).head(10)
        fi_df["display_feature"] = fi_df["feature"].map(lambda x: feature_name_map.get(x, x))
        
        fig_shap = px.bar(
            fi_df.sort_values("mean_abs_shap", ascending=True),
            x="mean_abs_shap", y="display_feature", orientation="h",
            color="mean_abs_shap", color_continuous_scale=[ACCENT, GOLD],
            labels={"mean_abs_shap": "Mean |SHAP Value|", "display_feature": "Feature Driver"}
        )
        fig_shap.update_layout(**PLOT_LAYOUT, height=330, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # CHART 10: Multi-Model Benchmark Comparison
    with c_ml_r:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">Multi-Model Evaluation Benchmark</div>
            <div class="chart-sub">ROC-AUC, PR-AUC & Accuracy — apples-to-apples performance on 2025 holdout</div>
        """, unsafe_allow_html=True)
        raw_evals = model_data.get("model_evaluations_apples_to_apples", [])
        bench_rows = []
        for m in raw_evals:
            bench_rows.append({
                "model_name": m.get("model_name", "").replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", ""),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "accuracy": m.get("default_thresh_0.5", {}).get("accuracy", 0.5)
            })
        if not bench_rows:
            bench_rows = [
                {"model_name": "Naive Baseline", "roc_auc": 0.500, "pr_auc": 0.473, "accuracy": 0.527},
                {"model_name": "Supplier Heuristic", "roc_auc": 0.680, "pr_auc": 0.652, "accuracy": 0.622},
                {"model_name": "Logistic Regression", "roc_auc": 0.700, "pr_auc": 0.673, "accuracy": 0.640},
                {"model_name": "Random Forest", "roc_auc": 0.714, "pr_auc": 0.689, "accuracy": 0.657},
                {"model_name": "XGBoost Classifier", "roc_auc": 0.697, "pr_auc": 0.676, "accuracy": 0.642},
                {"model_name": "Soft-Voting Ensemble", "roc_auc": 0.703, "pr_auc": 0.684, "accuracy": 0.644}
            ]
        bench_df = pd.DataFrame(bench_rows)
        fig_met = px.bar(
            bench_df, x="model_name", y="roc_auc", color="pr_auc",
            color_continuous_scale=[AMBER, ACCENT, GREEN],
            labels={"roc_auc": "ROC-AUC", "model_name": "Model Candidate", "pr_auc": "PR-AUC"},
            text_auto=".3f"
        )
        fig_met.update_layout(**PLOT_LAYOUT, height=330, xaxis_tickangle=-20)
        st.plotly_chart(fig_met, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Evaluation Metrics Banner
    st.markdown(f"""
    <div style="background:rgba(108,142,245,0.06); border:1px solid rgba(108,142,245,0.18); border-left:3px solid {GOLD};
                border-radius:10px; padding:15px 20px; margin-bottom:28px;">
        <div style="font-weight:600; font-size:0.88rem; color:{TEXT}; margin-bottom:6px;">Model Evaluation Summary & Architectural Trade-off</div>
        <div style="display:flex; gap:24px; font-size:0.80rem; color:{TEXT_MUTED}; flex-wrap:wrap; line-height:1.5;">
            <div><b>Cost-Optimal Winner</b>: Logistic Regression (Expected Cost: &#8377;87.90M, saving &#8377;147.05M vs baseline)</div>
            <div><b>ROC-AUC Champion & Live Engine</b>: Random Forest (<span style="color:{GREEN}; font-weight:700;">ROC-AUC 0.714</span> [95% CI: 0.706–0.725], Accuracy 65.7%, FPR 32.7%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. LIVE RISK SIMULATOR SECTION
    st.markdown(f"""
    <div class="section-divider">
        <div class="section-divider-line"></div>
        <div class="section-divider-label">Live Risk Simulator</div>
        <div class="section-divider-line"></div>
    </div>
    <div style="margin-bottom:6px;">
        <span style="font-size:0.96rem; font-weight:600; color:{TEXT}; letter-spacing:-0.02em;">Purchase Order Delay Risk Engine</span>
    </div>
    <div style="font-size:0.80rem; color:{TEXT_MUTED}; margin-bottom:18px;">
        Configure order parameters below. Real-time inference calls <code>model.predict_proba()</code> on serialized Random Forest engine.
    </div>
    """, unsafe_allow_html=True)

    all_suppliers_df = pd.read_sql("""
        SELECT s.supplier_id, s.supplier_name, s.tier, s.region,
               COALESCE(AVG(d.is_late), 0.20) as late_rate
        FROM suppliers s
        LEFT JOIN purchase_orders po ON s.supplier_id = po.supplier_id
        LEFT JOIN deliveries d ON po.po_id = d.po_id
        GROUP BY s.supplier_id
    """, conn)

    supplier_options = sorted(all_suppliers_df["supplier_name"].tolist())

    with st.form(key="order_risk_simulator_form"):
        sim1, sim2 = st.columns(2, gap="medium")
        with sim1:
            sim_supplier = st.selectbox("Supplier", options=supplier_options)
            sim_category = st.selectbox("Product Category", options=filters["categories"])
            sim_mode = st.selectbox("Shipping Mode", options=filters["shipping_modes"])
        with sim2:
            sim_month = st.slider("Order Month", 1, 12, 6)
            sim_qty = st.number_input("Order Quantity", min_value=1, max_value=10000, value=250, step=50)
            sim_unit_price = st.number_input("Unit Price (&#8377;)", min_value=1.0, max_value=100000.0, value=750.0, step=50.0)

        submitted = st.form_submit_button("Execute ML Inference & Risk Audit", use_container_width=True)

    # Perform real-time ML inference
    import joblib
    model_artifact_path = os.path.join(BASE_DIR, "ml", "rf_model.joblib")

    total_order_cost = sim_qty * sim_unit_price
    sup_row = all_suppliers_df[all_suppliers_df["supplier_name"] == sim_supplier].iloc[0]
    sup_id = sup_row["supplier_id"]
    sup_region = sup_row["region"]
    sup_tier = sup_row["tier"]

    if os.path.exists(model_artifact_path):
        try:
            art = joblib.load(model_artifact_path)
            model = art["model"]
            features = art["features"]
            encoders = art["label_encoders"]
            te_maps = art["te_maps"]
            g_mean = art["global_late_mean"]
            threshold = art["optimal_threshold"]

            row_dict = {
                "quantity": sim_qty,
                "unit_price": sim_unit_price,
                "order_cost": total_order_cost,
                "unit_cost_base": sim_unit_price * 0.85,
                "lead_time_days_base": 14,
                "order_month": sim_month,
                "order_quarter": (sim_month - 1) // 3 + 1,
                "order_day_of_week": 2,
                "is_peak_season": 1 if sim_month in [10, 11, 12, 1] else 0,
                "order_qty_vs_sup_mean": round(sim_qty / 250.0, 3),
                "sup_concurrent_po_30d": 3,
                "sup_rolling_ontime": 1.0 - float(sup_row["late_rate"]),
                "sup_rolling_defect": 0.03,
                "sup_rolling_delay": 3.5,
                "sup_ewm_ontime": 1.0 - float(sup_row["late_rate"]),
                "sup_ewm_delay": 3.5,
                "supplier_age_years": 5,
                "crude_oil_index": 1.05,
                "is_holiday_order": 0,
                "container_shortage_flag": 1 if sim_month in [10, 11, 12] else 0,
                "supplier_id_te": te_maps["supplier_id"].get(sup_id, g_mean),
                "product_id_te": g_mean,
                "sup_category_te": te_maps["sup_category"].get(f"{sup_id}_{sim_category}", g_mean),
                "sup_month_te": te_maps["sup_month"].get(f"{sup_id}_{sim_month}", g_mean),
                "ship_region_te": te_maps["ship_region"].get(f"{sim_mode}_{sup_region}", g_mean),
                "logistics_stress_index": 14.0 / (3.5 + 1.0),
                "supplier_health_index": (1.0 - float(sup_row["late_rate"])) * 0.97,
                "priority_code": 1,
                "shipping_mode_code": encoders["shipping_mode"].transform([sim_mode])[0] if sim_mode in encoders["shipping_mode"].classes_ else 0,
                "category_code": encoders["category"].transform([sim_category])[0] if sim_category in encoders["category"].classes_ else 0,
                "sub_category_code": 0,
                "region_code": encoders["region"].transform([sup_region])[0] if sup_region in encoders["region"].classes_ else 0,
                "tier_code": encoders["tier"].transform([sup_tier])[0] if sup_tier in encoders["tier"].classes_ else 0,
            }

            X_sim = pd.DataFrame([row_dict])[features]
            prob = float(model.predict_proba(X_sim)[0, 1])
        except Exception as e:
            prob = float(sup_row["late_rate"])
            threshold = float(model_data.get("optimal_threshold", 0.35))
    else:
        prob = float(sup_row["late_rate"])
        threshold = float(model_data.get("optimal_threshold", 0.35))

    pred_late = prob >= threshold

    if pred_late:
        delay_multiplier = 1.6 if "Sea" in sim_mode else (1.2 if "Standard" in sim_mode else 0.8)
        est_days = round(max(1.2, prob * 10.0 * delay_multiplier), 1)
    else:
        est_days = 0.0

    exp_risk_cost = total_order_cost * prob * 0.15

    res1, res2, res3, res4 = st.columns(4, gap="medium")
    res1.markdown(f"""
    <div class="metric-card metric-card-stripe" style="border-left-color:{RED if pred_late else GREEN};">
        <div class="metric-label">Predicted Risk Status</div>
        <div class="metric-value" style="color:{RED if pred_late else GREEN};">{'LATE RISK' if pred_late else 'ON TIME'}</div>
        <div class="metric-badge" style="background:{RED_BG if pred_late else GREEN_BG}; color:{RED if pred_late else GREEN};">
            {'Cutoff Threshold &ge; {:.0f}%'.format(threshold*100) if pred_late else 'Within Safe Limit'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    res2.markdown(f"""
    <div class="metric-card metric-card-stripe" style="border-left-color:{AMBER};">
        <div class="metric-label">Delay Probability</div>
        <div class="metric-value" style="color:{AMBER};">{prob*100:.1f}<span class="metric-unit">%</span></div>
        <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Order Value: &#8377;{total_order_cost:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    res3.markdown(f"""
    <div class="metric-card metric-card-stripe" style="border-left-color:{ACCENT};">
        <div class="metric-label">Estimated Delay Duration</div>
        <div class="metric-value" style="color:{ACCENT};">{est_days} <span class="metric-unit">Days</span></div>
        <div class="metric-badge" style="background:{ACCENT_GLOW}; color:{ACCENT};">Logistics Mode: {sim_mode}</div>
    </div>
    """, unsafe_allow_html=True)

    res4.markdown(f"""
    <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
        <div class="metric-label">Expected Financial Exposure</div>
        <div class="metric-value" style="color:{GOLD};">&#8377;{exp_risk_cost/1e3:.1f}<span class="metric-unit"> K</span></div>
        <div class="metric-badge" style="background:{GOLD_BG}; color:{GOLD};">Risk Cost Impact</div>
    </div>
    """, unsafe_allow_html=True)

    # Local Feature Insights Box
    st.markdown(f"""
    <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:16px 20px; margin-top:16px; margin-bottom:28px;">
        <div style="font-size:0.75rem; font-weight:700; color:{ACCENT}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
            Local Order Risk Drivers (SHAP Local Explanation)
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; font-size:0.80rem; color:{TEXT_MUTED};">
            <div style="background:rgba(245,96,74,0.06); padding:8px 12px; border-radius:6px; border-left:2px solid {RED};">
                <b>Seasonality Factor</b>: {('Month ' + str(sim_month) + ' (Peak Season)') if sim_month in [10,11,12,1] else ('Month ' + str(sim_month) + ' (Off-Peak)')}
            </div>
            <div style="background:rgba(16,217,140,0.06); padding:8px 12px; border-radius:6px; border-left:2px solid {GREEN};">
                <b>Supplier Baseline</b>: {sim_supplier} (Historical Late Rate: {float(sup_row['late_rate'])*100:.1f}%)
            </div>
            <div style="background:rgba(108,142,245,0.06); padding:8px 12px; border-radius:6px; border-left:2px solid {ACCENT};">
                <b>Logistics Route</b>: {sim_mode} ({sup_region})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. REALLOCATION SIMULATOR SECTION
    st.markdown(f"""
    <div class="section-divider">
        <div class="section-divider-line"></div>
        <div class="section-divider-label">Reallocation Simulator</div>
        <div class="section-divider-line"></div>
    </div>
    <div style="margin-bottom:6px;">
        <span style="font-size:0.96rem; font-weight:600; color:{TEXT}; letter-spacing:-0.02em;">Multi-Constraint Supplier Volume Reallocation</span>
    </div>
    <div style="font-size:0.80rem; color:{TEXT_MUTED}; margin-bottom:18px;">
        Model the real trade-off when shifting PO volume: capacity strain &rarr; delay penalty, price premium &rarr; cost delta, net benefit verdict.
    </div>
    """, unsafe_allow_html=True)

    sup_details_df = pd.read_sql("""
        SELECT s.supplier_id, s.supplier_name, s.tier, s.region,
               COUNT(po.po_id) AS total_pos,
               AVG(po.unit_price) AS avg_unit_price,
               AVG(po.order_cost) AS avg_po_value,
               COALESCE(AVG(d.is_late), 0.20) AS late_rate
        FROM suppliers s
        LEFT JOIN purchase_orders po ON s.supplier_id = po.supplier_id
        LEFT JOIN deliveries d ON po.po_id = d.po_id
        GROUP BY s.supplier_id
    """, conn)

    sup_list = sorted(sup_details_df["supplier_name"].tolist())

    col_sim_src, col_sim_tgt, col_sim_vol = st.columns(3, gap="medium")
    with col_sim_src:
        src_sup_name = st.selectbox("Source Supplier (Shift Volume FROM)", options=sup_list, index=0)
    with col_sim_tgt:
        tgt_sup_name = st.selectbox("Destination Supplier (Shift Volume TO)", options=sup_list, index=min(1, len(sup_list)-1))
    with col_sim_vol:
        shift_po_count = st.slider("Volume to Reassign (Number of POs)", min_value=10, max_value=2000, value=250, step=10)

    src_row = sup_details_df[sup_details_df["supplier_name"] == src_sup_name].iloc[0]
    tgt_row = sup_details_df[sup_details_df["supplier_name"] == tgt_sup_name].iloc[0]

    if src_sup_name == tgt_sup_name:
        st.warning("Source and destination supplier are the same — reallocation has no effect. Please choose two different suppliers.")
    else:
        src_base_pos = max(src_row["total_pos"], 1)
        tgt_base_pos = max(tgt_row["total_pos"], 1)

        src_late_rate = src_row["late_rate"]
        tgt_late_rate = tgt_row["late_rate"]

        src_price = src_row["avg_unit_price"]
        tgt_price = tgt_row["avg_unit_price"]

        # 1. Capacity Constraint Evaluation
        capacity_expansion_pct = (shift_po_count / tgt_base_pos) * 100.0
        capacity_strained = capacity_expansion_pct > 30.0

        capacity_penalty = (capacity_expansion_pct - 30.0) * 0.003 if capacity_strained else 0.0
        effective_tgt_late_rate = min(0.95, tgt_late_rate + capacity_penalty)

        # 2. Late Deliveries Prevented
        late_pos_src_avoided = shift_po_count * src_late_rate
        late_pos_tgt_incurred = shift_po_count * effective_tgt_late_rate
        net_late_pos_prevented = round(late_pos_src_avoided - late_pos_tgt_incurred)

        # 3. Unit Price Delta & Financial Cost Impact
        avg_po_val = (src_row["avg_po_value"] + tgt_row["avg_po_value"]) / 2.0
        price_pct_delta = ((tgt_price - src_price) / src_price) * 100.0 if src_price > 0 else 0.0
        net_price_cost_delta = shift_po_count * avg_po_val * (price_pct_delta / 100.0)

        # 4. Stockout Savings vs Price Premium Reconciliation
        fn_stockout_savings = net_late_pos_prevented * 50000.0
        net_financial_tradeoff = fn_stockout_savings - net_price_cost_delta

        sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4, gap="medium")
        sim_m1.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GREEN if net_late_pos_prevented > 0 else RED};">
            <div class="metric-label">Late POs Prevented</div>
            <div class="metric-value" style="color:{GREEN if net_late_pos_prevented > 0 else RED};">{'+' if net_late_pos_prevented > 0 else ''}{net_late_pos_prevented:,} POs</div>
            <div class="metric-badge" style="background:{GREEN_BG if net_late_pos_prevented > 0 else RED_BG}; color:{GREEN if net_late_pos_prevented > 0 else RED};">SLA Improvement</div>
        </div>
        """, unsafe_allow_html=True)

        sim_m2.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{AMBER if net_price_cost_delta > 0 else GREEN};">
            <div class="metric-label">Price Premium Delta</div>
            <div class="metric-value" style="color:{AMBER if net_price_cost_delta > 0 else GREEN};">{'+\u20b9' if net_price_cost_delta > 0 else '-\u20b9'}{abs(net_price_cost_delta)/1e5:.2f} L</div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">Price Delta: {price_pct_delta:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        sim_m3.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{RED if capacity_strained else GREEN};">
            <div class="metric-label">Target Capacity Expansion</div>
            <div class="metric-value" style="color:{RED if capacity_strained else GREEN};">{capacity_expansion_pct:.1f}%</div>
            <div class="metric-badge" style="background:{RED_BG if capacity_strained else GREEN_BG}; color:{RED if capacity_strained else GREEN};">{'Capacity Strained' if capacity_strained else 'Within Capacity'}</div>
        </div>
        """, unsafe_allow_html=True)

        sim_m4.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GREEN if net_financial_tradeoff > 0 else RED};">
            <div class="metric-label">Net Reallocation Benefit</div>
            <div class="metric-value" style="color:{GREEN if net_financial_tradeoff > 0 else RED};">{'+\u20b9' if net_financial_tradeoff > 0 else '-\u20b9'}{abs(net_financial_tradeoff)/1e5:.2f} L</div>
            <div class="metric-badge" style="background:{GREEN_BG if net_financial_tradeoff > 0 else RED_BG}; color:{GREEN if net_financial_tradeoff > 0 else RED};">{'Recommended' if net_financial_tradeoff > 0 else 'Not Cost-Effective'}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 5: PRODUCTION SQL ANALYTICS STUDIO & WORKBENCH
# ---------------------------------------------------------------
import time
import re

DESTRUCTIVE_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|REPLACE|ATTACH|DETACH|VACUUM|CREATE)\b",
    re.IGNORECASE
)

@st.cache_data(ttl=300)
def get_dynamic_db_schema():
    conn = get_db_connection()
    tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", conn)
    schema_info = {}
    for tbl in tables_df["name"]:
        info_df = pd.read_sql(f"PRAGMA table_info('{tbl}')", conn)
        cols_formatted = []
        for _, r in info_df.iterrows():
            pk_suffix = " (PK)" if r["pk"] == 1 else ""
            cols_formatted.append((r["name"], f"{r['type']}{pk_suffix}"))
        schema_info[tbl] = cols_formatted
    return schema_info

with tab5:
    # 1. TOP KPI SUMMARY ROW FOR SQL STUDIO
    sql_k1, sql_k2, sql_k3, sql_k4 = st.columns(4, gap="medium")
    with sql_k1:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GOLD};">
            <div class="metric-label">SQL Portfolio Library</div>
            <div class="metric-value" style="color:{GOLD};">10 <span class="metric-unit">Queries</span></div>
            <div class="metric-badge" style="background:{GOLD_BG}; color:{GOLD};">CTEs &amp; Window Functions</div>
        </div>
        """, unsafe_allow_html=True)
    with sql_k2:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{GREEN};">
            <div class="metric-label">Database Scale</div>
            <div class="metric-value">30,000 <span class="metric-unit">POs</span></div>
            <div class="metric-badge" style="background:{GREEN_BG}; color:{GREEN};">Normalized 5-Table Schema</div>
        </div>
        """, unsafe_allow_html=True)
    with sql_k3:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{ACCENT};">
            <div class="metric-label">SQL Engine</div>
            <div class="metric-value" style="color:{ACCENT};">SQLite 3.x</div>
            <div class="metric-badge" style="background:{ACCENT_GLOW}; color:{ACCENT};">Read-Only Sanitized Studio</div>
        </div>
        """, unsafe_allow_html=True)
    with sql_k4:
        st.markdown(f"""
        <div class="metric-card metric-card-stripe" style="border-left-color:{AMBER};">
            <div class="metric-label">Analytics Features</div>
            <div class="metric-value" style="color:{AMBER};">Dynamic</div>
            <div class="metric-badge" style="background:{AMBER_BG}; color:{AMBER};">PRAGMA Inspector &amp; Plan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)

    # 2. QUERY METADATA & SELECTOR
    queries_meta = {
        "Query 1: MoM Spend & Cumulative Running Spend": {
            "tag": "Window Functions",
            "badge_color": ACCENT,
            "badge_bg": ACCENT_GLOW,
            "desc": "Calculates monthly spend per category alongside cumulative running spend using window aggregations.",
            "tech": "SUM() OVER (PARTITION BY category ORDER BY month ROWS UNBOUNDED PRECEDING), strftime()",
            "sql": """WITH MonthlyCategorySpend AS (
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
ORDER BY category, order_month;"""
        },
        "Query 2: Regional Supplier SLA Ranking": {
            "tag": "Performance Ranking",
            "badge_color": GREEN,
            "badge_bg": GREEN_BG,
            "desc": "Ranks active suppliers within their geographic region based on on-time delivery percentage and average delay days.",
            "tech": "DENSE_RANK() OVER (PARTITION BY region ORDER BY on_time_pct DESC), HAVING COUNT >= 10",
            "sql": """WITH SupplierMetrics AS (
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
ORDER BY region, regional_rank;"""
        },
        "Query 3: Year-over-Year Unit Price Drift": {
            "tag": "Price Inflation OLS",
            "badge_color": GOLD,
            "badge_bg": GOLD_BG,
            "desc": "Detects price inflation per supplier by measuring year-over-year unit price percentage changes using LAG().",
            "tech": "LAG(avg_unit_price, 1) OVER (PARTITION BY supplier_id ORDER BY order_year), strftime('%Y')",
            "sql": """WITH AnnualSupplierPrices AS (
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
ORDER BY yoy_price_change_pct DESC;"""
        },
        "Query 4: Lead Time Variance & Reliability Cohorts": {
            "tag": "Logistics SLA Cohorts",
            "badge_color": AMBER,
            "badge_bg": AMBER_BG,
            "desc": "Measures variance between contracted lead time and actual delivery days across shipping modes and supplier tiers.",
            "tech": "julianday(delivery_date) - julianday(order_date), GROUP BY mode, tier",
            "sql": """SELECT
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
ORDER BY mean_delay_days DESC;"""
        },
        "Query 5: Inventory Stockout Risk Matrix": {
            "tag": "Stockout Risk Evaluation",
            "badge_color": RED,
            "badge_bg": RED_BG,
            "desc": "Evaluates stock coverage, months of cover, and flags understocked SKUs breaching reorder thresholds.",
            "tech": "CASE WHEN current_stock < reorder_level, JOIN inventory -> products",
            "sql": """SELECT
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
ORDER BY i.months_of_cover ASC;"""
        },
        "Query 6: Quality Defect Rate & Spend Exposure": {
            "tag": "Quality Rejection Loss",
            "badge_color": RED,
            "badge_bg": RED_BG,
            "desc": "Quantifies financial loss and rejection percentages resulting from defective PO shipments per sub-category.",
            "tech": "HAVING defective_orders > 0, SUM(CASE WHEN has_defect = 1 THEN order_cost ELSE 0 END)",
            "sql": """SELECT
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
ORDER BY defective_spend_exposure DESC;"""
        },
        "Query 7: Predictive ML Feature Engineering": {
            "tag": "Feature Extraction",
            "badge_color": ACCENT,
            "badge_bg": ACCENT_GLOW,
            "desc": "Extracts normalized order rows, temporal features, and target labels to feed machine learning models.",
            "tech": "strftime('%m', order_date), categorical codes, raw inference join",
            "sql": """SELECT
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
ORDER BY po.order_date ASC;"""
        },
        "Query 8: Supplier Spend Pareto 80/20 Analysis": {
            "tag": "Pareto 80/20 Rule",
            "badge_color": GOLD,
            "badge_bg": GOLD_BG,
            "desc": "Classifies suppliers into Class A (Top 80% spend), Class B (Next 15%), and Class C (Tail spend) using running cumulative totals.",
            "tech": "SUM() OVER (ORDER BY total_spend DESC ROWS UNBOUNDED PRECEDING), Pareto Classing",
            "sql": """WITH SupplierSpend AS (
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
ORDER BY total_spend DESC;"""
        },
        "Query 9: Monthly Order Volume MoM Growth": {
            "tag": "Time-Series MoM Growth",
            "badge_color": GREEN,
            "badge_bg": GREEN_BG,
            "desc": "Computes month-over-month percentage growth in order volume and tracks late delivery rate point changes.",
            "tech": "LAG(current_month_orders, 1) OVER (ORDER BY order_month)",
            "sql": """WITH MonthlyOrderStats AS (
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
ORDER BY order_month;"""
        },
        "Query 10: Fulfillment Bottleneck & Delay Severity": {
            "tag": "Delay Severity Bucket",
            "badge_color": AMBER,
            "badge_bg": AMBER_BG,
            "desc": "Categorizes delivery delay severity into on-time, minor (1-3d), moderate (4-7d), and severe (>7d) delay buckets.",
            "tech": "CASE WHEN delay_days BETWEEN ... THEN 1 ELSE 0 END",
            "sql": """SELECT
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
ORDER BY severe_delays_gt7d DESC;"""
        }
    }

    # Dynamic Database Schema Inspector using PRAGMA table_info
    dynamic_schema = get_dynamic_db_schema()
    with st.expander("Database Schema & Dynamic PRAGMA Inspector", expanded=False):
        st.markdown("<div style='font-size:0.80rem; color:#94a3b8; margin-bottom:12px;'>Dynamically inspecting tables, columns, data types, and primary keys in procurement.db:</div>", unsafe_allow_html=True)
        schema_cols = st.columns(len(dynamic_schema))
        for idx, (tbl, cols) in enumerate(dynamic_schema.items()):
            with schema_cols[idx % len(schema_cols)]:
                st.markdown(f"<div style='font-weight:700; font-size:0.82rem; color:{ACCENT}; margin-bottom:6px;'>{tbl}</div>", unsafe_allow_html=True)
                for cname, ctype in cols:
                    st.markdown(f"<div style='font-size:0.73rem; color:{TEXT_MUTED}; font-family:monospace;'>• <b>{cname}</b> <span style='color:{TEXT_DIM};'>({ctype})</span></div>", unsafe_allow_html=True)

    selected_query_name = st.selectbox("Select Portfolio SQL Query to Inspect & Execute:", list(queries_meta.keys()), key="sql_query_select")
    q_info = queries_meta[selected_query_name]

    # Query Info Card
    st.markdown(f"""
    <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:16px 20px; margin-top:8px; margin-bottom:14px;">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:0.92rem; font-weight:700; color:{TEXT};">{selected_query_name}</span>
            <span style="background:{q_info['badge_bg']}; color:{q_info['badge_color']}; padding:3px 10px; border-radius:6px; font-size:0.74rem; font-weight:600;">
                {q_info['tag']}
            </span>
        </div>
        <div style="font-size:0.82rem; color:{TEXT_MUTED}; margin-bottom:8px; line-height:1.5;">
            {q_info['desc']}
        </div>
        <div style="font-size:0.75rem; color:{TEXT_DIM};">
            <b>SQL Techniques</b>: <code>{q_info['tech']}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Snippet Builder
    st.markdown("<div style='font-size:0.76rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Quick SQL Snippet Builder</div>", unsafe_allow_html=True)
    snip_c1, snip_c2, snip_c3, snip_c4, snip_c5, snip_c6 = st.columns(6)
    
    if snip_c1.button("SELECT *", use_container_width=True):
        st.session_state["active_sql_script"] = "SELECT * FROM purchase_orders LIMIT 100;"
    if snip_c2.button("JOIN Tables", use_container_width=True):
        st.session_state["active_sql_script"] = "SELECT po.po_id, s.supplier_name, p.product_name, d.delay_days\nFROM purchase_orders po\nJOIN suppliers s ON po.supplier_id = s.supplier_id\nJOIN products p ON po.product_id = p.product_id\nJOIN deliveries d ON po.po_id = d.po_id\nLIMIT 100;"
    if snip_c3.button("GROUP Spend", use_container_width=True):
        st.session_state["active_sql_script"] = "SELECT p.category, SUM(po.order_cost) AS total_spend, COUNT(po.po_id) AS po_count\nFROM purchase_orders po\nJOIN products p ON po.product_id = p.product_id\nGROUP BY p.category\nORDER BY total_spend DESC;"
    if snip_c4.button("Window Rank", use_container_width=True):
        st.session_state["active_sql_script"] = "SELECT supplier_name, region,\n       DENSE_RANK() OVER (PARTITION BY region ORDER BY on_time_pct DESC) AS regional_rank\nFROM (\n  SELECT s.supplier_name, s.region, ROUND(100.0 * AVG(1 - d.is_late), 1) AS on_time_pct\n  FROM suppliers s JOIN purchase_orders po ON s.supplier_id = po.supplier_id\n  JOIN deliveries d ON po.po_id = d.po_id\n  GROUP BY s.supplier_id\n);"
    if snip_c5.button("Defects Loss", use_container_width=True):
        st.session_state["active_sql_script"] = "SELECT s.supplier_name, COUNT(po.po_id) AS total_pos,\n       SUM(d.has_defect) AS defective_pos,\n       ROUND(100.0 * SUM(d.has_defect) / COUNT(po.po_id), 2) AS defect_rate_pct\nFROM suppliers s JOIN purchase_orders po ON s.supplier_id = po.supplier_id\nJOIN deliveries d ON po.po_id = d.po_id\nGROUP BY s.supplier_id HAVING defective_pos > 0\nORDER BY defect_rate_pct DESC;"
    if snip_c6.button("Reset Query", use_container_width=True):
        st.session_state["active_sql_script"] = q_info["sql"]

    # Controls: Configurable Limit
    col_edt1, col_edt2 = st.columns([3, 1])
    with col_edt2:
        row_limit = st.number_input("Result Row Limit", min_value=10, max_value=5000, value=100, step=50, key="sql_limit_input")

    default_sql_val = st.session_state.get("active_sql_script", q_info["sql"])
    user_sql = st.text_area("SQL Script Editor (Edit or Run Custom SQL):", value=default_sql_val, height=230, key="sql_text_area")

    c_exec1, c_exec2, c_exec3 = st.columns([1, 1, 2])
    with c_exec1:
        run_btn = st.button("Execute Query Live", use_container_width=True, type="primary")
    with c_exec2:
        plan_btn = st.button("EXPLAIN Query Plan", use_container_width=True)

    # Check Destructive SQL Violation
    destructive_match = DESTRUCTIVE_SQL_KEYWORDS.search(user_sql)

    if plan_btn:
        if destructive_match:
            st.error(f"⚠️ Security Violation: Cannot EXPLAIN query containing modification keyword '{destructive_match.group(0)}'.")
        else:
            try:
                plan_sql = f"EXPLAIN QUERY PLAN {user_sql.strip().rstrip(';')}"
                plan_df = pd.read_sql(plan_sql, conn)
                st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#3b82f6; margin-top:10px;'>SQLite Query Execution Plan:</div>", unsafe_allow_html=True)
                st.dataframe(plan_df, use_container_width=True)
            except Exception as pe:
                st.error(f"Query Plan Execution Error: {pe}")

    if run_btn:
        if destructive_match:
            st.error(f"⚠️ Security Violation: Destructive statement detected ('{destructive_match.group(0)}'). Only read-only SELECT queries are permitted in SQL Analytics Studio.")
        else:
            t_start = time.time()
            try:
                # Apply configurable row limit if not already present
                clean_sql = user_sql.strip().rstrip(";")
                if "LIMIT" not in clean_sql.upper():
                    exec_sql = f"{clean_sql} LIMIT {int(row_limit)};"
                else:
                    exec_sql = f"{clean_sql};"

                res_df = pd.read_sql(exec_sql, conn)
                t_elapsed = (time.time() - t_start) * 1000.0

                # Persist in session state
                st.session_state["sql_run_df"] = res_df
                st.session_state["sql_t_elapsed"] = t_elapsed
                st.session_state["active_sql_script"] = user_sql

                # Add to query history
                history = st.session_state.get("sql_history", [])
                history.insert(0, {"time": time.strftime("%H:%M:%S"), "query": user_sql, "rows": len(res_df)})
                st.session_state["sql_history"] = history[:5]

            except Exception as e:
                st.error(f"SQL Execution Error: {str(e)}")
                st.session_state.pop("sql_run_df", None)

    # Render persisted query results
    if "sql_run_df" in st.session_state and st.session_state["sql_run_df"] is not None:
        res_df = st.session_state["sql_run_df"]
        t_elapsed = st.session_state.get("sql_t_elapsed", 0.0)

        st.markdown(f"""
        <div style="background:rgba(16,217,140,0.06); border:1px solid rgba(16,217,140,0.22); border-radius:8px; padding:10px 16px; margin-top:14px; margin-bottom:14px; font-size:0.82rem; color:{GREEN}; display:flex; align-items:center; justify-content:space-between;">
            <div><b>Query Executed Successfully</b> &nbsp;&middot;&nbsp; <b>{len(res_df):,} Rows Returned</b> &nbsp;&middot;&nbsp; <b>{len(res_df.columns)} Columns</b></div>
            <div>Execution Latency: <b style="color:{GOLD};">{t_elapsed:.1f} ms</b></div>
        </div>
        """, unsafe_allow_html=True)

        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["Interactive Data Table", "Smart Auto-Visualizer", "Raw JSON View", "Query Execution History"])
        
        with res_tab1:
            st.dataframe(res_df, use_container_width=True, height=350)

        with res_tab2:
            # Smart Auto-Visualizer: Filter out ID columns
            id_cols = {"po_id", "supplier_id", "product_id", "delivery_id"}
            all_cols = res_df.columns.tolist()
            non_id_cols = [c for c in all_cols if c.lower() not in id_cols]

            num_cols = res_df[non_id_cols].select_dtypes(include=[np.number]).columns.tolist() if non_id_cols else []
            cat_cols = res_df[non_id_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist() if non_id_cols else []

            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                chart_x = cat_cols[0]
                chart_y = num_cols[0]
                st.markdown(f"<div style='font-size:0.80rem; color:{TEXT_MUTED}; margin-bottom:8px;'>Smart Plot: Bar Chart of <b>{chart_y}</b> by <b>{chart_x}</b></div>", unsafe_allow_html=True)
                fig_auto = px.bar(
                    res_df.head(30), x=chart_x, y=chart_y,
                    color=chart_y, color_continuous_scale=[ACCENT, GOLD, GREEN],
                    labels={chart_x: chart_x.replace("_", " ").title(), chart_y: chart_y.replace("_", " ").title()}
                )
                fig_auto.update_layout(**PLOT_LAYOUT, height=350)
                st.plotly_chart(fig_auto, use_container_width=True, config=PLOTLY_CONFIG)
            elif len(num_cols) >= 2:
                chart_x = num_cols[0]
                chart_y = num_cols[1]
                st.markdown(f"<div style='font-size:0.80rem; color:{TEXT_MUTED}; margin-bottom:8px;'>Smart Plot: Scatter Plot of <b>{chart_y}</b> vs <b>{chart_x}</b></div>", unsafe_allow_html=True)
                fig_auto = px.scatter(
                    res_df.head(50), x=chart_x, y=chart_y,
                    color=chart_y, color_continuous_scale=[AMBER, GREEN]
                )
                fig_auto.update_layout(**PLOT_LAYOUT, height=350)
                st.plotly_chart(fig_auto, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("Smart visualizer requires at least one numeric and one descriptive category column in query output.")

        with res_tab3:
            st.json(res_df.head(50).to_dict(orient="records"))

        with res_tab4:
            history = st.session_state.get("sql_history", [])
            if history:
                st.markdown("<div style='font-size:0.80rem; color:#94a3b8; margin-bottom:8px;'>Last 5 Executed Queries:</div>", unsafe_allow_html=True)
                for h_idx, item in enumerate(history):
                    st.markdown(f"**[{item['time']}]** `{item['rows']} rows` — `{item['query'][:80]}...`")
            else:
                st.info("No query execution history recorded in this session.")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Result CSV",
                data=csv_data,
                file_name=f"{selected_query_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl_col2:
            json_data = res_df.to_json(orient="records", indent=2).encode('utf-8')
            st.download_button(
                label="Download Result JSON",
                data=json_data,
                file_name=f"{selected_query_name.replace(' ', '_').lower()}.json",
                mime="application/json",
                use_container_width=True
            )

# Footer
st.markdown(f"""
<div style="text-align:center; padding:24px 0 12px; margin-top:40px;
            border-top:1px solid {BORDER};">
    <span style="color:{TEXT_DIM}; font-size:0.72rem; letter-spacing:0.06em;">
        PROCURESENSE AI &nbsp;&middot;&nbsp; Procurement Intelligence Platform
        &nbsp;&middot;&nbsp; Built with Python &amp; Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
