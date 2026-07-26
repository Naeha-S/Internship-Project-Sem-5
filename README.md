# ProcureSense AI
### AI-Powered Procurement Analytics & Interactive BI Studio

An end-to-end data analytics platform built for Data Analyst job applications and portfolio showcases: synthetic multi-table procurement data → 10 advanced SQL queries → KPI health engine → XGBoost machine learning model → interactive Streamlit BI Studio → executive business recommendations.

---

## Project Structure & Architecture

| Layer | File / Location | Description |
|---|---|---|
| **Data Generation** | `data/generate_data.py` | Builds 5 linked synthetic relational CSVs (suppliers, products, purchase_orders, deliveries, inventory) across 15,000 purchase orders and 3 years (2023–2025). |
| **SQL Database** | `db/build_db.py` | Ingests CSVs into SQLite (`db/procurement.db`) with indexes and relational constraints. |
| **SQL Portfolio** | `docs/SQL_PORTFOLIO.md` | **10 Production-Grade SQL Queries** covering CTEs, window functions (`SUM() OVER`, `DENSE_RANK()`, `LAG()`), lead-time variance, inventory stockout risk, defect spend exposure, spend Pareto 80/20 analysis, and rolling 60-day ML feature extraction. |
| **KPI Engine** | `analysis/kpi_engine.py` | Computes the Procurement Health Score (49.8/100), supplier risk tiers, and business narrative insights. |
| **ML Predictive Model** | `ml/delay_prediction.py` | **XGBoost Classifier** predicting late deliveries at order time with SHAP explainability and ROC-AUC evaluation. |
| **Interactive BI Dashboard** | `dashboard/app.py` | Interactive Streamlit BI application featuring **10 Plotly visualizations**, dark/light mode toggle, dynamic global sidebar filters, live AI delay risk simulator, and an interactive SQL query explorer with live execution & CSV downloads. |
| **Executive Deliverable** | `report/business_report.md` | Quantified business findings, inflation alerts, supplier risk distribution, and strategic recommendations. |

---

## 10 Core SQL Portfolio Queries

1. **MoM Spend & Cumulative Running Spend per Category**: `SUM() OVER` window function over monthly spend.
2. **Regional Supplier SLA Ranking**: `DENSE_RANK() OVER` by regional delivery performance.
3. **Year-over-Year Unit Price Drift**: `LAG()` window function tracking price inflation percentages.
4. **Lead Time Variance & Reliability Cohort Analysis**: Date arithmetic (`julianday()`) analyzing contracted vs actual delivery times.
5. **Inventory Stockout Risk & Replenishment Matrix**: Evaluating stock cover against reorder thresholds.
6. **Quality Defect Rate & Spend Exposure Matrix**: Financial exposure calculation for defective shipments.
7. **Predictive ML Feature Engineering Query**: Rolling 60-day historical supplier reliability extraction.
8. **Supplier Spend Pareto 80/20 Analysis**: Cumulative spend share classification (Class A/B/C).
9. **Monthly Order Volume MoM Growth & Seasonal Variance**: Month-over-month volume shifts and late delivery correlations.
10. **Order Fulfillment Bottleneck & Delay Severity Ranking**: Delay severity breakdown (Minor, Moderate, Severe) across shipping modes and regions.

---

## How to Run the Project

### 1. Execute Pipeline & Verify Artifacts
```bash
# Activate virtual environment
.\.venv\Scripts\python.exe test_pipeline.py
```

### 2. Launch the Streamlit Interactive BI Dashboard
```bash
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```
Open **`http://localhost:8501`** in any browser to explore the dashboard, filter data interactively, run live SQL queries, and simulate delivery delay risk!

---

## Stack

- **Data Processing & SQL**: Python 3.10+, pandas, NumPy, SQLite 3 (`sqlite3`).
- **Machine Learning & Explainability**: XGBoost, scikit-learn, SHAP.
- **Interactive BI & Visualizations**: Streamlit, Plotly Express & Graph Objects.
