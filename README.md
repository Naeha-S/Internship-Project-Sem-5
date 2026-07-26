# ProcureSense AI — Executive Procurement Analytics & BI Studio
> **Enterprise Procurement Intelligence, Kaggle Benchmarking, 10 Production SQL Queries, XGBoost Delay Simulation & Interactive Streamlit BI Studio**

ProcureSense AI is an end-to-end data analytics platform engineered for supply chain and procurement intelligence. It seamlessly processes 30,000+ purchase order records across 5 relational tables, integrates real Kaggle benchmark datasets, runs 10 production SQL portfolio queries, and provides real-time machine learning late-delivery risk predictions via an interactive Streamlit BI Studio.

---

## 🌟 Key Platform Highlights

- **Kaggle & Real-World Data Integration**: Automatically downloads & integrates Kaggle Supply Chain & DataCo Smart Supply Chain datasets into SQLite (`procurement.db`).
- **Rich Relational Schema**: 30,000 purchase orders spanning 2023–2025 across 100 corporate suppliers (*Apex Precision*, *Bharat Heavy Metallics*, *Pacific Rim Semiconductor*) and 200 technical SKUs (*STM32 Microcontrollers*, *Aircraft Aluminum 6061-T6*).
- **10 Production SQL Portfolio Queries**: Full query collection covering window functions (`SUM() OVER`, `DENSE_RANK()`, `LAG()`), lead-time variance, defect spend exposure, spend Pareto 80/20 analysis, and rolling ML feature extraction.
- **XGBoost Machine Learning & SHAP Explainability**: Dual-stage late delivery classification ($ROC\text{-}AUC = 0.714$) and delay regressor with TreeSHAP feature importance.
- **Interactive Streamlit BI Studio**: 5-tab executive dashboard with dark/light themes, 10 Plotly charts, live SQL query execution with CSV exports, and a Live PO Late-Delivery Risk Simulator.

---

## 📂 Project Architecture

| Component | Path | Description |
|---|---|---|
| **Data Generation & Kaggle Loader** | [`data/generate_data.py`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/data/generate_data.py) | Generates 30,000 multi-table procurement records and integrates Kaggle benchmark datasets. |
| **Database & SQL Ingestion** | [`db/build_db.py`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/db/build_db.py) | Builds relational SQLite database (`db/procurement.db`) with indexes and Kaggle tables. |
| **KPI & Narrative Engine** | [`analysis/kpi_engine.py`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/analysis/kpi_engine.py) | Computes the Procurement Health Score (**49.6 / 100**), inflation flags, and automated executive narratives. |
| **ML Predictive Engine** | [`ml/delay_prediction.py`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/ml/delay_prediction.py) | Trains XGBoost ensemble model, evaluates ROC-AUC, generates SHAP explainability plots, and outputs `model_metrics.json`. |
| **Streamlit BI Studio** | [`dashboard/app.py`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/dashboard/app.py) | Unified Streamlit application featuring 10 Plotly visualizations, dark/light theme toggle, and Live Risk Simulator. |
| **SQL Portfolio Documentation** | [`docs/SQL_PORTFOLIO.md`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/docs/SQL_PORTFOLIO.md) | Complete documentation of all 10 production SQL queries with business rationale and schema descriptions. |
| **Executive Findings Report** | [`report/business_report.md`](file:///c:/Users/NAEHA/Desktop/Internship%20Project/report/business_report.md) | Comprehensive executive deliverable with quantified insights and strategic recommendations. |

---

## 💻 10 Production SQL Portfolio Queries

1. **MoM Spend & Cumulative Running Spend per Category**: Category expenditure tracking using `SUM() OVER(PARTITION BY ... ROWS UNBOUNDED PRECEDING)`.
2. **Regional Supplier SLA Ranking**: Regional delivery reliability ranking using `DENSE_RANK() OVER(PARTITION BY region ORDER BY on_time_pct DESC)`.
3. **Year-over-Year Unit Price Drift**: Supplier price inflation detection using `LAG(avg_unit_price, 1) OVER(...)`.
4. **Lead Time Variance & Reliability Cohort Analysis**: Contracted vs actual lead-time variance analysis via SQLite date arithmetic (`julianday()`).
5. **Inventory Stockout Risk & Replenishment Matrix**: Stock health classification (*Healthy*, *Understocked*, *Overstocked*, *Dead Stock*).
6. **Quality Defect Rate & Spend Exposure Matrix**: Direct financial exposure resulting from defective deliveries.
7. **Predictive ML Feature Engineering Query**: Extraction of rolling 60-day historical supplier performance indicators.
8. **Supplier Spend Pareto 80/20 Analysis**: Cumulative spend share classification into Class A (top 80%), Class B (next 15%), and Class C (tail spend).
9. **Monthly Order Volume MoM Growth & Seasonal Variance**: Month-over-month volume shifts and late delivery correlations.
10. **Order Fulfillment Bottleneck & Delay Severity Ranking**: Categorization of delays into Minor (1–3d), Moderate (4–7d), and Severe (>7d).

---

## 🤖 Machine Learning Model & Risk Simulator

- **Model**: Dual-stage XGBoost Classifier + Delay Days Regressor ($ROC\text{-}AUC = 0.714$, Accuracy $= 62.6\%$).
- **Explainability**: SHAP (SHapley Additive exPlanations) identifies supplier historical reliability, shipping mode, and order month as top risk drivers.
- **Live Simulator (`dashboard/app.py`)**: Allows supply chain managers to configure custom order parameters (Supplier, Product Category, Shipping Mode, Month, Quantity, Unit Price), submit the form, and receive real-time late risk predictions and estimated delay durations.

---

## 🚀 How to Run the Project

### 1. Run Pipeline & Verify Artifacts
```bash
# Execute end-to-end test pipeline
py test_pipeline.py
```

### 2. Launch the Streamlit Interactive BI Dashboard
```bash
# Launch Streamlit BI Studio
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in any web browser to interact with the executive studio.

---

## 🛠️ Technology Stack

- **Core & SQL**: Python 3.10+, pandas, NumPy, SQLite 3 (`sqlite3`).
- **Machine Learning & AI**: XGBoost, scikit-learn, SHAP, Optuna.
- **Interactive BI & Data Visualization**: Streamlit, Plotly Express, Plotly Graph Objects.
- **Data Integration**: Kaggle API (`kaggle`), Open-source Supply Chain Benchmark Datasets.
