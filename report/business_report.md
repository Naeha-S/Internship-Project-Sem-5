# ProcureSense AI — Executive Procurement Analytics Findings

> **Portfolio Showcase & Capability Project**: This report is produced by **ProcureSense AI**, an end-to-end supply chain analytics platform designed to demonstrate production Data Analyst competencies — including multi-table relational schema design, 10 production-grade SQL queries, automated KPI calculation, multi-model ML evaluation, and interactive BI dashboarding. All data is generated with realistic latent structures and benchmarked against real Kaggle supply chain datasets.

---

## Executive Summary

### Overall Procurement Health Score: **50.3 / 100**

Across 30,000 purchase orders spanning 78 active corporate suppliers (out of 98 total registered catalog suppliers), 200 technical product SKUs, and ₹50,967 Cr+ in tracked expenditure, the platform evaluates the operational health score at **50.3 / 100**.

```mermaid
pie title Active Supplier Operational Risk Tier Distribution (Percentile-Calibrated)
    "High Risk (Top 30% Riskiest Index)" : 24
    "Medium Risk (Middle 45%)" : 35
    "Low Risk (Bottom 25% Safest)" : 19
```

### Key Operational Findings:
1. **45.0% Late Delivery Rate**: 13,500+ purchase orders arrived past contracted delivery dates, representing a major performance drag across logistics channels.
2. **57.8% Expenditure Concentration at High-Risk Suppliers**: ₹29,481 Cr (57.8%) of total procurement spend is allocated to suppliers evaluated as **High Risk** under our continuous 4-component composite scoring model.
3. **Severe Inventory Stockout Exposure**: **18 of the 50 understocked SKUs (36.0%)** are primary-sourced from High-Risk suppliers, posing immediate assembly-line stoppage risk.
4. **Deteriorating Supplier Trajectories**: 3-year linear trend analysis ($\beta$) identifies **14 suppliers with escalating delivery delays**, pushing suppliers with decent historical averages into High Risk due to negative momentum.

---

## 1. Transparent Procurement Health Score Breakdown

The **Procurement Health Score (50.3 / 100)** is a composite metric combining 5 key operational dimensions. Each dimension is calculated on a 0–100 scale and weighted according to supply chain priority:

$$\text{Health Score} = \sum_{i=1}^{5} (W_i \times C_i)$$

$$\text{Health Score} = (0.25 \times \text{Reliability}) + (0.20 \times \text{Inventory Eff}) + (0.20 \times \text{Cost Opt}) + (0.20 \times \text{Delivery SLA}) + (0.15 \times \text{Risk Score})$$

### Health Score Component Breakdown:

| Dimension ($C_i$) | Component Formula / Definition | Score (0–100) | Weight ($W_i$) | Weighted Points | Operational Impact |
|---|---|---|---|---|---|
| **Supplier Reliability** | Mean On-Time Delivery % across all active suppliers | **55.0** | 25% | **13.75** | Severe SLA drag from Tier 2/3 delay rates |
| **Inventory Efficiency** | $100 - \% \text{Overstocked} - \% \text{DeadStock} - (0.75 \times \% \text{Understocked})$ | **81.2** | 20% | **16.24** | Good turnover, but 50 SKUs understocked |
| **Cost Optimisation** | $100 - (2.0 \times \text{Mean YoY Price Inflation \%})$ | **43.0** | 20% | **8.60** | High price drift across raw metals (+16.7%) |
| **Delivery Performance** | Overall Purchase Order On-Time SLA Fulfillment Rate | **55.0** | 20% | **11.00** | 45% of orders arrive late |
| **Supplier Risk Profile** | $100 - (2.0 \times \% \text{High Risk Suppliers}) - (0.75 \times \% \text{Medium Risk})$ | **4.7** | 15% | **0.71** | High concentration of High Risk suppliers (31%) |
| **TOTAL HEALTH SCORE** | **Weighted Sum** | — | **100%** | **50.30 ≈ 50.3** | **Needs Urgent Intervention** |

---

## 2. Advanced 4-Component Composite Supplier Risk Scoring & Trajectory Engine

To provide an objective, continuous, and defensible evaluation of supplier risk, we build a **4-Component Composite Risk Index** combined with **3-Year Linear Delay Trend ($\beta$) Analysis**:

$$\text{Composite Risk Index} = 0.35 \times C_{\text{late}} + 0.25 \times C_{\text{defect}} + 0.20 \times C_{\text{price\_vol}} + 0.20 \times C_{\text{delay\_trend\_slope}}$$

### Normalized Component Formulas (Scaled 0 to 1):
1. $C_{\text{late}}$ **Late Delivery Rate**: $1.0 - (\text{On-Time Delivery \%} / 100.0)$ (Weight $= 35\%$)
2. $C_{\text{defect}}$ **Quality Defect Rate**: $\text{Defect Rate \%} / 100.0$ (Weight $= 25\%$)
3. $C_{\text{price\_vol}}$ **Price Volatility**: Min-Max scaled Coefficient of Variation ($\text{std} / \text{mean}$) of unit prices (Weight $= 20\%$)
4. $C_{\text{delay\_trend\_slope}}$ **3-Year Delay Trend Slope ($\beta$)**: Min-Max scaled linear regression slope of monthly average delay days ($t = 1 \dots 36$) (Weight $= 20\%$)

---

## 3. Advanced Feature Engineering & Leakage Prevention

To maximize predictive signal without introducing future look-ahead bias, all features strictly utilize data available prior to the order date ($t < \text{order\_date}$ using `.shift(1)`):

1. **Order Quantity Spike Ratio (`order_qty_vs_sup_mean`)**: Ratio of current PO quantity to supplier's historical expanding mean PO quantity ($\text{qty} / \mu_{\text{sup\_qty}}$). Captures deprioritization during sudden order surges.
2. **Network Capacity Strain (`sup_concurrent_po_30d`)**: Count of active concurrent POs issued to the same supplier in the prior 30 days. Measures factory queue congestion.
3. **Supplier-Category Interaction (`sup_category_te`)**: Out-of-fold target encoding of supplier $\times$ product category interaction.
4. **Logistics Stress Index (`logistics_stress_index`)**: $\text{lead\_time\_days\_base} / (\text{sup\_rolling\_delay} + 1.0)$.

---

## 4. Machine Learning Delay Prediction, Walk-Forward Validation & Calibration

Evaluated **6 candidate models** on a held-out 2025 test set (9,944 purchase orders) using **100x Bootstrap Resampling** for 95% Confidence Intervals:

### Apples-to-Apples Benchmark Table:

| Model Candidate | ROC-AUC (95% CI) | PR-AUC (95% CI) | Default Thresh (0.50) Accuracy | Brier Score Loss | Expected Financial Risk Cost (₹) | Model Selection Role |
|---|---|---|---|---|---|---|
| **1. Naive Majority Baseline** | 0.500 [0.500–0.500] | 0.473 [0.473–0.473] | 52.7% | 0.2492 | ₹234.95 M | Baseline |
| **2. Supplier Historical Heuristic** | 0.680 [0.669–0.691] | 0.652 [0.638–0.666] | 62.2% | 0.2267 | ₹145.71 M | Heuristic Baseline |
| **3. Logistic Regression** | **0.700 [0.690–0.710]** | **0.673 [0.660–0.687]** | **64.0%** | 0.2230 | **₹87.90 M** | 🏆 **Cost-Optimal Winner** |
| **4. Random Forest Classifier** | **0.714 [0.706–0.725]** | **0.689 [0.676–0.703]** | **65.7%** | **0.2159** | **₹93.38 M** | 🏅 **ROC-AUC & Live Simulator Champion** |
| **5. Tuned XGBoost Classifier** | 0.697 [0.688–0.708] | 0.676 [0.666–0.688] | 64.2% | 0.2161 | ₹93.47 M | Tree Baseline |
| **6. Soft-Voting Ensemble** | 0.703 [0.694–0.713] | 0.684 [0.671–0.699] | 64.4% | 0.2160 | ₹93.37 M | Ensemble |

---

### Temporal Walk-Forward Validation (2025 Quarterly Expanding Window)

To prove model stability and rule out performance degradation over time, we perform a 4-quarter expanding window walk-forward validation across 2025:

| Validation Quarter | Training Window | Test Sample Size | ROC-AUC Score | PR-AUC Score | Classification Accuracy | Temporal Stability Status |
|---|---|---|---|---|---|---|
| **2025 Q1** | 2023–2024 (20,056 POs) | 2,482 POs | **0.720** | 0.657 | 66.6% | Stable |
| **2025 Q2** | 2023–Q1 2025 (22,538 POs) | 2,477 POs | **0.727** | 0.691 | 67.1% | Peak Stability |
| **2025 Q3** | 2023–Q2 2025 (25,015 POs) | 2,456 POs | **0.710** | 0.690 | 64.8% | Stable |
| **2025 Q4** | 2023–Q3 2025 (27,471 POs) | 2,529 POs | **0.722** | 0.740 | 62.4% | High Precision |
| **MEAN 2025 WALK-FORWARD** | **Expanding Window** | **9,944 POs Total** | **0.720 ($\pm 0.007$)** | **0.694** | **65.2%** | **Zero Concept Drift Degradation** |

---

### Per-Supplier Risk Tier Performance Disaggregation

Evaluating model accuracy and discrimination separately across **High**, **Medium**, and **Low Risk** suppliers proves that the ML engine provides actionable predictive value *within* each risk tier, rather than merely re-deriving static tiers:

| Supplier Risk Tier | Sub-Sample Size | Actual Late Rate | Model ROC-AUC | Model Accuracy | Model Precision | Model Recall | Key Operational Value |
|---|---|---|---|---|---|---|---|
| 🔴 **High Risk Suppliers** | 3,013 POs | **67.3%** | **0.652** | 67.4% | **70.6%** | **88.4%** | Catches 88.4% of actual late shipments in high-vulnerability segment |
| 🟡 **Medium Risk Suppliers** | 4,515 POs | **44.8%** | **0.633** | 60.0% | 56.1% | 49.5% | Discriminates SLA breaches in ambiguous middle tier |
| 🟢 **Low Risk Suppliers** | 2,416 POs | **26.7%** | **0.663** | **71.2%** | 42.0% | 20.0% | Minimizes false alarm expedites (71.2% accuracy) |

---

### Full Mathematical Cost Reconciliation (Net ₹5.49M Savings Analysis)

To understand why **Logistic Regression** achieves a lower total expected risk cost (₹87.90M) than **Random Forest** (₹93.38M), we perform a full two-sided financial reconciliation across missed late deliveries (False Negatives) and false alarms (False Positives):

- **Cost Matrix Parameters**:
  - Stockout Penalty ($C_{\text{FN}}$): ₹50,000 per unflagged late delivery line stoppage.
  - Expedite Cost ($C_{\text{FP}}$): ₹5,000 per false alarm expedite follow-up.

#### Two-Sided Financial Trade-off Reconciliation:

$$\text{Net Savings} = \text{FN Stoppage Savings} - \text{Extra FP Expediting Cost}$$

1. **FN Line-Stoppage Savings (Higher Recall)**:
   - Logistic Regression catches 141 more late shipments than Random Forest ($1,555$ missed vs $1,696$).
   - Savings $= (1,696 - 1,555) \times \text{₹}50,000 = 141 \times \text{₹}50,000 = \mathbf{+\text{₹}7,050,000}$
2. **Extra FP Expediting Cost (Higher False Alarm Rate)**:
   - Logistic Regression incurs 312 more false alarm expedites than Random Forest ($2,029$ false alarms vs $1,717$).
   - Penalty $= (2,029 - 1,717) \times \text{₹}5,000 = 312 \times \text{₹}5,000 = \mathbf{-\text{₹}1,560,000}$
3. **Net Expected Financial Risk Reduction**:
   - $\text{Net Savings} = \text{₹}7,050,000 - \text{₹}1,560,000 = \mathbf{+\text{₹}5,490,000 \approx \text{₹}5.49\text{M}}$
   - Reconciles exact model costs: $\text{₹}93.38\text{M} - \text{₹}87.90\text{M} = \mathbf{\text{₹}5.48\text{M} \approx \text{₹}5.49\text{M}}$

---

## 5. Honest Limitations & Technical Caveats

1. **Synthetic Data Scope**: While generated with realistic latent structures (supplier reliability drift, seasonality, macro commodity pressure) and benchmarked against real Kaggle DataCo supply chain datasets, the data originates from a synthetic generator (`data/generate_data.py`).
2. **Modest Predictive Power ($ROC\text{-}AUC = 0.700 – 0.714$)**: Predicting shipment delays at order creation time (weeks in advance) without live GPS/in-transit IoT telemetry is an inherently noisy problem. The model provides a useful decision-support signal rather than automated decision-making.
3. **Designed Composite Health Score**: The **Procurement Health Score (50.3/100)** is a custom-designed composite KPI for portfolio demonstration, not an official ISO industry standard. Weights are fully documented and adjustable.
