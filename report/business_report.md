# ProcureSense AI — Executive Procurement Analytics Findings
**Analyst:** Manus AI | **Data window:** Jan 2023 – Dec 2025 | **Scope:** 30,000 Purchase Orders across 100 Corporate Suppliers & 200 Industrial SKUs

---

## Executive Summary

**Overall Procurement Health Score: 49.6 / 100**

Across 30,000 purchase orders spanning 100 corporate suppliers, 200 technical product SKUs, and ₹50,967 Cr+ in tracked expenditure, several core operational risks emerge:

1. **45.0% Late Delivery Rate**: 13,500+ purchase orders arrived past contracted delivery dates, representing a severe performance drag across logistics channels.
2. **72.1% Expenditure Concentration at High-Risk Suppliers**: ₹36,752 Cr (72.1%) of total procurement spend is allocated to suppliers evaluated as **High Risk** under our multi-axis scoring methodology.
3. **Severe Inventory Stockout Exposure**: **36 of the 50 understocked SKUs (72.0%)** are primary-sourced from High-Risk suppliers, posing immediate line-stoppage risk.
4. **Distinct Risk Profiles (Reliability vs Quality)**: Operational failures are split across two distinct supplier risk axes: **78 suppliers exhibit Delivery Reliability Risk** (SLA breaches & long delay days), while **58 suppliers exhibit Quality Risk** (defect rate $>2\%$).

---

## 1. Supplier Risk Classification Methodology & Scoring Model

To avoid treating risk as an arbitrary assertion, supplier risk is calculated using a **transparent, weighted 3-axis scoring model**:

$$\text{Risk Points} = \text{Delivery SLA Points} + \text{Quality Points} + \text{Operational Points}$$

### Scoring Rules:
- **Delivery Reliability Axis**:
  - On-Time Delivery Rate $< 75.0\%$: $+2.0$ pts
  - On-Time Delivery Rate $< 85.0\%$: $+1.0$ pt
  - Average Delay $> 3.0$ days: $+1.0$ pt
- **Quality Exposure Axis**:
  - Defect Rate $> 5.0\%$: $+2.0$ pts
  - Defect Rate $> 2.0\%$: $+1.0$ pt
- **Operational & Structural Risk Axis**:
  - Tier 3 Supplier Classification: $+1.0$ pt
  - Overseas Import Region (e.g. China, Europe, SE Asia): $+0.5$ pt

### Risk Tier Thresholds:
- 🔴 **High Risk** ($\text{Score} \ge 4.0$ pts): **69 Suppliers**
- 🟡 **Medium Risk** ($2.0 \le \text{Score} < 4.0$ pts): **31 Suppliers**
- 🟢 **Low Risk** ($\text{Score} < 2.0$ pts): **0 Suppliers**

---

## 2. Multi-Axis Supplier Performance & Disaggregation

Comparing suppliers across separate **Reliability** vs **Quality** risk axes reveals critical operational differences:

| Supplier Name | Tier | Region | On-Time % | Avg Delay (d) | Defect % | Primary Risk Axis | Action Required |
|---|---|---|---|---|---|---|---|
| **Komatsu Earthmoving Parts** | Tier 2 | South India | **11.7%** | 10.09 days | 2.10% | **Reliability Risk** | Reallocate volume; SLA penalty enforcement |
| **Pinnacle Precision Castings** | Tier 2 | North India | 74.2% | 2.85 days | **18.21%** | **Quality Risk** | QA Audit; Batch rejection protocols |
| **Shanghai Industrial Silicon** | Tier 1 | East India | **79.8%** | 2.45 days | 4.67% | **Dual Risk** | Preferred partner for SLA; Quality remediation |

### Strategic Findings:
- **Reliability vs Quality Sanity Check**: Suppliers cannot be lumped into a single bucket. A supplier like *Pinnacle Precision Castings* delivers relatively on time (74.2%) but sends defective goods 18.21% of the time. Lumping this with *Komatsu Earthmoving Parts* (11.7% on-time, low defects) masks the underlying root cause.
- **Remediation**: Reliability failures require logistics rerouting & SLA penalties; Quality failures require factory QA audits & raw material testing.

---

## 3. Inventory Stockout & High-Risk Supplier Linkage

Linking product inventory health directly to primary supplier risk profiles uncovers severe supply chain vulnerability:

- **Total Catalog SKUs**: 200 SKUs
- **Healthy Stock Cover**: 150 SKUs (75.0%)
- **Understocked SKUs (Stock < Reorder Point)**: **50 SKUs (25.0%)**
- ⚠️ **Understocked SKUs Sourced from High-Risk Suppliers**: **36 SKUs (72.0%)**

### Stockout Exposure Impact:
72.0% of all understocked products rely entirely on High-Risk suppliers for replenishment. Any delivery delay or quality rejection for these 36 SKUs directly triggers production halts.

**Recommendation**: Immediately establish secondary (dual) sourcing contracts with Tier-1 Low-Risk suppliers for all 36 high-risk understocked SKUs.

---

## 4. Price Inflation & Cost Analysis

Year-over-Year (YoY 2024 → 2025) unit price drift tracking identifies severe price inflation flags:

| Supplier Name | Category | YoY Price Increase (%) | Total Annual Spend |
|---|---|---|---|
| **Shanghai Industrial Silicon** | Raw Metals | **+16.7%** | ₹84.2 Cr |
| **Bavaria Automotive Forgings** | Fasteners & Hardware | **+14.5%** | ₹62.1 Cr |
| **Tata Alloy & Steel Corp** | Raw Metals | **+12.8%** | ₹94.5 Cr |

**Recommendation**: Initiate formal cost breakdown audits and index-linked price renegotiations for suppliers exceeding $10\%$ annual price inflation.

---

## 5. Machine Learning Delay Prediction Benchmark

Evaluated **6 machine learning model candidates** on a 2025 chronological test set (9,944 orders):

| Model Candidate | Accuracy | ROC-AUC | PR-AUC | Precision | Recall (Late) | F1-Score |
|---|---|---|---|---|---|---|
| **1. Naive Majority Baseline** | 52.7% | 0.500 | 0.473 | 0.000 | 0.000 | 0.000 |
| **2. Supplier Historical Heuristic** | 60.1% | 0.680 | 0.652 | 0.554 | 0.796 | 0.653 |
| **3. Logistic Regression** | 64.0% | 0.700 | 0.673 | 0.608 | 0.669 | 0.637 |
| **4. Random Forest Classifier** | 65.7% | 0.714 | 0.689 | 0.636 | 0.639 | 0.638 |
| **5. Tuned XGBoost Classifier** | 64.2% | 0.697 | 0.676 | 0.616 | 0.642 | 0.629 |
| **6. Soft-Voting Ensemble (Optimal Thresh)** | **60.0%** | **0.703** | **0.683** | **0.549** | **0.867** | **0.672** |

### ML Key Takeaways:
- **PR-Curve Threshold Optimization ($0.328$)** achieves **86.7% Recall on Late Orders**, flagging **4,073 out of 4,699 late shipments** prior to dispatch.
- **Top Risk Drivers (TreeSHAP)**: Supplier OOF Target Encoding (`supplier_id_te`), Shipping Mode (`shipping_mode_code`), Order Month (`order_month`), and Shipping Mode × Region Interaction (`ship_region_te`).
