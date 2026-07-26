# ProcureSense AI — Procurement Analytics Findings
**Analyst:** Manus AI | **Data window:** Jan 2023 – Dec 2025 | **Dataset:** Synthetic (documented in README)

---

## Executive Summary

Overall Procurement Health Score: **59.6 / 100**

Across 15,000 purchase orders spanning 100 suppliers and ₹75.8 Cr+ in tracked spend, several key insights emerge:

1.  **41.5% of all orders arrive late** — this remains a significant drag on the overall health score, indicating systemic issues in delivery reliability.
2.  **A substantial portion of suppliers are classified as high risk (64 out of 100)**, contributing to potential disruptions and increased costs. Targeted intervention is crucial.
3.  **Price inflation is evident across multiple suppliers**, with some showing significant year-over-year increases. Proactive renegotiation and alternative sourcing are recommended.

---

## 1. Supplier Performance

| Supplier | On-Time % | Avg Delay (days) | Defect % | Risk Tier |
|---|---|---|---|---|
| Supplier 88 | 89.5% | 0.93 | 6.17% | Medium Risk |
| Supplier 80 | 15.3% | 10.57 | 1.39% | Medium Risk |

**Finding:** Supplier 80 is the weakest performer with an on-time delivery rate of 15.3%, significantly below the average. This supplier also contributes to substantial delays. Conversely, Supplier 88 demonstrates strong performance with an 89.5% on-time rate.

**Recommendation:** Prioritize shifting high-priority orders away from high-risk, low-performing suppliers like Supplier 80. Explore increasing engagement with reliable suppliers such as Supplier 88, and investigate the root causes of delays and defects with underperforming partners.

**Risk distribution:** The analysis reveals a high concentration of risk within the supplier base, with 64 suppliers classified as 'High Risk', 21 as 'Medium Risk', and only 3 as 'Low Risk'. This indicates a systemic issue requiring a comprehensive supplier management strategy.

---

## 2. Cost Analysis

Price inflation year-over-year (2024 → 2025) shows significant variation:

| Supplier | Latest Price Change |
|---|---|
| Supplier 3 | +35.4% |
| Supplier 42 | +27.0% |
| Supplier 41 | +27.0% |

**Recommendation:** Suppliers exhibiting high price increases, such as Supplier 3, 42, and 41, should be prioritized for renegotiation. A detailed cost analysis should be performed to understand the drivers of these increases and explore alternative sourcing options to mitigate future cost escalations.

---

## 3. Inventory

Inventory analysis indicates that 153 products are currently healthy, while 47 are understocked. No dead stock was identified, which is a positive indicator of inventory turnover. The primary risk remains on the supply side due to delays, rather than overstocking.

**Recommendation:** Implement enhanced monitoring for understocked SKUs, especially those sourced from high-risk suppliers. Proactive measures, such as safety stock adjustments or expedited shipping for critical items, should be considered to prevent stockouts.

---

## 4. Delivery Delay Prediction (ML)

An XGBoost classifier was trained to predict, at order time, whether a purchase order will arrive late. Model performance: **0.638 accuracy, 0.671 ROC-AUC** on held-out orders. This represents a modest but useful signal for proactive risk management.

**What actually drives the prediction (SHAP):**
1.  **Order Month:** Seasonality plays a significant role, with certain months (e.g., Nov-Jan, Jun-Jul) exhibiting higher delay probabilities.
2.  **Supplier's Rolling On-Time Rate:** Historical reliability of a supplier is a strong indicator of future performance.
3.  **Shipping Mode:** The chosen shipping method significantly impacts delivery timelines and potential delays.
4.  **Supplier's Rolling Average Delay:** Past average delay days for a supplier contribute to predicting future delays.

**Practical use:** The model provides valuable decision support by flagging orders with a higher probability of delay. This allows procurement teams to proactively engage with suppliers, consider alternative logistics, or inform internal stakeholders, thereby mitigating potential disruptions. The model is intended for decision support, not automated decision-making.

---

## Methodology Note

All figures and analyses are derived from an expanded synthetic dataset, generated with realistic latent structures (see `data/generate_data.py`). KPIs are computed using SQL queries and pandas in `analysis/kpi_engine.py`. The ML model for delay prediction is an XGBoost classifier, detailed in `ml/delay_prediction.py`. The dashboard (`dashboard/dashboard.html`) visualizes these insights. While the data is synthetic, the analytical pipeline is robust and designed to be directly applicable to real-world procurement data.
