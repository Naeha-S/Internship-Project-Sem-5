# ProcureSense AI — Production SQL Query Portfolio

Comprehensive documentation of the 10 production SQL queries powering the ProcureSense AI Procurement & Supply Chain Analytics Studio.

---

## 1. Month-over-Month Category Spend & Cumulative Running Spend
Tracks category expenditure over time along with running totals using window functions.
```sql
SELECT 
    category,
    strftime('%Y-%m', po.order_date) AS order_month,
    ROUND(SUM(po.order_cost), 2) AS monthly_spend,
    ROUND(SUM(SUM(po.order_cost)) OVER(PARTITION BY category ORDER BY strftime('%Y-%m', po.order_date) ROWS UNBOUNDED PRECEDING), 2) AS cumulative_running_spend
FROM purchase_orders po
JOIN products p ON po.product_id = p.product_id
GROUP BY category, order_month
ORDER BY category, order_month;
```

---

## 2. Regional Supplier SLA & Fulfillment Performance Ranking
Ranks corporate suppliers within their geographic region based on on-time delivery performance.
```sql
SELECT 
    s.region,
    s.supplier_name,
    s.tier,
    COUNT(po.po_id) AS total_orders,
    ROUND(100.0 * SUM(CASE WHEN d.is_late = 0 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS on_time_pct,
    DENSE_RANK() OVER(PARTITION BY s.region ORDER BY (100.0 * SUM(CASE WHEN d.is_late = 0 THEN 1 ELSE 0 END) / COUNT(po.po_id)) DESC) AS regional_rank
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY s.region, s.supplier_name, s.tier;
```

---

## 3. Year-over-Year Unit Price Drift & Price Volatility
Identifies price inflation and volatility across suppliers using `LAG()`.
```sql
WITH annual_prices AS (
    SELECT 
        s.supplier_name,
        strftime('%Y', po.order_date) AS order_year,
        ROUND(AVG(po.unit_price), 2) AS avg_unit_price
    FROM purchase_orders po
    JOIN suppliers s ON po.supplier_id = s.supplier_id
    GROUP BY s.supplier_name, order_year
)
SELECT 
    supplier_name,
    order_year,
    avg_unit_price,
    LAG(avg_unit_price, 1) OVER(PARTITION BY supplier_name ORDER BY order_year) AS prev_year_price,
    ROUND(avg_unit_price - LAG(avg_unit_price, 1) OVER(PARTITION BY supplier_name ORDER BY order_year), 2) AS yoy_price_change
FROM annual_prices;
```

---

## 4. Contracted vs. Actual Lead-Time Variance Analysis
Measures contracted lead time against actual delivery days to quantify delay variance.
```sql
SELECT 
    po.po_id,
    s.supplier_name,
    p.product_name,
    d.planned_lead_days,
    d.actual_lead_days,
    d.delay_days,
    CASE 
        WHEN d.delay_days <= 0 THEN 'On Time / Early'
        WHEN d.delay_days BETWEEN 1 AND 3 THEN 'Minor Delay (1-3d)'
        WHEN d.delay_days BETWEEN 4 AND 7 THEN 'Moderate Delay (4-7d)'
        ELSE 'Severe Delay (>7d)'
    END AS delay_severity
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN deliveries d ON po.po_id = d.po_id
JOIN products p ON po.product_id = p.product_id;
```

---

## 5. Inventory Stockout Risk & Replenishment Matrix
Evaluates current stock against safety stock thresholds and monthly demand.
```sql
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
        WHEN i.current_stock < i.reorder_level THEN 'Understocked'
        WHEN i.months_of_cover > 6.0 THEN 'Overstocked'
        ELSE 'Healthy'
    END AS stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id;
```

---

## 6. Quality Defect Rate & Defect Spend Exposure Matrix
Calculates financial risk resulting from non-conforming or defective orders.
```sql
SELECT 
    s.supplier_id,
    s.supplier_name,
    COUNT(po.po_id) AS total_orders,
    SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) AS defective_orders,
    ROUND(100.0 * SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 2) AS defect_rate_pct,
    ROUND(SUM(CASE WHEN d.has_defect = 1 THEN po.order_cost ELSE 0 END), 2) AS defect_spend_exposure
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY s.supplier_id, s.supplier_name
ORDER BY defect_spend_exposure DESC;
```

---

## 7. Predictive Machine Learning Feature Engineering Extraction
Extracts 60-day rolling supplier metrics for ML late delivery risk prediction.
```sql
SELECT 
    po.po_id,
    po.supplier_id,
    po.order_date,
    po.quantity,
    po.unit_price,
    po.order_cost,
    po.priority,
    po.shipping_mode,
    d.is_late
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id;
```

---

## 8. Supplier Spend Pareto (80/20 Rule) Concentration Analysis
Classifies suppliers into spend tiers (Class A, B, C) based on cumulative spend share.
```sql
WITH supplier_spend AS (
    SELECT 
        s.supplier_id,
        s.supplier_name,
        SUM(po.order_cost) AS total_spend
    FROM purchase_orders po
    JOIN suppliers s ON po.supplier_id = s.supplier_id
    GROUP BY s.supplier_id, s.supplier_name
),
spend_cum AS (
    SELECT 
        supplier_id,
        supplier_name,
        total_spend,
        SUM(total_spend) OVER(ORDER BY total_spend DESC) AS cum_spend,
        SUM(total_spend) OVER() AS grand_total
    FROM supplier_spend
)
SELECT 
    supplier_id,
    supplier_name,
    total_spend,
    ROUND(100.0 * cum_spend / grand_total, 2) AS cum_spend_pct,
    CASE 
        WHEN 100.0 * cum_spend / grand_total <= 80.0 THEN 'Class A (Top 80% Spend)'
        WHEN 100.0 * cum_spend / grand_total <= 95.0 THEN 'Class B (Next 15% Spend)'
        ELSE 'Class C (Tail 5% Spend)'
    END AS pareto_tier
FROM spend_cum;
```

---

## 9. Monthly Order Volume MoM Growth & Seasonal Variance
Analyzes order volume trends and late delivery rate correlation across months.
```sql
SELECT 
    strftime('%Y-%m', po.order_date) AS order_month,
    COUNT(po.po_id) AS total_pos,
    ROUND(SUM(po.order_cost), 2) AS total_spend,
    ROUND(100.0 * SUM(CASE WHEN d.is_late = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS late_rate_pct
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY order_month
ORDER BY order_month;
```

---

## 10. Order Fulfillment Bottleneck & Delay Severity Ranking
Categorizes and aggregates order fulfillment delays by severity tiers across categories.
```sql
SELECT 
    p.category,
    COUNT(po.po_id) AS total_orders,
    SUM(CASE WHEN d.delay_days BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS minor_delays,
    SUM(CASE WHEN d.delay_days BETWEEN 4 AND 7 THEN 1 ELSE 0 END) AS moderate_delays,
    SUM(CASE WHEN d.delay_days > 7 THEN 1 ELSE 0 END) AS severe_delays
FROM purchase_orders po
JOIN products p ON po.product_id = p.product_id
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY p.category;
```
