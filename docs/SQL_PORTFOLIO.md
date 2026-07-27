# ProcureSense AI — Production SQL Analytics Portfolio

Comprehensive documentation of the 10 enterprise SQL queries implemented in **ProcureSense AI SQL Analytics Studio**.

---

## 1. MoM Spend & Cumulative Running Spend
- **Technique**: Window Aggregations (`SUM() OVER (PARTITION BY ... ORDER BY ... ROWS UNBOUNDED PRECEDING)`), `strftime()`
- **Business Goal**: Tracks monthly procurement spend per category alongside cumulative running spend to monitor budget consumption.

```sql
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
ORDER BY category, order_month;
```

---

## 2. Regional Supplier SLA Ranking
- **Technique**: Dense Ranking (`DENSE_RANK() OVER (PARTITION BY region ORDER BY ...)`), `HAVING COUNT >= 10`
- **Business Goal**: Ranks active suppliers within geographic regions based on fulfillment reliability (on-time delivery % and average delay days).

```sql
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
ORDER BY region, regional_rank;
```

---

## 3. Year-over-Year Unit Price Drift (OLS Signal)
- **Technique**: Lag Functions (`LAG(avg_unit_price, 1) OVER (PARTITION BY ... ORDER BY ...)`), `strftime('%Y')`
- **Business Goal**: Identifies price inflation signals by measuring year-over-year unit price percentage changes per supplier.

```sql
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
ORDER BY yoy_price_change_pct DESC;
```

---

## 4. Lead Time Variance & Reliability Cohorts
- **Technique**: Julian Day Date Arithmetic (`julianday(delivery_date) - julianday(order_date)`), Multi-table Joins
- **Business Goal**: Measures discrepancy between contracted lead times and actual delivery days across logistics channels and commercial tiers.

```sql
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
```

---

## 5. Inventory Stockout Risk Matrix
- **Technique**: Conditional CASE Logic (`CASE WHEN current_stock < reorder_level ...`), Multi-table joins (`inventory` -> `products`)
- **Business Goal**: Evaluates stock cover, months of cover, and flags understocked SKUs breaching reorder thresholds.

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
        WHEN i.current_stock < i.reorder_level THEN 'Reorder Required'
        WHEN i.months_of_cover > 6 THEN 'Overstocked'
        ELSE 'Healthy'
    END AS stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id
ORDER BY i.months_of_cover ASC;
```

---

## 6. Quality Defect Rate & Spend Exposure
- **Technique**: Conditional Financial Aggregation (`SUM(CASE WHEN has_defect = 1 THEN order_cost ELSE 0 END)`)
- **Business Goal**: Quantifies financial loss resulting from defective shipments per product sub-category.

```sql
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
ORDER BY defective_spend_exposure DESC;
```

---

## 7. Predictive ML Feature Engineering
- **Technique**: Temporal Extraction (`strftime('%m', order_date)`), Raw Inference Joins
- **Business Goal**: Extracts normalized order rows, temporal features, and target labels to feed machine learning models.

```sql
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
ORDER BY po.order_date ASC;
```

---

## 8. Supplier Spend Pareto 80/20 Analysis
- **Technique**: Cumulative Window Totals (`SUM() OVER (ORDER BY total_spend DESC ROWS UNBOUNDED PRECEDING)`), ABC Pareto Classing
- **Business Goal**: Classifies suppliers into Class A (Top 80% spend), Class B (Next 15%), and Class C (Tail spend) using running cumulative totals.

```sql
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
ORDER BY total_spend DESC;
```

---

## 9. Monthly Order Volume MoM Growth
- **Technique**: Lag Functions (`LAG(current_month_orders, 1) OVER (ORDER BY order_month)`), NULLIF Handling
- **Business Goal**: Computes month-over-month percentage growth in order volume and tracks late delivery rate point changes.

```sql
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
ORDER BY order_month;
```

---

## 10. Fulfillment Bottleneck & Delay Severity
- **Technique**: Multi-bucket Conditional Sums (`SUM(CASE WHEN delay_days BETWEEN ...)`), Grouping by Mode & Region
- **Business Goal**: Categorizes delivery delay severity into on-time, minor (1-3d), moderate (4-7d), and severe (>7d) delay buckets across shipping channels.

```sql
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
```
