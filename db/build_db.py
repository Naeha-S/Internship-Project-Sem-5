"""
build_db.py
------------
Loads the 5 CSVs into a SQLite database (procurement.db) and runs the
core SQL queries that back the analytics dashboard. This is the part
of the project that actually demonstrates SQL competency — joins across
purchase_orders -> deliveries -> products -> suppliers.

Run standalone to regenerate procurement.db and print sample query output.
"""

import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")

conn = sqlite3.connect(DB_PATH)

tables = ["suppliers", "products", "purchase_orders", "deliveries", "inventory"]
for t in tables:
    df = pd.read_csv(f"{DATA}/{t}.csv", parse_dates=[c for c in
        (["order_date", "expected_delivery_date"] if t == "purchase_orders" else
         ["delivery_date"] if t == "deliveries" else [])])
    df.to_sql(t, conn, index=False, if_exists="replace")

conn.execute("DROP INDEX IF EXISTS idx_po_supplier")
conn.execute("DROP INDEX IF EXISTS idx_po_product")
conn.execute("DROP INDEX IF EXISTS idx_del_po")
conn.execute("CREATE INDEX IF NOT EXISTS idx_po_supplier ON purchase_orders(supplier_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_po_product ON purchase_orders(product_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_del_po ON deliveries(po_id)")
conn.commit()

# ---------------------------------------------------------------
# CORE SQL: Supplier Performance (joins po + deliveries + suppliers)
# ---------------------------------------------------------------
SUPPLIER_PERFORMANCE_SQL = """
SELECT
    s.supplier_id,
    s.supplier_name,
    s.region,
    s.tier,
    COUNT(po.po_id)                                      AS total_orders,
    ROUND(SUM(po.order_cost), 2)                          AS total_spend,
    ROUND(AVG(po.order_cost), 2)                          AS avg_order_value,
    ROUND(100.0 * SUM(CASE WHEN d.is_late = 0 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS on_time_pct,
    ROUND(AVG(d.delay_days), 2)                           AS avg_delay_days,
    ROUND(100.0 * SUM(CASE WHEN d.has_defect = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 2) AS defect_rate_pct
FROM purchase_orders po
JOIN suppliers s   ON po.supplier_id = s.supplier_id
JOIN deliveries d  ON po.po_id = d.po_id
GROUP BY s.supplier_id, s.supplier_name, s.region
ORDER BY on_time_pct DESC;
"""

# ---------------------------------------------------------------
# CORE SQL: Monthly spend + delay trend (procurement dashboard)
# ---------------------------------------------------------------
MONTHLY_TREND_SQL = """
SELECT
    strftime('%Y-%m', po.order_date) AS month,
    COUNT(po.po_id)                  AS order_count,
    ROUND(SUM(po.order_cost), 2)     AS total_spend,
    ROUND(100.0 * SUM(CASE WHEN d.is_late = 1 THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) AS late_pct
FROM purchase_orders po
JOIN deliveries d ON po.po_id = d.po_id
GROUP BY month
ORDER BY month;
"""

# ---------------------------------------------------------------
# CORE SQL: Inventory risk (dead stock / overstock / understock)
# ---------------------------------------------------------------
INVENTORY_RISK_SQL = """
SELECT
    i.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    i.current_stock,
    i.reorder_level,
    i.avg_monthly_demand,
    i.months_of_cover,
    CASE
        WHEN i.avg_monthly_demand = 0 THEN 'Dead Stock'
        WHEN i.current_stock < i.reorder_level THEN 'Understocked'
        WHEN i.months_of_cover > 6 THEN 'Overstocked'
        ELSE 'Healthy'
    END AS stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id
ORDER BY stock_status;
"""

# ---------------------------------------------------------------
# CORE SQL: Price trend per supplier (cost analysis / inflation)
# ---------------------------------------------------------------
PRICE_TREND_SQL = """
SELECT
    s.supplier_name,
    strftime('%Y', po.order_date) AS year,
    ROUND(AVG(po.unit_price), 2)  AS avg_unit_price,
    COUNT(*)                      AS n_orders
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
GROUP BY s.supplier_name, year
ORDER BY s.supplier_name, year;
"""

if __name__ == "__main__":
    print("=== Supplier Performance (top 5) ===")
    print(pd.read_sql(SUPPLIER_PERFORMANCE_SQL, conn).head())
    print("\n=== Monthly Trend (first 5 months) ===")
    print(pd.read_sql(MONTHLY_TREND_SQL, conn).head())
    print("\n=== Inventory Risk (status counts) ===")
    print(pd.read_sql(INVENTORY_RISK_SQL, conn)["stock_status"].value_counts())
    print(f"\nDatabase built at {DB_PATH}")

conn.close()
