"""
build_db.py (v3 — Schema DDL Integration & Incremental Load Support)
----------------------------------------------------------------------
Loads procurement CSV files into a SQLite database (procurement.db) with:
1. Formal DDL schema execution from db/schema.sql
2. PRAGMA foreign_keys = ON for referential integrity
3. Incremental load support (append vs replace mode)
4. Comprehensive multi-column indexing for fast query performance
5. Fixed SQL grouping by supplier_id (resolving supplier name collision bug)
"""

import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")

# ---------------------------------------------------------------
# CORE SQL QUERIES (Exported for KPI engine & Analytics Studio)
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
GROUP BY s.supplier_id, s.supplier_name, s.region, s.tier
ORDER BY on_time_pct DESC;
"""

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

PRICE_TREND_SQL = """
SELECT
    s.supplier_id,
    s.supplier_name,
    strftime('%Y', po.order_date) AS year,
    ROUND(AVG(po.unit_price), 2)  AS avg_unit_price,
    COUNT(*)                      AS n_orders
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
GROUP BY s.supplier_id, s.supplier_name, year
ORDER BY s.supplier_name, year;
"""

def build_db(db_path=DB_PATH, data_dir=DATA, schema_path=SCHEMA_PATH, incremental=False, verbose=True):
    """
    Builds procurement.db from CSV files, enforces foreign keys,
    applies DDL schema, supports incremental loading, and builds performance indexes.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    # Apply DDL schema if schema file exists
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as sf:
            conn.executescript(sf.read())
        if verbose:
            print(f"[OK] Applied database DDL schema from {schema_path}")

    table_pks = {
        "suppliers": "supplier_id",
        "products": "product_id",
        "purchase_orders": "po_id",
        "deliveries": "po_id",
        "inventory": "product_id"
    }

    tables = ["suppliers", "products", "purchase_orders", "deliveries", "inventory"]
    for t in tables:
        csv_file = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(csv_file):
            if verbose:
                print(f"[SKIP] CSV file not found: {csv_file}")
            continue

        date_cols = (
            ["order_date", "expected_delivery_date"] if t == "purchase_orders" else
            ["delivery_date"] if t == "deliveries" else []
        )
        df = pd.read_csv(csv_file, parse_dates=date_cols)

        if incremental:
            # Check if table already exists in database
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)
            ).fetchone() is not None

            if table_exists and t in table_pks:
                pk_col = table_pks[t]
                existing_ids = set(pd.read_sql(f"SELECT {pk_col} FROM {t}", conn)[pk_col])
                original_len = len(df)
                df = df[~df[pk_col].isin(existing_ids)]
                if verbose:
                    print(f"[INCREMENTAL] Table '{t}': {len(df):,} new rows to append ({original_len - len(df):,} duplicate PK rows skipped)")

            if not df.empty:
                df.to_sql(t, conn, index=False, if_exists="append")
            elif verbose:
                print(f"[SKIP] Table '{t}' up-to-date. Zero new rows added.")
        else:
            df.to_sql(t, conn, index=False, if_exists="replace")
            if verbose:
                print(f"[OK] Loaded table '{t}' ({len(df):,} rows, mode: replace)")

    # Optionally load Kaggle benchmark datasets if available
    for kt in ["kaggle_supply_chain", "kaggle_dataco_sample"]:
        kpath = os.path.join(data_dir, f"{kt}.csv")
        if os.path.exists(kpath):
            df_k = pd.read_csv(kpath, encoding="latin-1" if "dataco" in kt else "utf-8")
            sql_if_exists = "append" if incremental else "replace"
            df_k.to_sql(kt, conn, index=False, if_exists=sql_if_exists)
            if verbose:
                print(f"[OK] Loaded Kaggle table '{kt}' ({len(df_k):,} rows)")

    # INDEXING STRATEGY
    indexes = [
        ("idx_po_supplier", "purchase_orders", "supplier_id"),
        ("idx_po_product", "purchase_orders", "product_id"),
        ("idx_po_order_date", "purchase_orders", "order_date"),
        ("idx_del_po", "deliveries", "po_id"),
        ("idx_del_is_late", "deliveries", "is_late"),
        ("idx_sup_region", "suppliers", "region"),
        ("idx_sup_tier", "suppliers", "tier"),
        ("idx_prod_category", "products", "category")
    ]

    for idx_name, tbl, col in indexes:
        conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl}({col})")

    conn.commit()
    conn.close()
    if verbose:
        print(f"\n[OK] Database successfully built at {db_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build procurement SQLite database from CSV files.")
    parser.add_argument("--incremental", action="store_true", help="Append new CSV rows without duplicating existing PKs.")
    parser.add_argument("--db-path", default=DB_PATH, help="Target SQLite database path.")
    parser.add_argument("--data-dir", default=DATA, help="Directory containing CSV files.")
    args = parser.parse_args()

    build_db(db_path=args.db_path, data_dir=args.data_dir, incremental=args.incremental)

    conn = sqlite3.connect(args.db_path)
    print("\n=== Supplier Performance (top 5) ===")
    print(pd.read_sql(SUPPLIER_PERFORMANCE_SQL, conn).head())
    print("\n=== Monthly Trend (first 5 months) ===")
    print(pd.read_sql(MONTHLY_TREND_SQL, conn).head())
    print("\n=== Inventory Risk (status counts) ===")
    print(pd.read_sql(INVENTORY_RISK_SQL, conn)["stock_status"].value_counts())
    conn.close()
