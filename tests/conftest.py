"""
conftest.py — Pytest Fixtures & Synthetic Testing Setup
-------------------------------------------------------
Provides in-memory SQLite database connection and synthetic DataFrames
for testing KPI engine, ML pipeline, and dashboard components without
requiring the production 14 MB procurement.db file.
"""

import pytest
import sqlite3
import pandas as pd
import numpy as np

@pytest.fixture
def mock_db_conn():
    """In-memory SQLite database fixture with sample schema and synthetic data."""
    conn = sqlite3.connect(":memory:")
    
    # Create DDL schema
    conn.executescript("""
        CREATE TABLE suppliers (
            supplier_id TEXT PRIMARY KEY,
            supplier_name TEXT NOT NULL,
            region TEXT,
            tier TEXT
        );
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            sub_category TEXT,
            unit_cost_base REAL,
            lead_time_days_base INTEGER
        );
        CREATE TABLE purchase_orders (
            po_id TEXT PRIMARY KEY,
            order_date TEXT NOT NULL,
            product_id TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            order_cost REAL NOT NULL
        );
        CREATE TABLE deliveries (
            po_id TEXT PRIMARY KEY,
            is_late INTEGER NOT NULL,
            delay_days INTEGER DEFAULT 0,
            has_defect INTEGER DEFAULT 0
        );
        CREATE TABLE inventory (
            product_id TEXT PRIMARY KEY,
            current_stock INTEGER DEFAULT 0,
            reorder_level INTEGER DEFAULT 0,
            avg_monthly_demand REAL DEFAULT 0.0,
            months_of_cover REAL DEFAULT 999.0
        );
    """)

    # Populate sample rows
    conn.execute("INSERT INTO suppliers VALUES ('SUP1001', 'Acme Corp', 'North India', 'Tier 1');")
    conn.execute("INSERT INTO suppliers VALUES ('SUP1002', 'Global Tech', 'Import - China', 'Tier 2');")
    
    conn.execute("INSERT INTO products VALUES ('PRD2001', 'Steel Sheet', 'Raw Metals', 'Steel', 200.0, 10);")
    conn.execute("INSERT INTO products VALUES ('PRD2002', 'MCU Chip', 'Electronics', 'Microcontrollers', 500.0, 14);")

    conn.execute("INSERT INTO purchase_orders VALUES ('PO-001', '2023-05-10', 'PRD2001', 'SUP1001', 100, 200.0, 20000.0);")
    conn.execute("INSERT INTO purchase_orders VALUES ('PO-002', '2023-06-15', 'PRD2002', 'SUP1002', 50, 500.0, 25000.0);")

    conn.execute("INSERT INTO deliveries VALUES ('PO-001', 0, 0, 0);")
    conn.execute("INSERT INTO deliveries VALUES ('PO-002', 1, 5, 1);")

    conn.execute("INSERT INTO inventory VALUES ('PRD2001', 150, 100, 50.0, 3.0);")
    conn.execute("INSERT INTO inventory VALUES ('PRD2002', 20, 50, 30.0, 0.67);")

    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def sample_ml_df():
    """Synthetic DataFrame for testing ML out-of-fold target encoding and evaluation metrics."""
    n_rows = 100
    rng = np.random.default_rng(42)
    
    df = pd.DataFrame({
        "po_id": [f"PO-{i:03d}" for i in range(n_rows)],
        "order_year": rng.choice([2023, 2024, 2025], n_rows, p=[0.4, 0.4, 0.2]),
        "order_month": rng.integers(1, 13, n_rows),
        "supplier_id": rng.choice(["SUP1001", "SUP1002", "SUP1003"], n_rows),
        "product_id": rng.choice(["PRD2001", "PRD2002"], n_rows),
        "category": rng.choice(["Raw Metals", "Electronics"], n_rows),
        "shipping_mode": rng.choice(["Air Freight", "Sea Freight"], n_rows),
        "region": rng.choice(["North India", "Import - China"], n_rows),
        "tier": rng.choice(["Tier 1", "Tier 2"], n_rows),
        "quantity": rng.integers(10, 500, n_rows),
        "unit_price": rng.uniform(50.0, 500.0, n_rows),
        "order_cost": rng.uniform(500.0, 250000.0, n_rows),
        "unit_cost_base": rng.uniform(40.0, 450.0, n_rows),
        "lead_time_days_base": rng.integers(5, 30, n_rows),
        "is_late": rng.choice([0, 1], n_rows, p=[0.6, 0.4])
    })
    return df
