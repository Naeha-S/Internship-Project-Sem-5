-- ProcureSense AI — Formal Database DDL Schema Definition File
-- File: db/schema.sql
-- Description: Full relational schema definition for procurement.db with primary keys,
--              foreign key constraints, column definitions, and multi-column indexes.

PRAGMA foreign_keys = ON;

-- 1. SUPPLIERS TABLE
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    contact_email TEXT,
    country TEXT,
    region TEXT,
    tier TEXT,
    onboarded_year INTEGER,
    rating REAL,
    payment_terms TEXT,
    certifications TEXT
);

-- 2. PRODUCTS TABLE
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    sku TEXT UNIQUE NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT,
    sub_category TEXT,
    unit_of_measure TEXT,
    unit_cost_base REAL,
    primary_supplier_id TEXT,
    secondary_supplier_id TEXT,
    reorder_level INTEGER,
    lead_time_days_base INTEGER,
    FOREIGN KEY (primary_supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (secondary_supplier_id) REFERENCES suppliers(supplier_id)
);

-- 3. PURCHASE ORDERS TABLE
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id TEXT PRIMARY KEY,
    po_number TEXT UNIQUE NOT NULL,
    buyer_name TEXT,
    incoterms TEXT,
    order_date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    order_cost REAL NOT NULL,
    expected_delivery_date TEXT,
    priority TEXT,
    shipping_mode TEXT,
    crude_oil_index REAL,
    is_holiday_order INTEGER,
    container_shortage_flag INTEGER,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- 4. DELIVERIES TABLE
CREATE TABLE IF NOT EXISTS deliveries (
    po_id TEXT PRIMARY KEY,
    tracking_number TEXT,
    carrier TEXT,
    delivery_date TEXT,
    planned_lead_days INTEGER,
    actual_lead_days INTEGER,
    is_late INTEGER NOT NULL,
    delay_days INTEGER DEFAULT 0,
    has_defect INTEGER DEFAULT 0,
    inspection_status TEXT,
    status TEXT DEFAULT 'Delivered',
    FOREIGN KEY (po_id) REFERENCES purchase_orders(po_id)
);

-- 5. INVENTORY TABLE
CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT PRIMARY KEY,
    warehouse TEXT,
    current_stock INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 0,
    avg_monthly_demand REAL DEFAULT 0.0,
    months_of_cover REAL DEFAULT 999.0,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- INDEXES FOR QUERY OPTIMIZATION
CREATE INDEX IF NOT EXISTS idx_po_supplier ON purchase_orders(supplier_id);
CREATE INDEX IF NOT EXISTS idx_po_product ON purchase_orders(product_id);
CREATE INDEX IF NOT EXISTS idx_po_order_date ON purchase_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_del_po ON deliveries(po_id);
CREATE INDEX IF NOT EXISTS idx_del_is_late ON deliveries(is_late);
CREATE INDEX IF NOT EXISTS idx_sup_region ON suppliers(region);
CREATE INDEX IF NOT EXISTS idx_sup_tier ON suppliers(tier);
CREATE INDEX IF NOT EXISTS idx_prod_category ON products(category);
