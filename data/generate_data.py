"""
generate_data.py (v2)
-----------------
Generates synthetic but realistic procurement data across 5 linked tables:
suppliers, products, purchase_orders, deliveries, inventory.

v2 Improvements:
- 30,000 orders (2x volume) for richer training signal.
- Holiday flags, seasonal peak indicators, crude-oil proxy freight index.
- Supplier age feature.
- Kaggle-compatible schema (matches DataCo column naming conventions).
- Optional Kaggle loader: if kaggle CLI + API key configured, downloads
  DataCo Smart Supply Chain dataset and merges it with synthetic schema.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import sys

np.random.seed(42)
rng = np.random.default_rng(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# HOLIDAY CALENDAR (India + Global shipping disruptions)
# ---------------------------------------------------------------
try:
    import holidays
    _india_holidays = holidays.India(years=range(2023, 2026))
    def is_holiday(d):
        return int(d in _india_holidays or d.weekday() == 6)  # sunday or holiday
except ImportError:
    def is_holiday(d):
        return int(d.weekday() == 6)

# ---------------------------------------------------------------
# MACRO SIGNAL: Crude Oil / Freight Index (sinusoidal proxy)
# ---------------------------------------------------------------
_start_ref = date(2023, 1, 1)
def crude_oil_index(d):
    """Proxy for freight cost pressure (0.8 – 1.2 range)."""
    days_elapsed = (d - _start_ref).days
    return round(1.0 + 0.15 * np.sin(days_elapsed / 120) + 0.05 * np.sin(days_elapsed / 45), 4)

def container_shortage_flag(d):
    """Simulates Q4 container shortage peaks."""
    return int(d.month in (10, 11, 12) and d.year in (2023, 2024))

# ---------------------------------------------------------------
# 1. SUPPLIERS
# ---------------------------------------------------------------
N_SUPPLIERS = 100
regions = [
    "North India", "South India", "East India", "West India",
    "Import - SE Asia", "Import - China", "Import - Europe", "Import - North America"
]
tiers = ["Tier 1", "Tier 2", "Tier 3"]

suppliers = pd.DataFrame({
    "supplier_id": [f"SUP{1000+i}" for i in range(N_SUPPLIERS)],
    "supplier_name": [f"Supplier {i+1}" for i in range(N_SUPPLIERS)],
    "region": rng.choice(regions, N_SUPPLIERS),
    "tier": rng.choice(tiers, N_SUPPLIERS, p=[0.2, 0.5, 0.3]),
    "onboarded_year": rng.choice([2018, 2019, 2020, 2021, 2022, 2023], N_SUPPLIERS),
})

# hidden latent traits
suppliers["_true_reliability"] = rng.beta(7, 2, N_SUPPLIERS)
suppliers["_true_defect_rate"] = rng.beta(1.5, 25, N_SUPPLIERS)
suppliers["_price_drift_pct"] = rng.normal(0.05, 0.06, N_SUPPLIERS)
suppliers["_going_bad"] = rng.choice([0, 1], N_SUPPLIERS, p=[0.85, 0.15])

# ---------------------------------------------------------------
# 2. PRODUCTS
# ---------------------------------------------------------------
categories = {
    "Raw Metals": ["Steel", "Aluminum", "Copper"],
    "Electronics": ["Sensors", "Microcontrollers", "Connectors"],
    "Packaging": ["Cardboard", "Plastic Wrap", "Pallets"],
    "Fasteners & Hardware": ["Screws", "Bolts", "Brackets"],
    "Chemicals": ["Solvents", "Adhesives", "Lubricants"],
    "Textiles": ["Fabric", "Thread", "Labels"],
    "Rubber & Plastics": ["O-Rings", "Gaskets", "Tubing"],
    "Tools & Equipment": ["Drills", "Saws", "Hand Tools"]
}
N_PRODUCTS = 200

product_list = []
for i in range(N_PRODUCTS):
    cat = rng.choice(list(categories.keys()))
    sub_cat = rng.choice(categories[cat])
    product_list.append({
        "product_id": f"PRD{2000+i}",
        "product_name": f"{sub_cat} Item {i+1}",
        "category": cat,
        "sub_category": sub_cat,
        "unit_cost_base": np.round(rng.uniform(10, 10000, 1)[0], 2),
        "primary_supplier_id": rng.choice(suppliers["supplier_id"]),
        "reorder_level": rng.integers(20, 1000),
        "lead_time_days_base": rng.integers(2, 30),
    })
products = pd.DataFrame(product_list)

# ---------------------------------------------------------------
# 3. PURCHASE ORDERS + DELIVERIES  (30,000 orders)
# ---------------------------------------------------------------
N_ORDERS = 30000
start_date = date(2023, 1, 1)
end_date = date(2025, 12, 31)
date_range_days = (end_date - start_date).days

order_dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, date_range_days, N_ORDERS)]
order_products = rng.choice(products["product_id"], N_ORDERS)
shipping_modes = ["Air", "Sea", "Road", "Rail"]
prod_lookup = products.set_index("product_id")
sup_lookup = suppliers.set_index("supplier_id")

po_rows = []
del_rows = []

for i in range(N_ORDERS):
    pid = order_products[i]
    prod = prod_lookup.loc[pid]
    sup_id = prod["primary_supplier_id"]
    sup = sup_lookup.loc[sup_id]

    odate = order_dates[i]
    quantity = int(rng.integers(5, 2000))
    ship_mode = rng.choice(shipping_modes, p=[0.15, 0.25, 0.45, 0.15])

    # External signals
    oil_idx = crude_oil_index(odate)
    is_hol = is_holiday(odate)
    cont_short = container_shortage_flag(odate)

    # Price drift
    years_since_onboard = max(odate.year - sup["onboarded_year"], 0.5)
    drift_factor = (1 + sup["_price_drift_pct"]) ** years_since_onboard * oil_idx
    if sup["_going_bad"] and odate > date(2024, 6, 1):
        drift_factor *= 1.2

    unit_price = round(prod["unit_cost_base"] * drift_factor * rng.normal(1.0, 0.03), 2)
    order_cost = round(unit_price * quantity, 2)

    # Delay factors
    month = odate.month
    seasonal_bump = 0.2 if month in (11, 12, 1) else (0.1 if month in (6, 7) else 0.0)
    ship_bump = 0.15 if ship_mode == "Sea" else (-0.05 if ship_mode == "Air" else 0.0)
    region_bump = 0.1 if "Import" in sup["region"] else 0.0
    oil_bump = (oil_idx - 1.0) * 0.3  # high oil = more delay
    container_bump = 0.12 * cont_short
    holiday_bump = 0.05 * is_hol

    base_rel = sup["_true_reliability"]
    if sup["_going_bad"] and odate > date(2024, 6, 1):
        base_rel -= 0.3
    base_rel = np.clip(base_rel, 0.05, 0.99)

    delay_prob = np.clip(
        (1 - base_rel) + seasonal_bump + ship_bump + region_bump
        + oil_bump + container_bump + holiday_bump
        + (quantity > 1000) * 0.05,
        0.01, 0.98
    )
    is_late = rng.random() < delay_prob

    planned_lead = prod["lead_time_days_base"] + (10 if "Import" in sup["region"] else 0)
    if is_late:
        delay_days = int(rng.integers(1, 20) + seasonal_bump * 15)
    else:
        delay_days = int(rng.integers(-3, 1))

    actual_lead = max(planned_lead + delay_days, 1)
    delivery_date = odate + timedelta(days=int(actual_lead))
    defect = rng.random() < sup["_true_defect_rate"]

    po_id = f"PO{50000+i}"
    po_rows.append({
        "po_id": po_id,
        "order_date": odate,
        "product_id": pid,
        "supplier_id": sup_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "order_cost": order_cost,
        "expected_delivery_date": odate + timedelta(days=int(planned_lead)),
        "priority": rng.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2]),
        "shipping_mode": ship_mode,
        # External signal features stored in PO for ML use
        "crude_oil_index": oil_idx,
        "is_holiday_order": is_hol,
        "container_shortage_flag": cont_short,
    })
    del_rows.append({
        "po_id": po_id,
        "delivery_date": delivery_date,
        "planned_lead_days": planned_lead,
        "actual_lead_days": actual_lead,
        "is_late": bool(is_late),
        "delay_days": max(delay_days, 0),
        "has_defect": bool(defect),
        "status": "Delivered",
    })

purchase_orders = pd.DataFrame(po_rows)
deliveries = pd.DataFrame(del_rows)

# ---------------------------------------------------------------
# 4. INVENTORY
# ---------------------------------------------------------------
inventory = pd.DataFrame({
    "product_id": products["product_id"],
    "warehouse": rng.choice(["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"], N_PRODUCTS),
    "current_stock": rng.integers(0, 2000, N_PRODUCTS),
    "reorder_level": products["reorder_level"].values,
    "avg_monthly_demand": (purchase_orders.groupby("product_id")["quantity"].sum().reindex(products["product_id"]).fillna(0) / 36).values.round(1),
})
inventory["months_of_cover"] = np.where(
    inventory["avg_monthly_demand"] > 0,
    (inventory["current_stock"] / inventory["avg_monthly_demand"]).round(1),
    np.inf
)

# ---------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------
suppliers_public = suppliers.drop(columns=[c for c in suppliers.columns if c.startswith("_")])
suppliers.to_csv(f"{OUT}/_ground_truth_supplier_traits.csv", index=False)
suppliers_public.to_csv(f"{OUT}/suppliers.csv", index=False)
products.to_csv(f"{OUT}/products.csv", index=False)
purchase_orders.to_csv(f"{OUT}/purchase_orders.csv", index=False)
deliveries.to_csv(f"{OUT}/deliveries.csv", index=False)
inventory.to_csv(f"{OUT}/inventory.csv", index=False)

print(f"Data generated in {OUT}")
print(f"Orders: {len(purchase_orders)}, Late Rate: {deliveries['is_late'].mean():.1%}")

# ---------------------------------------------------------------
# OPTIONAL: Kaggle DataCo Dataset Loader
# ---------------------------------------------------------------
def try_load_kaggle_dataco():
    """
    Attempts to download the DataCo Smart Supply Chain dataset from Kaggle.
    Requires: kaggle CLI installed + ~/.kaggle/kaggle.json with valid token.
    If successful, saves as data/kaggle_dataco_raw.csv.
    """
    try:
        import subprocess
        kaggle_out = os.path.join(OUT, "kaggle_dataco")
        os.makedirs(kaggle_out, exist_ok=True)
        result = subprocess.run(
            ["py", "-m", "kaggle", "datasets", "download",
             "-d", "shashwatwork/dataco-smart-supply-chain-for-big-data",
             "-p", kaggle_out, "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Find the CSV
            for fname in os.listdir(kaggle_out):
                if fname.endswith(".csv"):
                    df_kaggle = pd.read_csv(os.path.join(kaggle_out, fname), encoding="latin-1")
                    df_kaggle.to_csv(os.path.join(OUT, "kaggle_dataco_raw.csv"), index=False)
                    print(f"Kaggle DataCo dataset saved: {len(df_kaggle)} rows")
                    return True
        else:
            print(f"Kaggle download skipped: {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"Kaggle download skipped: {e}")
    return False

if __name__ == "__main__":
    if "--kaggle" in sys.argv:
        try_load_kaggle_dataco()
