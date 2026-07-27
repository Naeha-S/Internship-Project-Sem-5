"""
generate_data.py (v4 — Parameterized, Multi-Sourcing & OU Stochastic Macro Signals)
----------------------------------------------------------------------------------
Generates synthetic but highly realistic procurement data across 5 linked tables:
suppliers, products, purchase_orders, deliveries, inventory.

v4 Features:
- CLI Parameterization via argparse (--orders, --suppliers, --start-date, --end-date, --tables, --version-stamp)
- Multi-Sourcing Support (primary_supplier_id + secondary_supplier_id split)
- Ornstein-Uhlenbeck (OU) Mean-Reverting Stochastic Process for Crude Oil Index with real-world noise
- Data Versioning & Manifest export (data/manifest.json)
- Selective Table Regeneration (--tables suppliers products purchase_orders deliveries inventory)
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# HOLIDAY CALENDAR (India + Global shipping disruptions)
# ---------------------------------------------------------------
def get_holiday_checker(start_yr, end_yr):
    try:
        import holidays
        _india_holidays = holidays.India(years=range(start_yr, end_yr + 1))
        def is_holiday(d):
            return int(d in _india_holidays or d.weekday() == 6)
        return is_holiday
    except ImportError:
        def is_holiday(d):
            return int(d.weekday() == 6)
        return is_holiday

# ---------------------------------------------------------------
# MACRO SIGNALS: ORNSTEIN-UHLENBECK STOCHASTIC PROCESS
# ---------------------------------------------------------------
def generate_crude_oil_map(start_d, end_d, rng):
    """
    Simulates crude oil freight cost index (0.8 – 1.25 range) using an
    Ornstein-Uhlenbeck mean-reverting stochastic process with seasonal trend.
    """
    total_days = (end_d - start_d).days + 1
    dt = 1.0
    theta = 0.03   # Mean-reversion speed
    sigma = 0.012  # Stochastic volatility
    ou_val = 0.0
    
    oil_map = {}
    for t in range(total_days):
        cur_date = start_d + timedelta(days=t)
        seasonal = 1.0 + 0.12 * np.sin(t / 120.0) + 0.04 * np.sin(t / 45.0)
        dW = rng.normal(0, np.sqrt(dt))
        ou_val += theta * (0.0 - ou_val) * dt + sigma * dW
        ou_val = np.clip(ou_val, -0.2, 0.2)
        oil_map[cur_date] = round(float(seasonal + ou_val), 4)
    return oil_map

def container_shortage_flag(d):
    """Simulates Q4 container shortage peaks."""
    return int(d.month in (10, 11, 12))

# ---------------------------------------------------------------
# 1. AUTHENTIC SUPPLIERS CATALOG
# ---------------------------------------------------------------
REAL_SUPPLIER_NAMES = [
    "Apex Precision Components Ltd", "Bharat Heavy Metallics", "Pacific Rim Semiconductor Corp",
    "Mahindra Industrial Logistics", "Vanguard Chemical Solutions", "Shenzhen Microelectronics Co",
    "Bavaria Automotive Forgings", "Global Freight Networks", "Nordic Polymer Materials",
    "Tata Alloy & Steel Corp", "Zenith Electronics Mfg", "Reliance Industrial Polymers",
    "Orion Aerospace Materials", "Kyoto Precision Instruments", "Hindustan Fasteners & Hardware",
    "Atlas Heavy Machinery Supplies", "DynaFlex Rubber & Elastomers", "Evergreen Shipping Lines",
    "Sterling Chemical Synthetics", "Alpha Tool & Die Works", "Kavita Packaging Solutions",
    "Shanghai Industrial Silicon", "Stuttgart Auto Components", "Larsen Structural Engineers",
    "Vertex Microchip Fab", "Deccan Mining & Metals", "Pinnacle Precision Castings",
    "Matrix Logistics & Forwarding", "Kalyani Forge & Foundry", "Tokyo Electronic Wire Ltd",
    "Aegis Defense Equipment", "Sonata Acoustic Sensors", "Indus Valley Packaging",
    "BlueStar Refrigeration Gear", "Taiwan Semiconductor Foundry", "Ganga Hydraulics & Valves",
    "Titanium Tech Aerospace", "Continental Rubber Works", "Godrej Material Handling",
    "Hanover Precision Bearings", "Nippon Steel Logistics", "Shree Ram Textiles & Weaving",
    "AeroTech Composite Systems", "Bengal Chemical Reagents", "Silicon Valley Micro Hardware",
    "Prabhat Fastener Industries", "Krupp Heavy Metallics GmbH", "Gujarat Petrochem Synthetics", "OmniVision Optoelectronics",
    "Ashok Leyland Logistics", "Samsung Component Solutions", "Delta Power Electronics",
    "Jindal Stainless & Alloy", "Yokohama Industrial Polymers", "Swastik Cardboard & Packaging",
    "Federal Mogul Auto Parts", "Kirloskar Pumps & Compressors", "Toshiba Memory Systems",
    "Siemens Industrial Automation", "Sundaram Fasteners Ltd", "Dow Petrochemical Corp",
    "Murata Manufacturing Co", "Thermax Boilers & Heat Systems", "Bosch Mobility Electronics",
    "Hindalco Aluminum Extrusions", "Schneider Electric Gear", "Texas Instrument Distributors",
    "SKF Precision Bearings", "NTPC Equipment Supplies", "Hitachi Industrial Machinery",
    "Caterpillar Heavy Logistics", "TVS Auto Accessories", "3M Industrial Science",
    "BASF Chemical Solutions", "L&T Heavy Engineering", "Eaton Hydraulic Systems",
    "Parker Hannifin Pneumatics", "Panasonic Battery Industrial", "Honeywell Control Systems",
    "Cummins Engine Components", "Wipro Infrastructure Eng", "ABB Automation Systems",
    "Emerson Industrial Motion", "Danfoss Climate Solutions", "Rockwell Automation Gear",
    "Festo Pneumatic Automation", "Molex Interconnect Systems", "TE Connectivity Products",
    "Kyocera Electronic Ceramics", "Amphenol Cable Systems", "Yaskawa Electric Drives",
    "Fanuc Robotics Supplies", "Mitsubishi Heavy Industries", "Kawasaki Heavy Machinery",
    "Komatsu Earthmoving Parts", "Hyundai Heavy Industries", "Doosan Industrial Equipment", "SANY Heavy Machinery"
]

region_country_map = {
    "North India": "India",
    "South India": "India",
    "East India": "India",
    "West India": "India",
    "Import - SE Asia": "Vietnam",
    "Import - China": "China",
    "Import - Europe": "Germany",
    "Import - North America": "USA"
}

PRODUCT_CATALOG = {
    ("Raw Metals", "Steel"): (
        ["Cold-Rolled Steel Sheet 2mm", "Hot-Rolled Structural I-Beam", "Stainless Steel Pipe 304",
         "High-Tensile Carbon Steel Rod", "Galvanized Steel Roofing Sheet", "Tool Steel Die Block D2"],
        "Kg", (150.0, 450.0)
    ),
    ("Raw Metals", "Aluminum"): (
        ["Aircraft Aluminum 6061-T6 Bar", "Extruded Aluminum Channel 40x40", "Aluminum Sheet 3mm Alloy 5052",
         "High-Purity Aluminum Ingot", "Cast Aluminum Housing Enclosure"],
        "Kg", (220.0, 750.0)
    ),
    ("Raw Metals", "Copper"): (
        ["Oxygen-Free Copper Wire Spool", "Bare Copper Busbar 50x6mm", "Beryllium Copper Contact Strip",
         "Insulated Copper Winding Wire 18 AWG"],
        "Spools", (400.0, 1800.0)
    ),
    ("Electronics", "Sensors"): (
        ["Precision Temperature Sensor Probe PT100", "Digital Pressure Transducer 0-10 Bar", "Optical Infrared Proximity Sensor",
         "Tri-Axis MEMS Accelerometer Module", "Ultrasonic Distance Sensor 24V"],
        "Units", (350.0, 4500.0)
    ),
    ("Electronics", "Microcontrollers"): (
        ["STM32 ARM Cortex-M4 Microcontroller IC", "ESP32 Dual-Core Wi-Fi/BT SoC", "Microchip PIC18 8-Bit MCU",
         "FPGA Programmable Logic Array Board", "8-Channel Relay Driver Controller Board"],
        "Units", (180.0, 6800.0)
    ),
    ("Electronics", "Connectors"): (
        ["Heavy-Duty Industrial Connector 16-Pin", "Gold-Plated Circular Aviation Plug", "DIN-Rail Terminal Block Assembly",
         "PCB Header Connector 2.54mm"],
        "Packs", (85.0, 1200.0)
    ),
    ("Packaging", "Cardboard"): (
        ["Double-Wall Corrugated Box 50x40x30", "Heavy-Duty Shipping Container Carton", "Printed Product Packaging Sleeve",
         "Kraft Paper Buffer Roll 100m"],
        "Boxes", (25.0, 350.0)
    ),
    ("Packaging", "Plastic Wrap"): (
        ["Industrial Stretch Film Wrap 500mm", "Anti-Static ESD Bubble Wrap Roll", "Heavy Polyethylene Heat-Shrink Tubing"],
        "Packs", (45.0, 480.0)
    ),
    ("Packaging", "Pallets"): (
        ["Euro-Spec Heavy Wooden Cargo Pallet", "HDPE Molded Plastic Shipping Pallet", "Heat-Treated Export Pine Wood Pallet"],
        "Pallets", (650.0, 3200.0)
    ),
    ("Fasteners & Hardware", "Screws"): (
        ["Grade 8.8 M8 Socket Head Cap Screws", "Stainless Steel Self-Tapping Screws", "Titanium Alloy Torx Screws"],
        "Boxes", (40.0, 850.0)
    ),
    ("Fasteners & Hardware", "Bolts"): (
        ["High-Strength Anchor Bolts M16x200", "Galvanized Hex Head Structural Bolts M12", "Carriage Bolt & Flange Nut Set"],
        "Boxes", (60.0, 1100.0)
    ),
    ("Fasteners & Hardware", "Brackets"): (
        ["Heavy-Duty L-Shape Steel Corner Bracket", "Adjustable Mounting Rail Bracket", "Vibration Isolation Rubber Mount"],
        "Units", (75.0, 650.0)
    ),
    ("Chemicals", "Solvents"): (
        ["Industrial Solvent Acetone 99.5%", "Isopropanol Alcohol Degreaser 20L", "Trichloroethylene Metal Cleaning Fluid"],
        "Liters", (120.0, 1800.0)
    ),
    ("Chemicals", "Adhesives"): (
        ["High-Bond Two-Part Epoxy Resin 5L", "Structural Polyurethane Adhesive Sealant", "Anaerobic Threadlocking Compound 250ml"],
        "Units", (320.0, 3400.0)
    ),
    ("Chemicals", "Lubricants"): (
        ["Synthetic Hydraulic Fluid ISO VG 46", "High-Temp Lithium Grease 18kg", "Food-Grade Gear Oil Synthetic 220"],
        "Liters", (280.0, 2900.0)
    ),
    ("Textiles", "Fabric"): (
        ["Industrial Grade Kevlar Aramid Weave", "Heavy Canvas Tarpaulin Waterproof 600GSM", "Non-Woven Polypropylene Filter Cloth"],
        "Meters", (150.0, 2200.0)
    ),
    ("Textiles", "Thread"): (
        ["High-Tensile Bonded Nylon Sewing Thread", "Heat-Resistant Nomex Industrial Thread", "Polyester Core-Spun Thread Spool"],
        "Spools", (65.0, 850.0)
    ),
    ("Textiles", "Labels"): (
        ["Thermal Transfer Barcode Label Roll", "Woven Polyester Fabric Care Tag Labels", "Tamper-Evident Security Seal Stickers"],
        "Packs", (35.0, 450.0)
    ),
    ("Rubber & Plastics", "O-Rings"): (
        ["Viton High-Temp O-Ring Seal Kit", "NBR Nitrile Rubber Hydraulic O-Rings", "Silicone Food-Grade Sealing Ring 50mm"],
        "Packs", (80.0, 1400.0)
    ),
    ("Rubber & Plastics", "Gaskets"): (
        ["Spiral Wound Graphite Pipe Flange Gasket", "Neoprene Sheet Gasket Material 3mm", "Compressed Fiber Steam Gasket Joint"],
        "Units", (90.0, 1250.0)
    ),
    ("Rubber & Plastics", "Tubing"): (
        ["Reinforced Flexible PVC Braided Tubing", "PTFE Chemical Resistant Hose 1/2-Inch", "Pneumatic Polyurethane Air Line Hose"],
        "Meters", (50.0, 980.0)
    ),
    ("Tools & Equipment", "Drills"): (
        ["Solid Carbide Twist Drill Bit 10mm", "Industrial Rotary Hammer Drill 1100W", "Diamond Core Drilling Bit 50mm"],
        "Units", (250.0, 7500.0)
    ),
    ("Tools & Equipment", "Saws"): (
        ["Bi-Metal Bandsaw Blade 27mm", "Tungsten Carbide Circular Saw 355mm", "Abrasive Metal Cut-Off Wheel 14-Inch"],
        "Units", (180.0, 4200.0)
    ),
    ("Tools & Equipment", "Hand Tools"): (
        ["Adjustable Torque Wrench 1/2-Inch 40-200Nm", "Insulated Screwdriver Set 1000V", "Heavy Hydraulic Cable Crimper Tool"],
        "Units", (450.0, 8900.0)
    )
}

def generate_email(name, sup_id):
    clean = "".join(c for c in name.split()[0] if c.isalnum()).lower()
    return f"procurement@{clean}-{sup_id.lower()}.com"

def load_and_integrate_kaggle_data():
    """Optionally downloads and integrates Kaggle datasets."""
    kaggle_dir = os.path.join(OUT, "kaggle_raw")
    dataco_dir = os.path.join(OUT, "kaggle_dataco")
    os.makedirs(kaggle_dir, exist_ok=True)
    os.makedirs(dataco_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        sc_file = os.path.join(kaggle_dir, "supply_chain_data.csv")
        if not os.path.exists(sc_file):
            print("Downloading Kaggle Supply Chain Analysis dataset...")
            api.dataset_download_files('harshsingh2209/supply-chain-analysis', path=kaggle_dir, unzip=True)

        dataco_file = os.path.join(dataco_dir, "DataCoSupplyChainDataset.csv")
        if not os.path.exists(dataco_file):
            print("Downloading Kaggle DataCo Smart Supply Chain dataset...")
            api.dataset_download_files('shashwatwork/dataco-smart-supply-chain-for-big-data-analysis', path=dataco_dir, unzip=True)

        if os.path.exists(sc_file):
            df_sc = pd.read_csv(sc_file)
            df_sc.to_csv(os.path.join(OUT, "kaggle_supply_chain.csv"), index=False)
            print(f"[OK] Integrated Kaggle Supply Chain dataset: {len(df_sc):,} records")

        if os.path.exists(dataco_file):
            df_dc = pd.read_csv(dataco_file, encoding="latin-1", nrows=10000)
            df_dc.to_csv(os.path.join(OUT, "kaggle_dataco_sample.csv"), index=False)
            print(f"[OK] Integrated Kaggle DataCo dataset: {len(df_dc):,} benchmark records")
    except Exception as e:
        print(f"Kaggle integration status: {e}")

# ---------------------------------------------------------------
# MAIN GENERATOR FUNCTION
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ProcureSense AI Procurement Data Generator")
    parser.add_argument("--orders", type=int, default=28765, help="Number of purchase orders to generate (default: 28765)")
    parser.add_argument("--suppliers", type=int, default=100, help="Number of suppliers to generate (default: 100)")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date YYYY-MM-DD (default: 2023-01-01)")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date YYYY-MM-DD (default: 2025-12-31)")
    parser.add_argument("--tables", nargs="+", default=["all"], help="Tables to generate: all, suppliers, products, purchase_orders, deliveries, inventory")
    parser.add_argument("--version-stamp", action="store_true", help="Generate timestamped files and write manifest.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    date_range_days = (end_date - start_date).days
    deterioration_start_date = start_date + timedelta(days=int(date_range_days * 0.5))

    is_holiday_fn = get_holiday_checker(start_date.year, end_date.year)
    oil_map = generate_crude_oil_map(start_date, end_date, rng)

    tables_to_gen = set(args.tables)
    gen_all = "all" in tables_to_gen

    # 1. SUPPLIERS
    n_sups = min(args.suppliers, len(REAL_SUPPLIER_NAMES))
    supplier_names_selected = REAL_SUPPLIER_NAMES[:n_sups]
    regions = list(region_country_map.keys())
    tiers = ["Tier 1", "Tier 2", "Tier 3"]
    payment_terms_options = ["Net 30", "Net 60", "Net 90", "2/10 Net 30", "Immediate / CIA"]
    certifications_options = ["ISO 9001", "ISO 14001", "IATF 16949", "AS9100", "ISO 45001"]

    assigned_regions = rng.choice(regions, n_sups)
    assigned_countries = [region_country_map[r] for r in assigned_regions]
    sup_ids = [f"SUP{1000+i}" for i in range(n_sups)]

    suppliers = pd.DataFrame({
        "supplier_id": sup_ids,
        "supplier_name": supplier_names_selected,
        "contact_email": [generate_email(supplier_names_selected[i], sup_ids[i]) for i in range(n_sups)],
        "country": assigned_countries,
        "region": assigned_regions,
        "tier": rng.choice(tiers, n_sups, p=[0.25, 0.50, 0.25]),
        "onboarded_year": rng.choice([2018, 2019, 2020, 2021, 2022, 2023], n_sups),
        "rating": np.round(rng.uniform(3.2, 4.9, n_sups), 1),
        "payment_terms": rng.choice(payment_terms_options, n_sups, p=[0.5, 0.3, 0.1, 0.05, 0.05]),
        "certifications": rng.choice(certifications_options, n_sups),
    })

    suppliers["_true_reliability"] = rng.beta(6, 3, n_sups)
    suppliers["_true_defect_rate"] = rng.beta(1.2, 45, n_sups)
    suppliers["_price_drift_pct"] = rng.normal(0.03, 0.04, n_sups)
    suppliers["_going_bad"] = rng.choice([0, 1], n_sups, p=[0.85, 0.15])

    # 2. PRODUCTS & MULTI-SOURCING ASSIGNMENTS
    n_prods = 200
    product_list = []
    catalog_keys = list(PRODUCT_CATALOG.keys())

    for i in range(n_prods):
        cat, sub_cat = catalog_keys[i % len(catalog_keys)]
        names_list, uom, price_range = PRODUCT_CATALOG[(cat, sub_cat)]
        base_name = rng.choice(names_list)
        product_name = f"{base_name} (v{rng.integers(1, 4)})" if i >= len(catalog_keys) else base_name
        sku = f"SKU-{cat[:3].upper()}-{2000+i}"

        active_sup_ids = suppliers["supplier_id"].iloc[:78]
        p_sup = rng.choice(active_sup_ids)
        s_sup_candidates = [s for s in active_sup_ids if s != p_sup]
        s_sup = rng.choice(s_sup_candidates)

        product_list.append({
            "product_id": f"PRD{2000+i}",
            "sku": sku,
            "product_name": product_name,
            "category": cat,
            "sub_category": sub_cat,
            "unit_of_measure": uom,
            "unit_cost_base": np.round(rng.uniform(price_range[0], price_range[1]), 2),
            "primary_supplier_id": p_sup,
            "secondary_supplier_id": s_sup,
            "reorder_level": int(rng.integers(300, 1800)),
            "lead_time_days_base": int(rng.integers(2, 20)),
        })

    products = pd.DataFrame(product_list)

    # 3. PURCHASE ORDERS & DELIVERIES
    n_orders = args.orders
    buyer_names = ["Ananya Sharma", "Rahul Verma", "Sarah Jenkins", "Michael Chang", "Priya Nair", "David Miller", "Vikram Patel", "Elena Rostova"]
    incoterms_options = ["FOB", "CIF", "DDP", "EXW", "FCA", "DAP"]
    shipping_modes = ["Expedited Air", "Air Freight", "Express Ground", "Standard Ground", "Sea Freight"]
    carrier_options = ["DHL Express", "FedEx Supply Chain", "Maersk Line", "Blue Dart Cargo", "DB Schenker Logistics", "Kuehne+Nagel"]

    order_dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, date_range_days, n_orders)]
    order_products = rng.choice(products["product_id"], n_orders)

    prod_lookup = products.set_index("product_id")
    sup_lookup = suppliers.set_index("supplier_id")

    po_rows = []
    del_rows = []

    for i in range(n_orders):
        pid = order_products[i]
        prod = prod_lookup.loc[pid]
        
        # Multi-sourcing allocation: 85% primary, 15% secondary supplier
        use_secondary = rng.random() < 0.15
        sup_id = prod["secondary_supplier_id"] if use_secondary else prod["primary_supplier_id"]
        sup = sup_lookup.loc[sup_id]

        odate = order_dates[i]
        quantity = int(rng.integers(5, 1500))
        ship_mode = rng.choice(shipping_modes, p=[0.10, 0.20, 0.25, 0.30, 0.15])
        buyer = rng.choice(buyer_names)
        incoterm = rng.choice(incoterms_options)

        oil_idx = oil_map.get(odate, 1.0)
        is_hol = is_holiday_fn(odate)
        cont_short = container_shortage_flag(odate)

        years_since_onboard = max(odate.year - sup["onboarded_year"], 0.5)
        drift_factor = (1 + sup["_price_drift_pct"]) ** years_since_onboard * oil_idx
        if sup["_going_bad"] and odate > deterioration_start_date:
            drift_factor *= 1.2

        unit_price = round(prod["unit_cost_base"] * drift_factor * rng.normal(1.0, 0.02), 2)
        order_cost = round(unit_price * quantity, 2)

        month = odate.month
        seasonal_bump = 0.15 if month in (11, 12, 1) else (0.08 if month in (6, 7) else 0.0)
        ship_bump = 0.14 if ship_mode == "Sea Freight" else (0.07 if ship_mode == "Standard Ground" else (-0.08 if "Air" in ship_mode else 0.0))
        region_bump = 0.08 if "Import" in sup["region"] else 0.0
        oil_bump = (oil_idx - 1.0) * 0.2
        container_bump = 0.10 * cont_short
        holiday_bump = 0.04 * is_hol

        base_rel = sup["_true_reliability"]
        if sup["_going_bad"] and odate > deterioration_start_date:
            base_rel -= 0.25
        base_rel = np.clip(base_rel, 0.08, 0.98)

        delay_prob = np.clip(
            (1 - base_rel) + seasonal_bump + ship_bump + region_bump
            + oil_bump + container_bump + holiday_bump
            + (quantity > 900) * 0.04,
            0.01, 0.95
        )
        is_late = rng.random() < delay_prob

        planned_lead = prod["lead_time_days_base"] + (5 if "Import" in sup["region"] else 0)
        if is_late:
            delay_days = int(rng.integers(1, 12) + seasonal_bump * 10)
        else:
            delay_days = int(rng.integers(-2, 1))

        actual_lead = max(planned_lead + delay_days, 1)
        delivery_date = odate + timedelta(days=int(actual_lead))
        defect = rng.random() < sup["_true_defect_rate"]

        po_id = f"PO{50000+i}"
        po_num = f"PO-{odate.year}-{10000+i}"
        trk_num = f"TRK-{rng.integers(10000000, 99999999)}"
        carrier = rng.choice(carrier_options)

        if defect:
            inspection_status = rng.choice(["Major Defect - Rejected", "Minor Defect - Accepted"], p=[0.6, 0.4])
        else:
            inspection_status = "Passed Inspection"

        delivery_status = "In Transit" if delivery_date > end_date else "Delivered"

        po_rows.append({
            "po_id": po_id,
            "po_number": po_num,
            "buyer_name": buyer,
            "incoterms": incoterm,
            "order_date": odate,
            "product_id": pid,
            "supplier_id": sup_id,
            "quantity": quantity,
            "unit_price": unit_price,
            "order_cost": order_cost,
            "expected_delivery_date": odate + timedelta(days=int(planned_lead)),
            "priority": rng.choice(["Standard", "Rush", "Urgent", "Critical"], p=[0.5, 0.3, 0.15, 0.05]),
            "shipping_mode": ship_mode,
            "crude_oil_index": oil_idx,
            "is_holiday_order": is_hol,
            "container_shortage_flag": cont_short,
        })

        del_rows.append({
            "po_id": po_id,
            "tracking_number": trk_num,
            "carrier": carrier,
            "delivery_date": delivery_date,
            "planned_lead_days": planned_lead,
            "actual_lead_days": actual_lead,
            "is_late": bool(is_late),
            "delay_days": max(delay_days, 0),
            "has_defect": bool(defect),
            "inspection_status": inspection_status,
            "status": delivery_status,
        })

    purchase_orders = pd.DataFrame(po_rows)
    deliveries = pd.DataFrame(del_rows)

    # 4. INVENTORY
    inventory = pd.DataFrame({
        "product_id": products["product_id"],
        "warehouse": rng.choice(["WH-North (Delhi)", "WH-South (Chennai)", "WH-East (Kolkata)", "WH-West (Mumbai)", "WH-Central (Nagpur)"], n_prods),
        "current_stock": rng.integers(800, 6000, n_prods),
        "reorder_level": products["reorder_level"].values,
        "avg_monthly_demand": (purchase_orders.groupby("product_id")["quantity"].sum().reindex(products["product_id"]).fillna(0) / 36).values.round(1),
    })
    inventory["months_of_cover"] = np.where(
        inventory["avg_monthly_demand"] > 0,
        (inventory["current_stock"] / inventory["avg_monthly_demand"]).round(1),
        999.0
    )

    load_and_integrate_kaggle_data()

    # 5. EXPORT & MANIFEST
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suppliers_public = suppliers.drop(columns=[c for c in suppliers.columns if c.startswith("_")])

    files_written = []
    row_counts = {}

    def write_csv(df, name):
        p_primary = os.path.join(OUT, f"{name}.csv")
        df.to_csv(p_primary, index=False)
        files_written.append(p_primary)
        row_counts[name] = len(df)
        if args.version_stamp:
            p_stamped = os.path.join(OUT, f"{name}_{timestamp_str}.csv")
            df.to_csv(p_stamped, index=False)
            files_written.append(p_stamped)

    if gen_all or "suppliers" in tables_to_gen:
        suppliers.to_csv(os.path.join(OUT, "_ground_truth_supplier_traits.csv"), index=False)
        write_csv(suppliers_public, "suppliers")

    if gen_all or "products" in tables_to_gen:
        write_csv(products, "products")

    if gen_all or "purchase_orders" in tables_to_gen:
        write_csv(purchase_orders, "purchase_orders")

    if gen_all or "deliveries" in tables_to_gen:
        write_csv(deliveries, "deliveries")

    if gen_all or "inventory" in tables_to_gen:
        write_csv(inventory, "inventory")

    manifest = {
        "dataset_version": "v4.0",
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "n_orders": args.orders,
            "n_suppliers": args.suppliers,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "tables_requested": args.tables,
            "random_seed": args.seed
        },
        "macro_signal_engine": "Ornstein-Uhlenbeck Stochastic Process + Seasonal Sine Wave",
        "multi_sourcing_policy": "85% Primary Supplier / 15% Secondary Backup Supplier Allocation",
        "row_counts": row_counts,
        "files_written": files_written
    }

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Procurement dataset generated successfully in {OUT}")
    print(f"  • Suppliers: {len(suppliers_public)} entities")
    print(f"  • Products: {len(products)} technical SKUs (Multi-Sourced)")
    print(f"  • Orders: {len(purchase_orders):,} POs (Date range: {args.start_date} to {args.end_date})")
    print(f"  • Manifest written to {os.path.join(OUT, 'manifest.json')}")

if __name__ == "__main__":
    main()
