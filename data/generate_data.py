"""
generate_data.py (v3 — Expanded & Enriched Dataset)
---------------------------------------------------
Generates synthetic but highly realistic procurement data across 5 linked tables:
suppliers, products, purchase_orders, deliveries, inventory.

v3 Enhancements:
- 100 Authentic Corporate Supplier Names (e.g. Apex Precision Components, Bharat Heavy Metallics, Pacific Rim Semiconductor)
- Realistic international/domestic geographic assignments (India, Germany, USA, Japan, China, Taiwan)
- Contact Emails, Ratings, Payment Terms (Net 30/60/90), ISO Certifications
- 200 Specific Industrial Technical Product Names with SKUs and Units of Measure
- Procurement Buyers, Incoterms (FOB, CIF, DDP), PO Numbers, Tracking Numbers, Carriers (DHL, FedEx, Maersk, Blue Dart)
- Inspection QA Statuses (Passed Inspection, Minor Defect - Accepted, Major Defect - Rejected)
- 30,000 Purchase Orders spanning 2023–2025
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
        return int(d in _india_holidays or d.weekday() == 6)  # Sunday or holiday
except ImportError:
    def is_holiday(d):
        return int(d.weekday() == 6)

# ---------------------------------------------------------------
# MACRO SIGNALS
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
# 1. AUTHENTIC SUPPLIERS (100 Companies)
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

regions = list(region_country_map.keys())
tiers = ["Tier 1", "Tier 2", "Tier 3"]
payment_terms_options = ["Net 30", "Net 60", "Net 90", "2/10 Net 30", "Immediate / CIA"]
certifications_options = ["ISO 9001", "ISO 14001", "IATF 16949", "AS9100", "ISO 45001"]

N_SUPPLIERS = len(REAL_SUPPLIER_NAMES)

assigned_regions = rng.choice(regions, N_SUPPLIERS)
assigned_countries = [region_country_map[r] for r in assigned_regions]

def generate_email(name):
    clean = "".join(c for c in name.split()[0] if c.isalnum()).lower()
    return f"procurement@{clean}corp.com"

suppliers = pd.DataFrame({
    "supplier_id": [f"SUP{1000+i}" for i in range(N_SUPPLIERS)],
    "supplier_name": REAL_SUPPLIER_NAMES,
    "contact_email": [generate_email(n) for n in REAL_SUPPLIER_NAMES],
    "country": assigned_countries,
    "region": assigned_regions,
    "tier": rng.choice(tiers, N_SUPPLIERS, p=[0.25, 0.50, 0.25]),
    "onboarded_year": rng.choice([2018, 2019, 2020, 2021, 2022, 2023], N_SUPPLIERS),
    "rating": np.round(rng.uniform(3.2, 4.9, N_SUPPLIERS), 1),
    "payment_terms": rng.choice(payment_terms_options, N_SUPPLIERS, p=[0.5, 0.3, 0.1, 0.05, 0.05]),
    "certifications": rng.choice(certifications_options, N_SUPPLIERS),
})

# Latent traits
suppliers["_true_reliability"] = rng.beta(7, 2, N_SUPPLIERS)
suppliers["_true_defect_rate"] = rng.beta(1.5, 25, N_SUPPLIERS)
suppliers["_price_drift_pct"] = rng.normal(0.05, 0.06, N_SUPPLIERS)
suppliers["_going_bad"] = rng.choice([0, 1], N_SUPPLIERS, p=[0.85, 0.15])

# ---------------------------------------------------------------
# 2. DETAILED PRODUCTS & TECHNICAL CATALOG
# ---------------------------------------------------------------
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

N_PRODUCTS = 200
product_list = []
catalog_keys = list(PRODUCT_CATALOG.keys())

for i in range(N_PRODUCTS):
    cat, sub_cat = catalog_keys[i % len(catalog_keys)]
    names_list, uom, price_range = PRODUCT_CATALOG[(cat, sub_cat)]
    base_name = rng.choice(names_list)
    product_name = f"{base_name} (v{rng.integers(1, 4)})" if i >= len(catalog_keys) else base_name
    sku = f"SKU-{cat[:3].upper()}-{2000+i}"

    product_list.append({
        "product_id": f"PRD{2000+i}",
        "sku": sku,
        "product_name": product_name,
        "category": cat,
        "sub_category": sub_cat,
        "unit_of_measure": uom,
        "unit_cost_base": np.round(rng.uniform(price_range[0], price_range[1]), 2),
        "primary_supplier_id": rng.choice(suppliers["supplier_id"]),
        "reorder_level": int(rng.integers(20, 1000)),
        "lead_time_days_base": int(rng.integers(2, 30)),
    })

products = pd.DataFrame(product_list)

# ---------------------------------------------------------------
# 3. PURCHASE ORDERS & DELIVERIES (30,000 Orders)
# ---------------------------------------------------------------
N_ORDERS = 30000
start_date = date(2023, 1, 1)
end_date = date(2025, 12, 31)
date_range_days = (end_date - start_date).days

buyer_names = ["Ananya Sharma", "Rahul Verma", "Sarah Jenkins", "Michael Chang", "Priya Nair", "David Miller", "Vikram Patel", "Elena Rostova"]
incoterms_options = ["FOB", "CIF", "DDP", "EXW", "FCA", "DAP"]
shipping_modes = ["Expedited Air", "Air Freight", "Express Ground", "Standard Ground", "Sea Freight"]
carrier_options = ["DHL Express", "FedEx Supply Chain", "Maersk Line", "Blue Dart Cargo", "DB Schenker Logistics", "Kuehne+Nagel"]

order_dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, date_range_days, N_ORDERS)]
order_products = rng.choice(products["product_id"], N_ORDERS)

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
    ship_mode = rng.choice(shipping_modes, p=[0.10, 0.20, 0.25, 0.30, 0.15])
    buyer = rng.choice(buyer_names)
    incoterm = rng.choice(incoterms_options)

    # Signals
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
    ship_bump = 0.18 if ship_mode == "Sea Freight" else (0.08 if ship_mode == "Standard Ground" else (-0.10 if "Air" in ship_mode else 0.0))
    region_bump = 0.10 if "Import" in sup["region"] else 0.0
    oil_bump = (oil_idx - 1.0) * 0.3
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
    po_num = f"PO-{odate.year}-{10000+i}"
    trk_num = f"TRK-{rng.integers(10000000, 99999999)}"
    carrier = rng.choice(carrier_options)

    if defect:
        inspection_status = rng.choice(["Major Defect - Rejected", "Minor Defect - Accepted"], p=[0.6, 0.4])
    else:
        inspection_status = "Passed Inspection"

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
        "status": "Delivered",
    })

purchase_orders = pd.DataFrame(po_rows)
deliveries = pd.DataFrame(del_rows)

# ---------------------------------------------------------------
# 4. INVENTORY
# ---------------------------------------------------------------
inventory = pd.DataFrame({
    "product_id": products["product_id"],
    "warehouse": rng.choice(["WH-North (Delhi)", "WH-South (Chennai)", "WH-East (Kolkata)", "WH-West (Mumbai)", "WH-Central (Nagpur)"], N_PRODUCTS),
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

print(f"Expanded dataset generated in {OUT}")
print(f"Suppliers: {len(suppliers_public)} corporate entities")
print(f"Products: {len(products)} technical SKUs")
print(f"Orders: {len(purchase_orders):,} (Late Rate: {deliveries['is_late'].mean():.1%})")
