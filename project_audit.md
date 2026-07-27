# ProcureSense AI — Full Project Audit

> **Scope**: Every component, script, and dashboard section analyzed for limitations, bugs, must-haves, and nice-to-haves.
> **Date**: July 2026

---

## 2. `db/build_db.py` — SQLite Database Builder

### 🔴 Current Limitations (Addressed & Resolved)
- **[Resolved] Foreign Key Constraints**: Enforced `PRAGMA foreign_keys = ON;` on connection.
- **[Resolved] Indexing Strategy**: Added indexes on `order_date`, `is_late`, `region`, `tier`, `category` alongside primary keys to avoid table scans.
- **[Resolved] Modular Build Design**: Wrapped table loading and index creation inside `build_db()` to prevent side-effect executions when imported.

### 🟡 Known Issues / Bugs (Resolved)
- [x] **Import Side-Effects**: `conn` creation and script execution moved inside `build_db()` function and `if __name__ == "__main__":` block. Importing `build_db.py` now runs silently without rebuilding tables.
- [x] **Supplier Grouping Collision**: Fixed `PRICE_TREND_SQL` to `GROUP BY s.supplier_id, s.supplier_name, year` preventing duplicate name merging.

### ✅ Must-Haves
- [x] Add `PRAGMA foreign_keys = ON` after connecting
- [x] Add indexes on `order_date`, `category`, `region`, `tier`, `is_late` for dashboard query performance
- [x] Move module-level code into an explicit `build_db()` function so imports don't trigger side effects
- [x] Fix `PRICE_TREND_SQL` to `GROUP BY supplier_id, supplier_name`

### 💡 Nice-to-Haves
- [x] Add row-count validation after each table load and print a summary
- [x] Support incremental loads (append new CSV rows via `incremental=True`, don't replace)
- [x] Add a `CREATE TABLE IF NOT EXISTS` schema definition file (`db/schema.sql`) for documentation & DDL

---

## 3. `analysis/kpi_engine.py` — KPI & Analytics Engine

### 🔴 Current Limitations (Addressed & Resolved)
- **[Resolved] Modular Function Design**: Wrapped execution in `compute_kpi_summary()` and `if __name__ == "__main__":` block.
- **[Resolved] Inflation Data Export**: Saved full list of all 100 suppliers in `price_ols_inflation_trend` (instead of truncating to `.head(10)`).
- **[Resolved] HTML Safety**: Added `html.escape()` for supplier names in executive narrative generation.
- **[Resolved] Configurable Weights**: Health Score weights are now parameterized (`weights=DEFAULT_HEALTH_WEIGHTS` dict override).
- **[Resolved] Safe Median Imputation**: Replaced zero-masking `fillna(0.0)` for low-observation suppliers with active cohort median imputation.

### 🟡 Known Issues / Bugs (Resolved)
- [x] **Import Execution**: `compute_kpi_summary()` moves calculation execution inside function scope so imports don't trigger automatic calculation runs.
- [x] **HTML Character Breakages**: Applied `html.escape()` on text variables before rendering HTML tags.
- [x] **Zero-Masking Artifacts**: Imputed missing delay slopes and price volatility CVs with cohort medians.

### ✅ Must-Haves
- [x] Wrap all module-level code in `if __name__ == "__main__"` or a `compute_kpi_summary()` function
- [x] Add HTML escaping (`html.escape()`) for all supplier names in narrative
- [x] Store all 100 suppliers in `price_ols_inflation_trend`, not just top 10
- [ ] Pass the active filter as a parameter so `delivery_performance` is filter-aware

### 💡 Nice-to-Haves
- [x] Expose health score weights as a configurable dictionary / JSON structure (`DEFAULT_HEALTH_WEIGHTS`)
- [ ] Add confidence intervals to the OLS slope estimates
- [ ] Version-stamp the JSON output (`"generated_at": "2026-07-27T..."`)
- [ ] Write unit tests for `composite_risk_index` with known fixture data

---

## 4. `ml/delay_prediction.py` — ML Model Training Pipeline

### 🔴 Current Limitations
- **`sup_concurrent_po_30d` uses O(n²) loop** — for 30k orders sorted by date, this runs in ~seconds but will hang on 500k+ orders.
- **`rf_model.joblib` is 63 MB** — this gets loaded into RAM every time the dashboard starts. On machines with <4 GB RAM, this is a bottleneck.
- **Label encoders are refitted on full dataset** (lines 520–524) but during training they were fitted per fold — the `te_maps` saved to joblib use full-dataset statistics, not OOF statistics. This is a minor data leakage into the artifact.
- **Walk-forward validation trains 4 separate RF models** with `n_estimators=100` — this is correct but doubles training time and is not parallelized.
- **No model versioning** — every run overwrites `rf_model.joblib` and `model_metrics.json`. Previous runs are lost.
- **SHAP is computed on only first 1,000 test rows** — may not represent the full test distribution.
- **`HAS_XGB` fallback silently uses RF probs for XGB slot** — the ensemble then effectively doubles Random Forest weight.

### 🟡 Known Issues / Bugs
- `cost_optimal_expected_cost_inr` in the JSON saves `res_m3["default_thresh_0.5"]["expected_cost_inr"]` but the field is named `"cost_optimal_model_name": "Logistic Regression"` — misleading because the cost used is the **default threshold** cost, not the **cost-optimal threshold** cost.
- `mathematical_reconciliation` values (`fn_savings_inr: 7050000`, `fp_penalty_inr: 1560000`) are **hard-coded**, not computed from the actual run's confusion matrix. They will be wrong if data changes.
- `"optimal_threshold"` saved in the joblib artifact is from the RF `cost_optimal` dict — but the dashboard banner text says threshold = 0.35, which may differ per run.

### ✅ Must-Haves
- [ ] Replace O(n²) concurrent PO loop with a vectorized rolling-window approach using `pd.merge_asof` or sorted numpy binary search
- [ ] Fix `mathematical_reconciliation` to compute values from actual metrics instead of hardcoding
- [ ] Add model versioning — save artifacts with a timestamp suffix and keep a `latest` symlink/pointer
- [ ] Fix XGBoost fallback to clearly signal it's using RF probs, not XGB probs

### 💡 Nice-to-Haves
- [ ] Add `LIME` explanations alongside SHAP for local interpretability
- [ ] Run SHAP on the full test set with `approximate=True` for speed
- [ ] Add `MLflow` or a simple JSON run log to track experiment history
- [ ] Add a `predict.py` CLI tool so the model can be used without the full dashboard
- [ ] Compress `rf_model.joblib` using `joblib.dump(..., compress=3)`

---

## 5. `dashboard/app.py` — Streamlit Dashboard (1,991 lines)

> Broken down by section / tab

---

### 5a. Global Setup & CSS

#### 🔴 Limitations
- **~260 lines of inline CSS in `st.markdown`** — unmaintainable, cannot be linted, and will break on Streamlit version upgrades that change internal DOM structure (e.g., `[data-testid="stAppViewContainer"]`).
- **Dark mode only** — `IS_DARK = True` is a stub; light mode is not implemented.
- **`get_db_connection()` uses `@st.cache_resource` with `check_same_thread=False`** — SQLite is not thread-safe in write mode. Works now because the app is read-only, but risky.
- **No connection pooling or retry logic** — if `procurement.db` is locked, the app crashes silently.

#### ✅ Must-Haves
- [ ] Move CSS to a separate `.css` file loaded via `st.markdown(open("style.css").read(), unsafe_allow_html=True)`
- [ ] Add a try/except around `get_db_connection()` with a user-friendly error message

#### 💡 Nice-to-Haves
- [ ] Implement light/dark toggle that actually works
- [ ] Add `@st.cache_data(ttl=300)` to the main filtered query so repeated filter interactions don't re-hit the DB

---

### 5b. Sidebar — Filters & Control Center

#### 🔴 Limitations
- **`get_filter_options()` is NOT cached** — runs 4 SQL queries on every page interaction/rerun.
- **SQL injection risk in filter clauses** — `_sql_in()` does basic escaping (`'` → `''`) but is home-grown; should use parameterized queries instead.
- **`preset_peak`** only sets `sel_ship` to Air/Express modes but doesn't change year slider or other filters — partial presets can confuse users.
- **Date range is a year slider (2023–2025) not a calendar date picker** — cannot filter by specific months or quarters.

#### 🟡 Known Issues
- `st.session_state` for filter defaults is only set on preset button clicks. If a user refreshes the page, all `session_state` is lost and defaults reset to full selection.
- `preset_peak` button sets `sel_ship` but the multiselect widget reads from `st.session_state.get("sel_ship", filters["shipping_modes"])` — the widget and session state can get out of sync on the same rerun.

#### ✅ Must-Haves
- [ ] Cache `get_filter_options()` with `@st.cache_data`
- [ ] Replace home-grown SQL injection protection with SQLAlchemy or proper parameterized queries
- [ ] Add a date range picker (full `date_input` with start/end) instead of year-only slider

#### 💡 Nice-to-Haves
- [ ] Save filter state to URL query params so users can share filtered views
- [ ] Add a "Supplier Search" text filter in the sidebar
- [ ] Add more presets: "High Risk Only", "Import Suppliers", "Critical Priority"

---

### 5c. Tab 1 — Executive Overview

#### 🔴 Limitations
- **`_hdr_orders` runs two separate `COUNT(*)` queries** (lines 367 and 387) for the same count — one for the header, one for the sidebar. Should use one cached value.
- **KPI cards are hardcoded HTML** — if Streamlit updates its rendering, they may lose styling.
- **`kpi_data.get('narrative_example')` renders raw HTML** — contains `&amp;`, `<b>`, `<br>` tags that depend on `kpi_engine.py` generating correct HTML. If the narrative is regenerated with a supplier name containing `<` or `>`, it will break.
- **Chart 2 (Risk Tier)** uses a simplified 3-factor heuristic locally in the dashboard tab that differs from the 4-component formula in `kpi_engine.py` — the two can disagree.
- **Chart 3 (Inventory)** does not account for dynamic ROP — uses simple `current_stock < reorder_level`, not the inflated lead time ROP from `kpi_engine.py`.

#### 🟡 Known Issues
- `monthly` groupby re-aggregates `df_filtered` on every render — for 30k rows this is fast, but if `@st.cache_data` is not applied, any sidebar interaction triggers a full recompute.
- `fig_trend` secondary Y-axis for Late Rate % has no explicit range — if late rate is 0% or very low, the axis min/max can be misleading.

#### ✅ Must-Haves
- [ ] Deduplicate the two COUNT queries into one cached variable
- [ ] Unify risk tier calculation — Tab 1 should use the same formula as `kpi_engine.py`, not a local approximation
- [ ] Add `html.escape()` or sanitize the narrative before rendering

#### 💡 Nice-to-Haves
- [ ] Add sparkline trend arrows (▲ / ▼) to KPI cards showing WoW or MoM change
- [ ] Add a "Drill-through" button on the health score card linking to Tab 2
- [ ] Make charts export-ready with proper axis labels and legends

---

### 5d. Tab 2 — Supplier & Region SLAs

#### 🔴 Limitations
- **Chart 5 (Supplier Ranking) uses `kpi_data["top_suppliers"]` and `kpi_data["bottom_suppliers"]`** — these are pre-computed from `kpi_engine.py` and do NOT respond to sidebar filters. A user filtering to "Import - China" region will still see all suppliers.
- **Only shows top 5 + bottom 5 suppliers** — 90 suppliers are completely invisible.
- **Chart 6 (Spend by Region/Tier) is the only dynamic chart in this tab** — the ranking bar chart is static.
- **No supplier detail drill-down** — clicking a supplier bar does nothing.

#### 🟡 Known Issues
- `comb_sup = pd.concat([top_sup, bot_sup]).drop_duplicates(subset="supplier_id")` — if a supplier is simultaneously top 5 and bottom 5 (impossible by definition) the logic is fine, but if `kpi_data` is stale, `supplier_id` may be missing and `.drop_duplicates()` fails silently.

#### ✅ Must-Haves
- [ ] Make Chart 5 respond to sidebar filters (re-compute from `df_filtered` not cached JSON)
- [ ] Paginate or searchably show all 100 suppliers, not just 10
- [ ] Add a supplier detail expandable panel or modal

#### 💡 Nice-to-Haves
- [ ] Add a supplier trajectory chart (delay trend over time for selected supplier)
- [ ] Add a heatmap of region × tier on-time performance
- [ ] Add a scatter plot: Spend vs. On-Time Rate (high-spend low-reliability is the danger quadrant)

---

### 5e. Tab 3 — Inventory Control

#### 🔴 Limitations
- **`inv_full_df` SQL query runs on every render** with a complex LEFT JOIN — this is the heaviest query in the dashboard (~5 table join). Not cached.
- **`COALESCE(del_stats.late_rate, 0.20)`** — 20% default for suppliers with no POs is arbitrary and misleads the High-Risk supplier flag.
- **Inventory monitor table is capped at `head(25)`** — 175 SKUs are hidden with no pagination.
- **Dynamic ROP in Tab 3** (`_inv_st()` on line 879) uses a simplified logic (no actual dynamic ROP formula from `kpi_engine.py`). The two tabs will report different "Understocked" counts.

#### 🟡 Known Issues
- `inv_active` can be empty if `df_filtered` has no matching product names. The `inv_active = inv_full_df.copy()` fallback is triggered, but this can happen silently without any user warning.
- `defect_spend_val` uses `df_filtered` (filtered by sidebar) while all other inventory metrics use `inv_active` — mixed data scopes on the same KPI row.

#### ✅ Must-Haves
- [ ] Cache `inv_full_df` with `@st.cache_data`
- [ ] Add pagination to the stockout monitor table (beyond head 25)
- [ ] Unify dynamic ROP formula between `kpi_engine.py` and Tab 3
- [ ] Show explicit warning when `inv_active` falls back to full inventory

#### 💡 Nice-to-Haves
- [ ] Add a "Reorder Now" action button that exports a pre-filled PO template CSV
- [ ] Add an ABC classification (A/B/C by spend value) to the inventory table
- [ ] Add a "Days Until Stockout" calculated column in the table

---

### 5f. Tab 4 — ML Risk Simulator

#### 🔴 Limitations
- **`joblib.load(model_artifact_path)` runs on every page render** (no `@st.cache_resource`) — loading a 63 MB model file on every user interaction is extremely slow.
- **ML Simulator has hardcoded feature assumptions**: `lead_time_days_base = 14`, `sup_concurrent_po_30d = 3`, `sup_rolling_defect = 0.03`, `sup_ewm_delay = 3.5` etc. — these are fixed constants regardless of the selected supplier. The model effectively only varies on `supplier_id_te`, `shipping_mode_code`, and month.
- **`order_qty_vs_sup_mean = sim_qty / 250.0`** — hardcoded denominator of 250 instead of the actual supplier mean quantity from DB.
- **`logistics_stress_index = 14.0 / (3.5 + 1.0)`** — fully hardcoded, not calculated from the selected order's actual values.
- **Estimated Delay Duration formula (`prob * 10.0 * delay_multiplier`)** is made up, not derived from any model.
- **"SHAP Local Explanation" box is not actual SHAP** — it shows hardcoded human-readable strings, not computed SHAP values for the specific prediction.

#### 🟡 Known Issues
- `sup_row = all_suppliers_df[...].iloc[0]` — if `sim_supplier` name has a special character or the query returns empty, this throws `IndexError` without a user-friendly error.
- `exp_risk_cost = total_order_cost * prob * 0.15` — the 15% factor is unexplained and not tied to the `COST_FN`/`COST_FP` cost matrix from training.
- The ML metric cards (ROC-AUC 0.714, Acc 65.7%) are hardcoded HTML, not pulled from `model_data` — they become stale if the model is retrained.

#### ✅ Must-Haves (Critical)
- [ ] Load model with `@st.cache_resource` — this is the single most impactful performance fix
- [ ] Pull actual supplier rolling stats from DB for the selected supplier instead of hardcoded defaults
- [ ] Pull ML metric cards from `model_data` dict, not hardcoded HTML strings
- [ ] Add a try/except with friendly error around `iloc[0]` for supplier lookup

#### 💡 Nice-to-Haves
- [ ] Compute actual local SHAP values for the submitted order using `explainer.shap_values(X_sim)`
- [ ] Add a waterfall chart showing each feature's contribution to the prediction
- [ ] Add confidence interval display on the delay probability
- [ ] Let users compare two suppliers side-by-side in the simulator

---

### 5g. Reallocation Simulator (within Tab 4)

#### 🔴 Limitations
- **`sup_details_df` SQL query** (lines 1408–1418) runs on every page render — not cached.
- **Capacity strain threshold of 30%** and **`capacity_penalty = (expansion - 30) * 0.003`** are completely arbitrary magic numbers with no sourcing.
- **Price premium delta** uses `avg_po_value` (average of source and target) not the actual product price — switching supplier categories changes product mix but this is ignored.
- **Net benefit assumes all late deliveries cost exactly ₹50,000** (the `COST_FN` from training) — makes no distinction by order size or product criticality.

#### ✅ Must-Haves
- [ ] Cache `sup_details_df` with `@st.cache_data`
- [ ] Make capacity strain threshold configurable (e.g., a slider `30–60%`)
- [ ] Clarify that stockout cost is the model training constant, not an actual contractual penalty

#### 💡 Nice-to-Haves
- [ ] Add a sensitivity analysis chart: show net benefit across a range of volume shifts (10–2000 POs)
- [ ] Weight stockout cost by product category criticality (Electronics > Packaging)
- [ ] Save reallocation scenarios to a comparison table

---

### 5h. Tab 5 — SQL Analytics Studio

#### 🔴 Limitations
- **SQL execution uses `pd.read_sql(user_sql, conn)` with no sanitization** — arbitrary SQL can be run, including destructive statements (`DROP TABLE`, `DELETE`). This is a significant security issue if the app is ever shared publicly.
- **Editable text area loses state on filter changes** — Streamlit reruns reset `user_sql` to the default, discarding user edits unless `st.session_state["active_sql_script"]` is set.
- **Auto-visualizer always picks first categorical + first numeric column** — often produces meaningless charts (e.g., `po_id` on X-axis).
- **Schema inspector hardcodes column types** (lines 1850–1855) — if `build_db.py` schema changes, the inspector becomes inaccurate.
- **All 10 queries have `LIMIT 20` or `LIMIT 25`** — users cannot see full results without running the query in another tool.

#### 🟡 Known Issues
- `if run_btn or "sql_run_df" in st.session_state:` — the `"sql_run_df"` key is never set in `st.session_state`, so this condition always reduces to `if run_btn:`. The intent to persist results across reruns is broken.
- `t_start = time.time()` import of `time` is inside the `if run_btn` block — not a bug but sloppy.

#### ✅ Must-Haves
- [ ] Block or confirm destructive SQL keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `TRUNCATE`) before execution
- [ ] Use `st.session_state` properly to persist query results across reruns
- [ ] Make LIMIT configurable (e.g., a `st.number_input` for row count)
- [ ] Pull schema info dynamically from `PRAGMA table_info(tablename)` instead of hardcoding

#### 💡 Nice-to-Haves
- [ ] Add syntax highlighting to the SQL editor (use `streamlit-ace` or similar)
- [ ] Add query execution history (last 5 queries run)
- [ ] Let users save their custom queries with a name
- [ ] Add a "Explain Query" button that shows the SQLite query plan

---

## 6. `requirements.txt`

### 🔴 Limitations
- `joblib` is used in the dashboard and ML pipeline but is **not listed** in `requirements.txt`. Works because it ships with scikit-learn but not guaranteed.
- `matplotlib` is used in `delay_prediction.py` for the SHAP plot but is **not listed**.
- `holidays` is used in `generate_data.py` (try/except) but is **not listed** — always falls back to Sunday-only detection.
- Version pins are minimum-only (`>=`) — no upper bound means breaking changes from future package versions can silently break the project.

### ✅ Must-Haves
- [ ] Add `joblib`, `matplotlib`, `holidays` to requirements
- [ ] Pin versions more tightly: `scikit-learn>=1.2.0,<2.0.0`

### 💡 Nice-to-Haves
- [ ] Add a `requirements-dev.txt` for testing/linting tools (`pytest`, `black`, `ruff`)
- [ ] Generate a `requirements-lock.txt` with exact pinned versions for reproducibility

---

## 7. `test_pipeline.py` — Integration Tests

### 🔴 Limitations
- Not reviewed in detail, but from the file size (7,971 bytes) it appears to be a single integration test file. 
- No unit tests for individual functions (KPI formulas, feature engineering, risk scoring).
- Tests likely depend on the full `procurement.db` being pre-built — can't run in CI without data setup.

### ✅ Must-Haves
- [ ] Add unit tests for `composite_risk_index`, `oof_target_encode`, `evaluate_model_full`
- [ ] Add a `conftest.py` with fixture data so tests don't require the full 14 MB database

### 💡 Nice-to-Haves
- [ ] Add a GitHub Actions CI pipeline that runs tests on push
- [ ] Add a smoke test that launches the Streamlit app and checks the header renders

---

## Summary Table

| Component | Severity | Top Priority Fix |
|---|---|---|
| `generate_data.py` | 🟡 Medium | Fix email collision; cap `inf` months of cover |
| `build_db.py` | 🔴 High | Add FK constraints; fix GROUP BY supplier_name bug |
| `kpi_engine.py` | 🔴 High | Wrap in `main()`; fix hardcoded reconciliation values |
| `delay_prediction.py` | 🔴 High | Fix hardcoded `mathematical_reconciliation`; model versioning |
| `app.py` — Global | 🟡 Medium | Cache model load with `@st.cache_resource` |
| `app.py` — Sidebar | 🟡 Medium | Cache filter options; proper parameterized SQL |
| `app.py` — Tab 1 | 🟡 Medium | Unify risk tier formula |
| `app.py` — Tab 2 | 🔴 High | Make supplier ranking filter-aware |
| `app.py` — Tab 3 | 🟡 Medium | Cache inventory query; unify dynamic ROP |
| `app.py` — Tab 4 ML | 🔴 Critical | `@st.cache_resource` for joblib; fix hardcoded features |
| `app.py` — Tab 5 SQL | 🔴 High | Block destructive SQL; fix session state persistence |
| `requirements.txt` | 🟡 Medium | Add missing packages |
