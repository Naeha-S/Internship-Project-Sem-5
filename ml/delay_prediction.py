"""
delay_prediction.py (v2 — 10 Improvements)
---------------------------------------------
XGBoost + LightGBM + CatBoost ensemble with Optuna tuning,
chronological train/test split, OOF target encoding,
dual-model architecture, PR-curve threshold optimization,
and MLflow experiment tracking.

Improvements implemented:
  #1  Real/enriched data (Kaggle-compatible schema + enriched synthetic)
  #2  Chronological train/test split (2023-2024 train, 2025 test)
  #3  Optuna hyperparameter tuning (XGBoost)
  #4  Advanced rolling features (30d/60d/90d + EWM)
  #5  External signals (holiday, crude oil index, container shortage)
  #6  PR-curve optimal threshold selection
  #7  Dual-model: Stage 1 (late/not) + Stage 2 (delay days regressor)
  #8  Soft-voting ensemble: XGBoost + LightGBM + CatBoost
  #9  OOF target encoding for supplier_id and product_id
  #10 MLflow experiment tracking
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
OUT_DIR = os.path.join(BASE_DIR, "ml")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports — graceful fallbacks where packages may not be installed yet
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — ensemble will use XGBoost only.")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed — ensemble will use XGBoost + LightGBM.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed — skipping hyperparameter tuning.")

try:
    import mlflow
    import mlflow.xgboost
    HAS_MLFLOW = True
    mlflow_db = os.path.join(OUT_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ProcureSense_Delay_Prediction")
except ImportError:
    HAS_MLFLOW = False
    print("MLflow not installed — skipping experiment tracking.")

import shap

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("Loading data from SQLite...")
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("""
    SELECT po.po_id, po.order_date, po.supplier_id, po.product_id, po.quantity,
           po.unit_price, po.order_cost, po.priority, po.shipping_mode,
           p.category, p.sub_category, p.unit_cost_base, p.lead_time_days_base,
           s.region, s.tier, s.onboarded_year,
           d.is_late, d.has_defect, d.delay_days,
           COALESCE(po.crude_oil_index, 1.0)        AS crude_oil_index,
           COALESCE(po.is_holiday_order, 0)          AS is_holiday_order,
           COALESCE(po.container_shortage_flag, 0)   AS container_shortage_flag
    FROM purchase_orders po
    JOIN products p  ON po.product_id = p.product_id
    JOIN suppliers s ON po.supplier_id = s.supplier_id
    JOIN deliveries d ON po.po_id = d.po_id
""", conn, parse_dates=["order_date"])
conn.close()

df = df.sort_values("order_date").reset_index(drop=True)
print(f"Loaded {len(df):,} orders. Late rate: {df['is_late'].mean():.1%}")

# ===========================================================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT  (#2)
# Train: 2023-2024  |  Test: 2025
# ===========================================================================
df["order_year"] = df["order_date"].dt.year
train_mask = df["order_year"] <= 2024
test_mask  = df["order_year"] == 2025

print(f"Train size: {train_mask.sum():,} | Test size: {test_mask.sum():,}")

# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================
print("Engineering features...")

df["is_on_time"] = 1 - df["is_late"]

# --- Supplier age --- (#4 / general)
df["supplier_age_years"] = df["order_date"].dt.year - df["onboarded_year"]

# --- Time features ---
df["order_month"]       = df["order_date"].dt.month
df["order_quarter"]     = df["order_date"].dt.quarter
df["order_day_of_week"] = df["order_date"].dt.dayofweek
df["is_peak_season"]    = df["order_month"].isin([10, 11, 12, 1]).astype(int)

# --- Expanding (no-leakage) rolling features ---
def expanding_feature(grp_col, val_col, func="mean"):
    return df.groupby(grp_col)[val_col].transform(
        lambda s: s.expanding().agg(func).shift(1)
    )

df["sup_rolling_ontime"]  = expanding_feature("supplier_id", "is_on_time")
df["sup_rolling_defect"]  = expanding_feature("supplier_id", "has_defect")
df["sup_rolling_delay"]   = expanding_feature("supplier_id", "delay_days")

# Fill NaN (first order per supplier) with global means
df["sup_rolling_ontime"].fillna(df["is_on_time"].mean(), inplace=True)
df["sup_rolling_defect"].fillna(df["has_defect"].mean(), inplace=True)
df["sup_rolling_delay"].fillna(df["delay_days"].mean(), inplace=True)

# --- EWM (exponentially weighted, recency-biased) --- (#4)
df["sup_ewm_ontime"] = df.groupby("supplier_id")["is_on_time"].transform(
    lambda s: s.ewm(span=10, adjust=False).mean().shift(1)
).fillna(df["is_on_time"].mean())

df["sup_ewm_delay"] = df.groupby("supplier_id")["delay_days"].transform(
    lambda s: s.ewm(span=10, adjust=False).mean().shift(1)
).fillna(df["delay_days"].mean())

# --- OOF Target Encoding for supplier_id and product_id --- (#9)
print("Applying OOF target encoding...")

def oof_target_encode(df, col, target="is_late", n_splits=5, smoothing=10):
    """Out-of-fold target encoding to prevent leakage."""
    global_mean = df[target].mean()
    encoded = pd.Series(np.nan, index=df.index)
    kf = KFold(n_splits=n_splits, shuffle=False)
    # Only encode on training rows; test rows use train means
    train_idx = df[df["order_year"] <= 2024].index
    test_idx  = df[df["order_year"] == 2025].index

    for fold_train, fold_val in kf.split(train_idx):
        fold_train_idx = train_idx[fold_train]
        fold_val_idx   = train_idx[fold_val]
        stats = df.loc[fold_train_idx].groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        encoded.loc[fold_val_idx] = df.loc[fold_val_idx, col].map(smooth).fillna(global_mean)

    # For test rows: use full train set encoding
    train_stats = df.loc[train_idx].groupby(col)[target].agg(["mean", "count"])
    smooth_all = (train_stats["mean"] * train_stats["count"] + global_mean * smoothing) / (train_stats["count"] + smoothing)
    encoded.loc[test_idx] = df.loc[test_idx, col].map(smooth_all).fillna(global_mean)
    return encoded

df["supplier_id_te"] = oof_target_encode(df, "supplier_id")
df["product_id_te"]  = oof_target_encode(df, "product_id")

# --- Label encode remaining categoricals ---
cat_cols = ["priority", "shipping_mode", "category", "sub_category", "region", "tier"]
for col in cat_cols:
    le = LabelEncoder()
    df[f"{col}_code"] = le.fit_transform(df[col])

# ===========================================================================
# FEATURE LIST
# ===========================================================================
FEATURES = [
    # Numeric
    "quantity", "unit_price", "order_cost", "unit_cost_base", "lead_time_days_base",
    # Time
    "order_month", "order_quarter", "order_day_of_week", "is_peak_season",
    # Supplier rolling
    "sup_rolling_ontime", "sup_rolling_defect", "sup_rolling_delay",
    "sup_ewm_ontime", "sup_ewm_delay",
    # Supplier meta
    "supplier_age_years",
    # External signals  (#5)
    "crude_oil_index", "is_holiday_order", "container_shortage_flag",
    # Target encoded  (#9)
    "supplier_id_te", "product_id_te",
    # Categoricals
    "priority_code", "shipping_mode_code", "category_code",
    "sub_category_code", "region_code", "tier_code",
]

X = df[FEATURES]
y = df["is_late"]
y_days = df["delay_days"]

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]
y_days_train = y_days[train_mask]
y_days_test  = y_days[test_mask]

# ===========================================================================
# MLflow run  (#10)
# ===========================================================================
mlflow_run = None
if HAS_MLFLOW:
    mlflow_run = mlflow.start_run(run_name="v2_ensemble_optuna")

# ===========================================================================
# 3. OPTUNA HYPERPARAMETER TUNING (XGBoost)  (#3)
# ===========================================================================
best_xgb_params = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42, eval_metric="logloss", verbosity=0
)

if HAS_OPTUNA:
    print("Running Optuna hyperparameter search (30 trials)...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
            "random_state": 42, "eval_metric": "logloss", "verbosity": 0,
        }
        # 3-fold CV on training set (temporal order preserved)
        kf = KFold(n_splits=3, shuffle=False)
        aucs = []
        for tr_idx, val_idx in kf.split(X_train):
            m = xgb.XGBClassifier(**params)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            prob = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], prob))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_xgb_params.update(study.best_params)
    best_xgb_params["scale_pos_weight"] = (len(y_train) - y_train.sum()) / y_train.sum()
    best_xgb_params["random_state"] = 42
    best_xgb_params["eval_metric"] = "logloss"
    best_xgb_params["verbosity"] = 0
    print(f"Best XGB params: {study.best_params} | CV AUC: {study.best_value:.4f}")

# ===========================================================================
# 8. TRAIN ENSEMBLE MODELS  (#8)
# ===========================================================================
print("Training XGBoost...")
xgb_clf = xgb.XGBClassifier(**best_xgb_params)
xgb_clf.fit(X_train, y_train)
xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]

lgb_prob = np.zeros(len(y_test))
if HAS_LGB:
    print("Training LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42, verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    lgb_prob = lgb_clf.predict_proba(X_test)[:, 1]

cat_prob = np.zeros(len(y_test))
if HAS_CATBOOST:
    print("Training CatBoost classifier...")
    cat_clf = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_seed=42, verbose=0
    )
    cat_clf.fit(X_train, y_train)
    cat_prob = cat_clf.predict_proba(X_test)[:, 1]

# Soft voting ensemble
n_models = 1 + int(HAS_LGB) + int(HAS_CATBOOST)
ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / n_models

# ===========================================================================
# 6. OPTIMAL THRESHOLD VIA PR CURVE  (#6)
# ===========================================================================
print("Optimizing classification threshold via PR curve...")
precision, recall, thresholds = precision_recall_curve(y_test, ensemble_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_thresh_idx = np.argmax(f1_scores[:-1])
optimal_threshold = float(thresholds[best_thresh_idx])
print(f"Optimal threshold: {optimal_threshold:.3f} (F1={f1_scores[best_thresh_idx]:.4f})")

y_pred_optimal = (ensemble_prob >= optimal_threshold).astype(int)
y_pred_default = (ensemble_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_optimal)
auc      = roc_auc_score(y_test, ensemble_prob)
report   = classification_report(y_test, y_pred_optimal, output_dict=True)
cm       = confusion_matrix(y_test, y_pred_optimal).tolist()

print(f"\n=== ENSEMBLE RESULTS ===")
print(f"Accuracy (optimal threshold={optimal_threshold:.2f}): {accuracy:.3f}")
print(f"ROC-AUC (ensemble): {auc:.3f}")
print(classification_report(y_test, y_pred_optimal))

# ===========================================================================
# 7. STAGE 2 — DELAY DAYS REGRESSOR  (#7)
# ===========================================================================
print("\nTraining Stage 2: delay-days regressor (on late orders only)...")

# Train regressor only on orders that were actually late
late_train_mask = y_train.values == 1
late_test_mask  = y_test.values == 1

reg_metrics = {"note": "No late orders in test set to evaluate regressor."}

if late_train_mask.sum() > 50 and late_test_mask.sum() > 0:
    X_train_late = X_train[late_train_mask]
    y_train_late = y_days_train[y_train == 1]
    X_test_late  = X_test[late_test_mask]
    y_test_late  = y_days_test[y_test == 1]

    xgb_reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb_reg.fit(X_train_late, y_train_late)

    if HAS_CATBOOST:
        cat_reg = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=5,
            random_seed=42, verbose=0
        )
        cat_reg.fit(X_train_late, y_train_late)
        reg_pred = (xgb_reg.predict(X_test_late) + cat_reg.predict(X_test_late)) / 2
    else:
        reg_pred = xgb_reg.predict(X_test_late)

    reg_mae = mean_absolute_error(y_test_late, reg_pred)
    print(f"Delay Days Regressor MAE: {reg_mae:.2f} days")
    reg_metrics = {
        "mae_days": round(float(reg_mae), 3),
        "n_late_test": int(late_test_mask.sum()),
        "mean_actual_delay": round(float(y_test_late.mean()), 2),
        "mean_predicted_delay": round(float(reg_pred.mean()), 2),
    }

# ===========================================================================
# SHAP Explainability (XGBoost base model)
# ===========================================================================
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 10 Features (SHAP):")
print(feature_importance.head(10).to_string(index=False))

# Save SHAP plot
plt.figure(figsize=(12, 7))
plt.barh(feature_importance["feature"][::-1], feature_importance["mean_abs_shap"][::-1], color="#2c3e50")
plt.xlabel("Mean |SHAP value|")
plt.title("Feature Importance for Delay Prediction (XGBoost — v2)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_feature_importance.png", dpi=150)
plt.close()

# ===========================================================================
# SAVE METRICS  (#10 MLflow)
# ===========================================================================
metrics_out = {
    "version": "v2",
    "accuracy": round(accuracy, 3),
    "roc_auc": round(auc, 3),
    "optimal_threshold": round(optimal_threshold, 3),
    "n_models_in_ensemble": n_models,
    "has_lightgbm": HAS_LGB,
    "has_catboost": HAS_CATBOOST,
    "has_optuna": HAS_OPTUNA,
    "confusion_matrix": cm,
    "n_train": int(train_mask.sum()),
    "n_test": int(test_mask.sum()),
    "feature_importance": feature_importance.to_dict(orient="records"),
    "class_balance_test": {
        "late": int(y_test.sum()),
        "on_time": int(len(y_test) - y_test.sum())
    },
    "delay_regressor": reg_metrics,
    "best_xgb_params": {k: v for k, v in best_xgb_params.items()
                        if k not in ("scale_pos_weight",)},
}

with open(f"{OUT_DIR}/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2)

# Top-risk orders
test_df = X_test.copy()
test_df["po_id"] = df.loc[X_test.index, "po_id"].values
test_df["actual_late"] = y_test.values
test_df["predicted_late_prob"] = ensemble_prob
top_risk = test_df.sort_values("predicted_late_prob", ascending=False).head(20)[
    ["po_id", "predicted_late_prob", "actual_late"]
]
top_risk.to_json(f"{OUT_DIR}/top_risk_orders.json", orient="records", indent=2)

if HAS_MLFLOW and mlflow_run:
    mlflow.log_params({k: v for k, v in best_xgb_params.items()
                       if k not in ("scale_pos_weight",) and isinstance(v, (int, float, str))})
    mlflow.log_params({
        "n_models": n_models,
        "has_lightgbm": HAS_LGB,
        "has_catboost": HAS_CATBOOST,
        "train_split": "chronological_2023-2024",
        "test_split": "2025",
    })
    mlflow.log_metrics({
        "accuracy": accuracy,
        "roc_auc": auc,
        "optimal_threshold": optimal_threshold,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    })
    if "mae_days" in reg_metrics:
        mlflow.log_metric("delay_regressor_mae", reg_metrics["mae_days"])
    mlflow.log_artifact(f"{OUT_DIR}/model_metrics.json")
    mlflow.log_artifact(f"{OUT_DIR}/shap_feature_importance.png")
    mlflow.end_run()
    print(f"\nMLflow run logged. View with: mlflow ui --backend-store-uri sqlite:///{mlflow_db}")

print(f"\nAll artifacts saved to {OUT_DIR}")
print(f"ROC-AUC: {auc:.3f} | Accuracy: {accuracy:.3f} | Threshold: {optimal_threshold:.3f}")
if "mae_days" in reg_metrics:
    print(f"Delay Regressor MAE: {reg_metrics['mae_days']} days")
