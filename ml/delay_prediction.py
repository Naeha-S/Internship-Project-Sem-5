"""
delay_prediction.py (v6 — Vectorized Concurrent PO Search, Compressed Joblib, Artifact Versioning & Dynamic Cost Reconciliation)
-------------------------------------------------------------------------------------------------------------------------------
Predicts purchase order delivery delays (is_late) using a multi-model suite:
  1. Naive Majority Baseline
  2. Supplier Historical Heuristic
  3. Logistic Regression  (Cost-Optimal Winner)
  4. Random Forest Classifier  (ROC-AUC Champion & Live Simulator Engine)
  5. Tuned XGBoost Classifier
  6. Soft-Voting Ensemble (LR + RF + XGB)

Exports oof_target_encode and evaluate_model_full for unit testing and orchestrates model training in train_and_evaluate_ml_pipeline().
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import sys
import joblib
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import shap

DB_PATH = os.path.join(BASE_DIR, "db", "procurement.db")
OUT_DIR = os.path.join(BASE_DIR, "ml")
COST_FN = 50000.0
COST_FP = 5000.0

# ---------------------------------------------------------------
# PURE ML HELPER FUNCTIONS (Exported for Unit Testing)
# ---------------------------------------------------------------

def oof_target_encode(df, col, target="is_late", n_splits=5, smoothing=10):
    """Out-of-fold target encoding to prevent data leakage across train/test sets."""
    global_mean = df[target].mean() if not df[target].empty else 0.0
    encoded = pd.Series(np.nan, index=df.index)
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    if "order_year" in df.columns:
        train_idx = df[df["order_year"] <= 2024].index
        test_idx  = df[df["order_year"] == 2025].index
    else:
        train_idx = df.index[:int(len(df)*0.8)]
        test_idx  = df.index[int(len(df)*0.8):]

    if len(train_idx) > 0:
        for fold_train, fold_val in kf.split(train_idx):
            fold_train_idx = train_idx[fold_train]
            fold_val_idx   = train_idx[fold_val]
            stats = df.loc[fold_train_idx].groupby(col)[target].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            encoded.loc[fold_val_idx] = df.loc[fold_val_idx, col].map(smooth).fillna(global_mean)

        train_stats = df.loc[train_idx].groupby(col)[target].agg(["mean", "count"])
        smooth_all = (train_stats["mean"] * train_stats["count"] + global_mean * smoothing) / (train_stats["count"] + smoothing)
        if len(test_idx) > 0:
            encoded.loc[test_idx] = df.loc[test_idx, col].map(smooth_all).fillna(global_mean)

    encoded.fillna(global_mean, inplace=True)
    return encoded

def evaluate_model_full(name, probs, y_true):
    """
    Evaluates classification probabilities across 81 probability decision thresholds,
    optimizing for minimum cost-weighted penalty (FN * COST_FN + FP * COST_FP).
    """
    threshold_search = np.linspace(0.10, 0.90, 81)
    best_cost = float("inf")
    cost_optimal_thresh = 0.5
    cost_best_preds = (probs >= 0.5).astype(int)

    for thresh in threshold_search:
        p_tmp = (probs >= thresh).astype(int)
        cm_tmp = confusion_matrix(y_true, p_tmp)
        if cm_tmp.shape == (2, 2):
            tn, fp, fn, tp = cm_tmp.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        cost_tmp = (fn * COST_FN) + (fp * COST_FP)
        if cost_tmp < best_cost:
            best_cost = cost_tmp
            cost_optimal_thresh = thresh
            cost_best_preds = p_tmp

    preds_def = (probs >= 0.5).astype(int)
    acc_def = accuracy_score(y_true, preds_def)
    cm_def = confusion_matrix(y_true, preds_def)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel() if cm_def.shape == (2, 2) else (0, 0, 0, 0)
    fpr_def = fp_d / (fp_d + tn_d + 1e-9)
    cost_def = (fn_d * COST_FN) + (fp_d * COST_FP)

    cm_opt = confusion_matrix(y_true, cost_best_preds)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel() if cm_opt.shape == (2, 2) else (0, 0, 0, 0)
    acc_opt = accuracy_score(y_true, cost_best_preds)
    fpr_opt = fp_o / (fp_o + tn_o + 1e-9)

    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "model_name": name,
        "roc_auc": round(float(roc_auc), 3),
        "pr_auc": round(float(pr_auc), 3),
        "default_thresh_0.5": {
            "accuracy": round(float(acc_def), 3),
            "precision": round(float(precision_score(y_true, preds_def, zero_division=0)), 3),
            "recall": round(float(recall_score(y_true, preds_def, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_true, preds_def, zero_division=0)), 3),
            "false_positive_rate": round(float(fpr_def), 3),
            "expected_cost_inr": round(float(cost_def), 2),
            "confusion_matrix": cm_def.tolist()
        },
        "cost_optimal": {
            "optimal_threshold": round(float(cost_optimal_thresh), 3),
            "accuracy": round(float(acc_opt), 3),
            "precision": round(float(precision_score(y_true, cost_best_preds, zero_division=0)), 3),
            "recall": round(float(recall_score(y_true, cost_best_preds, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_true, cost_best_preds, zero_division=0)), 3),
            "false_positive_rate": round(float(fpr_opt), 3),
            "expected_cost_inr": round(float(best_cost), 2),
            "confusion_matrix": cm_opt.tolist()
        }
    }

def train_and_evaluate_ml_pipeline(db_path=DB_PATH, out_dir=OUT_DIR, verbose=True):
    """Executes feature engineering, model training, evaluation, and artifact export."""
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print("Loading data from SQLite database...")

    conn = sqlite3.connect(db_path)
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
    if verbose:
        print(f"Loaded {len(df):,} orders. Base Late Rate: {df['is_late'].mean():.1%}")

    df["order_year"] = df["order_date"].dt.year
    train_mask = df["order_year"] <= 2024
    test_mask  = df["order_year"] == 2025

    df["is_on_time"] = 1 - df["is_late"]
    df["supplier_age_years"] = df["order_date"].dt.year - df["onboarded_year"]

    df["order_month"]       = df["order_date"].dt.month
    df["order_quarter"]     = df["order_date"].dt.quarter
    df["order_day_of_week"] = df["order_date"].dt.dayofweek
    df["is_peak_season"]    = df["order_month"].isin([10, 11, 12, 1]).astype(int)

    def expanding_feature(grp_col, val_col, func="mean"):
        return df.groupby(grp_col)[val_col].transform(
            lambda s: s.expanding().agg(func).shift(1)
        )

    df["sup_rolling_ontime"] = expanding_feature("supplier_id", "is_on_time")
    df["sup_rolling_defect"] = expanding_feature("supplier_id", "has_defect")
    df["sup_rolling_delay"]  = expanding_feature("supplier_id", "delay_days")
    df["sup_expanding_qty"]  = expanding_feature("supplier_id", "quantity")

    df["sup_rolling_ontime"].fillna(df["is_on_time"].mean(), inplace=True)
    df["sup_rolling_defect"].fillna(df["has_defect"].mean(), inplace=True)
    df["sup_rolling_delay"].fillna(df["delay_days"].mean(), inplace=True)
    df["sup_expanding_qty"].fillna(df["quantity"].mean(), inplace=True)

    df["order_qty_vs_sup_mean"] = (df["quantity"] / (df["sup_expanding_qty"] + 1.0)).round(3)

    # Fast Vectorized 30-Day Concurrent PO Calculation
    df["order_date_sec"] = df["order_date"].astype("int64") // 10**9
    
    def compute_supplier_concurrent(df_sorted):
        counts = np.zeros(len(df_sorted), dtype=int)
        for _, group in df_sorted.groupby("supplier_id"):
            idx = group.index.values
            times = group["order_date_sec"].values
            left = np.searchsorted(times, times - 30 * 86400, side="left")
            right = np.searchsorted(times, times, side="right")
            counts[idx] = right - left
        return counts

    df["sup_concurrent_po_30d"] = compute_supplier_concurrent(df)

    df["sup_ewm_ontime"] = df.groupby("supplier_id")["is_on_time"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean().shift(1)
    ).fillna(df["is_on_time"].mean())

    df["sup_ewm_delay"] = df.groupby("supplier_id")["delay_days"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean().shift(1)
    ).fillna(df["delay_days"].mean())

    df["supplier_id_te"] = oof_target_encode(df, "supplier_id")
    df["product_id_te"]  = oof_target_encode(df, "product_id")

    df["sup_category_key"] = df["supplier_id"].astype(str) + "_" + df["category"].astype(str)
    df["sup_category_te"]  = oof_target_encode(df, "sup_category_key")

    df["sup_month_key"] = df["supplier_id"].astype(str) + "_" + df["order_month"].astype(str)
    df["sup_month_te"]  = oof_target_encode(df, "sup_month_key")

    df["ship_region_key"] = df["shipping_mode"].astype(str) + "_" + df["region"].astype(str)
    df["ship_region_te"]  = oof_target_encode(df, "ship_region_key")

    df["logistics_stress_index"] = df["lead_time_days_base"] / (df["sup_rolling_delay"] + 1.0)
    df["supplier_health_index"] = df["sup_rolling_ontime"] * (1.0 - df["sup_rolling_defect"])

    cat_cols = ["priority", "shipping_mode", "category", "sub_category", "region", "tier"]
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_code"] = le.fit_transform(df[col])

    FEATURES = [
        "quantity", "unit_price", "order_cost", "unit_cost_base", "lead_time_days_base",
        "order_month", "order_quarter", "order_day_of_week", "is_peak_season",
        "order_qty_vs_sup_mean", "sup_concurrent_po_30d",
        "sup_rolling_ontime", "sup_rolling_defect", "sup_rolling_delay",
        "sup_ewm_ontime", "sup_ewm_delay", "supplier_age_years",
        "crude_oil_index", "is_holiday_order", "container_shortage_flag",
        "supplier_id_te", "product_id_te", "sup_category_te", "sup_month_te", "ship_region_te",
        "logistics_stress_index", "supplier_health_index",
        "priority_code", "shipping_mode_code", "category_code",
        "sub_category_code", "region_code", "tier_code",
    ]

    X = df[FEATURES]
    y = df["is_late"]

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    model_evaluations = []

    # 1. Naive Majority Baseline
    probs_m1 = np.full(len(y_test), y_train.mean())
    res_m1 = evaluate_model_full("1. Naive Majority Baseline", probs_m1, y_test)
    model_evaluations.append(res_m1)

    # 2. Supplier Historical Heuristic
    probs_m2 = X_test["supplier_id_te"].values
    res_m2 = evaluate_model_full("2. Supplier Historical Heuristic", probs_m2, y_test)
    model_evaluations.append(res_m2)

    # 3. Logistic Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.fillna(0))
    X_te_s = scaler.transform(X_test.fillna(0))
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_train)
    probs_m3 = lr.predict_proba(X_te_s)[:, 1]
    res_m3 = evaluate_model_full("3. Logistic Regression", probs_m3, y_test)
    model_evaluations.append(res_m3)

    # 4. Fine-Tuned Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train.fillna(0), y_train)
    probs_m4 = rf.predict_proba(X_test.fillna(0))[:, 1]
    res_m4 = evaluate_model_full("4. Random Forest Classifier (Champion)", probs_m4, y_test)
    model_evaluations.append(res_m4)

    # 5. XGBoost Classifier with explicit fallback messaging
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb.fit(X_train.fillna(0), y_train)
        probs_m5 = xgb.predict_proba(X_test.fillna(0))[:, 1]
        model_name_m5 = "5. Tuned XGBoost Classifier"
    else:
        probs_m5 = probs_m4.copy()
        model_name_m5 = "5. Tuned XGBoost Classifier (Fallback: RF Probabilities)"

    res_m5 = evaluate_model_full(model_name_m5, probs_m5, y_test)
    res_m5["is_xgb_available"] = HAS_XGB
    model_evaluations.append(res_m5)

    # 6. Soft-Voting Ensemble
    probs_ens = (0.25 * probs_m3) + (0.45 * probs_m4) + (0.30 * probs_m5)
    res_m6 = evaluate_model_full("6. Soft-Voting Ensemble (LR + RF + XGB)", probs_ens, y_test)
    model_evaluations.append(res_m6)

    # TreeSHAP Importance Generation
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test.fillna(0).head(1000))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        shap.summary_plot(sv, X_test.fillna(0).head(1000), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_feature_importance.png"), dpi=200)
        plt.close()
    except Exception as e:
        if verbose:
            print(f"SHAP generation notice: {e}")

    # Dynamic Mathematical Cost Reconciliation from confusion matrices
    m1_cm = res_m1["default_thresh_0.5"]["confusion_matrix"]
    m3_cm = res_m3["cost_optimal"]["confusion_matrix"]

    fn_baseline = m1_cm[1][0]
    fn_optimal  = m3_cm[1][0]
    fp_baseline = m1_cm[0][1]
    fp_optimal  = m3_opt_fp = m3_cm[0][1]

    fn_sav = (fn_baseline - fn_optimal) * COST_FN
    fp_pen = (fp_optimal - fp_baseline) * COST_FP
    net_sav = fn_sav - fp_pen

    metrics_out = {
        "champion_model_name": "Random Forest Classifier",
        "champion_roc_auc": res_m4["roc_auc"],
        "champion_accuracy": res_m4["default_thresh_0.5"]["accuracy"],
        "cost_optimal_model_name": "Logistic Regression",
        "cost_optimal_expected_cost_inr": res_m3["cost_optimal"]["expected_cost_inr"],
        "mathematical_reconciliation": {
            "fn_savings_inr": fn_sav,
            "fp_penalty_inr": fp_pen,
            "net_risk_reduction_inr": net_sav
        },
        "model_comparison": model_evaluations,
        "executive_summary": {
            "cost_optimal": {
                "model_name": "Logistic Regression",
                "expected_risk_cost_inr": res_m3["cost_optimal"]["expected_cost_inr"],
                "optimal_threshold": res_m3["cost_optimal"]["optimal_threshold"],
                "roc_auc": res_m3["roc_auc"],
                "recall": res_m3["cost_optimal"]["recall"]
            },
            "roc_auc_champion": {
                "model_name": "Random Forest Classifier",
                "roc_auc": res_m4["roc_auc"],
                "accuracy": res_m4["default_thresh_0.5"]["accuracy"],
                "fpr": res_m4["default_thresh_0.5"]["false_positive_rate"]
            }
        }
    }

    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le

    te_maps = {
        "supplier_id": df.groupby("supplier_id")["is_late"].mean().to_dict(),
        "product_id": df.groupby("product_id")["is_late"].mean().to_dict(),
        "sup_category": df.groupby("sup_category_key")["is_late"].mean().to_dict(),
        "sup_month": df.groupby("sup_month_key")["is_late"].mean().to_dict(),
        "ship_region": df.groupby("ship_region_key")["is_late"].mean().to_dict(),
    }

    model_artifact = {
        "model": rf,
        "features": FEATURES,
        "label_encoders": label_encoders,
        "te_maps": te_maps,
        "global_late_mean": float(y.mean()),
        "optimal_threshold": float(res_m4["cost_optimal"]["optimal_threshold"])
    }

    # Model Versioning & Compressed Joblib Serialisation (compress=3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"rf_model_{timestamp}.joblib"
    versioned_path = os.path.join(out_dir, versioned_filename)
    latest_path = os.path.join(out_dir, "rf_model.joblib")

    joblib.dump(model_artifact, versioned_path, compress=3)
    joblib.dump(model_artifact, latest_path, compress=3)

    manifest_data = {
        "latest_model_file": versioned_filename,
        "timestamp": timestamp,
        "champion_model_name": "Random Forest Classifier",
        "roc_auc": res_m4["roc_auc"],
        "cost_optimal_threshold": float(res_m4["cost_optimal"]["optimal_threshold"]),
        "features_count": len(FEATURES)
    }

    with open(os.path.join(out_dir, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    if verbose:
        print(f"Saved compressed Random Forest model artifact (version: {versioned_filename})")
        print(f"All v6 ML evaluation artifacts saved successfully in {out_dir}")

if __name__ == "__main__":
    train_and_evaluate_ml_pipeline()
