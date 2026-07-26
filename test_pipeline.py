"""
test_pipeline.py
-----------------
Runs the entire ProcureSense AI pipeline from data generation to dashboard
creation and verifies that all artifacts are correctly produced.
"""

import subprocess
import os
import sys
import json

def run_script(path):
    print(f"Running: {path}")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"Error running {path}:")
        print(result.stderr)
        return False
    print(f"Successfully ran {path}")
    return True

def test_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        os.path.join(base_dir, "data/generate_data.py"),
        os.path.join(base_dir, "db/build_db.py"),
        os.path.join(base_dir, "analysis/kpi_engine.py"),
        os.path.join(base_dir, "ml/delay_prediction.py"),
        os.path.join(base_dir, "dashboard/generate_dashboard.py")
    ]

    for script in scripts:
        if not run_script(script):
            print("Pipeline failed!")
            return

    # Verification of artifacts
    artifacts = [
        os.path.join(base_dir, "data/suppliers.csv"),
        os.path.join(base_dir, "db/procurement.db"),
        os.path.join(base_dir, "analysis/kpi_summary.json"),
        os.path.join(base_dir, "ml/model_metrics.json"),
        os.path.join(base_dir, "ml/shap_feature_importance.png"),
        os.path.join(base_dir, "docs/SQL_PORTFOLIO.md"),
        os.path.join(base_dir, "dashboard/app.py")
    ]

    print("\nVerifying artifacts:")
    all_exists = True
    for artifact in artifacts:
        if os.path.exists(artifact):
            print(f"  [OK] {artifact}")
        else:
            print(f"  [MISSING] {artifact}")
            all_exists = False

    if all_exists:
        print("\nPipeline test PASSED!")
        
        # Print some key metrics
        with open(os.path.join(base_dir, "analysis/kpi_summary.json")) as f:
            kpi = json.load(f)
            print(f"Overall Health Score: {kpi['overall_health_score']}")
            
        with open(os.path.join(base_dir, "ml/model_metrics.json")) as f:
            model = json.load(f)
            print(f"Model Accuracy: {model['accuracy']}")
            print(f"Model ROC-AUC: {model['roc_auc']}")
    else:
        print("\nPipeline test FAILED due to missing artifacts.")

if __name__ == "__main__":
    test_pipeline()
