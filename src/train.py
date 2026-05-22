import io
import zipfile
import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, precision_score, recall_score

def load_modeling_packages(zip_path="data/processed/modeling/modeling_matrices_package.zip"):
    """Extract and load the pre-split training and testing matrices."""
    print("📦 Unpacking modeling matrices...")
    matrices = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv'):
                var_name = file_name.replace('.csv', '')
                with z.open(file_name) as f:
                    matrices[var_name] = pd.read_csv(f)
    return matrices

def load_best_parameters(study_name):
    """Loads the optimal parameters from the JSON audit logs."""
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "tuning"
    file_path = log_dir / f"{study_name}_best_results.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"🚨 Tuning log not found at {file_path}. Did you run tune.py?")
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    print(f"✅ Loaded optimal parameters for {study_name}.")
    return data["best_params"]

def train_hurdle_model():
    # Ensure directory exists
    os.makedirs("models", exist_ok=True)

    matrices = load_modeling_packages()

    X_train = matrices["X_train_stage1"]
    X_test = matrices["X_test_unfiltered"]
    y_train_cls = matrices["y_train_stage1"].values.ravel()
    y_test_cls = matrices["y_test_stage1"].values.ravel()

    X_train_reg = matrices["X_train_stage2"]
    y_train_reg_log = matrices["y_train_stage2_log"].values.ravel()
    y_test_reg_raw = matrices["y_test_stage2_raw"].values.ravel()

    # Load parameters dynamically
    stage1_params = load_best_parameters("stage1_classifier")
    stage2_params = load_best_parameters("stage2_regressor")

    # -------------------------------------------------------------------------
    # STAGE 1: Classifier Hurdle (Did a financial loss occur?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 1 Classifier (Random Forest)...")

    # Unpack parameters using **
    clf = RandomForestClassifier(
        **stage1_params,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train_cls)

    # Evaluate Stage 1
    y_pred_cls = clf.predict(X_test)
    y_prob_cls = clf.predict_proba(X_test)[:, 1]

    print("\n📊 Default Stage 1 Classification Report (Threshold 0.50):")
    print(classification_report(y_test_cls, y_pred_cls))

    # Threshold Calibration Loop
    print("\n🔍 Phase 1: Testing Probability Thresholds for Precision-Recall Trade-off...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    print(f"{'Threshold':<12} | {'Precision (Class 1)':<20} | {'Recall (Class 1)':<20}")
    print("-" * 60)
    
    for thresh in thresholds:
        y_pred_custom = (y_prob_cls >= thresh).astype(int)
        prec = precision_score(y_test_cls, y_pred_custom, zero_division=0)
        rec = recall_score(y_test_cls, y_pred_custom, zero_division=0)
        print(f"{thresh:<12.2f} | {prec:<20.4f} | {rec:<20.4f}")
    print("-" * 60)
    # ---------------------------------------

    # -------------------------------------------------------------------------
    # STAGE 2: Regressor (If a loss occurred, how much did it cost?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 2 Regressor (Random Forest Regressor)...")

    # Unpack parameters using **
    reg = RandomForestRegressor(
        **stage2_params,
        random_state=42,
        n_jobs=-1
    )

    reg.fit(X_train_reg, y_train_reg_log)

    # Evaluate Stage 2 conditionally
    test_mask = y_test_reg_raw > 0
    X_test_reg = X_test[test_mask]
    y_test_reg_actual = y_test_reg_raw[test_mask]

    reg_preds_logged = reg.predict(X_test_reg)
    reg_preds_actual = np.expm1(reg_preds_logged)

    print("\n📊 Stage 2 Regression Metrics (Conditional on Cost > 0):")
    print(f"  - MAE:  ${mean_absolute_error(y_test_reg_actual, reg_preds_actual):,.2f}")
    print(f"  - R²:   {r2_score(y_test_reg_actual, reg_preds_actual):.3f}")

    # -------------------------------------------------------------------------
    # COMBINED HURDLE EVALUATION
    # -------------------------------------------------------------------------
    print("\n🧮 Calculating Combined Hurdle Expectations across the entire test set...")
    
    all_reg_preds_logged = reg.predict(X_test)
    all_reg_preds_actual = np.expm1(all_reg_preds_logged)
    
    expected_cost = y_prob_cls * all_reg_preds_actual
    
    overall_mae = mean_absolute_error(y_test_reg_raw, expected_cost)
    print(f"🏁 Overall Hurdle Model Unconditional MAE: ${overall_mae:,.2f}")
    
    # -------------------------------------------------------------------------
    # ARTIFACT STORAGE
    # -------------------------------------------------------------------------
    print("\n💾 Saving trained artifacts...")
    joblib.dump(clf, "models/stage1_classifier.joblib")
    joblib.dump(reg, "models/stage2_regressor.joblib")
    print("✨ Modeling artifacts saved successfully!")

if __name__ == "__main__":
    train_hurdle_model()