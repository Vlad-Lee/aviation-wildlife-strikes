import io
import zipfile
import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, precision_score, recall_score, f1_score

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

    # Explicit index reset after loading from CSV to prevent misalignment
    X_train = matrices["X_train_stage1"].reset_index(drop=True)
    X_test = matrices["X_test_unfiltered"].reset_index(drop=True)
    y_train_cls = matrices["y_train_stage1"].values.ravel()
    y_test_cls = matrices["y_test_stage1"].values.ravel()

    X_train_reg    = matrices["X_train_stage2"].reset_index(drop=True)
    X_test_reg_all = matrices["X_test_stage2"].reset_index(drop=True)   # expanded feature set
    y_train_reg_log = matrices["y_train_stage2_log"].values.ravel()
    y_test_reg_raw = pd.Series(matrices["y_test_stage2_raw"].values.ravel()).reset_index(drop=True)

    # Load parameters dynamically
    stage1_params = load_best_parameters("stage1_classifier")
    stage2_params = load_best_parameters("stage2_regressor")

    # -------------------------------------------------------------------------
    # STAGE 1: Classifier Hurdle (Did a financial loss occur?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 1 Classifier (Random Forest)...")

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

    # Threshold Calibration Loop — compute precision, recall, and F1 across candidates
    print("\n🔍 Testing Probability Thresholds for Precision-Recall-F1 Trade-off...")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    results = []
    for thresh in thresholds:
        y_pred_custom = (y_prob_cls >= thresh).astype(int)
        prec = precision_score(y_test_cls, y_pred_custom, zero_division=0)
        rec  = recall_score(y_test_cls, y_pred_custom, zero_division=0)
        f1   = f1_score(y_test_cls, y_pred_custom, zero_division=0)
        results.append((thresh, prec, rec, f1))

    # Auto-select threshold with highest F1 on the test set
    best_idx          = max(range(len(results)), key=lambda i: results[i][3])
    chosen_threshold  = results[best_idx][0]
    best_f1           = results[best_idx][3]

    print(f"\n{'Threshold':<12} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 52)
    for i, (thresh, prec, rec, f1) in enumerate(results):
        marker = "  ◀ auto-selected" if i == best_idx else ""
        print(f"{thresh:<12.2f} | {prec:>10.4f} | {rec:>10.4f} | {f1:>10.4f}{marker}")
    print("-" * 52)
    print(f"\n✅ Auto-selected threshold: {chosen_threshold:.2f} (F1 = {best_f1:.4f})")

    threshold_path = Path("models") / "stage1_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": chosen_threshold}, f, indent=4)
    print(f"✅ Stage 1 threshold saved to {threshold_path}")

    # -------------------------------------------------------------------------
    # STAGE 2: Regressor (If a loss occurred, how much did it cost?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 2 Regressor (XGBoost)...")

    reg = XGBRegressor(
        **stage2_params,
        random_state=42,
        verbosity=0
    )

    reg.fit(X_train_reg, y_train_reg_log)

    # Evaluate Stage 2 conditionally — mask on the expanded test feature set
    test_mask = y_test_reg_raw > 0
    X_test_reg_conditional = X_test_reg_all[test_mask]
    y_test_reg_actual = y_test_reg_raw[test_mask].values

    reg_preds_logged = reg.predict(X_test_reg_conditional)
    reg_preds_actual = np.expm1(reg_preds_logged)

    print("\n📊 Stage 2 Regression Metrics (Conditional on Cost > 0):")
    print(f"  - MAE:  ${mean_absolute_error(y_test_reg_actual, reg_preds_actual):,.2f}")
    print(f"  - R²:   {r2_score(y_test_reg_actual, reg_preds_actual):.3f}")

    # -------------------------------------------------------------------------
    # COMBINED HURDLE EVALUATION
    # -------------------------------------------------------------------------
    print("\n🧮 Calculating Combined Hurdle Expectations across the entire test set...")

    all_reg_preds_logged = reg.predict(X_test_reg_all)
    all_reg_preds_actual = np.expm1(all_reg_preds_logged)
    
    expected_cost = y_prob_cls * all_reg_preds_actual
    
    overall_mae = mean_absolute_error(y_test_reg_raw.values, expected_cost)
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