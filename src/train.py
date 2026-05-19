import io
import zipfile
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score


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


def train_hurdle_model():
    # Ensure directory exists
    os.makedirs("models", exist_ok=True)

    matrices = load_modeling_packages()

    X_train, X_test = matrices["X_train"], matrices["X_test"]

    y_train_cls = matrices["y_train_stage1"].values.ravel()
    y_test_cls = matrices["y_test_stage1"].values.ravel()

    y_train_reg_log = matrices["y_train_stage2"].values.ravel()
    X_train_reg = matrices["X_train_stage2"]

    y_test_reg_raw = matrices["y_test_stage2_raw"]

    # -------------------------------------------------------------------------
    # STAGE 1: Classifier Hurdle (Did a financial loss occur?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 1 Classifier (Random Forest)...")

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train_cls)

    # Evaluate Stage 1
    y_pred_cls = clf.predict(X_test)
    y_prob_cls = clf.predict_proba(X_test)[:, 1]

    print("\n📊 Stage 1 Classification Report:")
    print(classification_report(y_test_cls, y_pred_cls))

    # -------------------------------------------------------------------------
    # STAGE 2: Regressor (If a loss occurred, how much did it cost?)
    # -------------------------------------------------------------------------
    print("\n🚀 Training Stage 2 Regressor (Random Forest Regressor)...")

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=11,
        min_samples_leaf=4,
        max_features=0.5926834623509254,
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
    
    # The total expected cost is modeled as: E[Cost | X] = P(Damage = 1 | X) * E[Cost | Damage = 1, X]
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