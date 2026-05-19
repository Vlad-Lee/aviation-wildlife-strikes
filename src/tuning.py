import zipfile
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

def load_modeling_packages(zip_path="./data/processed/modeling/modeling_matrices_package.zip"):
    """Extract and load the pre-split training matrices for tuning."""
    print("📦 Unpacking modeling matrices for Optuna tuning...")
    matrices = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv') and 'train' in file_name: # Only need training data for CV
                var_name = file_name.replace('.csv', '')
                with z.open(file_name) as f:
                    matrices[var_name] = pd.read_csv(f)
    return matrices

# Load data globally for the Optuna objective functions
matrices = load_modeling_packages()

# Stage 1 Data
X_train_cls = matrices["X_train_stage1"]
y_train_cls = matrices["y_train_stage1"].values.ravel()

# Stage 2 Data (The Fiscal Hurdle Subset)
X_train_reg = matrices["X_train_stage2"]
y_train_reg_log = matrices["y_train_stage2_log"].values.ravel()

# -------------------------------------------------------------------------
# STAGE 1: Classifier Objective Function
# -------------------------------------------------------------------------
def objective_stage1(trial):
    """Optuna objective for the Stage 1 Classification Hurdle."""
    
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1 
    )
    
    # Use Stratified K-Fold to maintain the rare event ratio across all splits
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # We optimize for 'f1_macro' to ensure the model respects the minority class
    scores = cross_val_score(clf, X_train_cls, y_train_cls, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    return scores.mean()

# -------------------------------------------------------------------------
# STAGE 2: Regressor Objective Function
# -------------------------------------------------------------------------
def objective_stage2(trial):
    """Optuna objective for the Stage 2 Severity Regressor."""
    
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    # max_features is critical to prevent overfitting on dominant variables
    max_features = trial.suggest_float('max_features', 0.3, 1.0) 
    
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # We optimize for Negative Mean Absolute Error (closer to 0 is better)
    scores = cross_val_score(reg, X_train_reg, y_train_reg_log, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    return scores.mean()

# -------------------------------------------------------------------------
# EXECUTE TUNING STUDIES
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Initiating Optuna Tuning for Stage 1 Classifier...")
    # Direction is 'maximize' because a higher F1 score is better
    study_stage1 = optuna.create_study(direction='maximize', study_name="Stage1_Hurdle_Classifier")
    study_stage1.optimize(objective_stage1, n_trials=15) 
    
    print("\n🏆 Best Parameters for Stage 1 Classifier:")
    print(study_stage1.best_params)
    print(f"Best F1-Macro Score: {study_stage1.best_value:.4f}")
    
    print("\n---------------------------------------------------------")
    
    print("\n🚀 Initiating Optuna Tuning for Stage 2 Regressor...")
    # Direction is 'maximize' because scikit-learn outputs NEGATIVE MAE (so closer to 0 is "maximum")
    study_stage2 = optuna.create_study(direction='maximize', study_name="Stage2_Severity_Regressor")
    study_stage2.optimize(objective_stage2, n_trials=15)
    
    print("\n🏆 Best Parameters for Stage 2 Regressor:")
    print(study_stage2.best_params)
    print(f"Best Negative MAE (Log Space): {study_stage2.best_value:.4f}")
    
    print("\n✅ Tuning complete!")