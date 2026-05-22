import zipfile
import json
from pathlib import Path
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

def load_modeling_packages(zip_path="./data/processed/modeling/modeling_matrices_package.zip"):
    """Extracts and loads training matrices for tuning."""
    print("📦 Unpacking modeling matrices for Optuna tuning...")
    matrices = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv') and 'train' in file_name:
                var_name = file_name.replace('.csv', '')
                with z.open(file_name) as f:
                    matrices[var_name] = pd.read_csv(f)
    return matrices

# Initialize data
matrices = load_modeling_packages()
X_train_cls = matrices["X_train_stage1"]
y_train_cls = matrices["y_train_stage1"].values.ravel()
X_train_reg = matrices["X_train_stage2"]
y_train_reg_log = matrices["y_train_stage2_log"].values.ravel()

def save_hyperparameters(params_dict, best_value, study_name):
    """Saves best parameters and score to a JSON log file (Overwrites)."""
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    data_to_save = {
        "study_name": study_name,
        "best_value": best_value,
        "best_params": params_dict
    }
    
    file_path = log_dir / f"{study_name}_best_results.json"
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"✅ Production config for {study_name} saved to {file_path}")

def append_to_history(params_dict, best_value, study_name):
    """Appends every tuning run to a master history file with an iteration tracker."""
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_file = log_dir / "experiment_history.json"
    
    # Load existing history if it exists
    history = []
    if history_file.exists():
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
                
    # Calculate the next iteration number for this specific study
    study_runs = [run for run in history if run.get("study_name") == study_name]
    next_iteration = len(study_runs) + 1
    
    # Add current run
    history.append({
        "iteration": next_iteration,
        "study_name": study_name,
        "best_value": best_value,
        "best_params": params_dict,
        "timestamp": pd.Timestamp.now().isoformat()
    })
    
    # Save back
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📚 Iteration {next_iteration} appended to master history: {history_file}")

    
    # Save back
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📚 Run appended to master history: {history_file}")

def objective_stage1(trial):
    """Optuna objective for Stage 1 Classification."""
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
    'max_depth': trial.suggest_int('max_depth', 5, 22),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 25),
    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
    }
    
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_cls, y_train_cls, cv=cv, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

def objective_stage2(trial):
    """Optuna objective for Stage 2 Severity Regression."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0)
    }
    
    reg = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_train_reg, y_train_reg_log, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    return scores.mean()

if __name__ == "__main__":
    # Stage 1
    print("\n🚀 Starting Stage 1 Tuning...")
    study_stage1 = optuna.create_study(direction='maximize')
    study_stage1.optimize(objective_stage1, n_trials=20)
    save_hyperparameters(study_stage1.best_params, study_stage1.best_value, "stage1_classifier")
    append_to_history(study_stage1.best_params, study_stage1.best_value, "stage1_classifier")
    
    # Stage 2
    print("\n🚀 Starting Stage 2 Tuning...")
    study_stage2 = optuna.create_study(direction='maximize')
    study_stage2.optimize(objective_stage2, n_trials=20)
    save_hyperparameters(study_stage2.best_params, study_stage2.best_value, "stage2_regressor")
    append_to_history(study_stage2.best_params, study_stage2.best_value, "stage2_regressor")
    
    print("\n🎉 Tuning complete! Logs successfully updated.")