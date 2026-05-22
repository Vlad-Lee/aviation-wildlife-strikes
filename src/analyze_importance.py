import joblib
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io
from pathlib import Path

def get_feature_names(zip_path):
    """Extracts column names from the X_train_stage1 matrix."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("X_train_stage1.csv") as f:
            # Read only the header
            df_temp = pd.read_csv(f, nrows=0)
            return df_temp.columns

def plot_importance(model_path, feature_names, stage_name, output_folder):
    """Loads model, calculates importances, and saves a chart."""
    model = joblib.load(model_path)
    
    # Create DataFrame for visualization
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    # Take top 10
    top_10 = importances.head(10)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['feature'], top_10['importance'], color='#4c72b0')
    plt.gca().invert_yaxis()
    plt.title(f'Top 10 Feature Importances: {stage_name}')
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    output_path = output_folder / f"{stage_name.lower().replace(' ', '_')}_importance.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved chart to {output_path}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    output_folder = base_dir / "outputs" / "figures"
    zip_path = base_dir / "data" / "processed" / "modeling" / "modeling_matrices_package.zip"
    
    # Ensure figures folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load feature names from the matrix package
    feature_names = get_feature_names(zip_path)
    
    # Plot both stages
    plot_importance(model_dir / "stage1_classifier.joblib", feature_names, "Stage 1 Classifier", output_folder)
    plot_importance(model_dir / "stage2_regressor.joblib", feature_names, "Stage 2 Regressor", output_folder)

if __name__ == "__main__":
    main()