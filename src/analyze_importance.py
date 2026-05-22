import joblib
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path

def get_feature_names(zip_path, csv_name):
    """Extracts column names from the specified training matrix inside the zip."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_name) as f:
            return pd.read_csv(f, nrows=0).columns

def plot_importance(model_path, feature_names, stage_name, output_folder, top_n=20):
    """Loads model, plots top N feature importances, and saves the chart."""
    model = joblib.load(model_path)

    importances = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top = importances.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top['feature'], top['importance'], color='#4c72b0')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances: {stage_name}')
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = output_folder / f"{stage_name.lower().replace(' ', '_')}_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved chart to {output_path}")

def main():
    base_dir      = Path(__file__).resolve().parent.parent
    model_dir     = base_dir / "models"
    output_folder = base_dir / "outputs" / "figures"
    zip_path      = base_dir / "data" / "processed" / "modeling" / "modeling_matrices_package.zip"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Load feature names separately per stage — guards against the pipelines
    # diverging in future and producing silently mismatched importance plots.
    feature_names_s1 = get_feature_names(zip_path, "X_train_stage1.csv")
    feature_names_s2 = get_feature_names(zip_path, "X_train_stage2.csv")

    plot_importance(model_dir / "stage1_classifier.joblib", feature_names_s1, "Stage 1 Classifier", output_folder)
    plot_importance(model_dir / "stage2_regressor.joblib",  feature_names_s2, "Stage 2 Regressor",  output_folder)

if __name__ == "__main__":
    main()