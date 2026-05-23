import zipfile
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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

def load_threshold():
    """Load the chosen Stage 1 threshold."""
    # validate.py is in /src/, go up to project root
    project_root = Path(__file__).resolve().parent.parent
    threshold_path = project_root / "models" / "stage1_threshold.json"
    with open(threshold_path, "r") as f:
        return json.load(f)["threshold"]

# =========================================================================
# STAGE 1: CLASSIFICATION VALIDATION
# =========================================================================
def validate_stage1(clf, X_test, y_test, threshold, output_folder):
    """Generate all Stage 1 classification plots and metrics."""
    print("\n" + "="*70)
    print("STAGE 1: CLASSIFICATION VALIDATION")
    print("="*70)

    # Get predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # =====================================================================
    # 1. CLASS DISTRIBUTION IN TEST SET
    # =====================================================================
    print("\n1️⃣ Class Distribution in Test Set:")
    class_counts = pd.Series(y_test).value_counts()
    print(f"   No Damage (0): {class_counts[0]:,} ({100*class_counts[0]/len(y_test):.1f}%)")
    print(f"   Damage (1):    {class_counts[1]:,} ({100*class_counts[1]/len(y_test):.1f}%)")
    print(f"   Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")

    # =====================================================================
    # 2. CONFUSION MATRIX & BASIC METRICS
    # =====================================================================
    print(f"\n2️⃣ Confusion Matrix (at threshold {threshold:.2f}):")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"   TP: {tp:,}  |  FP: {fp:,}")
    print(f"   FN: {fn:,}  |  TN: {tn:,}")

    # =====================================================================
    # 3. SENSITIVITY & SPECIFICITY
    # =====================================================================
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n3️⃣ Sensitivity & Specificity (at threshold {threshold:.2f}):")
    print(f"   Sensitivity (Recall/True Positive Rate): {sensitivity:.4f}")
    print(f"   Specificity (True Negative Rate):        {specificity:.4f}")

    # =====================================================================
    # 4. CLASSIFICATION METRICS
    # =====================================================================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"\n4️⃣ Classification Metrics:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc_score:.4f}")

    # =====================================================================
    # 5. ROC CURVE
    # =====================================================================
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2ca02c', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Stage 1: ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "stage1_roc_curve.png", dpi=300)
    plt.close()
    print(f"\n   ✅ Saved: stage1_roc_curve.png")

    # =====================================================================
    # 6. PRECISION-RECALL CURVE
    # =====================================================================
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='#ff7f0e', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=class_counts[1]/len(y_test), color='gray', linestyle='--', lw=1.5, label='Baseline (Random)')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Stage 1: Precision-Recall Curve (Best for Imbalanced Data)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_folder / "stage1_precision_recall_curve.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage1_precision_recall_curve.png")

    # =====================================================================
    # 7. CONFUSION MATRIX HEATMAP
    # =====================================================================
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Damage', 'Damage'],
                yticklabels=['No Damage', 'Damage'],
                annot_kws={'size': 14})
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.title(f'Stage 1: Confusion Matrix (Threshold = {threshold:.2f})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_folder / "stage1_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage1_confusion_matrix.png")

    # =====================================================================
    # 8. THRESHOLD ANALYSIS
    # =====================================================================
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='o', markersize=4, lw=2)
    plt.plot(thresholds, recalls, label='Recall', marker='s', markersize=4, lw=2)
    plt.plot(thresholds, f1s, label='F1-Score', marker='^', markersize=4, lw=2)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Chosen Threshold ({threshold:.2f})')
    plt.xlabel('Decision Threshold', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.title('Stage 1: Threshold Trade-offs', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "stage1_threshold_analysis.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage1_threshold_analysis.png")


# =========================================================================
# STAGE 2: REGRESSION VALIDATION
# =========================================================================
def validate_stage2(reg, X_test, y_test_raw, threshold, y_prob_stage1, output_folder):
    """Generate all Stage 2 regression plots and metrics."""
    print("\n" + "="*70)
    print("STAGE 2: REGRESSION VALIDATION")
    print("="*70)

    # Filter to cases with damage (y > 0)
    mask = y_test_raw > 0
    X_test_damage = X_test[mask]
    y_test_actual = y_test_raw[mask].values

    print(f"\n📊 Evaluating on {len(y_test_actual):,} test cases with damage (TOTAL_COST > 0)")

    # Get predictions (on log scale, then exponentiate)
    y_pred_logged = reg.predict(X_test_damage)
    y_pred_actual = np.expm1(y_pred_logged)

    # =====================================================================
    # 1. RESIDUALS & ERRORS
    # =====================================================================
    residuals = y_test_actual - y_pred_actual
    abs_errors = np.abs(residuals)
    percent_errors = 100 * np.abs(y_test_actual - y_pred_actual) / y_test_actual

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    median_error = np.median(abs_errors)

    print(f"\n1️⃣ Regression Metrics (Conditional on Cost > 0):")
    print(f"   MAE:           ${mae:,.2f}")
    print(f"   RMSE:          ${rmse:,.2f}")
    print(f"   Median Error:  ${median_error:,.2f}")
    print(f"   R²:            {r2:.4f}")
    print(f"   Median % Error: {np.median(percent_errors):.1f}%")

    # =====================================================================
    # 2. PREDICTED vs ACTUAL SCATTER PLOT
    # =====================================================================
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, s=30, edgecolors='none')

    # Perfect prediction line
    max_val = max(y_test_actual.max(), y_pred_actual.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Cost ($)', fontsize=11)
    plt.ylabel('Predicted Cost ($)', fontsize=11)
    plt.title(f'Stage 2: Predicted vs Actual Cost (n={len(y_test_actual):,})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Add text box with metrics
    textstr = f'MAE: ${mae:,.0f}\nRMSE: ${rmse:,.0f}\nR²: {r2:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_folder / "stage2_predicted_vs_actual.png", dpi=300)
    plt.close()
    print(f"\n   ✅ Saved: stage2_predicted_vs_actual.png")

    # =====================================================================
    # 3. RESIDUAL PLOT (Residuals vs Predicted)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_actual, residuals, alpha=0.5, s=30, edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Cost ($)', fontsize=11)
    plt.ylabel('Residuals ($)', fontsize=11)
    plt.title('Stage 2: Residual Plot (Residuals vs Predicted)', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "stage2_residuals.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage2_residuals.png")

    # =====================================================================
    # 4. RESIDUAL DISTRIBUTION & NORMALITY
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2, label='Mean = 0')
    axes[0].set_xlabel('Residuals ($)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Q-Q plot (normality test)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Test for Normality)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_folder / "stage2_residual_distribution.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage2_residual_distribution.png")

    # Normality test
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuals[:min(len(residuals), 5000)])  # Shapiro limited to 5000 samples
    print(f"\n2️⃣ Residual Normality Test (Shapiro-Wilk):")
    print(f"   p-value: {p_value:.4f} {'✓ Normal' if p_value > 0.05 else '✗ Not Normal'}")
    print(f"   Mean of residuals: ${residuals.mean():,.2f} (should be ≈ 0)")
    print(f"   Std of residuals:  ${residuals.std():,.2f}")

    # =====================================================================
    # 5. ERROR DISTRIBUTION
    # =====================================================================
    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='#ff7f0e')
    plt.xlabel('Absolute Error ($)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'Stage 2: Distribution of Absolute Errors (Median = ${median_error:,.0f})', fontsize=13, fontweight='bold')
    plt.axvline(x=median_error, color='r', linestyle='--', lw=2, label=f'Median = ${median_error:,.0f}')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "stage2_error_distribution.png", dpi=300)
    plt.close()
    print(f"\n3️⃣ Absolute Error Distribution:")
    print(f"   ✅ Saved: stage2_error_distribution.png")

    # =====================================================================
    # 6. OUTLIER DETECTION
    # =====================================================================
    Q1 = np.percentile(abs_errors, 25)
    Q3 = np.percentile(abs_errors, 75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    outliers = (abs_errors > outlier_threshold).sum()

    print(f"\n4️⃣ Outlier Analysis (IQR Method):")
    print(f"   Q1 (25th %ile):  ${Q1:,.2f}")
    print(f"   Q3 (75th %ile):  ${Q3:,.2f}")
    print(f"   IQR:             ${IQR:,.2f}")
    print(f"   Outlier threshold: ${outlier_threshold:,.2f}")
    print(f"   Number of outliers: {outliers:,} ({100*outliers/len(abs_errors):.1f}%)")

    # Visualize outliers
    plt.figure(figsize=(10, 6))
    is_outlier = abs_errors > outlier_threshold
    plt.scatter(range(len(abs_errors)), abs_errors, alpha=0.5, s=20, label='Normal', color='#1f77b4')
    plt.scatter(np.where(is_outlier)[0], abs_errors[is_outlier], alpha=0.7, s=50,
                label='Outliers', color='red', marker='X')
    plt.axhline(y=outlier_threshold, color='r', linestyle='--', lw=2, label=f'Threshold = ${outlier_threshold:,.0f}')
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Absolute Error ($)', fontsize=11)
    plt.title('Stage 2: Outlier Detection (IQR Method)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "stage2_outliers.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved: stage2_outliers.png")

    # Top 10 worst predictions
    worst_idx = np.argsort(abs_errors)[-10:]
    print(f"\n5️⃣ Top 10 Worst Predictions:")
    for i, idx in enumerate(worst_idx[::-1], 1):
        print(f"   {i}. Actual: ${y_test_actual[idx]:,.2f} | Predicted: ${y_pred_actual[idx]:,.2f} | Error: ${abs_errors[idx]:,.2f}")


# =========================================================================
# MAIN VALIDATION RUNNER
# =========================================================================
def main():
    print("\n" + "="*70)
    print("🔍 COMPREHENSIVE HURDLE MODEL VALIDATION")
    print("="*70)

    # Setup — validate.py is in /src/, go up to project root
    project_root = Path(__file__).resolve().parent.parent
    output_folder = project_root / "outputs" / "figures"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load data and models
    print("\n📥 Loading data and models...")
    zip_path = str(project_root / "data" / "processed" / "modeling" / "modeling_matrices_package.zip")
    matrices = load_modeling_packages(zip_path)
    X_test_s1 = matrices["X_test_unfiltered"].reset_index(drop=True)
    y_test_s1 = matrices["y_test_stage1"].values.ravel()

    X_test_s2 = matrices["X_test_stage2"].reset_index(drop=True)
    y_test_s2_raw = pd.Series(matrices["y_test_stage2_raw"].values.ravel()).reset_index(drop=True)

    clf = joblib.load(project_root / "models" / "stage1_classifier.joblib")
    reg = joblib.load(project_root / "models" / "stage2_regressor.joblib")
    threshold = load_threshold()

    # Align test data to match model's expected features
    print("   Aligning features to model specifications...")
    model_features_s1 = clf.feature_names_in_
    X_test_s1 = X_test_s1.reindex(columns=model_features_s1, fill_value=0)

    model_features_s2 = reg.feature_names_in_
    X_test_s2 = X_test_s2.reindex(columns=model_features_s2, fill_value=0)

    # Get Stage 1 probabilities for later use
    y_prob_s1 = clf.predict_proba(X_test_s1)[:, 1]

    # Run validations
    validate_stage1(clf, X_test_s1, y_test_s1, threshold, output_folder)
    validate_stage2(reg, X_test_s2, y_test_s2_raw, threshold, y_prob_s1, output_folder)

    print("\n" + "="*70)
    print("✨ VALIDATION COMPLETE!")
    print(f"📁 All plots saved to: outputs/figures/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()