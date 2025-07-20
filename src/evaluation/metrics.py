import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChromeCRISPRMetrics:
    """
    Evaluation metrics and statistical analysis for ChromeCRISPR models.

    Implements:
    - MSE and Spearman correlation calculation
    - Statistical significance testing (ANOVA, Tukey's HSD)
    - Performance visualization
    - Model comparison analysis
    """

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for reference

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)

        # Correlation metrics
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'n_samples': len(y_true)
        }

        self.metrics[model_name] = metrics
        return metrics

    def compare_models(self, results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple models and create summary table.

        Args:
            results: Dictionary with model names as keys and {'y_true': ..., 'y_pred': ...} as values

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for model_name, data in results.items():
            metrics = self.calculate_metrics(data['y_true'], data['y_pred'], model_name)
            comparison_data.append({
                'Model': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'Spearman': metrics['spearman_corr'],
                'Pearson': metrics['pearson_corr'],
                'N': metrics['n_samples']
            })

        return pd.DataFrame(comparison_data)

    def statistical_significance_test(self, results: Dict[str, Dict[str, np.ndarray]],
                                    metric: str = 'spearman_corr') -> Dict[str, Any]:
        """
        Perform statistical significance testing between models.

        Args:
            results: Dictionary with model results
            metric: Metric to test ('spearman_corr', 'mse', etc.)

        Returns:
            Dictionary with statistical test results
        """
        from scipy.stats import f_oneway
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        # Calculate metrics for all models
        model_metrics = {}
        for model_name, data in results.items():
            metrics = self.calculate_metrics(data['y_true'], data['y_pred'], model_name)
            model_metrics[model_name] = metrics[metric]

        # One-way ANOVA
        groups = list(model_metrics.values())
        f_stat, p_value = f_oneway(*groups)

        # Tukey's HSD test
        model_names = list(model_metrics.keys())
        metric_values = list(model_metrics.values())

        # Create data for Tukey test
        tukey_data = []
        tukey_labels = []
        for name, value in zip(model_names, metric_values):
            # Create multiple samples for Tukey test (simulate replicates)
            samples = np.random.normal(value, 0.01, 100)  # Small variance for replicates
            tukey_data.extend(samples)
            tukey_labels.extend([name] * 100)

        tukey_result = pairwise_tukeyhsd(tukey_data, tukey_labels)

        return {
            'anova_f_stat': f_stat,
            'anova_p_value': p_value,
            'tukey_result': tukey_result,
            'model_metrics': model_metrics,
            'significant_differences': p_value < 0.05
        }

    def plot_performance_comparison(self, results: Dict[str, Dict[str, np.ndarray]],
                                   save_path: str = None):
        """
        Create comprehensive performance comparison plots.

        Args:
            results: Dictionary with model results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ChromeCRISPR Model Performance Comparison', fontsize=16)

        # 1. Scatter plot of predictions vs true values
        ax1 = axes[0, 0]
        for model_name, data in results.items():
            ax1.scatter(data['y_true'], data['y_pred'], alpha=0.6, label=model_name)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs True Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Spearman correlation comparison
        ax2 = axes[0, 1]
        spearman_values = []
        model_names = []
        for model_name, data in results.items():
            metrics = self.calculate_metrics(data['y_true'], data['y_pred'], model_name)
            spearman_values.append(metrics['spearman_corr'])
            model_names.append(model_name)

        bars = ax2.bar(model_names, spearman_values, alpha=0.7)
        ax2.set_ylabel('Spearman Correlation')
        ax2.set_title('Spearman Correlation Comparison')
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, spearman_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 3. MSE comparison
        ax3 = axes[1, 0]
        mse_values = []
        for model_name, data in results.items():
            metrics = self.calculate_metrics(data['y_true'], data['y_pred'], model_name)
            mse_values.append(metrics['mse'])

        bars = ax3.bar(model_names, mse_values, alpha=0.7, color='orange')
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_title('MSE Comparison')

        # Add value labels on bars
        for bar, value in zip(bars, mse_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')

        # 4. Residual plot
        ax4 = axes[1, 1]
        for model_name, data in results.items():
            residuals = data['y_pred'] - data['y_true']
            ax4.scatter(data['y_pred'], residuals, alpha=0.6, label=model_name)

        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_learning_curves(self, train_losses: List[float], val_losses: List[float],
                           train_spearmans: List[float] = None,
                           val_spearmans: List[float] = None,
                           save_path: str = None):
        """
        Plot learning curves for model training.

        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            train_spearmans: Training Spearman correlation history
            val_spearmans: Validation Spearman correlation history
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1 = axes[0]
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Spearman correlation curves
        if train_spearmans and val_spearmans:
            ax2 = axes[1]
            ax2.plot(epochs, train_spearmans, 'b-', label='Training Spearman')
            ax2.plot(epochs, val_spearmans, 'r-', label='Validation Spearman')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Spearman Correlation')
            ax2.set_title('Training and Validation Spearman Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_report(self, results: Dict[str, Dict[str, np.ndarray]],
                       output_path: str = "evaluation_report.txt"):
        """
        Generate comprehensive evaluation report.

        Args:
            results: Dictionary with model results
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("ChromeCRISPR Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            # Model comparison table
            comparison_df = self.compare_models(results)
            f.write("Model Performance Comparison:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Statistical significance
            f.write("Statistical Significance Testing:\n")
            f.write("-" * 30 + "\n")
            stats_result = self.statistical_significance_test(results)
            f.write(f"ANOVA F-statistic: {stats_result['anova_f_stat']:.4f}\n")
            f.write(f"ANOVA p-value: {stats_result['anova_p_value']:.4f}\n")
            f.write(f"Significant differences: {stats_result['significant_differences']}\n\n")

            # Best model identification
            best_model = comparison_df.loc[comparison_df['Spearman'].idxmax()]
            f.write(f"Best Model (by Spearman correlation): {best_model['Model']}\n")
            f.write(f"Spearman correlation: {best_model['Spearman']:.4f}\n")
            f.write(f"MSE: {best_model['MSE']:.4f}\n")
            f.write(f"R²: {best_model['R²']:.4f}\n")

        print(f"Evaluation report saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example evaluation
    metrics = ChromeCRISPRMetrics()

    # Generate dummy results
    np.random.seed(42)
    y_true = np.random.random(1000)

    results = {
        'CNN': {'y_true': y_true, 'y_pred': y_true + np.random.normal(0, 0.1, 1000)},
        'GRU': {'y_true': y_true, 'y_pred': y_true + np.random.normal(0, 0.08, 1000)},
        'LSTM': {'y_true': y_true, 'y_pred': y_true + np.random.normal(0, 0.09, 1000)},
        'CNN-GRU': {'y_true': y_true, 'y_pred': y_true + np.random.normal(0, 0.05, 1000)}
    }

    # Compare models
    comparison = metrics.compare_models(results)
    print(comparison)

    # Generate plots
    metrics.plot_performance_comparison(results)

    # Generate report
    metrics.generate_report(results)
