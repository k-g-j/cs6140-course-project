"""Script for evaluating and visualizing model performance."""
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to evaluate and visualize model performance."""

    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models_dir = Path(self.config['model_paths']['output_dir'])
        self.figures_dir = Path('figures/models')
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self) -> dict:
        """Load training results."""
        results_path = self.models_dir / 'training_results.yaml'
        with open(results_path, 'r') as f:
            return yaml.safe_load(f)

    def plot_model_comparison(self, results: dict):
        """Plot comparison of model performances."""
        metrics = ['r2', 'rmse', 'mae']
        model_types = ['baseline', 'advanced']

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            data = []
            models = []
            scores = []

            for model_type in model_types:
                for model, metrics_dict in results[model_type].items():
                    if metric in metrics_dict:
                        data.append({
                            'Model Type': model_type,
                            'Model': model,
                            'Score': metrics_dict[metric]
                        })

            df = pd.DataFrame(data)
            sns.barplot(data=df, x='Model', y='Score', hue='Model Type')
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'model_comparison_{metric}.png')
            plt.close()

    def plot_feature_importance(self, results: dict):
        """Plot feature importance for models that support it."""
        for model_type in ['advanced']:
            for model, metrics in results[model_type].items():
                if 'feature_importances' in metrics:
                    importances = pd.DataFrame({
                        'Feature': list(metrics['feature_importances'].keys()),
                        'Importance': list(metrics['feature_importances'].values())
                    }).sort_values('Importance', ascending=False)

                    plt.figure(figsize=(12, 6))
                    sns.barplot(data=importances, x='Importance', y='Feature')
                    plt.title(f'Feature Importance - {model}')
                    plt.tight_layout()
                    plt.savefig(self.figures_dir / f'feature_importance_{model}.png')
                    plt.close()

    def plot_predictions(self, X_test: pd.DataFrame, y_test: pd.Series, models: Dict):
        """Plot actual vs predicted values for each model."""
        time_col = self.config['training'].get('time_column', 'year')
        feature_cols = [col for col in X_test.columns if col != time_col]

        # Sort by time for visualization
        sorted_idx = X_test[time_col].sort_values().index
        actual_values = y_test[sorted_idx]
        times = X_test.loc[sorted_idx, time_col]

        plt.figure(figsize=(15, 8))
        plt.plot(times, actual_values, 'k-', label='Actual', linewidth=2)

        for model_type, model_dict in models.items():
            for name, model in model_dict.items():
                if name != 'arima':
                    predictions = model.predict(X_test[feature_cols].loc[sorted_idx])
                    plt.plot(times, predictions, '--', label=f'{name}', alpha=0.7)
                else:
                    # Handle ARIMA predictions differently
                    try:
                        predictions = model.forecast(steps=len(y_test))
                        plt.plot(times, predictions, '--', label='ARIMA', alpha=0.7)
                    except Exception as e:
                        logger.warning(f"Could not plot ARIMA predictions: {str(e)}")

        plt.title('Model Predictions vs Actual Values')
        plt.xlabel('Year')
        plt.ylabel('Renewable Generation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'predictions_comparison.png')
        plt.close()

        # Plot prediction error distributions
        plt.figure(figsize=(12, 6))
        for model_type, model_dict in models.items():
            for name, model in model_dict.items():
                if name != 'arima':
                    predictions = model.predict(X_test[feature_cols])
                    errors = y_test - predictions
                    sns.kdeplot(errors, label=name)

        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / 'prediction_errors.png')
        plt.close()

    def generate_prediction_metrics(self, X_test: pd.DataFrame, y_test: pd.Series, models: Dict):
        """Generate detailed prediction metrics for each model."""
        feature_cols = [col for col in X_test.columns if
                        col != self.config['training'].get('time_column', 'year')]
        metrics_report = []

        metrics_report.append("\nDetailed Prediction Metrics")
        metrics_report.append("=" * 50)

        for model_type, model_dict in models.items():
            metrics_report.append(f"\n{model_type.upper()} MODELS")
            metrics_report.append("-" * 20)

            for name, model in model_dict.items():
                if name != 'arima':
                    predictions = model.predict(X_test[feature_cols])

                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    metrics_report.extend([
                        f"\n{name}:",
                        f"  Mean Squared Error: {mse:.4f}",
                        f"  Root Mean Squared Error: {rmse:.4f}",
                        f"  Mean Absolute Error: {mae:.4f}",
                        f"  R² Score: {r2:.4f}"
                    ])

        report_path = self.models_dir / 'prediction_metrics.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(metrics_report))

    def generate_evaluation_report(self, results: dict):
        """Generate text report of model performance."""
        report = ["Model Evaluation Report", "=" * 50, ""]

        for model_type in results:
            report.append(f"\n{model_type.upper()} MODELS")
            report.append("-" * 20)

            for model, metrics in results[model_type].items():
                report.append(f"\n{model}:")
                for metric, value in metrics.items():
                    if metric != 'feature_importances':
                        report.append(f"  {metric}: {value:.4f}")

        report_path = self.models_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

    def run_evaluation(self, X_test: pd.DataFrame = None, y_test: pd.Series = None,
                       models: Dict = None):
        """Run complete evaluation pipeline."""
        try:
            logger.info("Starting model evaluation...")

            # Load results
            results = self.load_results()

            # Create visualizations
            logger.info("Generating performance comparison plots...")
            self.plot_model_comparison(results)

            logger.info("Generating feature importance plots...")
            self.plot_feature_importance(results)

            # Plot predictions if test data is provided
            if all(v is not None for v in [X_test, y_test, models]):
                logger.info("Generating prediction plots...")
                self.plot_predictions(X_test, y_test, models)
                self.generate_prediction_metrics(X_test, y_test, models)

            # Generate report
            logger.info("Generating evaluation report...")
            self.generate_evaluation_report(results)

            logger.info("Model evaluation completed successfully!")

        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        config_path = Path('config.yaml')
        evaluator = ModelEvaluator(str(config_path))
        evaluator.run_evaluation()

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
