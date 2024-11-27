"""Script for creating comprehensive visualizations of model results."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def construct_undefined(self, node):
        if isinstance(node.value, (int, float, str)):
            return node.value
        return None


class VisualizationCreator:
    def __init__(self, results_path: Path, refined_results_path: Path = None):
        self.results_path = results_path
        self.refined_results_path = refined_results_path
        self.output_dir = Path('figures/final_analysis')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load results
        try:
            with open(results_path) as f:
                self.results = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            self.results = {}

        self.refined_results = None
        if refined_results_path and refined_results_path.exists():
            try:
                with open(refined_results_path) as f:
                    self.refined_results = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
            except Exception as e:
                logger.error(f"Error loading refined results: {e}")

    def _convert_value(self, value):
        """Convert any numpy/non-standard types to Python native types."""
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._convert_value(item) for item in value]
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def plot_model_comparison(self):
        """Create model performance comparison visualizations."""
        # Extract and convert metrics
        data = []

        for model_type in ['baseline', 'advanced']:
            if model_type in self.results:
                for model_name, metrics in self.results[model_type].items():
                    # Add metrics to data
                    data.append({
                        'Model': model_name,
                        'Category': model_type,
                        'R²': self._convert_value(metrics.get('r2', 0)),
                        'RMSE': self._convert_value(metrics.get('rmse', 0)),
                        'MAE': self._convert_value(metrics.get('mae', 0))
                    })

        if not data:
            logger.warning("No model results to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(data)

        # Create performance comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # R² scores
        sns.barplot(data=df, x='Model', y='R²', hue='Category', ax=ax1)
        ax1.set_title('Model Comparison - R² Score')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # RMSE scores
        sns.barplot(data=df, x='Model', y='RMSE', hue='Category', ax=ax2)
        ax2.set_title('Model Comparison - RMSE')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # MAE scores
        sns.barplot(data=df, x='Model', y='MAE', hue='Category', ax=ax3)
        ax3.set_title('Model Comparison - MAE')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self):
        """Create feature importance visualizations."""
        importance_data = {}
        for model_type in ['baseline', 'advanced']:
            for model_name, model_results in self.results[model_type].items():
                if 'feature_importances' in model_results:
                    importance_data[f"{model_type}_{model_name}"] = model_results[
                        'feature_importances']

        if importance_data:
            for model, importances in importance_data.items():
                # Sort importances
                sorted_importances = dict(
                    sorted(importances.items(), key=lambda x: x[1], reverse=True))

                # Create bar plot
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(sorted_importances)), list(sorted_importances.values()))
                plt.xticks(range(len(sorted_importances)), list(sorted_importances.keys()),
                           rotation=45)
                plt.title(f'Feature Importance - {model}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'feature_importance_{model}.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

    def plot_learning_curves(self):
        """Create learning curve visualizations."""
        if self.results.get('learning_curves'):
            for model, curves in self.results['learning_curves'].items():
                plt.figure(figsize=(10, 6))
                plt.plot(curves['train_scores'], label='Training Score')
                plt.plot(curves['test_scores'], label='Validation Score')
                plt.xlabel('Training Examples')
                plt.ylabel('Score')
                plt.title(f'Learning Curve - {model}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'learning_curve_{model}.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

    def plot_prediction_analysis(self):
        """Create prediction analysis visualizations."""
        if 'predictions' in self.results:
            for model, pred_data in self.results['predictions'].items():
                # Actual vs Predicted scatter plot
                plt.figure(figsize=(8, 8))
                plt.scatter(pred_data['actual'], pred_data['predicted'], alpha=0.5)
                plt.plot([min(pred_data['actual']), max(pred_data['actual'])],
                         [min(pred_data['actual']), max(pred_data['actual'])],
                         'r--')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Actual vs Predicted - {model}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'actual_vs_predicted_{model}.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

                # Residual plot
                residuals = np.array(pred_data['predicted']) - np.array(pred_data['actual'])
                plt.figure(figsize=(8, 6))
                plt.scatter(pred_data['predicted'], residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title(f'Residual Plot - {model}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'residuals_{model}.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

    def plot_refinement_comparison(self):
        """Compare original and refined model results."""
        if self.refined_results:
            # Extract metrics
            models = []
            original_r2 = []
            refined_r2 = []

            for model_type in ['baseline', 'advanced']:
                for model_name in self.results[model_type].keys():
                    if model_name in self.refined_results[model_type]:
                        models.append(model_name)
                        original_r2.append(self.results[model_type][model_name].get('r2', 0))
                        refined_r2.append(self.refined_results[model_type][model_name].get('r2', 0))

            # Create comparison plot
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(np.arange(len(models)) - width / 2, original_r2, width, label='Original')
            ax.bar(np.arange(len(models)) + width / 2, refined_r2, width, label='Refined')

            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance: Original vs Refined')
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(models, rotation=45)
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / 'refinement_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_all_visualizations(self):
        """Create all visualizations."""
        logger.info("Creating model comparison plots...")
        self.plot_model_comparison()

        logger.info("Creating feature importance plots...")
        self.plot_feature_importance()

        logger.info("Creating learning curve plots...")
        self.plot_learning_curves()

        logger.info("Creating prediction analysis plots...")
        self.plot_prediction_analysis()

        if self.refined_results:
            logger.info("Creating refinement comparison plots...")
            self.plot_refinement_comparison()

        logger.info("All visualizations created successfully!")


def main():
    """Main execution function."""
    try:
        results_path = Path('models/training_results.yaml')
        refined_results_path = Path('models/refined_results.yaml')

        visualizer = VisualizationCreator(results_path, refined_results_path)
        visualizer.create_all_visualizations()

        logger.info(f"Visualizations saved to {visualizer.output_dir}")

    except Exception as e:
        logger.error(f"Visualization creation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
