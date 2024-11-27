"""Script for refining models based on ablation study results."""
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRefiner:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load ablation results
        self.ablation_results = self._load_ablation_results()

        # Initialize preprocessors
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def _load_ablation_results(self):
        """Load ablation study results."""
        ablation_path = Path('figures/ablation_studies/ablation_study_report.txt')
        if not ablation_path.exists():
            logger.warning("No ablation results found")
            return None

        with open(ablation_path) as f:
            return f.read()

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Preprocess data by handling missing values and scaling features.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Tuple of (processed_X, processed_y)
        """
        logger.info("Preprocessing data...")

        # Handle missing values in features
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )

        # Handle missing values in target if any
        if y.isnull().any():
            y = pd.Series(
                self.imputer.fit_transform(y.values.reshape(-1, 1)).ravel(),
                index=y.index
            )

        return X_scaled, y

    def refine_hyperparameters(self):
        """Refine model hyperparameters based on ablation results."""
        if 'training' not in self.config:
            self.config['training'] = {}

        # Update hyperparameters based on ablation findings
        if self.ablation_results:
            # Example: If model complexity ablation showed better results with deeper trees
            rf_params = {
                'rf_n_estimators': 300,
                'rf_max_depth': 15,  # Increased from default
                'rf_min_samples_split': 2
            }

            gb_params = {
                'gb_n_estimators': 300,
                'gb_max_depth': 5,  # Adjusted based on ablation
                'gb_learning_rate': 0.1
            }

            # Update config
            self.config['advanced_models'] = {
                **rf_params,
                **gb_params
            }

    def refine_feature_selection(self):
        """Refine feature selection based on importance analysis."""
        if 'feature_engineering' not in self.config:
            self.config['feature_engineering'] = {}

        # Example: Update selected features based on importance scores
        important_features = [
            'renewable_generation_rolling_mean_3',
            'renewable_generation_lag_1',
            'renewable_generation_rolling_mean_6',
            'renewable_generation_rolling_mean_12',
            'renewable_generation_lag_3',
            'hydro_generation',
            'biofuel_generation',
            'solar_generation'
        ]

        self.config['training']['feature_columns'] = important_features

    def save_refined_config(self):
        """Save refined configuration."""
        output_path = self.config_path.parent / 'config_refined.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Refined configuration saved to {output_path}")
        return output_path

    def retrain_models(self, X: pd.DataFrame, y: pd.Series):
        """Retrain models with refined configuration."""
        try:
            # Preprocess data first
            X_processed, y_processed = self.preprocess_data(X, y)

            # Split data
            test_size = self.config['training'].get('test_size', 0.2)
            random_state = self.config['training'].get('random_state', 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed,
                test_size=test_size,
                random_state=random_state
            )

            # Initialize trainer with refined config
            trainer = ModelTrainer(str(self.config_path))

            # Train and evaluate models
            metrics = trainer.train_models(X_train, X_test, y_train, y_test)

            # Save results
            refined_results_path = Path('models/refined_results.yaml')
            with open(refined_results_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)

            logger.info(f"Refined model results saved to {refined_results_path}")
            return metrics

        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize refiner
        config_path = Path('config.yaml')
        refiner = ModelRefiner(config_path)

        # Refine hyperparameters and features
        refiner.refine_hyperparameters()
        refiner.refine_feature_selection()

        # Save refined configuration
        refined_config_path = refiner.save_refined_config()

        # Load data
        df = pd.read_csv('processed_data/final_processed_data.csv')

        # Prepare features and target
        target_col = refiner.config['training']['target_column']
        feature_cols = refiner.config['training']['feature_columns']

        X = df[feature_cols]
        y = df[target_col]

        # Retrain models
        refined_metrics = refiner.retrain_models(X, y)

        logger.info("Model refinement completed successfully!")

    except Exception as e:
        logger.error(f"Model refinement failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
