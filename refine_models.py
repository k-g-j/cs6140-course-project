"""Script for refining models based on ablation study results."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.train import ModelTrainer

logger = logging.getLogger(__name__)


def _convert_to_native(obj):
    """Convert numpy types to native Python types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ModelRefiner:
    """Class to handle model refinement and retraining."""

    def __init__(self, config_path: Path):
        """Initialize ModelRefiner."""
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load ablation results if they exist
        self.ablation_path = Path('figures/ablation_studies/ablation_study_report.txt')
        if self.ablation_path.exists():
            with open(self.ablation_path) as f:
                self.ablation_results = f.read()
        else:
            self.ablation_results = None

        # Initialize preprocessors
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Preprocess data by handling missing values and scaling."""
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
        rf_params = {
            'rf_n_estimators': 300,
            'rf_max_depth': 15,
            'rf_min_samples_split': 2
        }

        gb_params = {
            'gb_n_estimators': 300,
            'gb_max_depth': 5,
            'gb_learning_rate': 0.1
        }

        # Update config
        self.config['advanced_models'] = {
            **rf_params,
            **gb_params
        }

    def refine_feature_selection(self):
        """Refine feature selection based on importance analysis."""
        if 'training' not in self.config:
            self.config['training'] = {}

        # Update feature columns to match actual data
        self.config['training']['feature_columns'] = [
            'Hydroelectric Power',
            'Solar Energy',
            'Wind Energy',
            'Geothermal Energy',
            'Biomass Energy'
        ]

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

            # Convert numpy types to Python natives for YAML serialization
            serializable_metrics = _convert_to_native(metrics)

            # Save results
            refined_results_path = Path('models/refined_results.yaml')
            with open(refined_results_path, 'w') as f:
                yaml.dump(serializable_metrics, f, default_flow_style=False)

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
        data_path = Path('processed_data/final_processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find processed data at {data_path}")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Get features and target
        feature_cols = ['Hydroelectric Power', 'Solar Energy', 'Wind Energy',
                        'Geothermal Energy', 'Biomass Energy']
        target_col = 'Total Renewable Energy'

        # Basic validation
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info("Available columns:")
            logger.info(list(df.columns))
            raise ValueError(f"Missing required columns: {missing_cols}")

        X = df[feature_cols]
        y = df[target_col]

        # Retrain models
        refined_metrics = refiner.retrain_models(X, y)

        logger.info("Model refinement completed successfully!")

    except Exception as e:
        logger.error(f"Model refinement failed: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
