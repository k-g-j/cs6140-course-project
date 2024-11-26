"""Main training script for renewable energy prediction models."""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.models.advanced_models import AdvancedModels
from src.models.baseline_models import BaselineModels

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training and evaluation pipeline."""

    def __init__(self, config_path: str):
        """Initialize ModelTrainer with configuration."""
        logger.info(f"Loading configuration from {config_path}")
        self.config = self._load_config(config_path)

        # Get data directory, trying both processed_dir and output_dir
        data_paths = self.config.get('data_paths', {})
        if 'processed_dir' in data_paths:
            self.data_dir = Path(data_paths['processed_dir'])
        elif 'output_dir' in data_paths:
            self.data_dir = Path(data_paths['output_dir'])
        else:
            # Default to 'processed_data' if neither is specified
            self.data_dir = Path('processed_data')
            logger.warning("No data directory specified in config, using default: processed_data")

        # Get models directory
        model_paths = self.config.get('model_paths', {})
        self.models_dir = Path(model_paths.get('output_dir', 'models'))

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using data directory: {self.data_dir}")
        logger.info(f"Using models directory: {self.models_dir}")

        # Initialize models
        self.baseline_models = BaselineModels(self.config.get('baseline_models', {}))
        self.advanced_models = AdvancedModels(self.config.get('advanced_models', {}))

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Empty configuration file")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise

    def prepare_time_series(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Prepare time series data for ARIMA model.

        Args:
            X: DataFrame containing features including time column
            y: Target variable series

        Returns:
            pd.Series: Time series data resampled to yearly frequency
        """
        time_col = self.config['training'].get('time_column', 'year')

        if time_col not in X.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")

        try:
            # First normalize the years to a proper range
            years = X[time_col].values
            min_year = int(min(years))

            # Map years to a sequence starting from 2000 (arbitrary base year)
            year_mapping = {year: 2000 + idx for idx, year in enumerate(sorted(set(years)))}
            normalized_years = pd.Series(years).map(year_mapping)

            # Create datetime index from normalized years
            dates = pd.to_datetime(normalized_years.astype(int).astype(str), format='%Y')

            # Create time series with datetime index
            ts_data = pd.Series(y.values, index=dates)

            # Sort index and resample to yearly frequency
            # Using 'YE' (Year End) instead of deprecated 'Y'
            ts_data = ts_data.sort_index()
            yearly_data = ts_data.resample('YE').mean()

            logger.info(
                f"Created time series with {len(yearly_data)} points from {yearly_data.index.min().year} to {yearly_data.index.max().year}")
            return yearly_data

        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            raise

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> Dict:
        """
        Train all models and evaluate their performance.
        """
        try:
            metrics = {}

            # Train baseline models
            logger.info("Training baseline models...")
            time_col = self.config['training'].get('time_column', 'year')
            feature_cols = [col for col in X_train.columns if col != time_col]

            # Train linear models
            logger.info("Training linear models...")
            self.baseline_models.train_linear_models(X_train[feature_cols], y_train)

            # ARIMA model training
            if self.config['training'].get('train_arima', True):
                try:
                    logger.info("Training ARIMA model...")

                    # Prepare time series data
                    yearly_data = self.prepare_time_series(X_train, y_train)
                    n_points = len(yearly_data)
                    logger.info(f"Training ARIMA on {n_points} time points")

                    # Get ARIMA order from config and train
                    arima_order = tuple(
                        self.config['baseline_models'].get('arima_order', [1, 1, 1]))
                    logger.info(f"Using ARIMA order {arima_order}")

                    self.baseline_models.train_arima(yearly_data, order=arima_order)
                    logger.info("Successfully trained ARIMA model")

                except Exception as e:
                    logger.warning(f"Could not train ARIMA model: {str(e)}")
                    logger.warning("Continuing without ARIMA model...")

            # Evaluate baseline models
            logger.info("Evaluating baseline models...")
            baseline_metrics = self.baseline_models.evaluate_models(X_test[feature_cols], y_test)
            metrics['baseline'] = baseline_metrics

            # Train advanced models
            logger.info("Training advanced models...")
            tune_hyperparameters = self.config['training'].get('tune_hyperparameters', True)

            # Random Forest
            logger.info("Training Random Forest...")
            self.advanced_models.train_random_forest(
                X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained Random Forest model")

            # Gradient Boosting
            logger.info("Training Gradient Boosting...")
            self.advanced_models.train_gradient_boosting(
                X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained Gradient Boosting model")

            # SVR
            logger.info("Training SVR...")
            self.advanced_models.train_svr(
                X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained SVR model")

            # Evaluate advanced models
            logger.info("Evaluating advanced models...")
            advanced_metrics = self.advanced_models.evaluate_models(X_test[feature_cols], y_test)
            metrics['advanced'] = advanced_metrics

            # Log best models based on R2 score
            self._log_best_models(metrics)

            # Save model metrics
            logger.info("Saving training results...")
            self.save_results(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            raise

    def _log_best_models(self, metrics: Dict):
        """Log the best performing models based on R2 score."""
        # Combine all model metrics
        all_models = {}
        for model_type in metrics:
            for model_name, model_metrics in metrics[model_type].items():
                if 'r2' in model_metrics:
                    all_models[f"{model_type}_{model_name}"] = model_metrics['r2']

        # Sort models by R2 score
        sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)

        # Log results
        logger.info("\nModel Performance Summary (R² Score):")
        logger.info("-" * 40)
        for model_name, r2_score in sorted_models:
            logger.info(f"{model_name:30s}: {r2_score:.4f}")
        logger.info("-" * 40)

    def save_results(self, metrics: Dict):
        """Save training results to YAML file."""
        results_path = self.models_dir / 'training_results.yaml'
        logger.info(f"Saving training results to {results_path}")

        # Convert numpy values to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_metrics = convert_to_native(metrics)

        with open(results_path, 'w') as f:
            yaml.dump(serializable_metrics, f, default_flow_style=False)


def main():
    try:
        # Load configuration
        config_path = 'config.yaml'
        trainer = ModelTrainer(str(config_path))

        # Load data
        df = pd.read_csv('processed_data/final_processed_data.csv')

        # Get features and target
        target_col = trainer.config['training']['target_column']
        feature_cols = trainer.config['training']['feature_columns']
        time_col = trainer.config['training'].get('time_column', 'year')
        all_features = feature_cols + [time_col]

        X = df[all_features]
        y = df[target_col]

        # Handle missing values before splitting
        logger.info("Preprocessing data...")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)

        # Split data
        test_size = trainer.config['training'].get('test_size', 0.2)
        random_state = trainer.config['training'].get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info("Training set shape: {}".format(X_train.shape))
        logger.info("Test set shape: {}".format(X_test.shape))

        # Train models
        logger.info("Starting model training...")
        metrics = trainer.train_models(X_train, X_test, y_train, y_test)

        # Save results
        trainer.save_results(metrics)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
