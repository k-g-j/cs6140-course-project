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

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                     y_test: pd.Series) -> Dict:
        """Train and evaluate all models."""
        try:
            metrics = {}
            time_col = self.config['training'].get('time_column', 'year')
            feature_cols = self.config['training'].get('feature_columns',
                                                       [col for col in X_train.columns if
                                                        col != time_col])

            # Train baseline models
            logger.info("Training baseline models...")

            # Train linear models
            logger.info("Training linear models...")
            self.baseline_models.train_linear_models(X_train[feature_cols], y_train)
            logger.info("Linear models training complete")

            # Train ARIMA model
            logger.info("Training ARIMA model...")

            # Create simple integer index
            index = np.arange(len(y_train))
            time_series = pd.Series(data=y_train.values, index=index, name='value')

            # Get ARIMA parameters from config
            arima_order = tuple(
                self.config.get('baseline_models', {}).get('arima_order', [1, 1, 1]))
            try:
                self.baseline_models.train_arima(time_series, order=arima_order)
                logger.info(f"ARIMA model training complete with order {arima_order}")
            except Exception as e:
                logger.error(f"ARIMA training failed: {str(e)}")

            # Evaluate baseline models
            logger.info("Evaluating baseline models...")
            baseline_metrics = self.baseline_models.evaluate_models(X_test[feature_cols], y_test)
            metrics['baseline'] = baseline_metrics
            logger.info("Baseline models evaluation complete")

            # Train advanced models
            logger.info("Training advanced models...")

            # Train Random Forest
            logger.info("Training Random Forest...")
            tune_hyperparameters = self.config['training'].get('tune_hyperparameters', True)
            self.advanced_models.train_random_forest(
                X_train[feature_cols],
                y_train,
                tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained Random Forest model")

            # Train Gradient Boosting
            logger.info("Training Gradient Boosting...")
            self.advanced_models.train_gradient_boosting(
                X_train[feature_cols],
                y_train,
                tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained Gradient Boosting model")

            # Train SVR
            logger.info("Training SVR...")
            self.advanced_models.train_svr(
                X_train[feature_cols],
                y_train,
                tune_hyperparameters=tune_hyperparameters
            )
            logger.info("Trained SVR model")

            # Evaluate advanced models
            logger.info("Evaluating advanced models...")
            advanced_metrics = self.advanced_models.evaluate_models(X_test[feature_cols], y_test)
            metrics['advanced'] = advanced_metrics
            logger.info("Advanced models evaluation complete")

            # Print model performance summary
            logger.info("\nModel Performance Summary (R² Score):")
            logger.info("-" * 40)

            # Combine all metrics
            all_models = {}

            # Add baseline metrics
            for model_name, model_metrics in baseline_metrics.items():
                if isinstance(model_metrics, dict) and 'r2' in model_metrics:
                    all_models[f"baseline_{model_name}"] = model_metrics['r2']

            # Add advanced metrics
            for model_name, model_metrics in advanced_metrics.items():
                if isinstance(model_metrics, dict) and 'r2' in model_metrics:
                    all_models[f"advanced_{model_name}"] = model_metrics['r2']

            # Sort by R² score
            sorted_models = dict(sorted(all_models.items(), key=lambda x: x[1], reverse=True))

            for model, r2 in sorted_models.items():
                logger.info(f"{model:30s}: {r2:.4f}")
            logger.info("-" * 40)

            # Save results
            logger.info("Saving training results...")
            results_path = self.models_dir / 'training_results.yaml'

            # Convert numpy values to Python native types
            serializable_metrics = self._convert_to_serializable(metrics)

            with open(results_path, 'w') as f:
                yaml.dump(serializable_metrics, f, default_flow_style=False)

            logger.info(f"Saving training results to {results_path}")

            return metrics

        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            raise

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for YAML serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

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
