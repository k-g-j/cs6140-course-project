import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.advanced_models import AdvancedModels
from src.models.baseline_models import BaselineModels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training and evaluation pipeline."""

    def __init__(self, config_path: str):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_paths']['processed_dir'])
        self.models_dir = Path(self.config['model_paths']['output_dir'])

        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_models = BaselineModels(self.config.get('baseline_models', {}))
        self.advanced_models = AdvancedModels(self.config.get('advanced_models', {}))

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_data(self) -> pd.DataFrame:
        """
        Load processed data for training.

        Returns:
            Processed DataFrame
        """
        data_path = self.data_dir / 'final_processed_data.csv'
        logger.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting features and target.

        Args:
            df: Input DataFrame

        Returns:
            Training and testing splits (X_train, X_test, y_train, y_test)
        """
        try:
            # Get target and feature columns from config
            target_col = self.config['training']['target_column']
            feature_cols = self.config['training']['feature_columns']
            time_col = self.config['training'].get('time_column', 'year')

            logger.info(f"Target column: {target_col}")
            logger.info(f"Feature columns: {feature_cols}")
            logger.info(f"Time column: {time_col}")

            # Verify columns exist in dataframe
            missing_cols = [col for col in feature_cols + [target_col, time_col] if
                            col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in dataset: {missing_cols}")

            # Split features and target
            X = df[feature_cols + [time_col]].copy()  # Include time column
            y = df[target_col].copy()

            # Display initial data info
            logger.info(f"Initial shapes - X: {X.shape}, y: {y.shape}")
            logger.info(f"Data types:\n{X.dtypes}")

            # Handle missing values
            logger.info("Handling missing values...")

            # Get column types
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = X.select_dtypes(exclude=['float64', 'int64']).columns

            # Handle time column first
            if time_col in numeric_cols:
                logger.info(f"Processing time column: {time_col}")
                try:
                    # Create proper year values and ensure they're numeric
                    years = pd.to_numeric(df[time_col], errors='coerce')
                    years = years.fillna(years.median())  # Fill any NaN values with median
                    unique_years = sorted(years.unique())

                    logger.info(f"Original year range: {min(unique_years)} to {max(unique_years)}")

                    # Create mapping from year to index
                    year_mapping = {year: idx for idx, year in enumerate(unique_years)}

                    # Map the year values
                    X[time_col] = X[time_col].map(year_mapping).fillna(
                        0)  # Use 0 for any unmapped values

                    logger.info(f"Processed year range: {X[time_col].min()} to {X[time_col].max()}")

                except Exception as e:
                    logger.error(f"Error processing time column: {str(e)}")
                    raise

            # Handle missing values in other numeric columns
            for col in numeric_cols:
                if col != time_col:
                    missing = X[col].isna().sum()
                    if missing > 0:
                        logger.info(f"Filling {missing} missing values in {col}")
                        X[col] = X[col].fillna(X[col].median())

            # Handle missing values in categorical columns
            for col in categorical_cols:
                missing = X[col].isna().sum()
                if missing > 0:
                    logger.info(f"Filling {missing} missing values in {col}")
                    mode_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                    X[col] = X[col].fillna(mode_value)

            # Remove any rows where target is NaN
            initial_rows = len(y)
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            dropped_rows = initial_rows - len(y)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing target values")

            logger.info(f"After handling missing values - X shape: {X.shape}")

            # Split into train and test sets
            test_size = self.config['training'].get('test_size', 0.2)
            random_state = self.config['training'].get('random_state', 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=None  # Add stratification if needed
            )

            # Scale features if specified
            if self.config['training'].get('scale_features', True):
                logger.info("Scaling features...")
                scaler = StandardScaler()

                # Exclude time column from scaling
                features_to_scale = [col for col in X_train.columns if col != time_col]
                logger.info(f"Scaling columns: {features_to_scale}")

                # Scale features
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train[features_to_scale]),
                    columns=features_to_scale,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test[features_to_scale]),
                    columns=features_to_scale,
                    index=X_test.index
                )

                # Add back the time column
                X_train_scaled[time_col] = X_train[time_col]
                X_test_scaled[time_col] = X_test[time_col]

                X_train = X_train_scaled
                X_test = X_test_scaled

                logger.info("Feature scaling completed")

            logger.info(f"Final shapes - Training: {X_train.shape}, Testing: {X_test.shape}")

            # Verify data quality
            assert not X_train.isna().any().any(), "Missing values found in X_train"
            assert not X_test.isna().any().any(), "Missing values found in X_test"
            assert not y_train.isna().any(), "Missing values found in y_train"
            assert not y_test.isna().any(), "Missing values found in y_test"

            logger.info("Data preparation completed successfully")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> Dict:
        """
        Train all models and evaluate their performance.
        """
        metrics = {}

        # Train baseline models
        logger.info("Training baseline models...")
        time_col = self.config['training'].get('time_column', 'year')
        feature_cols = [col for col in X_train.columns if col != time_col]

        # Train linear models using only feature columns
        logger.info("Training linear models...")
        self.baseline_models.train_linear_models(X_train[feature_cols], y_train)

        # ARIMA model training
        if self.config['training'].get('train_arima', True):
            try:
                logger.info("Training ARIMA model...")
                # Group data by year and calculate mean values
                time_series_data = pd.DataFrame({
                    'year': X_train[time_col],
                    'value': y_train
                }).groupby('year')['value'].mean()

                # Create proper datetime index
                time_series_data.index = pd.date_range(
                    start=f"{int(time_series_data.index.min())}-01-01",
                    end=f"{int(time_series_data.index.max())}-12-31",
                    freq='YE'
                )[:len(time_series_data)]

                # Train ARIMA model
                logger.info(
                    f"Training ARIMA on {len(time_series_data)} time points from {time_series_data.index.min().year} to {time_series_data.index.max().year}")
                arima_order = tuple(self.config['baseline_models'].get('arima_order', [1, 1, 1]))
                self.baseline_models.train_arima(time_series_data, order=arima_order)
                logger.info("Successfully trained ARIMA model")

            except Exception as e:
                logger.warning(f"Could not train ARIMA model: {str(e)}")
                logger.warning("Continuing without ARIMA model...")

        # Evaluate baseline models
        baseline_metrics = self.baseline_models.evaluate_models(X_test[feature_cols], y_test)
        metrics['baseline'] = baseline_metrics

        # Train advanced models
        logger.info("Training advanced models...")
        tune_hyperparameters = self.config['training'].get('tune_hyperparameters', True)

        self.advanced_models.train_random_forest(
            X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
        )
        logger.info("Trained Random Forest model")

        self.advanced_models.train_gradient_boosting(
            X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
        )
        logger.info("Trained Gradient Boosting model")

        self.advanced_models.train_svr(
            X_train[feature_cols], y_train, tune_hyperparameters=tune_hyperparameters
        )
        logger.info("Trained SVR model")

        advanced_metrics = self.advanced_models.evaluate_models(X_test[feature_cols], y_test)
        metrics['advanced'] = advanced_metrics

        return metrics

    def save_results(self, metrics: Dict) -> None:
        """
        Save training results and metrics.

        Args:
            metrics: Dictionary of model metrics
        """
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

    def run_training_pipeline(self) -> None:
        """Execute the complete training pipeline."""
        try:
            logger.info("Starting model training pipeline...")

            # Load and prepare data
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(df)

            # Train and evaluate models
            metrics = self.train_models(X_train, X_test, y_train, y_test)

            # Save results
            self.save_results(metrics)

            logger.info("Model training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        # Load configuration
        config_path = Path('config.yaml')

        # Initialize and run training pipeline
        trainer = ModelTrainer(str(config_path))

        # Train models and get data splits
        X_train, X_test, y_train, y_test = trainer.prepare_data(trainer.load_data())
        metrics = trainer.train_models(X_train, X_test, y_train, y_test)

        # Save results before evaluation
        trainer.save_results(metrics)

        # Run evaluation
        from src.models.evaluate import ModelEvaluator
        evaluator = ModelEvaluator(str(config_path))
        evaluator.run_evaluation(X_test, y_test, {
            'baseline': trainer.baseline_models.models,
            'advanced': trainer.advanced_models.models
        })

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
