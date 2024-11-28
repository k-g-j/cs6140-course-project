"""Main script to run model training pipeline."""
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.train import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, target_col: str, feature_cols: list, time_col: str):
    """Preprocess data by handling missing values and scaling features."""
    logger.info("Starting data preprocessing...")
    logger.info(f"Available columns: {list(df.columns)}")
    logger.info(f"Requested feature columns: {feature_cols}")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Time column: {time_col}")

    # Check if all required columns exist
    missing_cols = [col for col in feature_cols + [target_col, time_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Separate features and target
    X = df[feature_cols + [time_col]].copy()
    y = df[target_col].copy()

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X[feature_cols] = imputer.fit_transform(X[feature_cols])

    # Scale features
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])

    # Handle missing values in target
    y = pd.Series(
        imputer.fit_transform(y.values.reshape(-1, 1)).ravel(),
        index=y.index
    )

    return X, y


def main():
    """Main execution function."""
    try:
        # Load configuration
        config_path = 'config.yaml'
        trainer = ModelTrainer(str(config_path))

        # Load data
        data_path = Path('processed_data/final_processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Get features and target
        target_col = trainer.config['training']['target_column']
        feature_cols = trainer.config['training']['feature_columns']
        time_col = trainer.config['training'].get('time_column', 'year')

        try:
            X, y = preprocess_data(df, target_col, feature_cols, time_col)
        except ValueError as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.info("Available columns in data:")
            logger.info(list(df.columns))
            raise

        # Split data
        test_size = trainer.config['training'].get('test_size', 0.2)
        random_state = trainer.config['training'].get('random_state', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        # Train models
        logger.info("Starting model training...")
        metrics = trainer.train_models(X_train, X_test, y_train, y_test)

        # Save results
        trainer.save_results(metrics)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
