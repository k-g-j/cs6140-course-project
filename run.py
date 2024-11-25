"""Main script to run model training pipeline."""
import logging

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
        df = pd.read_csv('processed_data/final_processed_data.csv')

        # Get features and target
        target_col = trainer.config['training']['target_column']
        feature_cols = trainer.config['training']['feature_columns']
        time_col = trainer.config['training'].get('time_column', 'year')

        # Preprocess data
        logger.info("Preprocessing data...")
        X, y = preprocess_data(df, target_col, feature_cols, time_col)

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
        raise


if __name__ == "__main__":
    main()
