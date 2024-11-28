"""Script to run ablation studies on trained models."""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.models.ablation_studies import AblationStudy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    try:
        # Initialize ablation study
        config_path = Path('config.yaml')
        ablation = AblationStudy(str(config_path))

        # Load your processed data
        data_path = Path('processed_data/final_processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Print available columns
        logger.info("Available columns in data:")
        logger.info(list(df.columns))

        # Get features and target from config
        target_col = ablation.config['training']['target_column']
        feature_cols = ablation.config['training']['feature_columns']

        logger.info(f"Target column from config: {target_col}")
        logger.info(f"Feature columns from config: {feature_cols}")

        # Basic validation
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            # Print the first few rows of data to help debug
            logger.info("First few rows of data:")
            logger.info(df.head())
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare features and target
        X = df[feature_cols]
        y = df[target_col]

        # Run all ablation studies
        logger.info("Starting ablation studies...")
        ablation.run_all_studies(X, y)

        logger.info("Ablation studies completed successfully!")

    except Exception as e:
        logger.error(f"Error in ablation studies: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
