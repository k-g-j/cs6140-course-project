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
        logging.FileHandler('logs/ablation.log'),
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

        # Load processed data
        data_path = Path('processed_data/final_processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Prepare features and target
        target_col = ablation.config['training']['target_column']
        feature_cols = ablation.config['training']['feature_columns']

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data.")
            raise ValueError(f"Target column '{target_col}' not found in data.")

        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.error(f"Missing feature columns in data: {missing_features}")
            raise ValueError(f"Missing feature columns in data: {missing_features}")

        X = df[feature_cols]
        y = df[target_col]

        # Run ablation studies
        ablation.run_all_studies(X, y)

    except Exception as e:
        logger.error(f"Ablation studies failed: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
