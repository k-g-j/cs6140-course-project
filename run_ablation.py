"""Script to run ablation studies on trained models."""

import logging
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

        # Get features and target from config
        target_col = ablation.config['training']['target_column']
        feature_cols = ablation.config['training']['feature_columns']
        time_col = ablation.config['training'].get('time_column', 'year')

        # Include time column in features
        all_features = feature_cols + [time_col]

        # Prepare features and target
        X = df[all_features]
        y = df[target_col]

        # Run all ablation studies
        logger.info("Starting ablation studies...")
        ablation.run_all_studies(X, y)

        logger.info("Ablation studies completed successfully!")

    except Exception as e:
        logger.error(f"Error in ablation studies: {str(e)}")
        raise


if __name__ == "__main__":
    main()
