"""Main script for data processing pipeline."""
import logging
import sys
from pathlib import Path


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to get more detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log', mode='w'),  # 'w' mode overwrites the file
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test logging configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")
    return logger


def main():
    """Main execution function."""
    # Set up logging first
    logger = setup_logging()

    try:
        # Import after logging is configured
        from data.load_data import (
            load_global_energy_data,
            load_worldwide_renewable_data,
            load_weather_conditions,
            load_us_renewable_data
        )
        from data.preprocess import DataPreprocessor
        from data.feature_engineering import FeatureEngineer

        logger.info("Starting data processing pipeline...")

        # Create necessary directories
        Path('processed_data').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)

        # Load configuration
        config_path = Path('config.yaml')
        if not config_path.exists():
            logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        # Initialize pipeline components
        data_dir = Path('data')
        logger.info(f"Loading data from {data_dir}")

        # Load datasets
        logger.info("Loading global energy data...")
        global_data = load_global_energy_data(data_dir)

        logger.info("Loading worldwide renewable data...")
        worldwide_data = load_worldwide_renewable_data(data_dir)

        logger.info("Loading weather conditions data...")
        weather_data = load_weather_conditions(data_dir)

        logger.info("Loading US renewable data...")
        us_data = load_us_renewable_data(data_dir)

        logger.info("All data loaded successfully")

        # Initialize preprocessor and feature engineer
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()

        # Start processing steps...
        logger.info("Data processing steps would continue here...")

        logger.info("Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
