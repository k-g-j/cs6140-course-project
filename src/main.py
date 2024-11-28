"""Main script for data processing pipeline."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

# Global logger variable
logger = None


def setup_logging():
    """Set up logging configuration."""
    global logger

    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        filename='logs/pipeline.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        mode='a'
    )
    file_handler.setFormatter(formatter)

    # Create stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create module logger
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")

    return logger


def validate_config(config: Dict) -> None:
    """
    Validate configuration file contents.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    global logger

    # Required top-level keys
    required_keys = [
        'data_paths',
        'model_paths',
        'training',
        'preprocessing',
        'feature_engineering'
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate data paths
    required_data_paths = ['base_dir', 'processed_dir', 'output_dir']
    for path in required_data_paths:
        if path not in config['data_paths']:
            raise ValueError(f"Missing required data path: {path}")

    # Validate training configuration
    required_training = [
        'target_column',
        'feature_columns',
        'test_size',
        'random_state'
    ]
    for param in required_training:
        if param not in config['training']:
            raise ValueError(f"Missing required training parameter: {param}")

    # Validate preprocessing configuration
    required_preprocessing = [
        'handle_missing',
        'numeric_missing_strategy',
        'categorical_missing_strategy',
        'remove_outliers',
        'outlier_method'
    ]
    for param in required_preprocessing:
        if param not in config['preprocessing']:
            raise ValueError(f"Missing required preprocessing parameter: {param}")

    # Validate feature engineering configuration
    required_feature_eng = [
        'create_weather_features',
        'renewable_cols',
        'total_energy_col'
    ]
    for param in required_feature_eng:
        if param not in config['feature_engineering']:
            raise ValueError(f"Missing required feature engineering parameter: {param}")

    # Validate data types
    if not isinstance(config['training']['feature_columns'], list):
        raise ValueError("Feature columns must be a list")

    if not isinstance(config['training']['test_size'], (int, float)):
        raise ValueError("Test size must be a number")

    if not isinstance(config['training']['random_state'], int):
        raise ValueError("Random state must be an integer")

    logger.info("Configuration validation successful")


def process_us_renewable_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process US renewable energy data.

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame
    """
    global logger

    try:
        # Filter residential sector
        df_residential = df[df['Sector'] == 'Residential'].copy()

        # Keep relevant columns
        cols_to_keep = [
            'Year', 'Month',
            'Hydroelectric Power',
            'Solar Energy',
            'Wind Energy',
            'Geothermal Energy',
            'Biomass Energy',
            'Total Renewable Energy'
        ]

        df_processed = df_residential[cols_to_keep].copy()

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        df_processed[cols_to_keep] = imputer.fit_transform(df_processed[cols_to_keep])

        # Convert to float type
        for col in cols_to_keep:
            df_processed[col] = df_processed[col].astype(float)

        return df_processed

    except Exception as e:
        logger.error(f"Error processing US renewable data: {str(e)}")
        raise


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    global logger

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError("Empty configuration file")

        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def main():
    """Main execution function."""
    global logger

    try:
        # Set up logging first
        logger = setup_logging()
        logger.info("Starting data processing pipeline...")

        # Load and validate configuration
        config_path = Path('config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        config = load_config(config_path)
        validate_config(config)
        logger.info(f"Configuration loaded and validated successfully")

        # Create necessary directories
        for dir_path in ['processed_data', 'models', 'figures', 'logs']:
            Path(dir_path).mkdir(exist_ok=True)

        # Import data loading functions after logging is configured
        from data.load_data import (
            load_global_energy_data,
            load_worldwide_renewable_data,
            load_weather_conditions,
            load_us_renewable_data
        )

        # Load datasets
        data_dir = Path(config['data_paths']['base_dir'])
        logger.info(f"Loading data from {data_dir}")

        logger.info("Loading global energy data...")
        global_data = load_global_energy_data(data_dir)

        logger.info("Loading worldwide renewable data...")
        worldwide_data = load_worldwide_renewable_data(data_dir)

        logger.info("Loading weather conditions data...")
        weather_data = load_weather_conditions(data_dir)

        logger.info("Loading US renewable data...")
        us_data = load_us_renewable_data(data_dir)

        logger.info("All data loaded successfully")

        # Process US renewable data
        logger.info("Processing US renewable data...")
        processed_us_data = process_us_renewable_data(us_data)

        # Log column information
        logger.info("Columns in processed data:")
        logger.info(list(processed_us_data.columns))

        # Save processed data
        output_path = Path(config['data_paths']['output_dir']) / 'final_processed_data.csv'
        processed_us_data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        # Validate output
        if not output_path.exists():
            raise FileNotFoundError(f"Failed to save processed data to {output_path}")

        if output_path.stat().st_size == 0:
            raise ValueError(f"Output file {output_path} is empty")

        logger.info("Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
