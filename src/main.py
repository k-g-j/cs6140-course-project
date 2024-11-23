import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from data.feature_engineering import FeatureEngineer
from data.load_data import (
    load_global_energy_data,
    load_worldwide_renewable_data,
    load_weather_conditions,
    load_us_renewable_data
)
# Local imports
from data.preprocess import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class DataPipeline:
    """
    Main class to orchestrate the entire data processing pipeline.
    """

    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_paths']['base_dir'])
        self.output_dir = Path(self.config['data_paths']['output_dir'])
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        """
        logger.info("Loading all datasets...")

        try:
            datasets = {}

            # Load global energy data
            global_energy_data = load_global_energy_data(self.data_dir)
            datasets.update(global_energy_data)

            # Load worldwide renewable data
            worldwide_renewable_data = load_worldwide_renewable_data(self.data_dir)
            datasets.update(worldwide_renewable_data)

            # Load weather conditions data
            weather_data = load_weather_conditions(self.data_dir)
            datasets['weather_conditions'] = weather_data

            # Load US renewable data
            us_renewable_data = load_us_renewable_data(self.data_dir)
            datasets['us_renewable'] = us_renewable_data

            logger.info("Successfully loaded all datasets")
            return datasets

        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge different datasets based on configuration.
        """
        logger.info("Merging datasets...")

        try:
            # Start with renewable generation data
            if 'renewable_gen' not in datasets:
                raise KeyError("Missing required dataset: renewable_gen")

            merged_df = datasets['renewable_gen'].copy()
            logger.info(f"Starting with renewable_gen. Shape: {merged_df.shape}")

            # Merge with country consumption data
            if 'country_consumption' in datasets:
                logger.info("Merging with country_consumption")
                merged_df = merged_df.merge(
                    datasets['country_consumption'],
                    on=['country', 'Year'],
                    how='left'
                )
                logger.info(f"After country consumption merge. Shape: {merged_df.shape}")

            # Merge with renewable share data
            if 'renewable_share' in datasets:
                logger.info("Merging with renewable_share")
                renewable_share = datasets['renewable_share'].copy()
                # Rename Entity to country for merge
                if 'Entity' in renewable_share.columns:
                    renewable_share = renewable_share.rename(columns={'Entity': 'country'})
                merged_df = merged_df.merge(
                    renewable_share,
                    on=['country', 'Year'],
                    how='left'
                )
                logger.info(f"After renewable share merge. Shape: {merged_df.shape}")

            # Merge with renewable consumption data
            if 'renewable_consumption' in datasets:
                logger.info("Merging with renewable_consumption")
                renewable_consumption = datasets['renewable_consumption'].copy()
                if 'Entity' in renewable_consumption.columns:
                    renewable_consumption = renewable_consumption.rename(
                        columns={'Entity': 'country'})
                merged_df = merged_df.merge(
                    renewable_consumption,
                    on=['country', 'Year'],
                    how='left'
                )
                logger.info(f"After renewable consumption merge. Shape: {merged_df.shape}")

            # Calculate total renewable generation
            twh_columns = [col for col in merged_df.columns if 'TWh' in col or 'Generation' in col]
            logger.info(f"Columns used for total generation: {twh_columns}")

            if twh_columns:
                merged_df['total_renewable_generation'] = merged_df[twh_columns].sum(axis=1)

            # Rename columns for consistency
            column_mapping = {
                'Year': 'year',
                'Hydro(TWh)': 'hydro_generation',
                'Solar PV (TWh)': 'solar_generation',
                'Biofuel(TWh)': 'biofuel_generation',
                'Geothermal (TWh)': 'geothermal_generation',
                'consumption_twh': 'total_energy_consumption',
                'Renewables (% equivalent primary energy)': 'renewable_share',
                'Solar Generation - TWh': 'solar_generation_alt',
                'Wind Generation - TWh': 'wind_generation',
                'Hydro Generation - TWh': 'hydro_generation_alt'
            }

            # Only rename columns that exist
            columns_to_rename = {k: v for k, v in column_mapping.items() if k in merged_df.columns}
            merged_df = merged_df.rename(columns=columns_to_rename)

            logger.info(f"Final merged dataset shape: {merged_df.shape}")
            logger.info(f"Final columns: {list(merged_df.columns)}")

            return merged_df

        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            logger.error("Merge operation failed. Dumping dataset info:")
            for name, df in datasets.items():
                logger.error(f"{name} shape: {df.shape}, columns: {list(df.columns)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the data before processing.
        """
        logger.info("Validating data...")

        # Check for required columns
        required_columns = self.config['data_validation']['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for minimum number of rows
        min_rows = self.config['data_validation']['min_rows']
        if len(df) < min_rows:
            raise ValueError(f"Dataset has fewer than {min_rows} rows")

        # Check date range
        if 'date' in df.columns:
            min_date = pd.to_datetime(self.config['data_validation']['min_date'])
            max_date = pd.to_datetime(self.config['data_validation']['max_date'])
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].min() > min_date or df['date'].max() < max_date:
                raise ValueError("Data does not cover required date range")

        logger.info("Data validation successful")

    def save_processed_data(self,
                            df: pd.DataFrame,
                            filename: str,
                            save_metadata: bool = True) -> None:
        """
        Save processed data and metadata.
        """
        # Save processed dataset
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        if save_metadata:
            # Create metadata
            metadata = {
                'processing_date': datetime.now().isoformat(),
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'configuration': self.config
            }

            # Save metadata
            metadata_path = self.output_dir / f"{filename.split('.')[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved metadata to {metadata_path}")

    def run_pipeline(self) -> None:
        """
        Execute the complete data processing pipeline.
        """
        try:
            # Load datasets
            logger.info("Starting data processing pipeline...")
            datasets = self.load_all_datasets()

            # Merge datasets
            merged_df = self.merge_datasets(datasets)

            # Validate data
            self.validate_data(merged_df)

            # Preprocess data
            logger.info("Preprocessing data...")
            preprocessed_df = self.preprocessor.preprocess_dataset(
                merged_df,
                self.config['preprocessing']
            )

            # Save intermediate results
            self.save_processed_data(
                preprocessed_df,
                'preprocessed_data.csv'
            )

            # Engineer features
            logger.info("Engineering features...")
            final_df = self.feature_engineer.create_all_features(
                preprocessed_df,
                self.config['feature_engineering']
            )

            # Save final results
            self.save_processed_data(
                final_df,
                'final_processed_data.csv'
            )

            logger.info("Data processing pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def create_default_config() -> Dict:
    """
    Create default configuration for the pipeline.
    """
    config = {
        'data_paths': {
            'base_dir': 'data',
            'output_dir': 'processed_data'
        },
        'data_validation': {
            'required_columns': [
                'country',
                'year',
                'renewable_generation',
                'total_energy_consumption'
            ],
            'min_rows': 1000,
            'min_date': '1965-01-01',
            'max_date': '2022-12-31'
        },
        'preprocessing': {
            'handle_missing': True,
            'numeric_missing_strategy': 'knn',
            'categorical_missing_strategy': 'mode',
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'date_column': 'date',
            'target_column': 'renewable_generation',
            'lag_periods': [1, 3, 6, 12],
            'rolling_windows': [3, 6, 12],
            'group_column': 'country',
            'normalize': True,
            'normalization_method': 'standard'
        },
        'feature_engineering': {
            'renewable_cols': [
                'solar_generation',
                'wind_generation',
                'hydro_generation'
            ],
            'total_energy_col': 'total_energy_consumption',
            'create_weather_features': True,
            'gdp_col': 'gdp',
            'population_col': 'population',
            'policy_cols': [
                'feed_in_tariff',
                'renewable_portfolio_standard',
                'carbon_tax'
            ],
            'capacity_cols': [
                'solar_capacity',
                'wind_capacity',
                'hydro_capacity'
            ],
            'feature_pairs': [
                ['gdp_per_capita', 'renewable_share'],  # Changed from tuple to list
                ['policy_support_index', 'renewable_growth']  # Changed from tuple to list
            ],
            'region_col': 'continent',
            'regional_target_cols': [
                'renewable_share',
                'energy_intensity'
            ]
        }
    }
    return config


def main():
    """
    Main execution function.
    """
    try:
        config_path = Path('config.yaml')

        if not config_path.exists():
            logger.info("Creating default configuration...")
            config = create_default_config()
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info("Created default configuration file")

        # Initialize and run pipeline
        pipeline = DataPipeline(str(config_path))
        pipeline.run_pipeline()

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
