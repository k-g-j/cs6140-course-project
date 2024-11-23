import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_global_energy_data(data_dir):
    """
    Load and combine global energy consumption and renewable generation data
    """
    try:
        base_path = Path(data_dir) / "Global Energy Consumption & Renewable Generation"
        logger.info(f"Loading data from {base_path}")

        # Load country consumption data and melt it to long format
        country_consumption = pd.read_csv(base_path / "Country_Consumption_TWH.csv")
        country_consumption_melted = country_consumption.melt(
            id_vars=['Year'],
            var_name='country',
            value_name='consumption_twh'
        )

        # Load renewable generation data
        renewable_gen = pd.read_csv(base_path / "renewablePowerGeneration97-17.csv")
        # Since this is global data, we'll replicate it for each country
        countries = country_consumption_melted['country'].unique()
        renewable_gen_expanded = pd.DataFrame()
        for country in countries:
            temp_df = renewable_gen.copy()
            temp_df['country'] = country
            renewable_gen_expanded = pd.concat([renewable_gen_expanded, temp_df])

        # Load other datasets
        nonrenewable_gen = pd.read_csv(base_path / "nonRenewablesTotalPowerGeneration.csv")
        continent_consumption = pd.read_csv(base_path / "Continent_Consumption_TWH.csv")

        return {
            'country_consumption': country_consumption_melted,
            'renewable_gen': renewable_gen_expanded,
            'nonrenewable_gen': nonrenewable_gen,
            'continent_consumption': continent_consumption
        }

    except Exception as e:
        logger.error(f"Error loading global energy data: {str(e)}")
        raise


def load_worldwide_renewable_data(data_dir):
    """
    Load worldwide renewable energy data from 1965-2022
    """
    try:
        base_path = Path(data_dir) / "Renewable Energy World Wide 1965-2022"
        logger.info(f"Loading data from {base_path}")

        # Define files to load with column mappings
        files = {
            'renewable_share': {
                'file': "01 renewable-share-energy.csv",
                'rename': {'Entity': 'country'}
            },
            'renewable_consumption': {
                'file': "02 modern-renewable-energy-consumption.csv",
                'rename': {'Entity': 'country'}
            },
            'renewable_production': {
                'file': "03 modern-renewable-prod.csv",
                'rename': {'Entity': 'country'}
            },
            'hydro_consumption': {
                'file': "05 hydropower-consumption.csv",
                'rename': {'Entity': 'country'}
            },
            'wind_generation': {
                'file': "08 wind-generation.csv",
                'rename': {'Entity': 'country'}
            },
            'solar_consumption': {
                'file': "12 solar-energy-consumption.csv",
                'rename': {'Entity': 'country'}
            }
        }

        datasets = {}
        for key, info in files.items():
            logger.info(f"Loading {key} from {info['file']}")
            df = pd.read_csv(base_path / info['file'])

            # Rename columns according to mapping
            if 'rename' in info:
                df = df.rename(columns=info['rename'])

            logger.info(f"{key} columns after renaming: {list(df.columns)}")
            datasets[key] = df

        return datasets

    except Exception as e:
        logger.error(f"Error loading worldwide renewable data: {str(e)}")
        raise


def load_worldwide_renewable_data(data_dir):
    """
    Load worldwide renewable energy data from 1965-2022
    """
    try:
        base_path = Path(data_dir) / "Renewable Energy World Wide 1965-2022"

        logger.info(f"Loading data from {base_path}")

        # Define files to load
        files = {
            'renewable_share': "01 renewable-share-energy.csv",
            'renewable_consumption': "02 modern-renewable-energy-consumption.csv",
            'renewable_production': "03 modern-renewable-prod.csv",
            'hydro_consumption': "05 hydropower-consumption.csv",
            'wind_generation': "08 wind-generation.csv",
            'solar_consumption': "12 solar-energy-consumption.csv"
        }

        datasets = {}
        for key, filename in files.items():
            file_path = base_path / filename
            logger.info(f"Loading {key} from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"{key} columns: {list(df.columns)}")
            datasets[key] = df

        return datasets

    except Exception as e:
        logger.error(f"Error loading worldwide renewable data: {str(e)}")
        raise


def load_weather_conditions(data_dir):
    """
    Load renewable energy and weather conditions data
    """
    try:
        file_path = Path(data_dir) / "renewable_energy_and_weather_conditions.csv"
        logger.info(f"Loading weather conditions data from {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Weather conditions columns: {list(df.columns)}")

        # Standardize date column name if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    except Exception as e:
        logger.error(f"Error loading weather conditions data: {str(e)}")
        raise


def load_us_renewable_data(data_dir):
    """
    Load US renewable energy consumption data
    """
    try:
        file_path = Path(data_dir) / "us_renewable_energy_consumption.csv"
        logger.info(f"Loading US renewable data from {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"US renewable data columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"Error loading US renewable data: {str(e)}")
        raise
