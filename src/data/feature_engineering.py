import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class to handle feature engineering for renewable energy data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Add output directory for plots
        self.output_dir = Path('figures')
        self.output_dir.mkdir(exist_ok=True)

    def create_renewable_features(self,
                                  df: pd.DataFrame,
                                  renewable_cols: List[str],
                                  total_energy_col: str) -> pd.DataFrame:
        """
        Create features related to renewable energy generation and consumption.

        Args:
            df: Input DataFrame
            renewable_cols: List of columns containing renewable energy data
            total_energy_col: Column containing total energy data

        Returns:
            DataFrame with additional renewable energy features
        """
        df_new = df.copy()

        # Calculate total renewable energy
        df_new['total_renewable'] = df_new[renewable_cols].sum(axis=1)

        # Calculate renewable energy share
        df_new['renewable_share'] = df_new['total_renewable'] / df_new[total_energy_col]

        # Calculate renewable energy mix
        for col in renewable_cols:
            df_new[f'{col}_share'] = df_new[col] / df_new['total_renewable']

        # Calculate year-over-year growth
        df_new['renewable_yoy_growth'] = df_new.groupby(['country'])['total_renewable'].pct_change()

        logger.info("Created renewable energy features")
        return df_new

    def create_weather_features(self,
                                df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features for renewable energy analysis.

        Args:
            df: Input DataFrame containing weather data

        Returns:
            DataFrame with additional weather features
        """
        df_weather = df.copy()

        # Create temperature features
        if 'temperature' in df_weather.columns:
            df_weather['temp_squared'] = df_weather['temperature'] ** 2
            df_weather['temp_range'] = df_weather.groupby('date')['temperature'].transform('max') - \
                                       df_weather.groupby('date')['temperature'].transform('min')

        # Create wind features
        if 'wind_speed' in df_weather.columns:
            df_weather['wind_energy'] = 0.5 * 1.225 * (
                    df_weather['wind_speed'] ** 3)  # Wind power density
            df_weather['wind_direction_bins'] = pd.qcut(df_weather['wind_direction'], q=8,
                                                        labels=False)

        # Create solar features
        if all(col in df_weather.columns for col in ['cloud_cover', 'solar_radiation']):
            df_weather['clear_sky_index'] = 1 - (df_weather['cloud_cover'] / 100)
            df_weather['solar_potential'] = df_weather['solar_radiation'] * df_weather[
                'clear_sky_index']

        logger.info("Created weather-related features")
        return df_weather

    def create_economic_features(self,
                                 df: pd.DataFrame,
                                 gdp_col: str,
                                 population_col: str) -> pd.DataFrame:
        """
        Create economic features for renewable energy analysis.

        Args:
            df: Input DataFrame
            gdp_col: Column containing GDP data
            population_col: Column containing population data

        Returns:
            DataFrame with additional economic features
        """
        df_econ = df.copy()

        # Calculate per capita metrics
        df_econ['gdp_per_capita'] = df_econ[gdp_col] / df_econ[population_col]
        df_econ['energy_per_capita'] = df_econ['total_energy_consumption'] / df_econ[population_col]
        df_econ['renewable_per_capita'] = df_econ['total_renewable'] / df_econ[population_col]

        # Calculate energy intensity
        df_econ['energy_intensity'] = df_econ['total_energy_consumption'] / df_econ[gdp_col]
        df_econ['renewable_intensity'] = df_econ['total_renewable'] / df_econ[gdp_col]

        # Calculate growth rates
        df_econ['gdp_growth'] = df_econ.groupby('country')[gdp_col].pct_change()
        df_econ['population_growth'] = df_econ.groupby('country')[population_col].pct_change()

        logger.info("Created economic features")
        return df_econ

    def create_policy_features(self,
                               df: pd.DataFrame,
                               policy_cols: List[str]) -> pd.DataFrame:
        """
        Create features related to renewable energy policies and incentives.

        Args:
            df: Input DataFrame
            policy_cols: List of columns containing policy indicators

        Returns:
            DataFrame with additional policy features
        """
        df_policy = df.copy()

        # Create policy index
        df_policy['policy_support_index'] = df_policy[policy_cols].sum(axis=1)

        # Calculate policy changes
        df_policy['policy_changes'] = df_policy.groupby('country')['policy_support_index'].diff()

        # Create policy stability indicator
        df_policy['policy_stability'] = df_policy.groupby('country')['policy_changes'].rolling(
            window=5, min_periods=1
        ).std().reset_index(0, drop=True)

        logger.info("Created policy-related features")
        return df_policy

    def create_technical_features(self,
                                  df: pd.DataFrame,
                                  capacity_cols: List[str]) -> pd.DataFrame:
        """
        Create technical features related to renewable energy infrastructure.

        Args:
            df: Input DataFrame
            capacity_cols: List of columns containing capacity data

        Returns:
            DataFrame with additional technical features
        """
        df_tech = df.copy()

        # Calculate capacity factors
        for col in capacity_cols:
            generation_col = col.replace('capacity', 'generation')
            if generation_col in df_tech.columns:
                df_tech[f'{col}_factor'] = df_tech[generation_col] / (
                        df_tech[col] * 8760)  # 8760 hours in a year

        # Calculate capacity growth
        for col in capacity_cols:
            df_tech[f'{col}_growth'] = df_tech.groupby('country')[col].pct_change()

        # Calculate total capacity and its growth
        df_tech['total_capacity'] = df_tech[capacity_cols].sum(axis=1)
        df_tech['total_capacity_growth'] = df_tech.groupby('country')['total_capacity'].pct_change()

        logger.info("Created technical features")
        return df_tech

    def create_interaction_features(self,
                                    df: pd.DataFrame,
                                    feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs of features.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact

        Returns:
            DataFrame with additional interaction features
        """
        df_inter = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df_inter.columns and feat2 in df_inter.columns:
                df_inter[f'{feat1}_{feat2}_interaction'] = df_inter[feat1] * df_inter[feat2]

        logger.info("Created interaction features")
        return df_inter

    def create_regional_features(self,
                                 df: pd.DataFrame,
                                 region_col: str,
                                 target_cols: List[str]) -> pd.DataFrame:
        """
        Create regional aggregate features.

        Args:
            df: Input DataFrame
            region_col: Column containing regional information
            target_cols: List of columns to aggregate

        Returns:
            DataFrame with additional regional features
        """
        df_region = df.copy()

        # Calculate regional averages
        for col in target_cols:
            regional_avg = df_region.groupby(region_col)[col].transform('mean')
            df_region[f'{col}_regional_avg'] = regional_avg
            df_region[f'{col}_regional_diff'] = df_region[col] - regional_avg

        # Calculate regional rankings
        for col in target_cols:
            df_region[f'{col}_regional_rank'] = df_region.groupby(region_col)[col].rank(pct=True)

        logger.info("Created regional features")
        return df_region

    def visualize_and_save_features(self, df: pd.DataFrame) -> None:
        """Create and save visualizations of engineered features."""

        logger.info("Creating feature visualizations...")

        # Create directories if they don't exist
        feature_analysis_dir = self.output_dir / 'feature_analysis'
        feature_analysis_dir.mkdir(parents=True, exist_ok=True)

        # Save technical features plot
        if any(col for col in df.columns if 'capacity' in col or 'generation' in col):
            plt.figure(figsize=(12, 6))
            technical_cols = [col for col in df.columns if
                              any(x in col for x in ['capacity', 'generation'])]
            sns.boxplot(data=df[technical_cols])
            plt.xticks(rotation=45)
            plt.title('Technical Features Distribution')
            plt.tight_layout()
            plt.savefig(feature_analysis_dir / 'technical_features_dist.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

        # Save weather features correlation plot
        weather_cols = [col for col in df.columns if
                        any(x in col for x in ['temp', 'wind', 'solar'])]
        if weather_cols:
            plt.figure(figsize=(15, 8))
            sns.heatmap(df[weather_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Weather Features Correlation')
            plt.tight_layout()
            plt.savefig(feature_analysis_dir / 'weather_features_correlation.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

        logger.info(f"Saved feature visualizations to {self.output_dir}")

    def create_all_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply all feature engineering steps and create visualizations.
        """

        logger.info("Starting feature engineering pipeline...")

        df_features = df.copy()

        # [Your existing feature creation code...]

        # Add visualization step at the end
        self.visualize_and_save_features(df_features)

        logger.info("Feature engineering pipeline completed successfully")
        return df_features


def main():
    """
    Example usage of the FeatureEngineer class.
    """
    # Sample feature engineering configuration
    config = {
        'renewable_cols': ['solar_generation', 'wind_generation', 'hydro_generation'],
        'total_energy_col': 'total_energy_consumption',
        'create_weather_features': True,
        'gdp_col': 'gdp',
        'population_col': 'population',
        'policy_cols': ['feed_in_tariff', 'renewable_portfolio_standard', 'carbon_tax'],
        'capacity_cols': ['solar_capacity', 'wind_capacity', 'hydro_capacity'],
        'feature_pairs': [
            ('gdp_per_capita', 'renewable_share'),
            ('policy_support_index', 'renewable_growth')
        ],
        'region_col': 'continent',
        'regional_target_cols': ['renewable_share', 'energy_intensity']
    }

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    try:
        # Load your preprocessed data from the correct path
        processed_data_path = Path('processed_data') / 'final_processed_data.csv'
        if not processed_data_path.exists():
            logger.error(f"Processed data file not found at {processed_data_path}")
            raise FileNotFoundError(f"Could not find {processed_data_path}")

        logger.info(f"Loading processed data from {processed_data_path}")
        df = pd.read_csv(processed_data_path)

        # Apply feature engineering pipeline with visualizations
        engineered_df = feature_engineer.create_all_features(df, config)

        # Save engineered features to processed_data directory
        output_path = Path('processed_data') / 'engineered_features.csv'
        engineered_df.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")

        logger.info("Feature engineering and visualizations completed successfully")

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
