import logging
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle all preprocessing steps for renewable energy data.
    """

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)

    def remove_outliers(self,
                        df: pd.DataFrame,
                        columns: List[str],
                        method: str = 'iqr',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns using either IQR or z-score method.

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            method: Either 'iqr' or 'zscore'
            threshold: Z-score threshold or IQR multiplier

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()

        for column in columns:
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) &
                    (df_clean[column] <= upper_bound)
                    ]

            elif method == 'zscore':
                z_scores = np.abs((df_clean[column] - df_clean[column].mean()) /
                                  df_clean[column].std())
                df_clean = df_clean[z_scores < threshold]

        logger.info(f"Removed outliers from {columns} using {method} method")
        return df_clean

    def handle_missing_values(self,
                              df: pd.DataFrame,
                              numeric_strategy: str = 'knn',
                              categorical_strategy: str = 'mode') -> pd.DataFrame:
        """
        Handle missing values in the dataset using specified strategies.

        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'knn')
            categorical_strategy: Strategy for categorical columns ('mode', 'constant')

        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()

        # Separate numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

        # Handle numeric columns
        if len(numeric_cols) > 0:
            if numeric_strategy == 'knn':
                df_clean[numeric_cols] = self.knn_imputer.fit_transform(df_clean[numeric_cols])
            else:
                self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                df_clean[numeric_cols] = self.numeric_imputer.fit_transform(df_clean[numeric_cols])

        # Handle categorical columns
        if len(categorical_cols) > 0:
            if categorical_strategy == 'mode':
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = self.categorical_imputer.fit_transform(
                df_clean[categorical_cols])

        logger.info("Handled missing values using specified strategies")
        return df_clean

    def normalize_columns(self,
                          df: pd.DataFrame,
                          columns: List[str],
                          method: str = 'standard') -> Tuple[
        pd.DataFrame, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        """
        Normalize specified columns using the chosen method.

        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: 'standard', 'minmax', or 'robust'

        Returns:
            Tuple of (normalized DataFrame, fitted scaler)
        """
        df_normalized = df.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")

        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        logger.info(f"Normalized {columns} using {method} scaling")

        return df_normalized, scaler

    def create_temporal_features(self,
                                 df: pd.DataFrame,
                                 date_column: str) -> pd.DataFrame:
        """
        Create temporal features from date column.

        Args:
            df: Input DataFrame
            date_column: Name of the date column

        Returns:
            DataFrame with additional temporal features
        """
        df_temporal = df.copy()

        # Convert to datetime if not already
        df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])

        # Extract basic temporal features
        df_temporal['year'] = df_temporal[date_column].dt.year
        df_temporal['month'] = df_temporal[date_column].dt.month
        df_temporal['quarter'] = df_temporal[date_column].dt.quarter
        df_temporal['day_of_week'] = df_temporal[date_column].dt.dayofweek
        df_temporal['week_of_year'] = df_temporal[date_column].dt.isocalendar().week

        # Create seasonal features
        df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)
        df_temporal['season'] = df_temporal['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        logger.info("Created temporal features")
        return df_temporal

    def create_lag_features(self,
                            df: pd.DataFrame,
                            target_column: str,
                            lag_periods: List[int],
                            group_column: Optional[str] = None) -> pd.DataFrame:
        """
        Create lagged features for the target column.

        Args:
            df: Input DataFrame
            target_column: Column to create lags for
            lag_periods: List of periods to lag
            group_column: Column to group by (optional)

        Returns:
            DataFrame with additional lag features
        """
        df_lag = df.copy()

        if group_column:
            for lag in lag_periods:
                df_lag[f'{target_column}_lag_{lag}'] = (
                    df_lag.groupby(group_column)[target_column]
                    .shift(lag)
                )
        else:
            for lag in lag_periods:
                df_lag[f'{target_column}_lag_{lag}'] = df_lag[target_column].shift(lag)

        logger.info(f"Created lag features for {target_column}")
        return df_lag

    def create_rolling_features(self,
                                df: pd.DataFrame,
                                target_column: str,
                                windows: List[int],
                                group_column: Optional[str] = None) -> pd.DataFrame:
        """
        Create rolling window features for the target column.

        Args:
            df: Input DataFrame
            target_column: Column to create rolling features for
            windows: List of window sizes
            group_column: Column to group by (optional)

        Returns:
            DataFrame with additional rolling features
        """
        df_rolling = df.copy()

        for window in windows:
            if group_column:
                df_rolling[f'{target_column}_rolling_mean_{window}'] = (
                    df_rolling.groupby(group_column)[target_column]
                    .rolling(window=window)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df_rolling[f'{target_column}_rolling_std_{window}'] = (
                    df_rolling.groupby(group_column)[target_column]
                    .rolling(window=window)
                    .std()
                    .reset_index(0, drop=True)
                )
            else:
                df_rolling[f'{target_column}_rolling_mean_{window}'] = (
                    df_rolling[target_column]
                    .rolling(window=window)
                    .mean()
                )
                df_rolling[f'{target_column}_rolling_std_{window}'] = (
                    df_rolling[target_column]
                    .rolling(window=window)
                    .std()
                )

        logger.info(f"Created rolling features for {target_column}")
        return df_rolling

    def preprocess_dataset(self,
                           df: pd.DataFrame,
                           config: Dict) -> pd.DataFrame:
        """
        Main preprocessing pipeline that applies all necessary transformations.

        Args:
            df: Input DataFrame
            config: Dictionary containing preprocessing configurations

        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")

        # Handle missing values
        if config.get('handle_missing', True):
            df = self.handle_missing_values(
                df,
                numeric_strategy=config.get('numeric_missing_strategy', 'knn'),
                categorical_strategy=config.get('categorical_missing_strategy', 'mode')
            )

        # Remove outliers if specified
        if config.get('remove_outliers', True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = self.remove_outliers(
                df,
                columns=numeric_cols,
                method=config.get('outlier_method', 'iqr'),
                threshold=config.get('outlier_threshold', 3.0)
            )

        # Create temporal features if date column specified
        if 'date_column' in config:
            df = self.create_temporal_features(df, config['date_column'])

        # Create lag features if specified
        if 'target_column' in config and 'lag_periods' in config:
            df = self.create_lag_features(
                df,
                config['target_column'],
                config['lag_periods'],
                config.get('group_column')
            )

        # Create rolling features if specified
        if 'target_column' in config and 'rolling_windows' in config:
            df = self.create_rolling_features(
                df,
                config['target_column'],
                config['rolling_windows'],
                config.get('group_column')
            )

        # Normalize numeric columns if specified
        if config.get('normalize', True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df, _ = self.normalize_columns(
                df,
                columns=numeric_cols,
                method=config.get('normalization_method', 'standard')
            )

        logger.info("Preprocessing pipeline completed successfully")
        return df


def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Sample preprocessing configuration
    config = {
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
    }

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Example usage with sample data
    try:
        # Load your data here
        df = pd.read_csv('path_to_your_data.csv')

        # Apply preprocessing pipeline
        processed_df = preprocessor.preprocess_dataset(df, config)

        # Save processed data
        processed_df.to_csv('processed_data.csv', index=False)
        logger.info("Data preprocessing completed and saved successfully")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
