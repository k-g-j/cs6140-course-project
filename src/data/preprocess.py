import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import logging
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """A class to handle all preprocessing steps for renewable energy data."""

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)

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

    def remove_outliers(self,
                        df: pd.DataFrame,
                        columns: List[str],
                        method: str = 'iqr',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection

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

    def normalize_columns(self,
                          df: pd.DataFrame,
                          columns: List[str],
                          method: str = 'standard') -> Tuple[
        pd.DataFrame, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        """
        Normalize specified columns.

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

    def create_temporal_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create temporal features from year or date column.

        Args:
            df: Input DataFrame
            time_column: Name of the time column (year or date)

        Returns:
            DataFrame with additional temporal features
        """
        df_temporal = df.copy()

        if time_column == 'date' and 'date' in df_temporal.columns:
            # Convert to datetime if it's a date column
            df_temporal['date'] = pd.to_datetime(df_temporal['date'])
            df_temporal['year'] = df_temporal['date'].dt.year
            df_temporal['month'] = df_temporal['date'].dt.month
            df_temporal['quarter'] = df_temporal['date'].dt.quarter
            df_temporal['day_of_week'] = df_temporal['date'].dt.dayofweek
            df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)

        elif time_column == 'year' and 'year' in df_temporal.columns:
            # If we only have year data, create decade and period features
            df_temporal['decade'] = (df_temporal['year'] // 10) * 10
            df_temporal['period'] = pd.cut(
                df_temporal['year'],
                bins=[1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                labels=['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
            )

        logger.info("Created temporal features")
        return df_temporal

    def create_lag_features(self,
                            df: pd.DataFrame,
                            target_column: str,
                            lag_periods: List[int],
                            group_column: Optional[str] = None) -> pd.DataFrame:
        """
        Create lagged features with support for yearly data.

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
            # Sort by year within each group
            df_lag = df_lag.sort_values([group_column, 'year'])
            for lag in lag_periods:
                df_lag[f'{target_column}_lag_{lag}'] = (
                    df_lag.groupby(group_column)[target_column]
                    .shift(lag)
                )
        else:
            # Sort by year
            df_lag = df_lag.sort_values('year')
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
        Create rolling window features.

        Args:
            df: Input DataFrame
            target_column: Column to create rolling features for
            windows: List of window sizes
            group_column: Column to group by (optional)

        Returns:
            DataFrame with additional rolling features
        """
        df_rolling = df.copy()

        if group_column:
            # Sort by year within each group
            df_rolling = df_rolling.sort_values([group_column, 'year'])
            for window in windows:
                df_rolling[f'{target_column}_rolling_mean_{window}'] = (
                    df_rolling.groupby(group_column)[target_column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df_rolling[f'{target_column}_rolling_std_{window}'] = (
                    df_rolling.groupby(group_column)[target_column]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
        else:
            # Sort by year
            df_rolling = df_rolling.sort_values('year')
            for window in windows:
                df_rolling[f'{target_column}_rolling_mean_{window}'] = (
                    df_rolling[target_column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                df_rolling[f'{target_column}_rolling_std_{window}'] = (
                    df_rolling[target_column]
                    .rolling(window=window, min_periods=1)
                    .std()
                )

        logger.info(f"Created rolling features for {target_column}")
        return df_rolling

    def preprocess_dataset(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            df: Input DataFrame
            config: Dictionary containing preprocessing configurations

        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")

        df_processed = df.copy()

        # Handle missing values
        if config.get('handle_missing', True):
            df_processed = self.handle_missing_values(
                df_processed,
                numeric_strategy=config.get('numeric_missing_strategy', 'knn'),
                categorical_strategy=config.get('categorical_missing_strategy', 'mode')
            )

        # Remove outliers if specified
        if config.get('remove_outliers', True):
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed = self.remove_outliers(
                df_processed,
                columns=numeric_cols,
                method=config.get('outlier_method', 'iqr'),
                threshold=config.get('outlier_threshold', 3.0)
            )

        # Create temporal features
        time_column = 'year' if 'year' in df_processed.columns else config.get('date_column')
        if time_column in df_processed.columns:
            df_processed = self.create_temporal_features(df_processed, time_column)

        # Create lag features
        if 'target_column' in config and 'lag_periods' in config:
            df_processed = self.create_lag_features(
                df_processed,
                config['target_column'],
                config['lag_periods'],
                config.get('group_column')
            )

        # Create rolling features
        if 'target_column' in config and 'rolling_windows' in config:
            df_processed = self.create_rolling_features(
                df_processed,
                config['target_column'],
                config['rolling_windows'],
                config.get('group_column')
            )

        # Normalize if specified
        if config.get('normalize', True):
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed, _ = self.normalize_columns(
                df_processed,
                columns=numeric_cols,
                method=config.get('normalization_method', 'standard')
            )

        logger.info("Preprocessing pipeline completed successfully")
        return df_processed


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
