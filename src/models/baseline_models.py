"""Implementation of baseline models for renewable energy prediction."""
import logging
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class BaselineModels:
    """Implementation of baseline models for renewable energy prediction."""

    def __init__(self, config: Dict = None):
        """Initialize baseline models with configuration."""
        self.config = config or {}
        self.models = {}
        self.metrics = {}
        self._arima_params = {}

    def train_linear_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train linear regression models (Linear, Ridge, Lasso).
        """
        logger.info("Training linear models...")

        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=self.config.get('ridge_alpha', 1.0)),
            'lasso': Lasso(alpha=self.config.get('lasso_alpha', 1.0))
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            logger.info(f"Trained {name} regression model")

        return self.models

    def train_arima(self, timeseries: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """
        Train ARIMA model for time series prediction.
        """
        logger.info(f"Training ARIMA model with order {order}...")

        try:
            # Ensure time series is properly sorted
            timeseries = timeseries.sort_index()

            # Train model
            model = ARIMA(
                timeseries,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit()

            # Store model and parameters
            self.models['arima'] = fitted_model
            self._arima_params = {
                'order': order,
                'training_start': timeseries.index.min(),
                'training_end': timeseries.index.max(),
                'initial_level': timeseries.iloc[0] if len(timeseries) > 0 else 0
            }

            logger.info("Successfully trained ARIMA model")
            return {'arima': fitted_model}

        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate all trained models using multiple metrics.
        """
        metrics = {}

        # Evaluate linear models
        for name, model in self.models.items():
            if name != 'arima':
                y_pred = model.predict(X_test)
                metrics[name] = self._calculate_metrics(y_test, y_pred)

        # Evaluate ARIMA model if available
        if 'arima' in self.models:
            try:
                # Generate forecast for test period
                forecast = self.models['arima'].forecast(steps=len(y_test))

                # Convert to numpy array and handle NaNs
                forecast_array = np.array(forecast)
                test_array = np.array(y_test)

                # Check for and handle NaN values
                mask = ~np.isnan(forecast_array) & ~np.isnan(test_array)
                if np.any(mask):
                    metrics['arima'] = self._calculate_metrics(
                        test_array[mask],
                        forecast_array[mask]
                    )
                else:
                    logger.warning("No valid (non-NaN) predictions from ARIMA model")
                    metrics['arima'] = {
                        'mse': np.inf,
                        'rmse': np.inf,
                        'mae': np.inf,
                        'r2': -np.inf
                    }
                logger.info("ARIMA evaluation complete")

            except Exception as e:
                logger.error(f"Error evaluating ARIMA model: {str(e)}")
                metrics['arima'] = {
                    'mse': np.inf,
                    'rmse': np.inf,
                    'mae': np.inf,
                    'r2': -np.inf
                }

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        try:
            # Remove any NaN values
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            if len(y_true_clean) == 0:
                raise ValueError("No valid predictions after removing NaNs")

            mse = float(mean_squared_error(y_true_clean, y_pred_clean))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
            r2 = float(r2_score(y_true_clean, y_pred_clean))

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'r2': -np.inf
            }

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Generate predictions using all trained models.
        """
        predictions = {}

        # Linear model predictions
        for name, model in self.models.items():
            if name != 'arima':
                predictions[name] = model.predict(X)

        # ARIMA predictions if available
        if 'arima' in self.models:
            forecast_steps = len(X)
            predictions['arima'] = self.models['arima'].forecast(steps=forecast_steps)

        return predictions


def main():
    """Example usage of BaselineModels class."""
    # Sample configuration
    config = {
        'ridge_alpha': 1.0,
        'lasso_alpha': 0.1
    }

    try:
        # Initialize models
        baseline = BaselineModels(config)

        # Load your data here
        # X_train, X_test, y_train, y_test = load_and_split_data()

        # Train models
        # baseline.train_linear_models(X_train, y_train)

        # For time series
        # baseline.train_arima(y_train)

        # Evaluate models
        # metrics = baseline.evaluate_models(X_test, y_test)

        logger.info("Baseline models training completed successfully")

    except Exception as e:
        logger.error(f"Error in baseline models training: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
