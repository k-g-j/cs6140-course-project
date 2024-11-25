import logging
from typing import Dict, Tuple, Union

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
        self._arima_params = {}  # Store ARIMA parameters

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
            # Store parameters
            self._arima_params = {
                'order': order,
                'initial_level': timeseries.iloc[0] if len(timeseries) > 0 else 0
            }

            # Handle NaN values
            timeseries = timeseries.ffill().bfill()

            # Train model
            model = ARIMA(
                timeseries,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit()

            # Store model
            self.models['arima'] = fitted_model
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
        if 'arima' in self.models and self._arima_params:
            try:
                # Get predictions for test period
                arima_model = self.models['arima']
                forecast = arima_model.forecast(steps=len(y_test))

                # Handle differencing based on stored parameters
                if self._arima_params['order'][1] > 0:  # if d > 0
                    forecast = forecast.cumsum() + self._arima_params['initial_level']

                metrics['arima'] = self._calculate_metrics(y_test, forecast)
                logger.info("Successfully evaluated ARIMA model")
            except Exception as e:
                logger.warning(f"Could not evaluate ARIMA model: {str(e)}")

        self.metrics = metrics
        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]) -> Dict:
        """Calculate standard regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions using trained models.

        Args:
            X: Features for prediction

        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}

        for name, model in self.models.items():
            if name != 'arima':
                predictions[name] = model.predict(X)

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
        raise


if __name__ == "__main__":
    main()
