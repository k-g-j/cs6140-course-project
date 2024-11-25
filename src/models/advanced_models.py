import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

logger = logging.getLogger(__name__)


class AdvancedModels:
    """Implementation of advanced machine learning models for renewable energy prediction."""

    def __init__(self, config: Dict = None):
        """Initialize advanced models with configuration."""
        self.config = config or {}
        self.models = {}
        self.metrics = {}

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            tune_hyperparameters: bool = True) -> Dict:
        """
        Train Random Forest model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target values
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model...")

        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            self.models['random_forest'] = grid_search.best_estimator_
            logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")

        else:
            rf = RandomForestRegressor(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', None),
                random_state=42
            )
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf

        return {'random_forest': self.models['random_forest']}

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                tune_hyperparameters: bool = True) -> Dict:
        """
        Train Gradient Boosting model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target values
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained Gradient Boosting model
        """
        logger.info("Training Gradient Boosting model...")

        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10]
            }

            gb = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            self.models['gradient_boosting'] = grid_search.best_estimator_
            logger.info(f"Best Gradient Boosting parameters: {grid_search.best_params_}")

        else:
            gb = GradientBoostingRegressor(
                n_estimators=self.config.get('gb_n_estimators', 100),
                learning_rate=self.config.get('gb_learning_rate', 0.1),
                random_state=42
            )
            gb.fit(X_train, y_train)
            self.models['gradient_boosting'] = gb

        return {'gradient_boosting': self.models['gradient_boosting']}

    def train_svr(self, X_train: pd.DataFrame, y_train: pd.Series,
                  tune_hyperparameters: bool = True) -> Dict:
        """
        Train Support Vector Regression model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target values
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained SVR model
        """
        logger.info("Training SVR model...")

        if tune_hyperparameters:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }

            svr = SVR()
            grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            self.models['svr'] = grid_search.best_estimator_
            logger.info(f"Best SVR parameters: {grid_search.best_params_}")

        else:
            svr = SVR(
                C=self.config.get('svr_C', 1.0),
                kernel=self.config.get('svr_kernel', 'rbf')
            )
            svr.fit(X_train, y_train)
            self.models['svr'] = svr

        return {'svr': self.models['svr']}

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate all trained models using multiple metrics.

        Args:
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary of evaluation metrics for each model
        """
        metrics = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

            if hasattr(model, 'feature_importances_'):
                metrics[name]['feature_importances'] = dict(zip(
                    X_test.columns, model.feature_importances_
                ))

        self.metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions using all trained models.

        Args:
            X: Features for prediction

        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        return predictions


def main():
    """Example usage of AdvancedModels class."""
    # Sample configuration
    config = {
        'rf_n_estimators': 200,
        'gb_learning_rate': 0.1,
        'svr_C': 1.0
    }

    try:
        # Initialize models
        advanced = AdvancedModels(config)

        # Load your data here
        # X_train, X_test, y_train, y_test = load_and_split_data()

        # Train models with hyperparameter tuning
        # advanced.train_random_forest(X_train, y_train, tune_hyperparameters=True)
        # advanced.train_gradient_boosting(X_train, y_train, tune_hyperparameters=True)
        # advanced.train_svr(X_train, y_train, tune_hyperparameters=True)

        # Evaluate models
        # metrics = advanced.evaluate_models(X_test, y_test)

        logger.info("Advanced models training completed successfully")

    except Exception as e:
        logger.error(f"Error in advanced models training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
