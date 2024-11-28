"""Implementation of ensemble methods for renewable energy prediction."""
import logging
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class EnsembleModels:
    """Implementation of ensemble methods combining multiple models."""

    def __init__(self, base_models: Dict = None, config: Dict = None):
        """
        Initialize ensemble models.

        Args:
            base_models: Dictionary of base models to ensemble
            config: Configuration dictionary
        """
        self.base_models = base_models or {}
        self.config = config or {}
        self.models = {}
        self.metrics = {}

    def create_voting_ensemble(self,
                               models: Dict[str, object],
                               weights: Optional[List[float]] = None) -> VotingRegressor:
        """
        Create a voting ensemble from given models.

        Args:
            models: Dictionary of models to ensemble
            weights: Optional list of weights for each model

        Returns:
            VotingRegressor instance
        """
        estimators = [(name, model) for name, model in models.items()]

        return VotingRegressor(
            estimators=estimators,
            weights=weights
        )

    def create_stacking_ensemble(self,
                                 models: Dict[str, object],
                                 final_estimator: object = None) -> StackingRegressor:
        """
        Create a stacking ensemble from given models.

        Args:
            models: Dictionary of models to ensemble
            final_estimator: Meta-learner model (defaults to LinearRegression)

        Returns:
            StackingRegressor instance
        """
        estimators = [(name, model) for name, model in models.items()]

        if final_estimator is None:
            final_estimator = LinearRegression()

        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )

    def train_voting_ensemble(self,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              models: Dict[str, object] = None,
                              weights: Optional[List[float]] = None) -> Dict:
        """
        Train voting ensemble model.

        Args:
            X_train: Training features
            y_train: Training target values
            models: Models to ensemble (uses base_models if not provided)
            weights: Optional weights for models

        Returns:
            Dictionary containing trained ensemble
        """
        logger.info("Training voting ensemble...")

        if models is None:
            models = self.base_models

        voting_ensemble = self.create_voting_ensemble(models, weights)
        voting_ensemble.fit(X_train, y_train)

        self.models['voting'] = voting_ensemble
        return {'voting': voting_ensemble}

    def train_stacking_ensemble(self,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                models: Dict[str, object] = None,
                                final_estimator: object = None) -> Dict:
        """
        Train stacking ensemble model.

        Args:
            X_train: Training features
            y_train: Training target values
            models: Base models for stacking (uses base_models if not provided)
            final_estimator: Meta-learner model

        Returns:
            Dictionary containing trained ensemble
        """
        logger.info("Training stacking ensemble...")

        if models is None:
            models = self.base_models

        stacking_ensemble = self.create_stacking_ensemble(models, final_estimator)
        stacking_ensemble.fit(X_train, y_train)

        self.models['stacking'] = stacking_ensemble
        return {'stacking': stacking_ensemble}

    def train_all_ensembles(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            weights: Optional[List[float]] = None,
                            final_estimator: object = None) -> Dict:
        """
        Train all ensemble methods.

        Args:
            X_train: Training features
            y_train: Training target values
            weights: Optional weights for voting ensemble
            final_estimator: Optional meta-learner for stacking ensemble

        Returns:
            Dictionary containing all trained ensembles
        """
        logger.info("Training all ensemble models...")

        # Train voting ensemble
        self.train_voting_ensemble(X_train, y_train, weights=weights)

        # Train stacking ensemble
        self.train_stacking_ensemble(X_train, y_train, final_estimator=final_estimator)

        return self.models

    def evaluate_ensembles(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate trained ensemble models.

        Args:
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary of evaluation metrics
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

            # Add cross-validation scores
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
            metrics[name]['cv_score_mean'] = cv_scores.mean()
            metrics[name]['cv_score_std'] = cv_scores.std()

        self.metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions using trained ensemble models.

        Args:
            X: Features for prediction

        Returns:
            Dictionary of predictions from each ensemble
        """
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        return predictions

    def get_model_weights(self, ensemble_name: str) -> Dict[str, float]:
        """
        Get the weights or importance of each base model in the ensemble.

        Args:
            ensemble_name: Name of the ensemble model

        Returns:
            Dictionary of model weights/importance
        """
        if ensemble_name not in self.models:
            raise ValueError(f"Ensemble model '{ensemble_name}' not found")

        model = self.models[ensemble_name]
        weights = {}

        if ensemble_name == 'voting':
            if hasattr(model, 'weights_') and model.weights_ is not None:
                for name, weight in zip(self.base_models.keys(), model.weights_):
                    weights[name] = weight
            else:
                # If no weights specified, assume equal weights
                equal_weight = 1.0 / len(self.base_models)
                weights = {name: equal_weight for name in self.base_models.keys()}

        elif ensemble_name == 'stacking':
            # For stacking, we can use the coefficients of the final estimator
            # if it's a linear model
            if hasattr(model.final_estimator_, 'coef_'):
                for name, coef in zip(self.base_models.keys(), model.final_estimator_.coef_):
                    weights[name] = abs(coef)  # Use absolute value for importance

                # Normalize weights
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_ensemble_summary(self) -> Dict:
        """
        Get a summary of the ensemble models and their performance.

        Returns:
            Dictionary containing ensemble summary information
        """
        summary = {
            'ensembles': list(self.models.keys()),
            'base_models': list(self.base_models.keys()),
            'metrics': self.metrics
        }

        # Add model weights/importance for each ensemble
        summary['model_weights'] = {}
        for ensemble_name in self.models.keys():
            try:
                summary['model_weights'][ensemble_name] = self.get_model_weights(ensemble_name)
            except Exception as e:
                logger.warning(f"Could not get weights for {ensemble_name}: {str(e)}")

        return summary


def main():
    """Example usage of EnsembleModels class."""
    try:
        # Create some sample base models
        base_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }

        # Initialize ensemble models
        ensemble = EnsembleModels(base_models)

        # Load your data here
        # X_train, X_test, y_train, y_test = load_and_split_data()

        # Train ensembles
        # ensemble.train_all_ensembles(X_train, y_train)

        # Evaluate ensembles
        # metrics = ensemble.evaluate_ensembles(X_test, y_test)

        # Get ensemble summary
        # summary = ensemble.get_ensemble_summary()

        logger.info("Ensemble models training completed successfully")

    except Exception as e:
        logger.error(f"Error in ensemble models training: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
