"""Implementation of deep learning models for renewable energy prediction."""
import logging
import sys
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)


class DeepLearningModels:
    """Implementation of deep learning models for renewable energy prediction."""

    def __init__(self, config: Dict = None):
        """Initialize deep learning models with configuration."""
        self.config = config or {}
        self.models = {}
        self.metrics = {}
        self.scalers = {}

        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def create_lstm_model(self, input_shape: tuple) -> Sequential:
        """Create an LSTM model architecture."""
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def create_cnn_model(self, input_shape: tuple) -> Sequential:
        """Create a CNN model architecture."""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(units=32, activation='relu'),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> tuple:
        """Prepare sequences for time series models."""
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    def train_lstm(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   sequence_length: int = 10,
                   validation_split: float = 0.2,
                   epochs: int = 100,
                   batch_size: int = 32) -> Dict:
        """Train LSTM model."""
        logger.info("Training LSTM model...")

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
        self.scalers['lstm'] = scaler

        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, sequence_length)

        # Create and train model
        model = self.create_lstm_model((sequence_length, X_train.shape[1]))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'models/lstm_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        history = model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.models['lstm'] = model
        return {'lstm': model, 'history': history.history}

    def train_cnn(self,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  sequence_length: int = 10,
                  validation_split: float = 0.2,
                  epochs: int = 100,
                  batch_size: int = 32) -> Dict:
        """Train CNN model."""
        logger.info("Training CNN model...")

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
        self.scalers['cnn'] = scaler

        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, sequence_length)

        # Create and train model
        model = self.create_cnn_model((sequence_length, X_train.shape[1]))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'models/cnn_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        history = model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.models['cnn'] = model
        return {'cnn': model, 'history': history.history}

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                        sequence_length: int = 10) -> Dict:
        """Evaluate trained models."""
        metrics = {}

        for name, model in self.models.items():
            # Scale test data
            scaler = self.scalers[name]
            X_scaled = scaler.transform(X_test)
            y_scaled = scaler.transform(y_test.values.reshape(-1, 1))

            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(X_scaled, sequence_length)

            # Get predictions
            y_pred_scaled = model.predict(X_seq)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            # Calculate metrics
            metrics[name] = {
                'mse': mean_squared_error(y_test[sequence_length:], y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test[sequence_length:], y_pred)),
                'mae': mean_absolute_error(y_test[sequence_length:], y_pred),
                'r2': r2_score(y_test[sequence_length:], y_pred)
            }

        self.metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame, sequence_length: int = 10) -> Dict[str, np.ndarray]:
        """Generate predictions using trained models."""
        predictions = {}

        for name, model in self.models.items():
            # Scale input data
            scaler = self.scalers[name]
            X_scaled = scaler.transform(X)

            # Prepare sequences
            X_seq, _ = self.prepare_sequences(X_scaled, sequence_length)

            # Get predictions
            y_pred_scaled = model.predict(X_seq)
            predictions[name] = scaler.inverse_transform(y_pred_scaled)

        return predictions


def main():
    """Example usage of DeepLearningModels class."""
    try:
        # Initialize models
        deep_learning = DeepLearningModels()

        # Load your data here
        # X_train, X_test, y_train, y_test = load_and_split_data()

        # Train models
        # lstm_results = deep_learning.train_lstm(X_train, y_train)
        # cnn_results = deep_learning.train_cnn(X_train, y_train)

        # Evaluate models
        # metrics = deep_learning.evaluate_models(X_test, y_test)

        logger.info("Deep learning models training completed successfully")

    except Exception as e:
        logger.error(f"Error in deep learning models training: {str(e)}")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        logger.info("Execution completed")


if __name__ == "__main__":
    main()
