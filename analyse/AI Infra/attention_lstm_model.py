"""
LSTM Model with Attention Mechanism for Pattern Detection
Allows visualization of which days in 30-day window are most important
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. Please install: pip install tensorflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for LSTM

    Computes attention weights across time steps, showing which days
    in the sequence are most important for the prediction.
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build attention layer

        Args:
            input_shape: (batch_size, time_steps, features)
        """
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass

        Args:
            inputs: (batch_size, time_steps, features)

        Returns:
            context_vector: Weighted sum of inputs
            attention_weights: Weights for each time step
        """
        # Compute attention scores
        # (batch_size, time_steps, features) @ (features, 1) = (batch_size, time_steps, 1)
        e = K.tanh(K.dot(inputs, self.W) + self.b)

        # Apply softmax to get attention weights
        # (batch_size, time_steps, 1)
        attention_weights = K.softmax(e, axis=1)

        # Compute context vector (weighted sum)
        # (batch_size, time_steps, features) * (batch_size, time_steps, 1)
        # = (batch_size, time_steps, features)
        context_vector = inputs * attention_weights

        # Sum across time steps
        # (batch_size, features)
        context_vector = K.sum(context_vector, axis=1)

        return context_vector, attention_weights

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class LSTMAttentionModel:
    """
    LSTM Model with Attention Mechanism

    Architecture:
    1. LSTM layers to process 30-day sequences
    2. Attention layer to weight important days
    3. Dense layers for classification

    The attention weights show which days the model focuses on.
    """

    def __init__(self,
                 window_size: int = 30,
                 n_features: int = 20,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM Attention Model

        Args:
            window_size: Number of days in input sequence (default: 30)
            n_features: Number of features per day (default: 20)
            lstm_units: Number of LSTM units (default: 64)
            dropout_rate: Dropout rate for regularization (default: 0.3)
            learning_rate: Learning rate for optimizer (default: 0.001)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM Attention Model")

        self.window_size = window_size
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.attention_model = None  # Model for extracting attention weights
        self.history = None

    def build_model(self) -> keras.Model:
        """
        Build LSTM model with attention mechanism

        Returns:
            Compiled Keras model
        """
        # Input layer: (batch_size, time_steps, features)
        inputs = layers.Input(shape=(self.window_size, self.n_features), name='input_sequence')

        # First LSTM layer (return sequences for attention)
        lstm1 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            name='lstm_1'
        )(inputs)
        lstm1 = layers.Dropout(self.dropout_rate)(lstm1)

        # Second LSTM layer (return sequences for attention)
        lstm2 = layers.LSTM(
            self.lstm_units // 2,
            return_sequences=True,
            name='lstm_2'
        )(lstm1)
        lstm2 = layers.Dropout(self.dropout_rate)(lstm2)

        # Attention layer
        context_vector, attention_weights = AttentionLayer(name='attention')(lstm2)

        # Dense layers
        dense1 = layers.Dense(32, activation='relu', name='dense_1')(context_vector)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)

        dense2 = layers.Dense(16, activation='relu', name='dense_2')(dense1)

        # Output layer (binary classification: winner vs not)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(dense2)

        # Build model
        model = models.Model(inputs=inputs, outputs=outputs, name='lstm_attention_model')

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        self.model = model

        # Build attention extraction model
        self.attention_model = models.Model(
            inputs=model.input,
            outputs=model.get_layer('attention').output[1]  # Get attention weights
        )

        logger.info("LSTM Attention Model built successfully")
        logger.info(f"Architecture: {self.window_size} days × {self.n_features} features")
        logger.info(f"LSTM units: {self.lstm_units}")

        return model

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32,
             verbose: int = 1) -> Dict:
        """
        Train the model

        Args:
            X_train: Training sequences (n_samples, window_size, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )

        # Train model
        logger.info(f"Training on {len(X_train):,} samples")
        logger.info(f"Validation on {len(X_val):,} samples")
        logger.info(f"Positive rate: Train={y_train.mean():.1%}, Val={y_val.mean():.1%}")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )

        # Evaluate
        train_loss, train_acc, train_auc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_auc = self.model.evaluate(X_val, y_val, verbose=0)

        logger.info(f"\nTraining Results:")
        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")

        return {
            'history': self.history.history,
            'train_metrics': {'loss': train_loss, 'accuracy': train_acc, 'auc': train_auc},
            'val_metrics': {'loss': val_loss, 'accuracy': val_acc, 'auc': val_auc}
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input sequences (n_samples, window_size, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built or trained")

        return self.model.predict(X, verbose=0).flatten()

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights for input sequences

        Args:
            X: Input sequences (n_samples, window_size, n_features)

        Returns:
            Attention weights (n_samples, window_size)
        """
        if self.attention_model is None:
            raise ValueError("Model not built")

        # Get attention weights from attention layer
        attention_weights = self.attention_model.predict(X, verbose=0)

        # Shape: (n_samples, window_size, 1) -> (n_samples, window_size)
        return attention_weights.squeeze()

    def analyze_prediction(self,
                          sequence: np.ndarray,
                          dates: Optional[List] = None) -> Dict:
        """
        Analyze a single prediction with attention weights

        Args:
            sequence: Single sequence (window_size, n_features)
            dates: Optional list of dates for the sequence

        Returns:
            Dictionary with prediction and attention analysis
        """
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Get prediction
        prediction = self.predict(sequence)[0]

        # Get attention weights
        attention = self.get_attention_weights(sequence)[0]

        # Find most important days
        top_k = 5
        top_indices = np.argsort(attention)[-top_k:][::-1]

        analysis = {
            'prediction': float(prediction),
            'predicted_class': 'WINNER' if prediction > 0.5 else 'NOT_WINNER',
            'confidence': float(abs(prediction - 0.5) * 2),  # 0-1 scale
            'attention_weights': attention.tolist(),
            'attention_mean': float(attention.mean()),
            'attention_std': float(attention.std()),
            'attention_max': float(attention.max()),
            'attention_min': float(attention.min()),
            'top_k_days': {
                'indices': top_indices.tolist(),
                'weights': attention[top_indices].tolist(),
                'dates': [dates[i] if dates else f"Day {i}" for i in top_indices]
            }
        }

        return analysis

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )

        # Rebuild attention extraction model
        self.attention_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('attention').output[1]
        )

        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built"

        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def prepare_sequences(df: pd.DataFrame,
                     window_size: int = 30,
                     feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training

    Args:
        df: DataFrame with time-series data (sorted by date)
        window_size: Number of days in each sequence
        feature_columns: List of feature column names

    Returns:
        X: Sequences (n_samples, window_size, n_features)
        y: Labels (n_samples,)
        dates: Date information for each sequence
    """
    if feature_columns is None:
        # Use volume and technical features
        feature_columns = [col for col in df.columns if any(x in col for x in [
            'vol_', 'bbw', 'adx', 'rsi', 'obv', 'accum_', 'consec_', 'sequence'
        ])]

    # Sort by time
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    sequences = []
    labels = []
    dates = []

    # Create sequences for each symbol
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)

        # Need at least window_size + 1 days
        if len(symbol_data) < window_size + 1:
            continue

        # Create overlapping windows
        for i in range(len(symbol_data) - window_size):
            # Get window of features
            window = symbol_data.iloc[i:i+window_size][feature_columns].values

            # Get label from next day after window
            label = symbol_data.iloc[i+window_size]['is_winner']

            # Get date range
            window_dates = symbol_data.iloc[i:i+window_size]['timestamp'].tolist()

            sequences.append(window)
            labels.append(label)
            dates.append(window_dates)

    X = np.array(sequences)
    y = np.array(labels)

    logger.info(f"Created {len(X):,} sequences")
    logger.info(f"Sequence shape: {X.shape}")
    logger.info(f"Positive rate: {y.mean():.1%}")

    return X, y, dates


if __name__ == "__main__":
    logger.info("LSTM Attention Model Module")
    logger.info("This module provides LSTM with attention for pattern detection")
    logger.info("\nFeatures:")
    logger.info("  • Attention mechanism showing important days")
    logger.info("  • 30-day sequence modeling")
    logger.info("  • Attention weight visualization support")
    logger.info("  • Per-prediction analysis")

    if not TF_AVAILABLE:
        logger.error("\nTensorFlow not available!")
        logger.error("Install with: pip install tensorflow")
    else:
        logger.info(f"\nTensorFlow version: {tf.__version__}")
        logger.info("Ready to train attention-based models!")
