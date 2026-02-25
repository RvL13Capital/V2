"""
Data Utilities for Temporal Architecture
=========================================

Helper functions for data loading, preprocessing, normalization, and splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def normalize_sequences(
    sequences: np.ndarray,
    method: str = 'standardize',
    scaler: Optional[object] = None,
    fit: bool = True
) -> Tuple[np.ndarray, object]:
    """
    Normalize sequences along feature dimension.

    Args:
        sequences: (n_samples, seq_len, n_features) array
        method: 'standardize' (z-score) or 'minmax' (0-1 range)
        scaler: Pre-fitted scaler (for test data)
        fit: Whether to fit scaler (True for train, False for test)

    Returns:
        normalized_sequences: Same shape as input
        scaler: Fitted scaler object (for test data)
    """
    n_samples, seq_len, n_features = sequences.shape

    # Reshape to (n_samples * seq_len, n_features)
    flat_sequences = sequences.reshape(-1, n_features)

    # Initialize scaler if not provided
    if scaler is None:
        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # Fit and/or transform
    if fit:
        normalized_flat = scaler.fit_transform(flat_sequences)
    else:
        normalized_flat = scaler.transform(flat_sequences)

    # Reshape back to (n_samples, seq_len, n_features)
    normalized_sequences = normalized_flat.reshape(n_samples, seq_len, n_features)

    return normalized_sequences, scaler


def split_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
    use_temporal_split: bool = False,
    temporal_date_column: str = 'sequence_end_date'
) -> dict:
    """
    Split sequences into train/val/test sets.

    Args:
        sequences: (n_samples, seq_len, n_features)
        labels: (n_samples,)
        metadata: Optional metadata DataFrame (required for temporal split)
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining)
        stratify: Whether to stratify by labels (only for random split)
        random_state: Random seed (only for random split)
        use_temporal_split: If True, use temporal split to prevent look-ahead bias
        temporal_date_column: Column name in metadata containing dates for temporal split

    Returns:
        Dictionary with train/val/test splits

    Raises:
        ValueError: If temporal split requested but metadata not provided
    """
    # Check if temporal split is requested
    if use_temporal_split:
        if metadata is None:
            raise ValueError("Metadata required for temporal split. Ensure metadata contains temporal information.")

        if temporal_date_column not in metadata.columns:
            raise ValueError(f"Date column '{temporal_date_column}' not found in metadata. "
                           f"Available columns: {metadata.columns.tolist()}")

        # Import temporal split utilities
        from .temporal_split import temporal_train_test_split

        # Get dates from metadata
        dates = metadata[temporal_date_column]

        # First temporal split: train+val vs test
        train_val_idx, test_idx = temporal_train_test_split(
            dates=dates,
            split_ratio=1.0 - test_size,
            strategy='percentage'
        )

        X_test = sequences[test_idx]
        y_test = labels[test_idx]

        # Get the train+val data
        X_temp = sequences[train_val_idx]
        y_temp = labels[train_val_idx]
        dates_temp = dates.iloc[train_val_idx]

        # Second temporal split: train vs val
        val_ratio_of_trainval = val_size / (1.0 - test_size)
        train_idx_rel, val_idx_rel = temporal_train_test_split(
            dates=dates_temp.reset_index(drop=True),
            split_ratio=1.0 - val_ratio_of_trainval,
            strategy='percentage'
        )

        X_train = X_temp[train_idx_rel]
        X_val = X_temp[val_idx_rel]
        y_train = y_temp[train_idx_rel]
        y_val = y_temp[val_idx_rel]

    else:
        # Original random split implementation
        # First split: train+val vs test
        stratify_labels = labels if stratify else None

        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_temp,
            random_state=random_state
        )

    # Split metadata if provided
    meta_train = meta_val = meta_test = None
    if metadata is not None:
        indices = np.arange(len(sequences))
        idx_temp, idx_test = train_test_split(
            indices, test_size=test_size, stratify=stratify_labels, random_state=random_state
        )
        idx_train, idx_val = train_test_split(
            idx_temp, test_size=val_size_adjusted, stratify=y_temp if stratify else None, random_state=random_state
        )

        meta_train = metadata.iloc[idx_train].reset_index(drop=True)
        meta_val = metadata.iloc[idx_val].reset_index(drop=True)
        meta_test = metadata.iloc[idx_test].reset_index(drop=True)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'meta_train': meta_train,
        'meta_val': meta_val,
        'meta_test': meta_test
    }


def balance_check(labels: np.ndarray, verbose: bool = True) -> dict:
    """
    Check class balance in labels.

    Args:
        labels: (n_samples,) array
        verbose: Whether to print distribution

    Returns:
        Dictionary with class counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    balance = {}
    if verbose:
        print("Class distribution:")

    for cls, count in zip(unique, counts):
        pct = 100 * count / total
        balance[f'K{cls}'] = {'count': int(count), 'percentage': pct}
        if verbose:
            print(f"  K{cls}: {count:,} ({pct:.2f}%)")

    return balance


def calculate_class_weights(labels: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Calculate class weights for imbalanced data.

    Args:
        labels: (n_samples,) array
        method: 'balanced' or 'sqrt' (square root of inverse frequency)

    Returns:
        (n_classes,) array of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_samples = len(labels)

    if method == 'balanced':
        # sklearn balanced: n_samples / (n_classes * n_samples_class)
        weights = n_samples / (n_classes * counts)
    elif method == 'sqrt':
        # Square root smoothing (less aggressive)
        freq = counts / n_samples
        weights = 1.0 / np.sqrt(freq)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to average 1.0
    weights = weights / weights.mean()

    return weights


def create_sequence_windows(
    data: np.ndarray,
    window_size: int = 20,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding windows from sequential data.

    Args:
        data: (n_timesteps, n_features) array
        window_size: Window length
        stride: Step size between windows

    Returns:
        (n_windows, window_size, n_features) array
    """
    n_timesteps, n_features = data.shape
    n_windows = (n_timesteps - window_size) // stride + 1

    windows = np.zeros((n_windows, window_size, n_features))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]

    return windows


def pad_sequences(
    sequences: list,
    max_length: int = 20,
    padding: str = 'post',
    value: float = 0.0
) -> np.ndarray:
    """
    Pad variable-length sequences to fixed length.

    Args:
        sequences: List of (seq_len, n_features) arrays
        max_length: Target sequence length
        padding: 'pre' or 'post'
        value: Padding value

    Returns:
        (n_samples, max_length, n_features) array
    """
    n_samples = len(sequences)
    n_features = sequences[0].shape[1]

    padded = np.full((n_samples, max_length, n_features), value, dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)

        if padding == 'post':
            padded[i, :seq_len] = seq[:seq_len]
        elif padding == 'pre':
            padded[i, -seq_len:] = seq[:seq_len]
        else:
            raise ValueError(f"Unknown padding: {padding}")

    return padded


def augment_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    shift_range: Tuple[int, int] = (-2, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to sequences.

    Augmentation types:
    - Gaussian noise
    - Scaling (amplitude)
    - Time shifting

    Args:
        sequences: (n_samples, seq_len, n_features)
        labels: (n_samples,)
        noise_std: Standard deviation of Gaussian noise
        scale_range: (min, max) scaling factors
        shift_range: (min, max) timestep shifts

    Returns:
        augmented_sequences: (n_samples * 3, seq_len, n_features)
        augmented_labels: (n_samples * 3,)
    """
    n_samples, seq_len, n_features = sequences.shape

    # Original + 3 augmentations = 4x data
    aug_sequences = []
    aug_labels = []

    for i in range(n_samples):
        seq = sequences[i]
        label = labels[i]

        # 1. Original
        aug_sequences.append(seq)
        aug_labels.append(label)

        # 2. Add noise
        noise = np.random.normal(0, noise_std, seq.shape)
        aug_sequences.append(seq + noise)
        aug_labels.append(label)

        # 3. Scale
        scale = np.random.uniform(*scale_range)
        aug_sequences.append(seq * scale)
        aug_labels.append(label)

        # 4. Time shift (circular)
        shift = np.random.randint(*shift_range)
        shifted = np.roll(seq, shift, axis=0)
        aug_sequences.append(shifted)
        aug_labels.append(label)

    return np.array(aug_sequences), np.array(aug_labels)


if __name__ == "__main__":
    # Test utilities
    print("Testing data utilities...")

    # Generate sample data
    sequences = np.random.randn(100, 20, 14)
    labels = np.random.randint(0, 6, 100)

    # Test normalization
    print("\n1. Normalization:")
    norm_seqs, scaler = normalize_sequences(sequences, method='standardize')
    print(f"   Original shape: {sequences.shape}")
    print(f"   Normalized shape: {norm_seqs.shape}")
    print(f"   Mean: {norm_seqs.mean():.4f}, Std: {norm_seqs.std():.4f}")

    # Test splitting
    print("\n2. Train/Val/Test Split:")
    splits = split_sequences(sequences, labels, test_size=0.2, val_size=0.1)
    print(f"   Train: {splits['X_train'].shape}")
    print(f"   Val: {splits['X_val'].shape}")
    print(f"   Test: {splits['X_test'].shape}")

    # Test balance check
    print("\n3. Class Balance:")
    balance_check(labels)

    # Test class weights
    print("\n4. Class Weights:")
    weights = calculate_class_weights(labels, method='balanced')
    for i, w in enumerate(weights):
        print(f"   K{i}: {w:.2f}")

    print("\n✓ All tests passed!")
