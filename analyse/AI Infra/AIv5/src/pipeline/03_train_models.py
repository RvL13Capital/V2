"""
Train models with Focal Loss for K4/K3 Precision Optimization - OPTIMIZED VERSION

OPTIMIZATIONS:
- Adaptive architecture (auto-scales based on feature count)
- Reduced epochs (100 instead of 300, early stopping ~40)
- Larger batch size (1024 vs 256) for 4x speedup
- Mixed precision training (2-3x speedup)
- Reduced early stopping patience (15 vs 30)
- Smaller model for 11 features, larger for 30-40 features

EXPECTED SPEEDUP: 10-37x faster (75 hours -> 2-7 hours)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras import mixed_precision
    import tensorflow.keras.backend as K

    # Mixed precision DISABLED due to focal loss numerical instability (NaN loss)
    # Focal loss uses log() operations that are unstable in float16
    # Trade-off: Lose 2x speedup, but maintain numerical stability
    # mixed_precision.set_global_policy('mixed_float16')  # DISABLED
    logger.info("[OPTIMIZATION] Mixed precision DISABLED (focal loss stability)")
    HAS_TF = True
except ImportError:
    logger.warning("TensorFlow not installed - will skip Deep Learning training")
    HAS_TF = False

# Configuration
OUTPUT_DIR = Path('output/focal_loss_training')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategic values for Expected Value calculation
# WINNER-ONLY FOCUS: Only K3/K4 matter, rest is neutral/avoid
STRATEGIC_VALUES = {
    'K0_STAGNANT': 0.0,      # Neutral - won't trade anyway
    'K1_MINIMAL': 0.0,       # Neutral - won't trade anyway
    'K2_QUALITY': 0.0,       # Neutral - won't trade anyway
    'K3_STRONG': 8.0,        # HIGH - big winner target
    'K4_EXCEPTIONAL': 10.0,  # MAXIMUM - ultimate goal
    'K5_FAILED': -10.0       # Avoid - don't trade losers
}

# FOCAL LOSS ALPHA VALUES - WINNER-ONLY FOCUS
# K3/K4 detection is 200-300X more important than K0/K1/K2
FOCAL_LOSS_ALPHA = {
    'K0_STAGNANT': 0.1,      # Don't waste model capacity
    'K1_MINIMAL': 0.1,       # Don't waste model capacity
    'K2_QUALITY': 0.1,       # Don't waste model capacity
    'K3_STRONG': 20.0,       # EXTREME FOCUS - 200X more important
    'K4_EXCEPTIONAL': 30.0,  # MAXIMUM FOCUS - 300X more important
    'K5_FAILED': 5.0         # Moderate - avoid predicting
}

# Focal loss gamma (focusing parameter)
# Higher gamma = more aggressive down-weighting of easy examples
FOCAL_LOSS_GAMMA = 3.0  # Aggressive focusing on hard K4/K3 examples


def load_and_prepare_data(patterns_file):
    """Load patterns and prepare features for training."""
    logger.info(f"Loading patterns from: {patterns_file}")
    df = pd.read_csv(patterns_file)
    logger.info(f"Loaded {len(df):,} patterns")

    # Verify snapshot_date column exists
    if 'snapshot_date' not in df.columns:
        raise ValueError("NO snapshot_date column - this is OLD system data! Re-scan with scan_patterns_aiv4_v2.py")

    logger.info("[OK] snapshot_date column verified")

    # Check outcome distribution
    logger.info("\nOutcome distribution:")
    outcome_counts = df['outcome_class'].value_counts().sort_index()
    for outcome, count in outcome_counts.items():
        logger.info(f"  {outcome:20s}: {count:,} ({count/len(df)*100:.1f}%)")

    return df


def engineer_features(df):
    """Engineer features for model training (NO DATA LEAKAGE - uses enhanced features from CSV if available)."""
    logger.info("\nEngineering features (using ONLY snapshot-time information)...")

    features = pd.DataFrame()

    # === BASIC TEMPORAL/PRICE FEATURES ===
    features['days_since_activation'] = df['days_since_activation']

    range_width = df['upper_boundary'] - df['lower_boundary']
    features['price_in_range'] = (df['current_price'] - df['lower_boundary']) / (range_width + 1e-10)
    features['snapshot_gain_pct'] = ((df['current_price'] - df['start_price']) / df['start_price']) * 100
    features['distance_to_power'] = df['power_boundary'] - df['current_price']
    features['distance_to_power_pct'] = features['distance_to_power'] / df['current_price']

    # === BASIC TECHNICAL INDICATORS ===
    features['current_bbw_20'] = df['current_bbw_20']
    features['current_bbw_percentile'] = df['current_bbw_percentile']
    features['current_adx'] = df['current_adx']
    features['current_volume_ratio_20'] = df['current_volume_ratio_20']
    features['current_range_ratio'] = df['current_range_ratio']
    features['range_width_pct'] = (range_width / df['start_price']) * 100

    # === ENHANCED FEATURES (if present from scanner) ===
    # Auto-detect and include ALL enhanced features from CSV
    enhanced_feature_columns = [
        # Compression features
        'avg_bbw_20d', 'avg_adx_20d', 'avg_volume_ratio_20d', 'avg_range_20d',
        'baseline_bbw_avg', 'baseline_bbw_std', 'baseline_volume_avg',
        'baseline_volatility', 'baseline_range_avg',
        'bbw_compression_ratio', 'volume_compression_ratio',
        'volatility_compression_ratio', 'overall_compression',
        # Trend features
        'bbw_slope_20d', 'adx_slope_20d', 'volume_slope_20d',
        # Volatility features
        'bbw_std_20d', 'volume_std_20d', 'price_volatility_20d',
        'volume_spikes_20d',
        # Price position features
        'price_position_in_range', 'price_distance_from_upper_pct',
        'price_distance_from_lower_pct', 'distance_from_power_pct',
        # Quality score
        'consolidation_quality_score'
    ]

    # Add enhanced features if they exist in the dataframe
    enhanced_count = 0
    for col in enhanced_feature_columns:
        if col in df.columns:
            features[col] = df[col]
            enhanced_count += 1

    # Drop NaN
    features = features.fillna(0)

    logger.info(f"Engineered {len(features.columns)} features ({enhanced_count} enhanced)")
    logger.info(f"Feature columns: {sorted(list(features.columns))}")

    return features


def focal_loss_fixed(y_true, y_pred, alpha, gamma=2.0):
    """
    Focal Loss for multi-class classification (TensorFlow/Keras)

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        alpha: Array of class-specific weights
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        Focal loss value
    """
    # Clip predictions to prevent log(0)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    # Calculate cross entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate focal loss weight: (1 - pt)^gamma
    # where pt is the probability of the true class
    pt = K.sum(y_true * y_pred, axis=-1, keepdims=True)
    focal_weight = K.pow(1.0 - pt, gamma)

    # Apply class-specific alpha weights
    alpha_tensor = K.constant(alpha, dtype='float32')
    alpha_weight = K.sum(y_true * alpha_tensor, axis=-1, keepdims=True)

    # Combine: FL = -α * (1-pt)^γ * log(pt)
    focal_loss = alpha_weight * focal_weight * cross_entropy

    return K.mean(K.sum(focal_loss, axis=-1))


def calculate_expected_value(y_pred_proba, class_labels):
    """Calculate Expected Value from prediction probabilities."""
    ev_values = []

    for probs in y_pred_proba:
        ev = sum(probs[i] * STRATEGIC_VALUES[class_labels[i]]
                for i in range(len(class_labels)))
        ev_values.append(ev)

    return np.array(ev_values)


def evaluate_k4_k3_precision(y_true_labels, y_pred_proba, class_labels, confidence_threshold=0.5):
    """
    Evaluate K4/K3 precision with confidence thresholding.
    PRIMARY OBJECTIVE: High precision on K4 and K3 predictions
    """
    y_pred_labels = [class_labels[i] for i in np.argmax(y_pred_proba, axis=1)]

    # Get indices for K4 and K3
    k4_idx = class_labels.index('K4_EXCEPTIONAL') if 'K4_EXCEPTIONAL' in class_labels else None
    k3_idx = class_labels.index('K3_STRONG') if 'K3_STRONG' in class_labels else None

    results = {}

    # K4 metrics
    if k4_idx is not None:
        k4_true = [i for i, label in enumerate(y_true_labels) if label == 'K4_EXCEPTIONAL']
        k4_pred = [i for i, label in enumerate(y_pred_labels) if label == 'K4_EXCEPTIONAL']
        k4_correct = len(set(k4_true) & set(k4_pred))

        k4_precision = k4_correct / len(k4_pred) if len(k4_pred) > 0 else 0.0
        k4_recall = k4_correct / len(k4_true) if len(k4_true) > 0 else 0.0

        # Confident K4 predictions (probability > threshold)
        k4_confident_pred = [i for i in range(len(y_pred_proba))
                           if np.argmax(y_pred_proba[i]) == k4_idx
                           and y_pred_proba[i][k4_idx] >= confidence_threshold]
        k4_confident_correct = len([i for i in k4_confident_pred if y_true_labels[i] == 'K4_EXCEPTIONAL'])
        k4_confident_precision = k4_confident_correct / len(k4_confident_pred) if len(k4_confident_pred) > 0 else 0.0

        results['k4_precision'] = float(k4_precision)
        results['k4_recall'] = float(k4_recall)
        results['k4_predictions'] = len(k4_pred)
        results['k4_confident_precision'] = float(k4_confident_precision)
        results['k4_confident_predictions'] = len(k4_confident_pred)

    # K3 metrics
    if k3_idx is not None:
        k3_true = [i for i, label in enumerate(y_true_labels) if label == 'K3_STRONG']
        k3_pred = [i for i, label in enumerate(y_pred_labels) if label == 'K3_STRONG']
        k3_correct = len(set(k3_true) & set(k3_pred))

        k3_precision = k3_correct / len(k3_pred) if len(k3_pred) > 0 else 0.0
        k3_recall = k3_correct / len(k3_true) if len(k3_true) > 0 else 0.0

        # Confident K3 predictions
        k3_confident_pred = [i for i in range(len(y_pred_proba))
                           if np.argmax(y_pred_proba[i]) == k3_idx
                           and y_pred_proba[i][k3_idx] >= confidence_threshold]
        k3_confident_correct = len([i for i in k3_confident_pred if y_true_labels[i] == 'K3_STRONG'])
        k3_confident_precision = k3_confident_correct / len(k3_confident_pred) if len(k3_confident_pred) > 0 else 0.0

        results['k3_precision'] = float(k3_precision)
        results['k3_recall'] = float(k3_recall)
        results['k3_predictions'] = len(k3_pred)
        results['k3_confident_precision'] = float(k3_confident_precision)
        results['k3_confident_predictions'] = len(k3_confident_pred)

    return results


def train_xgboost_focal(X_train, y_train, X_val, y_val, class_labels):
    """Train XGBoost with focal loss approximation via sample weights."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST WITH FOCAL LOSS WEIGHTS")
    logger.info("="*70)

    # Encode labels
    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    y_train_encoded = np.array([label_to_idx[y] for y in y_train])
    y_val_encoded = np.array([label_to_idx[y] for y in y_val])

    # Calculate focal loss-inspired sample weights
    # Weight = alpha * (1 - p_easy)^gamma where p_easy is approximated
    sample_weights = np.array([FOCAL_LOSS_ALPHA[y] for y in y_train])

    # Train model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(class_labels),
        max_depth=8,
        learning_rate=0.03,  # Slower learning for precision
        n_estimators=1000,   # More trees for better K4/K3 detection
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.5,           # Increased regularization
        min_child_weight=5,  # Higher to prevent overfitting on rare K4/K3
        random_state=42,
        early_stopping_rounds=100,
        eval_metric='mlogloss'
    )

    logger.info("Training XGBoost with focal loss weights...")
    logger.info(f"  K4 alpha weight: {FOCAL_LOSS_ALPHA['K4_EXCEPTIONAL']}")
    logger.info(f"  K3 alpha weight: {FOCAL_LOSS_ALPHA['K3_STRONG']}")

    model.fit(
        X_train, y_train_encoded,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_encoded)],
        verbose=50
    )

    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    y_pred_labels = [class_labels[i] for i in y_pred]

    # Metrics
    accuracy = accuracy_score(y_val_encoded, y_pred)
    logger.info(f"\nValidation Accuracy: {accuracy*100:.2f}%")

    # Expected Value
    ev_values = calculate_expected_value(y_pred_proba, class_labels)
    logger.info(f"Mean Expected Value: {np.mean(ev_values):.3f}")
    logger.info(f"Median Expected Value: {np.median(ev_values):.3f}")

    # K4/K3 Precision Analysis
    logger.info("\n" + "="*70)
    logger.info("K4/K3 PRECISION ANALYSIS")
    logger.info("="*70)

    for confidence in [0.3, 0.5, 0.7]:
        k4k3_metrics = evaluate_k4_k3_precision(y_val, y_pred_proba, class_labels, confidence)
        logger.info(f"\nConfidence Threshold: {confidence}")
        logger.info(f"  K4 Precision: {k4k3_metrics.get('k4_precision', 0)*100:.1f}% ({k4k3_metrics.get('k4_predictions', 0)} predictions)")
        logger.info(f"  K4 Confident Precision: {k4k3_metrics.get('k4_confident_precision', 0)*100:.1f}% ({k4k3_metrics.get('k4_confident_predictions', 0)} confident)")
        logger.info(f"  K3 Precision: {k4k3_metrics.get('k3_precision', 0)*100:.1f}% ({k4k3_metrics.get('k3_predictions', 0)} predictions)")
        logger.info(f"  K3 Confident Precision: {k4k3_metrics.get('k3_confident_precision', 0)*100:.1f}% ({k4k3_metrics.get('k3_confident_predictions', 0)} confident)")

    # Classification report
    logger.info("\n" + "="*70)
    logger.info("FULL CLASSIFICATION REPORT")
    logger.info("="*70)
    print(classification_report(y_val, y_pred_labels, zero_division=0))

    # Get K4/K3 metrics at 0.5 confidence for return
    k4k3_final = evaluate_k4_k3_precision(y_val, y_pred_proba, class_labels, 0.5)

    return model, {
        'accuracy': accuracy,
        'mean_ev': float(np.mean(ev_values)),
        'median_ev': float(np.median(ev_values)),
        'k4_precision': k4k3_final.get('k4_precision', 0),
        'k3_precision': k4k3_final.get('k3_precision', 0),
        'k4_confident_precision': k4k3_final.get('k4_confident_precision', 0),
        'k3_confident_precision': k4k3_final.get('k3_confident_precision', 0)
    }


def get_adaptive_architecture(n_features):
    """
    Return optimal architecture based on feature count.

    Returns: (layer_sizes, description)
    """
    if n_features <= 15:
        # Small feature set (11-15 features)
        return [256, 128, 64, 32], "SMALL (11-15 features, ~46K params, 4x faster)"
    elif n_features <= 30:
        # Medium feature set (16-30 features)
        return [384, 192, 96, 48], "MEDIUM (16-30 features, ~103K params, 2x faster)"
    else:
        # Large feature set (31-50 features)
        return [512, 256, 128, 64], "LARGE (31+ features, ~182K params, baseline)"


def train_deep_learning_focal(X_train, y_train, X_val, y_val, class_labels):
    """Train Deep Learning model with Focal Loss - OPTIMIZED VERSION."""
    if not HAS_TF:
        logger.warning("Skipping Deep Learning - TensorFlow not installed")
        return None, {}

    logger.info("\n" + "="*70)
    logger.info("TRAINING DEEP LEARNING WITH FOCAL LOSS (OPTIMIZED)")
    logger.info("="*70)

    # Encode labels
    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    y_train_encoded = np.array([label_to_idx[y] for y in y_train])
    y_val_encoded = np.array([label_to_idx[y] for y in y_val])

    # One-hot encode
    y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(class_labels))
    y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded, num_classes=len(class_labels))

    # Create alpha array matching class order
    alpha_array = np.array([FOCAL_LOSS_ALPHA[label] for label in class_labels])

    logger.info(f"Alpha weights by class:")
    for i, label in enumerate(class_labels):
        logger.info(f"  {label}: {alpha_array[i]}")
    logger.info(f"Gamma (focusing parameter): {FOCAL_LOSS_GAMMA}")

    # ADAPTIVE ARCHITECTURE based on feature count
    n_features = X_train.shape[1]
    layer_sizes, arch_desc = get_adaptive_architecture(n_features)

    logger.info(f"\n[OPTIMIZATION] Adaptive architecture selected:")
    logger.info(f"  Features: {n_features}")
    logger.info(f"  Architecture: {' -> '.join(map(str, layer_sizes))} -> {len(class_labels)}")
    logger.info(f"  Description: {arch_desc}")

    # Build adaptive model
    model = models.Sequential([
        layers.Dense(layer_sizes[0], activation='relu', input_shape=(n_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(layer_sizes[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(layer_sizes[2], activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(layer_sizes[3], activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(len(class_labels), activation='softmax', dtype='float32')  # Force float32 output
    ])

    # Create focal loss function
    def focal_loss(y_true, y_pred):
        return focal_loss_fixed(y_true, y_pred, alpha_array, gamma=FOCAL_LOSS_GAMMA)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR for stability
        loss=focal_loss,
        metrics=['accuracy']
    )

    logger.info("\nModel architecture:")
    model.summary(print_fn=logger.info)

    # OPTIMIZED Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # REDUCED from 30 (2x faster early stopping)
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,  # REDUCED from 15
        min_lr=1e-6
    )

    # Train
    logger.info("\nOPTIMIZATIONS:")
    logger.info("  - Epochs: 100 (vs 300, early stop expected ~40)")
    logger.info("  - Batch size: 1024 (vs 256, 4x faster)")
    logger.info("  - Early stop patience: 15 (vs 30, 2x faster)")
    logger.info("  - Mixed precision: DISABLED (focal loss stability)")
    logger.info("  - Adaptive architecture: Optimized for feature count")
    logger.info("\nExpected speedup: 5-20x (75 hours -> 4-15 hours)")
    logger.info("\nTraining Deep Learning model with Focal Loss...")

    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=100,  # REDUCED from 300 (will early stop around 40)
        batch_size=1024,  # INCREASED from 256 (4x speedup)
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val_onehot, verbose=0)
    logger.info(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    logger.info(f"Validation Loss (Focal): {val_loss:.3f}")
    logger.info(f"Epochs trained: {len(history.history['loss'])} (early stopped)")

    # Predictions
    y_pred_proba = model.predict(X_val, verbose=0)

    # Expected Value
    ev_values = calculate_expected_value(y_pred_proba, class_labels)
    logger.info(f"Mean Expected Value: {np.mean(ev_values):.3f}")
    logger.info(f"Median Expected Value: {np.median(ev_values):.3f}")

    # K4/K3 Precision Analysis
    logger.info("\n" + "="*70)
    logger.info("K4/K3 PRECISION ANALYSIS")
    logger.info("="*70)

    for confidence in [0.3, 0.5, 0.7]:
        k4k3_metrics = evaluate_k4_k3_precision(y_val, y_pred_proba, class_labels, confidence)
        logger.info(f"\nConfidence Threshold: {confidence}")
        logger.info(f"  K4 Precision: {k4k3_metrics.get('k4_precision', 0)*100:.1f}% ({k4k3_metrics.get('k4_predictions', 0)} predictions)")
        logger.info(f"  K4 Confident Precision: {k4k3_metrics.get('k4_confident_precision', 0)*100:.1f}% ({k4k3_metrics.get('k4_confident_predictions', 0)} confident)")
        logger.info(f"  K3 Precision: {k4k3_metrics.get('k3_precision', 0)*100:.1f}% ({k4k3_metrics.get('k3_predictions', 0)} predictions)")
        logger.info(f"  K3 Confident Precision: {k4k3_metrics.get('k3_confident_precision', 0)*100:.1f}% ({k4k3_metrics.get('k3_confident_predictions', 0)} confident)")

    # Get K4/K3 metrics at 0.5 confidence for return
    k4k3_final = evaluate_k4_k3_precision(y_val, y_pred_proba, class_labels, 0.5)

    return model, {
        'accuracy': float(val_accuracy),
        'loss': float(val_loss),
        'mean_ev': float(np.mean(ev_values)),
        'median_ev': float(np.median(ev_values)),
        'k4_precision': k4k3_final.get('k4_precision', 0),
        'k3_precision': k4k3_final.get('k3_precision', 0),
        'k4_confident_precision': k4k3_final.get('k4_confident_precision', 0),
        'k3_confident_precision': k4k3_final.get('k3_confident_precision', 0),
        'epochs_trained': len(history.history['loss']),
        'architecture': arch_desc
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--patterns', type=str, required=True, help='Path to labeled patterns CSV')
    args = parser.parse_args()

    print("="*70)
    print("TRAIN WITH FOCAL LOSS - K4/K3 OPTIMIZATION (OPTIMIZED)")
    print("="*70)
    print(f"Focal Loss Alpha (K4): {FOCAL_LOSS_ALPHA['K4_EXCEPTIONAL']}")
    print(f"Focal Loss Alpha (K3): {FOCAL_LOSS_ALPHA['K3_STRONG']}")
    print(f"Focal Loss Gamma: {FOCAL_LOSS_GAMMA}")
    print("="*70)
    print()

    # Load data
    df = load_and_prepare_data(args.patterns)

    # Engineer features
    X = engineer_features(df)
    y = df['outcome_class'].values

    # Get class labels
    class_labels = sorted(df['outcome_class'].unique())
    logger.info(f"\nClasses: {class_labels}")

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nTraining samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")

    # Train XGBoost with Focal Loss
    xgb_model, xgb_metrics = train_xgboost_focal(X_train, y_train, X_val, y_val, class_labels)

    # Save XGBoost
    xgb_path = OUTPUT_DIR / 'xgboost_focal_loss_optimized.json'
    xgb_model.save_model(str(xgb_path))
    logger.info(f"\n[OK] XGBoost (Focal Loss) saved to: {xgb_path}")

    # Train Deep Learning with Focal Loss
    dl_model, dl_metrics = train_deep_learning_focal(X_train, y_train, X_val, y_val, class_labels)

    # Save Deep Learning
    if dl_model:
        dl_path = OUTPUT_DIR / 'deep_learning_focal_loss_optimized.h5'
        dl_model.save(str(dl_path))
        logger.info(f"[OK] Deep Learning (Focal Loss) saved to: {dl_path}")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'patterns_file': str(args.patterns),
        'optimization_version': 'v2_adaptive_fast',
        'focal_loss_config': {
            'gamma': FOCAL_LOSS_GAMMA,
            'alpha_weights': FOCAL_LOSS_ALPHA
        },
        'data': {
            'total_patterns': len(df),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'features': len(X.columns),
            'classes': len(class_labels)
        },
        'xgboost': xgb_metrics,
        'deep_learning': dl_metrics if dl_model else None
    }

    report_path = OUTPUT_DIR / 'focal_loss_training_report_optimized.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n[OK] Training report saved to: {report_path}")

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - FOCAL LOSS OPTIMIZATION (OPTIMIZED)")
    print("="*70)
    print(f"XGBoost:")
    print(f"  EV: {xgb_metrics['mean_ev']:.3f}")
    print(f"  K4 Precision: {xgb_metrics['k4_precision']*100:.1f}%")
    print(f"  K3 Precision: {xgb_metrics['k3_precision']*100:.1f}%")
    if dl_model:
        print(f"\nDeep Learning ({dl_metrics.get('architecture', 'N/A')}):")
        print(f"  EV: {dl_metrics['mean_ev']:.3f}")
        print(f"  K4 Precision: {dl_metrics['k4_precision']*100:.1f}%")
        print(f"  K3 Precision: {dl_metrics['k3_precision']*100:.1f}%")
        print(f"  Epochs: {dl_metrics.get('epochs_trained', 'N/A')}")
    print(f"\nModels saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
