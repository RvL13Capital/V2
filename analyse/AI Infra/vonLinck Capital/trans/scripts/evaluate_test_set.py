#!/usr/bin/env python3
"""
Test Set Evaluation Script

Evaluates a trained model on the held-out test set to calculate:
- Top-15% Precision (Target Rate)
- Lift vs Baseline
- Class distribution analysis

Usage:
    python scripts/evaluate_test_set.py \
        --model output/models/us_v20_lstm_20260121_215107.pt \
        --sequences output/sequences/us_quality/sequences_20260121_050213.h5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import h5py
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.temporal_hybrid_v18 import HybridFeatureNetwork
from models.temporal_hybrid_unified import HybridFeatureNetwork as HybridFeatureNetworkV20, create_v20_model
from config.context_features import NUM_CONTEXT_FEATURES, VOLUME_RATIO_INDICES


def load_model(model_path: str, device: torch.device):
    """Load model from checkpoint with automatic architecture detection."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_version = config.get('model_version', 'v18')
    ablation_mode = config.get('ablation_mode', 'lstm')
    input_features = config.get('input_features', 10)
    use_conditioned_lstm = config.get('use_conditioned_lstm', False)
    context_features = NUM_CONTEXT_FEATURES

    print(f"Loading {model_version.upper()} model (ablation_mode={ablation_mode})")
    print(f"  Input features: {input_features}, Context features: {context_features}")
    print(f"  use_conditioned_lstm: {use_conditioned_lstm}")

    if model_version == 'v20':
        model = create_v20_model(
            ablation_mode=ablation_mode,
            input_features=input_features,
            sequence_length=20,
            context_features=context_features,
            num_classes=3
        )
    else:
        model = HybridFeatureNetwork(
            input_features=input_features,
            sequence_length=20,
            context_features=context_features,
            num_classes=3,
            use_conditioned_lstm=use_conditioned_lstm
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get normalization params
    norm_params = checkpoint.get('norm_params', {})

    return model, norm_params, checkpoint


def load_test_data(h5_path: str, norm_params: dict):
    """Load test set data from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        sequences = f['sequences'][:]
        labels = f['labels'][:]
        context = f['context'][:] if 'context' in f else None

        # Get metadata for date-based splitting
        if 'metadata' in f:
            metadata_json = f['metadata'][()].decode('utf-8')
            metadata = json.loads(metadata_json)
        else:
            metadata = None

    print(f"Loaded data: {sequences.shape[0]} samples")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels distribution: {np.bincount(labels)}")

    # Apply normalization (from training set statistics)
    if norm_params:
        train_median = np.array(norm_params.get('median', [0]*sequences.shape[2]))
        train_iqr = np.array(norm_params.get('iqr', [1]*sequences.shape[2]))
        train_iqr = np.where(train_iqr < 1e-8, 1.0, train_iqr)

        # Robust scaling
        sequences = (sequences - train_median) / train_iqr

        # Context normalization
        if context is not None and 'context_median' in norm_params:
            context_median = np.array(norm_params['context_median'])
            context_iqr = np.array(norm_params['context_iqr'])
            context_iqr = np.where(context_iqr < 1e-8, 1.0, context_iqr)

            # Apply log1p to volume ratio indices
            log1p_indices = norm_params.get('context_log1p_indices', [])
            if log1p_indices:
                for idx in log1p_indices:
                    if idx < context.shape[1]:
                        context[:, idx] = np.sign(context[:, idx]) * np.log1p(np.abs(context[:, idx]))

            context = (context - context_median) / context_iqr

    return sequences, labels, context, metadata


def calculate_top_k_metrics(probs: np.ndarray, labels: np.ndarray, k_pct: float = 0.15):
    """
    Calculate Top-K% precision metrics.

    Args:
        probs: (N, 3) probability array [Danger, Noise, Target]
        labels: (N,) ground truth labels
        k_pct: Percentage of top predictions to evaluate (default 15%)

    Returns:
        dict with metrics
    """
    n_samples = len(labels)
    n_top_k = max(1, int(n_samples * k_pct))

    # Get class 2 (Target) probability
    target_probs = probs[:, 2]

    # Get indices of top-K predictions (highest Target probability)
    top_k_indices = np.argsort(target_probs)[-n_top_k:]

    # Labels in top-K
    top_k_labels = labels[top_k_indices]

    # Calculate rates
    target_rate = (top_k_labels == 2).sum() / n_top_k * 100
    noise_rate = (top_k_labels == 1).sum() / n_top_k * 100
    danger_rate = (top_k_labels == 0).sum() / n_top_k * 100

    # Baseline rates (overall dataset)
    baseline_target = (labels == 2).sum() / n_samples * 100
    baseline_noise = (labels == 1).sum() / n_samples * 100
    baseline_danger = (labels == 0).sum() / n_samples * 100

    # Lift calculations
    lift_target = target_rate / baseline_target if baseline_target > 0 else 0
    lift_danger = danger_rate / baseline_danger if baseline_danger > 0 else 0

    return {
        'n_samples': n_samples,
        'n_top_k': n_top_k,
        'k_pct': k_pct * 100,
        'top_k_target_rate': target_rate,
        'top_k_noise_rate': noise_rate,
        'top_k_danger_rate': danger_rate,
        'baseline_target_rate': baseline_target,
        'baseline_noise_rate': baseline_noise,
        'baseline_danger_rate': baseline_danger,
        'lift_target': lift_target,
        'lift_danger': lift_danger
    }


def evaluate_model(model, sequences, labels, context, device, batch_size=256):
    """Run inference and collect predictions."""
    model.eval()
    all_probs = []

    n_samples = len(sequences)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_seq = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
            batch_ctx = torch.FloatTensor(context[i:i+batch_size]).to(device) if context is not None else None

            logits = model(batch_seq, context=batch_ctx)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequences', type=str, required=True, help='Path to HDF5 sequences file')
    parser.add_argument('--test-split', type=str, default='2024-01-01',
                       help='Date cutoff for test set (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--top-k', type=float, default=0.15, help='Top-K percentage (default: 0.15)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    model, norm_params, checkpoint = load_model(args.model, device)

    # Load all data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    sequences, labels, context, metadata = load_test_data(args.sequences, norm_params)

    # Try to get pattern dates for temporal split
    # Check for metadata parquet file
    h5_dir = os.path.dirname(args.sequences)
    meta_files = [f for f in os.listdir(h5_dir) if f.startswith('metadata_') and f.endswith('.parquet')]

    test_mask = None
    if meta_files:
        meta_path = os.path.join(h5_dir, sorted(meta_files)[-1])
        print(f"Loading metadata from: {meta_path}")
        meta_df = pd.read_parquet(meta_path)

        # Check for various date column names
        date_col = None
        for col in ['pattern_end_date', 'end_date', 'date']:
            if col in meta_df.columns:
                date_col = col
                break

        if date_col:
            meta_df[date_col] = pd.to_datetime(meta_df[date_col])
            test_cutoff = pd.to_datetime(args.test_split)
            test_mask = meta_df[date_col] >= test_cutoff
            print(f"Using date column: {date_col}")
            print(f"Test cutoff: {args.test_split}")
            print(f"Test set size: {test_mask.sum()} samples")

    # If no temporal split available, use last 40% as test
    if test_mask is None:
        n_total = len(labels)
        test_start = int(n_total * 0.6)
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[test_start:] = True
        print(f"Using last 40% as test set: {test_mask.sum()} samples")

    # Extract test set
    test_seq = sequences[test_mask]
    test_labels = labels[test_mask]
    test_context = context[test_mask] if context is not None else None

    print(f"\nTest set: {len(test_labels)} samples")
    print(f"  Label distribution: {np.bincount(test_labels)}")
    print(f"  Class 0 (Danger): {(test_labels==0).sum()} ({(test_labels==0).mean()*100:.1f}%)")
    print(f"  Class 1 (Noise):  {(test_labels==1).sum()} ({(test_labels==1).mean()*100:.1f}%)")
    print(f"  Class 2 (Target): {(test_labels==2).sum()} ({(test_labels==2).mean()*100:.1f}%)")

    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    probs = evaluate_model(model, test_seq, test_labels, test_context, device, args.batch_size)

    # Calculate metrics
    metrics = calculate_top_k_metrics(probs, test_labels, k_pct=args.top_k)

    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"\nDataset: {metrics['n_samples']} samples")
    print(f"Top-{metrics['k_pct']:.0f}%: {metrics['n_top_k']} patterns")

    print(f"\n--- Top-{metrics['k_pct']:.0f}% Precision ---")
    print(f"  Target Rate: {metrics['top_k_target_rate']:.1f}%")
    print(f"  Noise Rate:  {metrics['top_k_noise_rate']:.1f}%")
    print(f"  Danger Rate: {metrics['top_k_danger_rate']:.1f}%")

    print(f"\n--- Baseline (Random) ---")
    print(f"  Target Rate: {metrics['baseline_target_rate']:.1f}%")
    print(f"  Noise Rate:  {metrics['baseline_noise_rate']:.1f}%")
    print(f"  Danger Rate: {metrics['baseline_danger_rate']:.1f}%")

    print(f"\n--- LIFT (Top-K / Baseline) ---")
    print(f"  Target Lift: {metrics['lift_target']:.2f}x")
    print(f"  Danger Lift: {metrics['lift_danger']:.2f}x")

    # Success criteria check
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    lift_threshold = 1.6
    if metrics['lift_target'] >= lift_threshold:
        print(f"[PASS] Lift {metrics['lift_target']:.2f}x >= {lift_threshold}x threshold")
    else:
        print(f"[FAIL] Lift {metrics['lift_target']:.2f}x < {lift_threshold}x threshold")
        print(f"  Gap to target: {lift_threshold - metrics['lift_target']:.2f}x")

    # Additional analysis - probability distribution
    print("\n--- Probability Distribution Analysis ---")
    target_probs = probs[:, 2]
    print(f"  Target prob mean: {target_probs.mean():.3f}")
    print(f"  Target prob std:  {target_probs.std():.3f}")
    print(f"  Target prob max:  {target_probs.max():.3f}")
    print(f"  Target prob min:  {target_probs.min():.3f}")

    # Calibration check
    predicted_target = (probs.argmax(axis=1) == 2)
    actual_target = (test_labels == 2)
    if predicted_target.sum() > 0:
        precision = (predicted_target & actual_target).sum() / predicted_target.sum() * 100
        print(f"\n  Predicted Class 2: {predicted_target.sum()} samples")
        print(f"  Precision (Class 2): {precision:.1f}%")

    return metrics


if __name__ == '__main__':
    main()
