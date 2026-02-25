#!/usr/bin/env python3
"""
Feature Importance Analysis for V20 Concat Model
=================================================

Analyzes which features drive model predictions using permutation importance.
Tests both temporal features (10) and context features (18).

Usage:
    python scripts/analyze_feature_importance.py \
        --model output/models/best_model_20260124_085624.pt \
        --sequences output/sequences/combined/sequences_combined.h5 \
        --metadata output/sequences/combined/metadata_combined.parquet

Outputs:
    - Feature importance rankings
    - Positive vs negative impact analysis
    - Regime-specific overfitting detection
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import h5py
from datetime import datetime
from tqdm import tqdm
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.temporal_hybrid_unified import HybridFeatureNetwork as HybridFeatureNetworkV20, create_v20_model
from models.temporal_hybrid_v18 import HybridFeatureNetwork
from config.context_features import NUM_CONTEXT_FEATURES, CONTEXT_FEATURES
from config.feature_registry import FeatureRegistry

# Temporal feature names (10 features)
TEMPORAL_FEATURES = [
    'open_normalized',       # 0
    'high_normalized',       # 1
    'low_normalized',        # 2
    'close_normalized',      # 3
    'volume_log',            # 4
    'bbw_20',                # 5
    'adx',                   # 6
    'volume_ratio_20',       # 7
    'upper_slope',           # 8
    'lower_slope'            # 9
]


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


def load_test_data(h5_path: str, metadata_path: str, norm_params: dict, test_cutoff: str = '2024-01-01'):
    """Load test set data from HDF5 file."""

    # Load sequences and labels
    with h5py.File(h5_path, 'r') as f:
        sequences = f['sequences'][:]
        labels = f['labels'][:]
        context = f['context'][:] if 'context' in f else None

    print(f"Loaded data: {sequences.shape[0]} samples")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels distribution: {np.bincount(labels)}")

    # Load metadata for temporal split
    metadata = pd.read_parquet(metadata_path)

    # Find date column
    date_col = None
    for col in ['pattern_end_date', 'pattern_start_date', 'end_date', 'date']:
        if col in metadata.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("No date column found in metadata")

    metadata[date_col] = pd.to_datetime(metadata[date_col])
    test_cutoff_dt = pd.to_datetime(test_cutoff)

    # Create test mask
    test_mask = metadata[date_col] >= test_cutoff_dt
    print(f"Test cutoff: {test_cutoff}")
    print(f"Test set size: {test_mask.sum()} samples")

    # Extract test set
    test_seq = sequences[test_mask.values]
    test_labels = labels[test_mask.values]
    test_context = context[test_mask.values] if context is not None else None
    test_metadata = metadata[test_mask].reset_index(drop=True)

    # Apply normalization
    if norm_params:
        train_median = np.array(norm_params.get('median', [0]*test_seq.shape[2]))
        train_iqr = np.array(norm_params.get('iqr', [1]*test_seq.shape[2]))
        train_iqr = np.where(train_iqr < 1e-8, 1.0, train_iqr)
        test_seq = (test_seq - train_median) / train_iqr

        if test_context is not None and 'context_median' in norm_params:
            context_median = np.array(norm_params['context_median'])
            context_iqr = np.array(norm_params['context_iqr'])
            context_iqr = np.where(context_iqr < 1e-8, 1.0, context_iqr)
            test_context = (test_context - context_median) / context_iqr

    return test_seq, test_labels, test_context, test_metadata


def get_predictions(model, sequences, context, device, batch_size=256):
    """Run inference and get class probabilities."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seq = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
            batch_ctx = torch.FloatTensor(context[i:i+batch_size]).to(device) if context is not None else None

            logits = model(batch_seq, context=batch_ctx)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def calculate_target_metric(probs: np.ndarray, labels: np.ndarray, k_pct: float = 0.15):
    """
    Calculate Top-K% Target precision.
    This is the primary metric for feature importance.
    """
    n_samples = len(labels)
    n_top_k = max(1, int(n_samples * k_pct))

    # Get Target (class 2) probability
    target_probs = probs[:, 2]

    # Get indices of top-K predictions
    top_k_indices = np.argsort(target_probs)[-n_top_k:]

    # Calculate target rate in top-K
    top_k_labels = labels[top_k_indices]
    target_rate = (top_k_labels == 2).sum() / n_top_k

    return target_rate


def permutation_importance_temporal(
    model, sequences, context, labels, device,
    n_repeats: int = 5, random_state: int = 42
):
    """
    Compute permutation importance for temporal features.

    For each temporal feature (across all timesteps), we shuffle it
    and measure the drop in Target precision.
    """
    np.random.seed(random_state)

    # Baseline metric
    probs_baseline = get_predictions(model, sequences, context, device)
    baseline_metric = calculate_target_metric(probs_baseline, labels)
    print(f"\nBaseline Top-15% Target Rate: {baseline_metric*100:.2f}%")

    n_samples, n_timesteps, n_features = sequences.shape

    importance_scores = {}

    for feat_idx in tqdm(range(n_features), desc="Temporal features"):
        feature_name = TEMPORAL_FEATURES[feat_idx] if feat_idx < len(TEMPORAL_FEATURES) else f"temporal_{feat_idx}"
        scores = []

        for rep in range(n_repeats):
            # Copy sequences and shuffle this feature across all timesteps
            seq_shuffled = sequences.copy()

            # Shuffle feature across samples (same shuffle for all timesteps)
            perm_idx = np.random.permutation(n_samples)
            seq_shuffled[:, :, feat_idx] = sequences[perm_idx, :, feat_idx]

            # Get predictions with shuffled feature
            probs_shuffled = get_predictions(model, seq_shuffled, context, device)
            shuffled_metric = calculate_target_metric(probs_shuffled, labels)

            # Importance = drop in metric
            importance = baseline_metric - shuffled_metric
            scores.append(importance)

        importance_scores[feature_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    return importance_scores, baseline_metric


def permutation_importance_context(
    model, sequences, context, labels, device,
    n_repeats: int = 5, random_state: int = 42
):
    """
    Compute permutation importance for context features.

    For each context feature, we shuffle it and measure the drop in Target precision.
    """
    np.random.seed(random_state)

    if context is None:
        print("No context features available")
        return {}, 0.0

    # Baseline metric
    probs_baseline = get_predictions(model, sequences, context, device)
    baseline_metric = calculate_target_metric(probs_baseline, labels)

    n_samples, n_context_features = context.shape

    importance_scores = {}

    for feat_idx in tqdm(range(n_context_features), desc="Context features"):
        feature_name = CONTEXT_FEATURES[feat_idx] if feat_idx < len(CONTEXT_FEATURES) else f"context_{feat_idx}"
        scores = []

        for rep in range(n_repeats):
            # Copy context and shuffle this feature
            ctx_shuffled = context.copy()

            # Shuffle feature across samples
            perm_idx = np.random.permutation(n_samples)
            ctx_shuffled[:, feat_idx] = context[perm_idx, feat_idx]

            # Get predictions with shuffled feature
            probs_shuffled = get_predictions(model, sequences, ctx_shuffled, device)
            shuffled_metric = calculate_target_metric(probs_shuffled, labels)

            # Importance = drop in metric
            importance = baseline_metric - shuffled_metric
            scores.append(importance)

        importance_scores[feature_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    return importance_scores, baseline_metric


def analyze_regime_sensitivity(
    model, sequences, context, labels, metadata, device
):
    """
    Analyze if certain features are regime-specific (overfit to bull/bear).

    Splits test data by market regime and checks if feature importance differs.
    """
    # Try to detect regime from metadata
    if 'trend_position' not in metadata.columns and context is not None:
        # Use trend_position from context via FeatureRegistry (single source of truth)
        trend_idx = FeatureRegistry.get_index('trend_position')
        trend_positions = context[:, trend_idx]
    elif 'trend_position' in metadata.columns:
        trend_positions = metadata['trend_position'].values
    else:
        print("Cannot detect market regime - skipping regime analysis")
        return {}

    # Define regimes: Bull (trend > 1.1), Bear (trend < 0.95), Sideways (0.95-1.1)
    bull_mask = trend_positions > 1.1
    bear_mask = trend_positions < 0.95
    sideways_mask = ~bull_mask & ~bear_mask

    print(f"\nRegime Distribution:")
    print(f"  Bull:     {bull_mask.sum()} samples ({bull_mask.mean()*100:.1f}%)")
    print(f"  Bear:     {bear_mask.sum()} samples ({bear_mask.mean()*100:.1f}%)")
    print(f"  Sideways: {sideways_mask.sum()} samples ({sideways_mask.mean()*100:.1f}%)")

    results = {}

    for regime_name, mask in [('bull', bull_mask), ('bear', bear_mask), ('sideways', sideways_mask)]:
        if mask.sum() < 100:
            print(f"  Skipping {regime_name} regime (too few samples)")
            continue

        regime_seq = sequences[mask]
        regime_ctx = context[mask] if context is not None else None
        regime_labels = labels[mask]

        probs = get_predictions(model, regime_seq, regime_ctx, device)
        target_rate = calculate_target_metric(probs, regime_labels)

        # Class distribution
        target_pct = (regime_labels == 2).mean() * 100
        danger_pct = (regime_labels == 0).mean() * 100

        results[regime_name] = {
            'n_samples': int(mask.sum()),
            'target_rate': target_rate * 100,
            'baseline_target': target_pct,
            'baseline_danger': danger_pct,
            'lift': target_rate * 100 / target_pct if target_pct > 0 else 0
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance for V20 model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequences', type=str, required=True, help='Path to HDF5 sequences file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata parquet')
    parser.add_argument('--test-split', type=str, default='2024-01-01',
                       help='Date cutoff for test set (YYYY-MM-DD)')
    parser.add_argument('--n-repeats', type=int, default=5,
                       help='Number of permutation repeats (default: 5)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default='output/analysis',
                       help='Directory to save results')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    model, norm_params, checkpoint = load_model(args.model, device)

    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    sequences, labels, context, metadata = load_test_data(
        args.sequences, args.metadata, norm_params, args.test_split
    )

    print(f"\nTest set: {len(labels)} samples")
    print(f"  Label distribution:")
    print(f"    Class 0 (Danger): {(labels==0).sum()} ({(labels==0).mean()*100:.1f}%)")
    print(f"    Class 1 (Noise):  {(labels==1).sum()} ({(labels==1).mean()*100:.1f}%)")
    print(f"    Class 2 (Target): {(labels==2).sum()} ({(labels==2).mean()*100:.1f}%)")

    # Permutation importance for temporal features
    print("\n" + "="*70)
    print("TEMPORAL FEATURE IMPORTANCE (10 features)")
    print("="*70)
    temporal_importance, baseline = permutation_importance_temporal(
        model, sequences, context, labels, device,
        n_repeats=args.n_repeats
    )

    # Permutation importance for context features
    print("\n" + "="*70)
    print("CONTEXT FEATURE IMPORTANCE (18 features)")
    print("="*70)
    context_importance, _ = permutation_importance_context(
        model, sequences, context, labels, device,
        n_repeats=args.n_repeats
    )

    # Combine and rank
    all_importance = {}
    for name, scores in temporal_importance.items():
        all_importance[f"T: {name}"] = scores
    for name, scores in context_importance.items():
        all_importance[f"C: {name}"] = scores

    # Sort by mean importance
    sorted_importance = sorted(
        all_importance.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )

    # Print results
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE RANKING")
    print("="*70)
    print("\nTop features that IMPROVE Target prediction when present:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Feature':<40} {'Importance':>12} {'Std':>10}")
    print("-" * 70)

    positive_features = [(n, s) for n, s in sorted_importance if s['mean'] > 0]
    for i, (name, scores) in enumerate(positive_features[:15], 1):
        print(f"{i:<5} {name:<40} {scores['mean']*100:>+10.2f}% {scores['std']*100:>10.2f}%")

    print("\n" + "="*70)
    print("Features that HURT Target prediction (potential noise/overfitting):")
    print("-" * 70)

    negative_features = [(n, s) for n, s in sorted_importance if s['mean'] < 0]
    negative_features.sort(key=lambda x: x[1]['mean'])  # Most negative first

    for i, (name, scores) in enumerate(negative_features[:10], 1):
        print(f"{i:<5} {name:<40} {scores['mean']*100:>+10.2f}% {scores['std']*100:>10.2f}%")

    # Regime analysis
    print("\n" + "="*70)
    print("REGIME SENSITIVITY ANALYSIS")
    print("="*70)
    regime_results = analyze_regime_sensitivity(
        model, sequences, context, labels, metadata, device
    )

    if regime_results:
        print("\nPerformance by Market Regime:")
        print("-" * 70)
        print(f"{'Regime':<12} {'Samples':>10} {'Top-15% Target':>15} {'Baseline':>12} {'Lift':>10}")
        print("-" * 70)

        for regime, stats in regime_results.items():
            print(f"{regime:<12} {stats['n_samples']:>10,} {stats['target_rate']:>14.1f}% "
                  f"{stats['baseline_target']:>11.1f}% {stats['lift']:>9.2f}x")

        # Check for regime-specific overfitting
        if 'bull' in regime_results and 'bear' in regime_results:
            bull_lift = regime_results['bull']['lift']
            bear_lift = regime_results['bear']['lift']

            print("\n" + "-" * 70)
            if bull_lift > bear_lift * 1.5:
                print("[WARNING] Model performs significantly better in BULL market")
                print(f"  Bull Lift: {bull_lift:.2f}x vs Bear Lift: {bear_lift:.2f}x")
                print("  This suggests overfitting to bull market patterns")
            elif bear_lift > bull_lift * 1.5:
                print("[WARNING] Model performs significantly better in BEAR market")
                print(f"  Bear Lift: {bear_lift:.2f}x vs Bull Lift: {bull_lift:.2f}x")
                print("  This suggests overfitting to bear market patterns")
            else:
                print("[OK] Model performance is balanced across regimes")
                print(f"  Bull Lift: {bull_lift:.2f}x, Bear Lift: {bear_lift:.2f}x")

    # Analysis summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    # Identify potentially problematic features
    print("\n1. TOP CONTRIBUTING FEATURES (keep these):")
    for name, scores in positive_features[:5]:
        feature_type = "Temporal" if name.startswith("T:") else "Context"
        clean_name = name.replace("T: ", "").replace("C: ", "")
        print(f"   - {clean_name} ({feature_type}): +{scores['mean']*100:.2f}% boost")

    print("\n2. POTENTIALLY HARMFUL FEATURES (consider removing):")
    harmful_threshold = -0.01  # More than 1% decrease
    harmful = [(n, s) for n, s in negative_features if s['mean'] < harmful_threshold]
    if harmful:
        for name, scores in harmful[:5]:
            feature_type = "Temporal" if name.startswith("T:") else "Context"
            clean_name = name.replace("T: ", "").replace("C: ", "")
            print(f"   - {clean_name} ({feature_type}): {scores['mean']*100:.2f}% (hurts prediction)")
    else:
        print("   None identified (all features have neutral or positive impact)")

    print("\n3. LOW-IMPACT FEATURES (candidates for simplification):")
    low_impact = [(n, s) for n, s in sorted_importance
                  if abs(s['mean']) < 0.005 and s['std'] < 0.01]
    if low_impact:
        for name, scores in low_impact[:5]:
            clean_name = name.replace("T: ", "").replace("C: ", "")
            print(f"   - {clean_name}: {scores['mean']*100:+.2f}% (negligible)")
    else:
        print("   None identified (all features have measurable impact)")

    # Save results
    results = {
        'baseline_target_rate': baseline * 100,
        'test_set_size': len(labels),
        'temporal_importance': {k: {'mean': v['mean']*100, 'std': v['std']*100}
                                for k, v in temporal_importance.items()},
        'context_importance': {k: {'mean': v['mean']*100, 'std': v['std']*100}
                               for k, v in context_importance.items()},
        'regime_analysis': regime_results,
        'timestamp': datetime.now().isoformat()
    }

    output_path = os.path.join(args.output_dir, 'feature_importance.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
