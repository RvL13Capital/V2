#!/usr/bin/env python3
"""
TRANS Visual Analysis & Improvement Dashboard
=============================================

Comprehensive visualization suite for model analysis, pattern diagnostics,
and performance improvement insights.

Usage:
    python scripts/visual_analysis.py --predictions output/predictions/predictions.parquet
    python scripts/visual_analysis.py --sequences output/sequences/sequences.npy --metadata output/sequences/metadata.parquet
    python scripts/visual_analysis.py --all  # Full analysis with all available data

Output:
    Saves figures to output/analysis/ directory
    Opens interactive dashboard if --interactive flag is set
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color schemes
COLORS = {
    'danger': '#e74c3c',
    'noise': '#95a5a6',
    'target': '#27ae60',
    'eu': '#3498db',
    'us': '#e67e22',
    'primary': '#2c3e50',
    'secondary': '#7f8c8d'
}

CLASS_NAMES = {0: 'Danger', 1: 'Noise', 2: 'Target'}
CLASS_COLORS = [COLORS['danger'], COLORS['noise'], COLORS['target']]

# Strategic values
STRATEGIC_VALUES = {0: -10, 1: -1, 2: +10}


def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions parquet file with column normalization."""
    df = pd.read_parquet(path)

    # Column mapping for different file formats
    column_map = {
        'expected_value': 'predicted_ev',
        'true_class': 'actual_class',
        'danger_prob': 'prob_0',
        'noise_prob': 'prob_1',
        'target_prob': 'prob_2',
        'label': 'actual_class'  # Alternative name
    }

    for old_name, new_name in column_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Calculate EV if not present
    if 'predicted_ev' not in df.columns and 'prob_0' in df.columns:
        df['predicted_ev'] = (
            df['prob_0'] * STRATEGIC_VALUES[0] +
            df['prob_1'] * STRATEGIC_VALUES[1] +
            df['prob_2'] * STRATEGIC_VALUES[2]
        )

    # Calculate actual value if not present
    if 'actual_value' not in df.columns and 'actual_class' in df.columns:
        df['actual_value'] = df['actual_class'].map(STRATEGIC_VALUES)

    return df


def load_sequences(seq_path: str, meta_path: str,
                   sample_size: Optional[int] = None,
                   mmap_mode: Optional[str] = 'r') -> Tuple[np.ndarray, pd.DataFrame]:
    """Load sequences and metadata with memory-efficient options.

    Args:
        seq_path: Path to sequences .npy file
        meta_path: Path to metadata .parquet file
        sample_size: If set, randomly sample N sequences (faster, less memory)
        mmap_mode: Memory-map mode ('r' for read-only, None to load into RAM)

    Returns:
        Tuple of (sequences array, metadata dataframe)
    """
    # Use memory mapping for large files (doesn't load into RAM until accessed)
    sequences = np.load(seq_path, mmap_mode=mmap_mode)
    metadata = pd.read_parquet(meta_path)

    # Sample if requested (for faster analysis on large datasets)
    if sample_size and sample_size < len(sequences):
        np.random.seed(42)  # Reproducible sampling
        indices = np.random.choice(len(sequences), sample_size, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        sequences = np.array(sequences[indices])  # Load sampled subset into RAM
        metadata = metadata.iloc[indices].reset_index(drop=True)
        print(f"   Sampled {sample_size:,} from {len(sequences):,} sequences")

    return sequences, metadata


def calculate_metrics_by_threshold(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    """Calculate metrics at different EV thresholds."""
    results = []

    for thresh in thresholds:
        mask = df['predicted_ev'] >= thresh
        subset = df[mask]

        if len(subset) == 0:
            continue

        results.append({
            'threshold': thresh,
            'n_signals': len(subset),
            'pct_signals': len(subset) / len(df) * 100,
            'target_rate': (subset['actual_class'] == 2).mean() * 100,
            'danger_rate': (subset['actual_class'] == 0).mean() * 100,
            'avg_actual_value': subset['actual_value'].mean(),
            'total_value': subset['actual_value'].sum()
        })

    return pd.DataFrame(results)


# =============================================================================
# PLOT 1: EV Calibration (Predicted vs Realized)
# =============================================================================
def plot_ev_calibration(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    EV Calibration Plot: Shows how well predicted EV matches realized EV.

    A well-calibrated model should show points along the diagonal.
    Points above diagonal = model is conservative (good)
    Points below diagonal = model is overconfident (bad)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Bin predictions by EV
    df['ev_bin'] = pd.cut(df['predicted_ev'], bins=20)
    calibration = df.groupby('ev_bin', observed=True).agg({
        'predicted_ev': 'mean',
        'actual_value': ['mean', 'std', 'count']
    }).dropna()

    calibration.columns = ['pred_ev', 'actual_ev', 'std', 'count']
    calibration = calibration[calibration['count'] >= 10]  # Min samples

    # Plot
    sizes = calibration['count'] / calibration['count'].max() * 300 + 50
    scatter = ax.scatter(
        calibration['pred_ev'],
        calibration['actual_ev'],
        s=sizes,
        c=calibration['actual_ev'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='white',
        linewidths=1
    )

    # Error bars
    ax.errorbar(
        calibration['pred_ev'],
        calibration['actual_ev'],
        yerr=calibration['std'] / np.sqrt(calibration['count']),
        fmt='none',
        color=COLORS['secondary'],
        alpha=0.5,
        capsize=3
    )

    # Perfect calibration line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Calibration')
    ax.fill_between(lims, [l-1 for l in lims], [l+1 for l in lims],
                    alpha=0.1, color='gray', label='±1 Tolerance')

    ax.set_xlabel('Predicted EV', fontsize=12)
    ax.set_ylabel('Realized EV', fontsize=12)
    ax.set_title('EV Calibration: Predicted vs Realized\n(bubble size = sample count)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    # Add correlation
    corr = df['predicted_ev'].corr(df['actual_value'])
    ax.text(0.95, 0.05, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.colorbar(scatter, ax=ax, label='Realized EV')

    return fig


# =============================================================================
# PLOT 2: Win Rate by EV Bucket
# =============================================================================
def plot_winrate_by_ev(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Win Rate Analysis: Shows target hit rate across EV buckets.

    Key insight: Higher EV should correlate with higher win rate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Create EV buckets
    bins = [-10, -5, -2, 0, 1, 2, 3, 4, 5, 6, 8, 10]
    labels = ['<-5', '-5:-2', '-2:0', '0:1', '1:2', '2:3', '3:4', '4:5', '5:6', '6:8', '8+']
    df['ev_bucket'] = pd.cut(df['predicted_ev'], bins=bins, labels=labels[:len(bins)-1])

    # Calculate stats per bucket
    bucket_stats = df.groupby('ev_bucket', observed=True).agg({
        'actual_class': lambda x: (x == 2).mean() * 100,  # Target rate
        'predicted_ev': 'count'
    }).rename(columns={'actual_class': 'target_rate', 'predicted_ev': 'count'})

    # Also get danger rate
    danger_rate = df.groupby('ev_bucket', observed=True)['actual_class'].apply(
        lambda x: (x == 0).mean() * 100
    )
    bucket_stats['danger_rate'] = danger_rate

    bucket_stats = bucket_stats.dropna()

    # Plot
    x = range(len(bucket_stats))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], bucket_stats['target_rate'],
                   width, label='Target Rate', color=COLORS['target'], alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], bucket_stats['danger_rate'],
                   width, label='Danger Rate', color=COLORS['danger'], alpha=0.8)

    # Add count labels
    for i, (idx, row) in enumerate(bucket_stats.iterrows()):
        ax.annotate(f'n={int(row["count"])}',
                   xy=(i, max(row['target_rate'], row['danger_rate']) + 2),
                   ha='center', fontsize=8, color=COLORS['secondary'])

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_stats.index, rotation=45, ha='right')
    ax.set_xlabel('Predicted EV Bucket', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Win Rate & Danger Rate by EV Bucket\n(higher EV should → higher win rate)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add threshold line at EV > 2
    for i, label in enumerate(bucket_stats.index):
        if label in ['2:3', '3:4', '4:5', '5:6', '6:8', '8+']:
            ax.axvspan(i - 0.5, len(bucket_stats) - 0.5, alpha=0.1, color=COLORS['target'])
            ax.text(i, 95, '← Trading Zone (EV > 2)', fontsize=9, color=COLORS['target'])
            break

    return fig


# =============================================================================
# PLOT 3: Confusion Matrix with Values
# =============================================================================
def plot_confusion_matrix(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Enhanced Confusion Matrix with strategic value overlay.

    Shows not just counts but the VALUE implications of each cell.
    """
    from sklearn.metrics import confusion_matrix

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig = ax.figure
        axes = [ax, ax]  # Fallback

    # Get predictions
    if 'predicted_class' not in df.columns:
        df['predicted_class'] = df[['prob_0', 'prob_1', 'prob_2']].values.argmax(axis=1)

    cm = confusion_matrix(df['actual_class'], df['predicted_class'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot 1: Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Danger', 'Noise', 'Target'],
                yticklabels=['Danger', 'Noise', 'Target'])
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Plot 2: Percentages with value annotations
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Danger', 'Noise', 'Target'],
                yticklabels=['Danger', 'Noise', 'Target'],
                center=50)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (% of Actual)', fontsize=14, fontweight='bold')

    # Add value annotation
    value_text = "Strategic Values:\nDanger=-10, Noise=-1, Target=+10"
    axes[1].text(1.02, 0.5, value_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 4: Temporal Feature Heatmap by Class
# =============================================================================
def plot_temporal_heatmap(sequences: np.ndarray, labels: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Temporal Feature Heatmap: Average feature values over 20 timesteps by class.

    Reveals which features and when they differentiate outcomes.
    """
    if feature_names is None:
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'bbw_20', 'adx', 'volume_ratio',
            'vol_dryup', 'var_score', 'nes_score', 'lpf_score',
            'upper_bound', 'lower_bound'
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for class_idx, class_name in CLASS_NAMES.items():
        class_mask = labels == class_idx
        class_sequences = sequences[class_mask]

        if len(class_sequences) == 0:
            continue

        # Average across samples: (n_samples, 20, 14) -> (20, 14)
        avg_sequence = class_sequences.mean(axis=0)

        # Transpose for better visualization: (14, 20)
        im = axes[class_idx].imshow(avg_sequence.T, aspect='auto', cmap='RdYlBu_r')

        axes[class_idx].set_xlabel('Timestep (Day)', fontsize=11)
        axes[class_idx].set_ylabel('Feature', fontsize=11)
        axes[class_idx].set_title(f'{class_name} Patterns\n(n={class_mask.sum():,})',
                                   fontsize=13, fontweight='bold',
                                   color=CLASS_COLORS[class_idx])

        axes[class_idx].set_yticks(range(len(feature_names)))
        axes[class_idx].set_yticklabels(feature_names, fontsize=9)
        axes[class_idx].set_xticks([0, 5, 10, 15, 19])
        axes[class_idx].set_xticklabels(['1', '6', '11', '16', '20'])

        plt.colorbar(im, ax=axes[class_idx], shrink=0.8)

    fig.suptitle('Average Temporal Feature Evolution by Outcome Class\n'
                 '(What patterns look like before Danger vs Target)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# =============================================================================
# PLOT 5: Feature Importance / Discriminative Power
# =============================================================================
def plot_feature_importance(sequences: np.ndarray, labels: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Feature Discriminative Power: Which features best separate Target from Danger.

    Uses simple statistical test (effect size) between classes.
    """
    if feature_names is None:
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'bbw_20', 'adx', 'volume_ratio',
            'vol_dryup', 'var_score', 'nes_score', 'lpf_score',
            'upper_bound', 'lower_bound'
        ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Get class masks
    danger_mask = labels == 0
    target_mask = labels == 2

    # Calculate effect size (Cohen's d) for each feature at each timestep
    effect_sizes = np.zeros((14, 20))

    danger_seqs = sequences[danger_mask]
    target_seqs = sequences[target_mask]

    for feat_idx in range(14):
        for time_idx in range(20):
            danger_vals = danger_seqs[:, time_idx, feat_idx]
            target_vals = target_seqs[:, time_idx, feat_idx]

            pooled_std = np.sqrt((danger_vals.std()**2 + target_vals.std()**2) / 2)
            if pooled_std > 0:
                effect_sizes[feat_idx, time_idx] = (target_vals.mean() - danger_vals.mean()) / pooled_std

    # Plot 1: Heatmap of effect sizes
    im = axes[0].imshow(effect_sizes, aspect='auto', cmap='RdYlGn',
                        vmin=-1, vmax=1)
    axes[0].set_xlabel('Timestep', fontsize=11)
    axes[0].set_ylabel('Feature', fontsize=11)
    axes[0].set_title("Effect Size (Cohen's d): Target vs Danger\n"
                      "(Green = higher in Target, Red = higher in Danger)",
                      fontsize=12, fontweight='bold')
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_yticklabels(feature_names, fontsize=9)
    plt.colorbar(im, ax=axes[0], label="Cohen's d")

    # Plot 2: Aggregated importance (mean absolute effect across time)
    mean_abs_effect = np.abs(effect_sizes).mean(axis=1)
    sorted_idx = np.argsort(mean_abs_effect)[::-1]

    colors = [COLORS['target'] if effect_sizes[i].mean() > 0 else COLORS['danger']
              for i in sorted_idx]

    bars = axes[1].barh(range(len(feature_names)), mean_abs_effect[sorted_idx], color=colors)
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    axes[1].set_xlabel('Mean |Effect Size|', fontsize=11)
    axes[1].set_title('Feature Discriminative Power\n(ranked by ability to separate outcomes)',
                      fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['target'], label='Higher in Target'),
        Patch(facecolor=COLORS['danger'], label='Higher in Danger')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 6: Threshold Analysis
# =============================================================================
def plot_threshold_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Threshold Sensitivity Analysis: How metrics change with EV threshold.

    Helps find optimal trading threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    thresholds = np.arange(-5, 8, 0.5)
    metrics = calculate_metrics_by_threshold(df, thresholds)

    if len(metrics) == 0:
        return fig

    # Plot 1: Target Rate vs Danger Rate
    ax = axes[0, 0]
    ax.plot(metrics['threshold'], metrics['target_rate'],
            'o-', color=COLORS['target'], label='Target Rate', linewidth=2)
    ax.plot(metrics['threshold'], metrics['danger_rate'],
            's-', color=COLORS['danger'], label='Danger Rate', linewidth=2)
    ax.axvline(x=2.0, color=COLORS['primary'], linestyle='--', alpha=0.7, label='EV=2 Threshold')
    ax.set_xlabel('EV Threshold', fontsize=11)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Target & Danger Rate vs Threshold', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of Signals
    ax = axes[0, 1]
    ax.fill_between(metrics['threshold'], metrics['pct_signals'], alpha=0.3, color=COLORS['primary'])
    ax.plot(metrics['threshold'], metrics['pct_signals'],
            color=COLORS['primary'], linewidth=2)
    ax.axvline(x=2.0, color=COLORS['danger'], linestyle='--', alpha=0.7)
    ax.set_xlabel('EV Threshold', fontsize=11)
    ax.set_ylabel('% of All Patterns', fontsize=11)
    ax.set_title('Signal Rate vs Threshold\n(% of patterns that generate signals)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Average Realized Value
    ax = axes[1, 0]
    ax.plot(metrics['threshold'], metrics['avg_actual_value'],
            'D-', color=COLORS['target'], linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=2.0, color=COLORS['danger'], linestyle='--', alpha=0.7, label='EV=2')
    ax.fill_between(metrics['threshold'], 0, metrics['avg_actual_value'],
                    where=metrics['avg_actual_value'] > 0, alpha=0.2, color=COLORS['target'])
    ax.fill_between(metrics['threshold'], 0, metrics['avg_actual_value'],
                    where=metrics['avg_actual_value'] < 0, alpha=0.2, color=COLORS['danger'])
    ax.set_xlabel('EV Threshold', fontsize=11)
    ax.set_ylabel('Average Realized Value', fontsize=11)
    ax.set_title('Realized Value vs Threshold\n(expected profit per trade)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Efficiency Frontier (Target Rate vs Signal Rate)
    ax = axes[1, 1]
    scatter = ax.scatter(metrics['pct_signals'], metrics['target_rate'],
                        c=metrics['threshold'], cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Signal Rate (%)', fontsize=11)
    ax.set_ylabel('Target Rate (%)', fontsize=11)
    ax.set_title('Efficiency Frontier\n(trade-off: more signals vs higher win rate)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='EV Threshold')

    # Annotate key thresholds
    for thresh in [1, 2, 3, 4]:
        row = metrics[metrics['threshold'] == thresh]
        if len(row) > 0:
            ax.annotate(f'EV>{thresh}',
                       (row['pct_signals'].values[0], row['target_rate'].values[0]),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 7: Regional Comparison
# =============================================================================
def plot_regional_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Regional Comparison: EU vs US performance differences.
    """
    # Detect region from ticker suffix
    EU_SUFFIXES = ['.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS',
                   '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI']

    if 'ticker' not in df.columns:
        # Try to infer from other columns
        print("Warning: No ticker column found, skipping regional comparison")
        return plt.figure()

    df['region'] = df['ticker'].apply(
        lambda x: 'EU' if any(x.endswith(s) for s in EU_SUFFIXES) else 'US'
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Class Distribution by Region
    ax = axes[0, 0]
    for i, region in enumerate(['EU', 'US']):
        region_df = df[df['region'] == region]
        if len(region_df) == 0:
            continue
        class_dist = region_df['actual_class'].value_counts(normalize=True).sort_index() * 100
        x_pos = np.arange(3) + i * 0.35
        bars = ax.bar(x_pos, [class_dist.get(c, 0) for c in range(3)],
                     width=0.35, label=region,
                     color=COLORS['eu'] if region == 'EU' else COLORS['us'],
                     alpha=0.8)
    ax.set_xticks(np.arange(3) + 0.175)
    ax.set_xticklabels(['Danger', 'Noise', 'Target'])
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Outcome Distribution by Region', fontsize=12, fontweight='bold')
    ax.legend()

    # Plot 2: EV Distribution by Region
    ax = axes[0, 1]
    for region, color in [('EU', COLORS['eu']), ('US', COLORS['us'])]:
        region_df = df[df['region'] == region]
        if len(region_df) > 0:
            ax.hist(region_df['predicted_ev'], bins=50, alpha=0.5,
                   label=f'{region} (n={len(region_df):,})', color=color, density=True)
    ax.axvline(x=2.0, color=COLORS['danger'], linestyle='--', label='EV=2 Threshold')
    ax.set_xlabel('Predicted EV', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('EV Distribution by Region', fontsize=12, fontweight='bold')
    ax.legend()

    # Plot 3: Win Rate Comparison at Different Thresholds
    ax = axes[1, 0]
    thresholds = [1, 2, 3, 4]
    width = 0.35

    for i, region in enumerate(['EU', 'US']):
        region_df = df[df['region'] == region]
        if len(region_df) == 0:
            continue

        target_rates = []
        for thresh in thresholds:
            subset = region_df[region_df['predicted_ev'] >= thresh]
            if len(subset) > 0:
                target_rates.append((subset['actual_class'] == 2).mean() * 100)
            else:
                target_rates.append(0)

        x_pos = np.arange(len(thresholds)) + i * width
        ax.bar(x_pos, target_rates, width, label=region,
              color=COLORS['eu'] if region == 'EU' else COLORS['us'])

    ax.set_xticks(np.arange(len(thresholds)) + width/2)
    ax.set_xticklabels([f'EV>{t}' for t in thresholds])
    ax.set_ylabel('Target Rate (%)', fontsize=11)
    ax.set_title('Target Rate by Region at Different Thresholds', fontsize=12, fontweight='bold')
    ax.legend()

    # Plot 4: Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = []
    for region in ['EU', 'US']:
        region_df = df[df['region'] == region]
        if len(region_df) == 0:
            continue

        ev2_df = region_df[region_df['predicted_ev'] >= 2]

        summary_data.append([
            region,
            f"{len(region_df):,}",
            f"{(region_df['actual_class'] == 0).mean()*100:.1f}%",
            f"{(region_df['actual_class'] == 2).mean()*100:.1f}%",
            f"{len(ev2_df):,} ({len(ev2_df)/len(region_df)*100:.1f}%)" if len(region_df) > 0 else "0",
            f"{(ev2_df['actual_class'] == 2).mean()*100:.1f}%" if len(ev2_df) > 0 else "N/A",
            f"{ev2_df['actual_value'].mean():.2f}" if len(ev2_df) > 0 else "N/A"
        ])

    if summary_data:
        table = ax.table(
            cellText=summary_data,
            colLabels=['Region', 'Patterns', 'Danger%', 'Target%', 'EV>2 Signals', 'Win Rate', 'Avg Value'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title('Summary Statistics by Region', fontsize=12, fontweight='bold', y=0.8)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 8: Pattern Characteristics
# =============================================================================
def plot_pattern_characteristics(metadata: pd.DataFrame) -> plt.Figure:
    """
    Pattern Characteristics: What pattern attributes predict success.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    required_cols = ['box_width', 'pattern_duration', 'outcome_class']
    available_cols = [c for c in required_cols if c in metadata.columns]

    if len(available_cols) < 2:
        print(f"Warning: Need at least 2 of {required_cols}, found {available_cols}")
        return fig

    # Plot 1: Box Width Distribution by Outcome
    if 'box_width' in metadata.columns:
        ax = axes[0, 0]
        for class_idx, class_name in CLASS_NAMES.items():
            class_data = metadata[metadata['outcome_class'] == class_idx]['box_width']
            if len(class_data) > 0:
                ax.hist(class_data * 100, bins=30, alpha=0.5,
                       label=f'{class_name} (μ={class_data.mean()*100:.1f}%)',
                       color=CLASS_COLORS[class_idx], density=True)
        ax.set_xlabel('Box Width (%)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Box Width Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 2: Pattern Duration Distribution
    if 'pattern_duration' in metadata.columns:
        ax = axes[0, 1]
        for class_idx, class_name in CLASS_NAMES.items():
            class_data = metadata[metadata['outcome_class'] == class_idx]['pattern_duration']
            if len(class_data) > 0:
                ax.hist(class_data, bins=30, alpha=0.5,
                       label=f'{class_name} (μ={class_data.mean():.0f}d)',
                       color=CLASS_COLORS[class_idx], density=True)
        ax.set_xlabel('Pattern Duration (days)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Pattern Duration Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 3: Box Width vs Duration scatter
    if 'box_width' in metadata.columns and 'pattern_duration' in metadata.columns:
        ax = axes[1, 0]
        for class_idx, class_name in CLASS_NAMES.items():
            class_data = metadata[metadata['outcome_class'] == class_idx]
            if len(class_data) > 0:
                ax.scatter(class_data['pattern_duration'], class_data['box_width'] * 100,
                          alpha=0.3, label=class_name, color=CLASS_COLORS[class_idx], s=20)
        ax.set_xlabel('Pattern Duration (days)', fontsize=11)
        ax.set_ylabel('Box Width (%)', fontsize=11)
        ax.set_title('Pattern Duration vs Box Width', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 4: Market Cap Distribution (if available)
    if 'market_cap_category' in metadata.columns:
        ax = axes[1, 1]
        cap_order = ['nano_cap', 'micro_cap', 'small_cap', 'mid_cap', 'large_cap', 'mega_cap']

        for class_idx, class_name in CLASS_NAMES.items():
            class_data = metadata[metadata['outcome_class'] == class_idx]['market_cap_category']
            if len(class_data) > 0:
                counts = class_data.value_counts(normalize=True)
                counts = counts.reindex(cap_order).fillna(0) * 100
                x_pos = np.arange(len(cap_order)) + class_idx * 0.25
                ax.bar(x_pos, counts.values, width=0.25, label=class_name,
                      color=CLASS_COLORS[class_idx], alpha=0.8)

        ax.set_xticks(np.arange(len(cap_order)) + 0.25)
        ax.set_xticklabels(['Nano', 'Micro', 'Small', 'Mid', 'Large', 'Mega'], rotation=45)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Market Cap Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Market cap data not available',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 9: Time Series Performance
# =============================================================================
def plot_performance_over_time(df: pd.DataFrame) -> plt.Figure:
    """
    Performance Over Time: Track model performance across time periods.

    Detects potential drift.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if 'pattern_end_date' not in df.columns and 'date' not in df.columns:
        print("Warning: No date column found for time series analysis")
        return fig

    date_col = 'pattern_end_date' if 'pattern_end_date' in df.columns else 'date'
    df['date'] = pd.to_datetime(df[date_col])
    df['year_month'] = df['date'].dt.to_period('M')

    # Monthly aggregation
    monthly = df.groupby('year_month').agg({
        'actual_class': [
            lambda x: (x == 2).mean() * 100,  # Target rate
            lambda x: (x == 0).mean() * 100,  # Danger rate
            'count'
        ],
        'predicted_ev': 'mean',
        'actual_value': 'mean'
    })
    monthly.columns = ['target_rate', 'danger_rate', 'count', 'avg_pred_ev', 'avg_actual_value']
    monthly = monthly[monthly['count'] >= 10]  # Min samples

    if len(monthly) == 0:
        return fig

    x = range(len(monthly))

    # Plot 1: Target & Danger Rate Over Time
    ax = axes[0, 0]
    ax.plot(x, monthly['target_rate'], 'o-', color=COLORS['target'],
            label='Target Rate', linewidth=2)
    ax.plot(x, monthly['danger_rate'], 's-', color=COLORS['danger'],
            label='Danger Rate', linewidth=2)
    ax.fill_between(x, monthly['target_rate'], alpha=0.2, color=COLORS['target'])
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([str(monthly.index[i]) for i in x[::max(1, len(x)//10)]], rotation=45)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Target & Danger Rate Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Pattern Count Over Time
    ax = axes[0, 1]
    ax.bar(x, monthly['count'], color=COLORS['primary'], alpha=0.7)
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([str(monthly.index[i]) for i in x[::max(1, len(x)//10)]], rotation=45)
    ax.set_ylabel('Pattern Count', fontsize=11)
    ax.set_title('Pattern Volume Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Predicted vs Actual Value Over Time
    ax = axes[1, 0]
    ax.plot(x, monthly['avg_pred_ev'], 'o-', color=COLORS['eu'],
            label='Avg Predicted EV', linewidth=2)
    ax.plot(x, monthly['avg_actual_value'], 's-', color=COLORS['us'],
            label='Avg Realized Value', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([str(monthly.index[i]) for i in x[::max(1, len(x)//10)]], rotation=45)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Predicted vs Realized Value Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Rolling Calibration
    ax = axes[1, 1]
    df_sorted = df.sort_values('date')
    window = min(500, len(df_sorted) // 10)
    if window >= 50:
        rolling_corr = df_sorted['predicted_ev'].rolling(window).corr(df_sorted['actual_value'])
        ax.plot(df_sorted['date'], rolling_corr, color=COLORS['primary'], alpha=0.7)
        ax.axhline(y=0.3, color=COLORS['target'], linestyle='--',
                  label='Target Correlation (0.3)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_ylabel('Rolling Correlation', fontsize=11)
        ax.set_title(f'Rolling Calibration (window={window})\n'
                    '(EV-Value correlation over time)', fontsize=12, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data for rolling correlation',
               ha='center', va='center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 10: Improvement Recommendations
# =============================================================================
def generate_improvement_report(df: pd.DataFrame, sequences: Optional[np.ndarray] = None,
                                 labels: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Generate visual improvement recommendations based on analysis.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Collect insights
    insights = []

    # 1. Calibration Analysis
    corr = df['predicted_ev'].corr(df['actual_value'])
    if corr < 0.2:
        insights.append(("🔴 CRITICAL", "Low EV correlation ({:.3f}). Model predictions poorly calibrated.".format(corr)))
    elif corr < 0.3:
        insights.append(("🟡 WARNING", "Moderate EV correlation ({:.3f}). Consider temperature scaling.".format(corr)))
    else:
        insights.append(("🟢 GOOD", "EV correlation ({:.3f}) meets target.".format(corr)))

    # 2. Class Balance
    class_dist = df['actual_class'].value_counts(normalize=True)
    if class_dist.get(2, 0) < 0.10:
        insights.append(("🟡 WARNING", "Low target rate ({:.1f}%). Consider adjusting R thresholds.".format(class_dist.get(2, 0)*100)))

    # 3. High EV Performance
    ev2_df = df[df['predicted_ev'] >= 2]
    if len(ev2_df) > 0:
        ev2_target_rate = (ev2_df['actual_class'] == 2).mean() * 100
        if ev2_target_rate < 50:
            insights.append(("🟡 WARNING", "EV>2 target rate ({:.1f}%) below 50% goal.".format(ev2_target_rate)))
        else:
            insights.append(("🟢 GOOD", "EV>2 target rate ({:.1f}%) meets goal.".format(ev2_target_rate)))

    # 4. Signal Rate
    signal_rate = len(ev2_df) / len(df) * 100 if len(df) > 0 else 0
    if signal_rate < 1:
        insights.append(("🟡 WARNING", "Low signal rate ({:.1f}%). Model may be too conservative.".format(signal_rate)))
    elif signal_rate > 10:
        insights.append(("🟡 WARNING", "High signal rate ({:.1f}%). Model may be overconfident.".format(signal_rate)))
    else:
        insights.append(("🟢 GOOD", "Signal rate ({:.1f}%) in healthy range.".format(signal_rate)))

    # Add recommendations based on insights
    recommendations = []

    for severity, msg in insights:
        if "Low EV correlation" in msg:
            recommendations.append("• Apply temperature scaling (T=1.5) to calibrate probabilities")
            recommendations.append("• Consider retraining with different class weights")
        if "Low target rate" in msg:
            recommendations.append("• Experiment with tighter R thresholds (e.g., +4R/-1.5R for EU)")
            recommendations.append("• Filter to small/micro cap stocks only")
        if "below 50%" in msg:
            recommendations.append("• Increase EV threshold to 3.0+ for trading")
            recommendations.append("• Focus on EU patterns (5x better than US)")
        if "too conservative" in msg:
            recommendations.append("• Lower EV threshold or adjust class weights")
        if "overconfident" in msg:
            recommendations.append("• Increase EV threshold or add regularization")

    # Plot insights
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')

    y_pos = 0.95
    ax.text(0.5, y_pos + 0.05, "📊 Analysis Results & Recommendations",
            fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)

    for severity, msg in insights:
        ax.text(0.1, y_pos, f"{severity}: {msg}", fontsize=11,
               transform=ax.transAxes, verticalalignment='top')
        y_pos -= 0.12

    y_pos -= 0.1
    ax.text(0.1, y_pos, "💡 Recommendations:", fontsize=13, fontweight='bold',
           transform=ax.transAxes)
    y_pos -= 0.12

    for rec in recommendations[:5]:  # Top 5 recommendations
        ax.text(0.12, y_pos, rec, fontsize=10, transform=ax.transAxes)
        y_pos -= 0.10

    # Key metrics summary
    ax2 = fig.add_subplot(gs[1, 0])
    metrics_summary = [
        ["Metric", "Value", "Target", "Status"],
        ["EV Correlation", f"{corr:.3f}", "> 0.30", "✅" if corr >= 0.3 else "❌"],
        ["Target Rate (all)", f"{class_dist.get(2, 0)*100:.1f}%", "> 10%", "✅" if class_dist.get(2, 0) >= 0.1 else "❌"],
        ["EV>2 Win Rate", f"{ev2_target_rate:.1f}%" if len(ev2_df) > 0 else "N/A", "> 50%", "✅" if len(ev2_df) > 0 and ev2_target_rate >= 50 else "❌"],
        ["Signal Rate (EV>2)", f"{signal_rate:.1f}%", "2-5%", "✅" if 1 <= signal_rate <= 10 else "❌"],
        ["Danger Rate (all)", f"{class_dist.get(0, 0)*100:.1f}%", "< 45%", "✅" if class_dist.get(0, 0) < 0.45 else "❌"],
    ]

    table = ax2.table(cellText=metrics_summary[1:], colLabels=metrics_summary[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax2.axis('off')
    ax2.set_title('Key Performance Metrics', fontsize=12, fontweight='bold', y=0.9)

    # Quick wins chart
    ax3 = fig.add_subplot(gs[1, 1])
    quick_wins = [
        ("Use EU only", 5.0),
        ("EV > 3.0 threshold", 3.5),
        ("+4R/-1.5R config", 3.0),
        ("Filter micro/small cap", 2.5),
        ("Temperature scaling", 2.0),
    ]

    y_pos = range(len(quick_wins))
    ax3.barh(y_pos, [x[1] for x in quick_wins], color=COLORS['target'], alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([x[0] for x in quick_wins])
    ax3.set_xlabel('Expected Impact (EV improvement)', fontsize=11)
    ax3.set_title('Quick Wins (Expected Impact)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()

    # Next steps
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    next_steps = """
    📋 Next Steps for Improvement:

    1. IMMEDIATE: Apply temperature scaling (T=1.5) to fix probability calibration
    2. SHORT-TERM: Retrain EU-only model with strided evolution (146k sequences)
    3. SHORT-TERM: Implement weekly drift monitoring (PSI > 0.25 → retrain)
    4. MEDIUM-TERM: Add context features (5 features for Branch B GRN)
    5. LONG-TERM: Expand universe and collect more US quality filters

    Run: python pipeline/evaluate_trading_performance.py --drift-only  (weekly)
    """

    ax4.text(0.1, 0.9, next_steps, fontsize=11, transform=ax4.transAxes,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def create_dashboard(predictions_path: Optional[str] = None,
                     sequences_path: Optional[str] = None,
                     metadata_path: Optional[str] = None,
                     labels_path: Optional[str] = None,
                     output_dir: str = "output/analysis",
                     sample_size: int = 25000,
                     fast_mode: bool = False) -> None:
    """
    Create complete visual analysis dashboard.

    Args:
        predictions_path: Path to predictions parquet
        sequences_path: Path to sequences npy
        metadata_path: Path to metadata parquet
        labels_path: Path to labels npy
        output_dir: Output directory for plots
        sample_size: Max samples to use (0 = all data). Default 25k for fast analysis.
        fast_mode: Skip slow plots, use 10k samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fast mode settings
    if fast_mode:
        sample_size = min(sample_size, 10000) if sample_size > 0 else 10000
        print("[FAST MODE] Using 10k samples, skipping slow plots")

    print("=" * 60)
    print("TRANS Visual Analysis Dashboard")
    print(f"Sample size: {sample_size:,}" if sample_size > 0 else "Sample size: ALL")
    print("=" * 60)

    df = None
    sequences = None
    labels = None
    metadata = None

    # Load data with sampling
    if predictions_path and Path(predictions_path).exists():
        print(f"\n[DATA] Loading predictions from {predictions_path}")
        df = load_predictions(predictions_path)
        total_preds = len(df)
        if sample_size > 0 and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"   Sampled {len(df):,} from {total_preds:,} predictions")
        else:
            print(f"   Loaded {len(df):,} predictions")

    if sequences_path and Path(sequences_path).exists():
        print(f"\n[DATA] Loading sequences from {sequences_path}")
        # Use memory-mapped loading
        sequences_mmap = np.load(sequences_path, mmap_mode='r')
        total_seqs = len(sequences_mmap)

        if sample_size > 0 and total_seqs > sample_size:
            np.random.seed(42)
            indices = np.random.choice(total_seqs, sample_size, replace=False)
            indices = np.sort(indices)
            sequences = np.array(sequences_mmap[indices])  # Copy sampled to RAM
            print(f"   Sampled {len(sequences):,} from {total_seqs:,} sequences")
            print(f"   Memory: {sequences.nbytes / (1024*1024):.1f} MB")
        else:
            sequences = np.array(sequences_mmap)  # Copy all to RAM
            print(f"   Loaded sequences: {sequences.shape}")
            print(f"   Memory: {sequences.nbytes / (1024*1024):.1f} MB")

        # Load matching labels
        if labels_path and Path(labels_path).exists():
            labels_all = np.load(labels_path, mmap_mode='r')
            if sample_size > 0 and total_seqs > sample_size:
                labels = np.array(labels_all[indices])
            else:
                labels = np.array(labels_all)
            print(f"   Loaded labels: {labels.shape}")

    if metadata_path and Path(metadata_path).exists():
        print(f"\n[DATA] Loading metadata from {metadata_path}")
        metadata = pd.read_parquet(metadata_path)
        if sample_size > 0 and 'indices' in dir() and len(metadata) > sample_size:
            metadata = metadata.iloc[indices].reset_index(drop=True)
        print(f"   Loaded {len(metadata):,} metadata records")

        # Use metadata labels if separate labels not provided
        if labels is None and 'outcome_class' in metadata.columns:
            labels = metadata['outcome_class'].values

    figures = []

    # Generate plots
    if df is not None:
        print("\n[PLOT] Generating prediction analysis plots...")

        print("   1/7 EV Calibration")
        fig = plot_ev_calibration(df)
        fig.savefig(output_path / f"01_ev_calibration_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("EV Calibration", fig))

        print("   2/7 Win Rate by EV")
        fig = plot_winrate_by_ev(df)
        fig.savefig(output_path / f"02_winrate_by_ev_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Win Rate by EV", fig))

        print("   3/7 Confusion Matrix")
        fig = plot_confusion_matrix(df)
        fig.savefig(output_path / f"03_confusion_matrix_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Confusion Matrix", fig))

        print("   4/7 Threshold Analysis")
        fig = plot_threshold_analysis(df)
        fig.savefig(output_path / f"04_threshold_analysis_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Threshold Analysis", fig))

        print("   5/7 Regional Comparison")
        fig = plot_regional_comparison(df)
        fig.savefig(output_path / f"05_regional_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Regional Comparison", fig))

        print("   6/7 Performance Over Time")
        fig = plot_performance_over_time(df)
        fig.savefig(output_path / f"06_performance_time_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Performance Over Time", fig))

        print("   7/7 Improvement Report")
        fig = generate_improvement_report(df, sequences, labels)
        fig.savefig(output_path / f"07_improvement_report_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Improvement Report", fig))

    if sequences is not None and labels is not None:
        print("\n[PLOT] Generating sequence analysis plots...")

        print("   8/9 Temporal Feature Heatmap")
        fig = plot_temporal_heatmap(sequences, labels)
        fig.savefig(output_path / f"08_temporal_heatmap_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Temporal Heatmap", fig))

        print("   9/9 Feature Importance")
        fig = plot_feature_importance(sequences, labels)
        fig.savefig(output_path / f"09_feature_importance_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Feature Importance", fig))

    if metadata is not None:
        print("\n[PLOT] Generating pattern characteristic plots...")

        print("   10/10 Pattern Characteristics")
        fig = plot_pattern_characteristics(metadata)
        fig.savefig(output_path / f"10_pattern_characteristics_{timestamp}.png", dpi=150, bbox_inches='tight')
        figures.append(("Pattern Characteristics", fig))

    print("\n" + "=" * 60)
    print(f"[OK] Generated {len(figures)} visualizations")
    print(f"[DIR] Saved to: {output_path.absolute()}")
    print("=" * 60)

    # Print file list
    print("\nGenerated files:")
    for name, _ in figures:
        print(f"   • {name}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description="TRANS Visual Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visual_analysis.py --predictions output/predictions/predictions.parquet
  python scripts/visual_analysis.py --sequences output/sequences/sequences.npy --labels output/sequences/labels.npy
  python scripts/visual_analysis.py --all
  python scripts/visual_analysis.py --interactive
        """
    )

    parser.add_argument('--predictions', type=str, help='Path to predictions parquet file')
    parser.add_argument('--sequences', type=str, help='Path to sequences npy file')
    parser.add_argument('--labels', type=str, help='Path to labels npy file')
    parser.add_argument('--metadata', type=str, help='Path to metadata parquet file')
    parser.add_argument('--output', type=str, default='output/analysis', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Load all available data from default paths')
    parser.add_argument('--interactive', action='store_true', help='Show plots interactively')
    parser.add_argument('--sample', type=int, default=25000,
                        help='Sample size for analysis (default: 25000, use 0 for all data)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: sample 10k, skip slow plots')

    args = parser.parse_args()

    # Default paths if --all specified
    if args.all:
        default_paths = {
            'predictions': 'output/predictions/predictions.parquet',
            'sequences': 'output/sequences/sequences.npy',
            'labels': 'output/sequences/labels.npy',
            'metadata': 'output/sequences/metadata.parquet'
        }

        for key, path in default_paths.items():
            if getattr(args, key) is None and Path(path).exists():
                setattr(args, key, path)

    # Validate inputs
    if not any([args.predictions, args.sequences]):
        print("Error: Please provide at least --predictions or --sequences")
        print("Run with --help for usage information")
        sys.exit(1)

    create_dashboard(
        predictions_path=args.predictions,
        sequences_path=args.sequences,
        metadata_path=args.metadata,
        labels_path=args.labels,
        output_dir=args.output,
        sample_size=args.sample if args.sample > 0 else 0,
        fast_mode=args.fast
    )

    if args.interactive:
        print("\n[SHOW] Displaying plots interactively...")
        plt.show()


if __name__ == "__main__":
    main()
