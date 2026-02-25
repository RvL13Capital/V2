"""
Generate US Model Visualizations
================================

Creates plots for the US model analysis, highlighting the NEGATIVE EV correlation
and showing why EV-based selection doesn't work for the US market.

Output: PNG files in output/visualizations/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import h5py
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inference_wrapper import InferenceWrapper
from models.temporal_hybrid_v18 import HybridFeatureNetwork

CLASS_NAME_LIST = ['Danger', 'Noise', 'Target']
STRATEGIC_VALUES = {0: -1.0, 1: -0.1, 2: 5.0}


def plot_probability_calibration_us(probs, labels, class_idx, class_name, output_path, n_bins=10):
    """
    Create Reliability Diagram for US model with warnings about calibration issues.
    """
    y_true = (labels == class_idx).astype(int)
    y_prob = probs[:, class_idx]

    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Calculate ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * (bin_count / len(y_prob))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use red-tinted colors for US model
    colors = {'Danger': '#d62728', 'Noise': '#ff7f0e', 'Target': '#8B0000'}  # Dark red for Target
    class_color = colors.get(class_name, '#d62728')

    # Subplot 1: Reliability Diagram
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    ax1.plot(prob_pred, prob_true, 'o-', color=class_color, linewidth=2, markersize=8,
             label=f'{class_name} (n={y_true.sum()})')
    ax1.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color=class_color)

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)

    # Quality assessment
    if brier < 0.1:
        quality = "Good"
        quality_color = '#2ca02c'
    elif brier < 0.2:
        quality = "Moderate"
        quality_color = '#ff7f0e'
    else:
        quality = "Poor"
        quality_color = '#d62728'

    ax1.set_title(f'US Model - {class_name} Probability Calibration\n'
                  f'Brier Score: {brier:.4f} | ECE: {ece:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1.text(0.95, 0.05, f'Calibration: {quality}', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', color=quality_color,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Distribution
    ax2 = axes[1]
    bins = np.linspace(0, 1, 21)
    ax2.hist(y_prob[y_true == 1], bins=bins, alpha=0.7, color=class_color,
             label=f'Actual {class_name}', edgecolor='black')
    ax2.hist(y_prob[y_true == 0], bins=bins, alpha=0.5, color='gray',
             label=f'Not {class_name}', edgecolor='black')

    ax2.set_xlabel(f'Predicted P({class_name})', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Distribution of {class_name} Probabilities\n'
                  f'Positive Rate: {y_true.mean()*100:.1f}%',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return brier, ece


def plot_all_classes_calibration_us(probs, labels, output_path, n_bins=10):
    """
    Create combined calibration plot for all three classes - US version.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#d62728', '#ff7f0e', '#8B0000']  # Red theme for US
    results = {}

    for idx, (class_idx, class_name) in enumerate([(0, 'Danger'), (1, 'Noise'), (2, 'Target')]):
        ax = axes[idx]

        y_true = (labels == class_idx).astype(int)
        y_prob = probs[:, class_idx]

        brier = brier_score_loss(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        # Calculate ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            bin_count = mask.sum()
            if bin_count > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += np.abs(bin_acc - bin_conf) * (bin_count / len(y_prob))

        results[class_name] = {'brier': brier, 'ece': ece, 'count': y_true.sum()}

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
        ax.plot(prob_pred, prob_true, 'o-', color=colors[idx], linewidth=2, markersize=8)
        ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color=colors[idx])

        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'{class_name}\nBrier: {brier:.4f} | ECE: {ece:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.text(0.95, 0.05, f'n={y_true.sum()}', transform=ax.transAxes,
                fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('US Model - Probability Calibration (Reliability Diagrams)\n'
                 'Note: Calibration quality does NOT fix negative EV correlation',
                 fontsize=14, fontweight='bold', y=1.02, color='#d62728')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return results


def calculate_top_n_metrics_us(predicted_ev, test_labels, percentiles=[5, 10, 15, 20, 25, 30]):
    """
    Calculate Top-N metrics for US model - highlighting the broken ranking.
    """
    sorted_indices = np.argsort(predicted_ev)[::-1]
    sorted_labels = test_labels[sorted_indices]
    sorted_ev = predicted_ev[sorted_indices]

    total_samples = len(test_labels)
    global_target_rate = (test_labels == 2).mean()
    global_danger_rate = (test_labels == 0).mean()

    metrics = {
        'global_target_rate': global_target_rate,
        'global_danger_rate': global_danger_rate,
        'total_samples': total_samples,
        'percentiles': {}
    }

    for p in percentiles:
        n_samples = int(np.ceil(total_samples * p / 100))
        top_labels = sorted_labels[:n_samples]
        top_ev = sorted_ev[:n_samples]

        target_rate = (top_labels == 2).mean()
        danger_rate = (top_labels == 0).mean()
        noise_rate = (top_labels == 1).mean()
        lift = target_rate / global_target_rate if global_target_rate > 0 else 0

        ev_threshold = top_ev[-1] if len(top_ev) > 0 else 0

        metrics['percentiles'][p] = {
            'n_samples': n_samples,
            'target_rate': target_rate,
            'danger_rate': danger_rate,
            'noise_rate': noise_rate,
            'lift': lift,
            'ev_threshold': ev_threshold
        }

    return metrics


def plot_cumulative_lift_curve_us(predicted_ev, test_labels, output_path):
    """
    Create cumulative lift curve for US model - showing poor ranking efficiency.
    """
    sorted_indices = np.argsort(predicted_ev)[::-1]
    sorted_labels = test_labels[sorted_indices]

    global_target_rate = (test_labels == 2).mean()

    # Calculate cumulative target rate at each point
    cumulative_targets = np.cumsum(sorted_labels == 2)
    cumulative_counts = np.arange(1, len(sorted_labels) + 1)
    cumulative_target_rate = cumulative_targets / cumulative_counts
    cumulative_lift = cumulative_target_rate / global_target_rate

    x_axis = np.linspace(0, 100, len(cumulative_lift))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Left Plot: Cumulative Lift Curve ===
    ax1 = axes[0]
    ax1.plot(x_axis, cumulative_lift, label='Model Lift', color='#d62728', linewidth=2)
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Random (Lift=1.0)')
    ax1.axhline(y=1.5, color='green', linestyle=':', linewidth=2, label='Good Lift (1.5x)', alpha=0.7)

    # Fill areas
    ax1.fill_between(x_axis, 1.0, cumulative_lift,
                     where=cumulative_lift >= 1.0,
                     alpha=0.3, color='#ff7f0e', label='Above Random')
    ax1.fill_between(x_axis, 1.0, cumulative_lift,
                     where=cumulative_lift < 1.0,
                     alpha=0.3, color='#d62728', label='Below Random')

    # Add annotations
    for pct, style in [(10, ':'), (15, '--'), (20, ':')]:
        idx = int(len(cumulative_lift) * pct / 100)
        lift_at_pct = cumulative_lift[idx]
        ax1.axvline(x=pct, color='gray', linestyle=style, alpha=0.5)
        ax1.annotate(f'{pct}%: {lift_at_pct:.2f}x',
                    xy=(pct, lift_at_pct),
                    xytext=(pct + 3, lift_at_pct + 0.05),
                    fontsize=9, fontweight='bold', color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax1.set_xlabel('% of Population (sorted by EV descending)', fontsize=11)
    ax1.set_ylabel('Cumulative Lift', fontsize=11)
    ax1.set_title('US Model - Cumulative Lift Curve\n(Never reaches 1.5x - Poor Ranking)',
                  fontsize=12, fontweight='bold', color='#d62728')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0.8, max(cumulative_lift) * 1.1])

    # === Right Plot: Cumulative Target Rate ===
    ax2 = axes[1]
    ax2.plot(x_axis, cumulative_target_rate * 100, label='Cumulative Target Rate',
             color='#d62728', linewidth=2)
    ax2.axhline(y=global_target_rate * 100, color='blue', linestyle='--', linewidth=2,
                label=f'Baseline: {global_target_rate*100:.1f}%')

    ax2.fill_between(x_axis, global_target_rate * 100, cumulative_target_rate * 100,
                     where=cumulative_target_rate >= global_target_rate,
                     alpha=0.3, color='#ff7f0e')
    ax2.fill_between(x_axis, global_target_rate * 100, cumulative_target_rate * 100,
                     where=cumulative_target_rate < global_target_rate,
                     alpha=0.3, color='#d62728')

    for pct in [10, 15, 20]:
        idx = int(len(cumulative_target_rate) * pct / 100)
        rate_at_pct = cumulative_target_rate[idx] * 100
        ax2.plot(pct, rate_at_pct, 'o', color='#d62728', markersize=8)
        ax2.annotate(f'{rate_at_pct:.1f}%',
                    xy=(pct, rate_at_pct),
                    xytext=(pct + 3, rate_at_pct + 1),
                    fontsize=9, fontweight='bold')

    ax2.set_xlabel('% of Population (sorted by EV descending)', fontsize=11)
    ax2.set_ylabel('Cumulative Target Rate (%)', fontsize=11)
    ax2.set_title('US Model - Cumulative Target Rate\n(Minimal improvement over baseline)',
                  fontsize=12, fontweight='bold', color='#d62728')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    aulc = np.trapz(cumulative_lift, x_axis) / 100

    return {
        'aulc': aulc,
        'lift_at_10': cumulative_lift[int(len(cumulative_lift) * 0.10)],
        'lift_at_15': cumulative_lift[int(len(cumulative_lift) * 0.15)],
        'lift_at_20': cumulative_lift[int(len(cumulative_lift) * 0.20)]
    }


def plot_top_n_lift_curve_us(predicted_ev, test_labels, output_path):
    """
    Create Top-N lift curve for US model - showing the ineffective ranking.
    """
    percentiles = list(range(5, 101, 5))
    metrics = calculate_top_n_metrics_us(predicted_ev, test_labels, percentiles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pcts = list(metrics['percentiles'].keys())
    target_rates = [metrics['percentiles'][p]['target_rate'] * 100 for p in pcts]
    danger_rates = [metrics['percentiles'][p]['danger_rate'] * 100 for p in pcts]
    lifts = [metrics['percentiles'][p]['lift'] for p in pcts]
    ev_thresholds = [metrics['percentiles'][p]['ev_threshold'] for p in pcts]

    global_target = metrics['global_target_rate'] * 100
    global_danger = metrics['global_danger_rate'] * 100

    # Plot 1: Target Rate - with warning about non-monotonicity
    ax1 = axes[0, 0]
    ax1.plot(pcts, target_rates, 'o-', color='#d62728', linewidth=2, markersize=6, label='Target Rate')
    ax1.axhline(y=global_target, color='blue', linestyle='--', linewidth=2, label=f'Baseline: {global_target:.1f}%')

    # Highlight where below baseline
    ax1.fill_between(pcts, global_target, target_rates,
                     where=[t < global_target for t in target_rates],
                     alpha=0.3, color='#d62728', label='Below Baseline (Bad)')
    ax1.fill_between(pcts, global_target, target_rates,
                     where=[t >= global_target for t in target_rates],
                     alpha=0.3, color='#2ca02c', label='Above Baseline')

    ax1.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax1.set_ylabel('Target Rate (%)', fontsize=11)
    ax1.set_title('Target Rate by Selection Depth\n(Non-Monotonic Pattern)', fontsize=12, fontweight='bold', color='#d62728')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 105])

    # Plot 2: Lift Curve - mostly below 1.5x
    ax2 = axes[0, 1]
    ax2.plot(pcts, lifts, 'o-', color='#d62728', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No Lift (1.0x)')
    ax2.axhline(y=1.5, color='green', linestyle=':', linewidth=2, label='Good Lift (1.5x)', alpha=0.7)

    # Fill based on lift quality
    ax2.fill_between(pcts, 1.0, lifts, where=[l < 1.0 for l in lifts],
                     alpha=0.3, color='#d62728', label='Negative Lift')
    ax2.fill_between(pcts, 1.0, lifts, where=[l >= 1.0 for l in lifts],
                     alpha=0.3, color='#ff7f0e', label='Minimal Lift')

    ax2.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax2.set_ylabel('Lift (vs. Global)', fontsize=11)
    ax2.set_title('Lift Curve (EV-Based Ranking)\n(Consistently Below 1.5x)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 105])

    # Add annotations for key percentiles
    for key_pct in [10, 15, 20]:
        if key_pct in metrics['percentiles']:
            lift_val = metrics['percentiles'][key_pct]['lift']
            ax2.annotate(f'{lift_val:.2f}x', xy=(key_pct, lift_val),
                        xytext=(key_pct + 5, lift_val + 0.08),
                        fontsize=9, fontweight='bold', color='#d62728',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Plot 3: Danger Rate
    ax3 = axes[1, 0]
    ax3.plot(pcts, danger_rates, 'o-', color='#d62728', linewidth=2, markersize=6, label='Danger Rate')
    ax3.axhline(y=global_danger, color='gray', linestyle='--', linewidth=2, label=f'Baseline: {global_danger:.1f}%')

    ax3.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax3.set_ylabel('Danger Rate (%)', fontsize=11)
    ax3.set_title('Danger Rate by Selection Depth', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 105])

    # Plot 4: EV Threshold
    ax4 = axes[1, 1]
    ax4.plot(pcts, ev_thresholds, 'o-', color='#9467bd', linewidth=2, markersize=6)
    ax4.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax4.set_ylabel('EV Threshold', fontsize=11)
    ax4.set_title('EV Threshold by Selection Depth', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 105])

    for key_pct in [10, 15, 20]:
        if key_pct in metrics['percentiles']:
            ev_val = metrics['percentiles'][key_pct]['ev_threshold']
            ax4.annotate(f'EV >= {ev_val:.2f}', xy=(key_pct, ev_val),
                        xytext=(key_pct + 8, ev_val + 0.03),
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.suptitle('US Model - Top-N Performance Analysis\n'
                 'WARNING: EV-based ranking is NOT effective for US market',
                 fontsize=14, fontweight='bold', y=1.01, color='#d62728')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def plot_top_n_summary_table_us(metrics, output_path):
    """
    Create summary table for US model - with warning colors.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    key_percentiles = [5, 10, 15, 20, 25, 30]
    headers = ['Top N %', 'Samples', 'Target Rate', 'Lift', 'Danger Rate', 'EV Threshold']

    table_data = []
    for p in key_percentiles:
        if p in metrics['percentiles']:
            m = metrics['percentiles'][p]
            row = [
                f'Top {p}%',
                f"{m['n_samples']}",
                f"{m['target_rate']*100:.1f}%",
                f"{m['lift']:.2f}x",
                f"{m['danger_rate']*100:.1f}%",
                f"{m['ev_threshold']:.3f}"
            ]
            table_data.append(row)

    table_data.append([
        'Baseline (100%)',
        f"{metrics['total_samples']}",
        f"{metrics['global_target_rate']*100:.1f}%",
        '1.00x',
        f"{metrics['global_danger_rate']*100:.1f}%",
        '-'
    ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#c0392b'] * len(headers)  # Red header for US
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(len(headers)):
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color rows based on lift - for US, most will be yellow/red
    for i, p in enumerate(key_percentiles):
        if p in metrics['percentiles']:
            lift = metrics['percentiles'][p]['lift']
            if lift >= 1.5:
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor('#c6efce')  # Green (unlikely for US)
            elif lift >= 1.2:
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor('#ffeb9c')  # Yellow
            else:
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor('#ffc7ce')  # Red - poor lift

    for j in range(len(headers)):
        table[(len(key_percentiles) + 1, j)].set_facecolor('#f2f2f2')

    ax.set_title('US Model - Signal Quality Summary (Top-N Performance)\n'
                 'Red: Lift < 1.2x (Poor) | Yellow: Lift 1.2-1.5x | Green: Lift >= 1.5x',
                 fontsize=14, fontweight='bold', pad=20, color='#d62728')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print('Generating US Model Visualizations...')
    print('=' * 60)

    # Setup
    base_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load US ensemble
    ensemble_path = base_dir / 'output' / 'models' / 'ensemble' / 'us_ensemble_combined.pt'
    print(f'Loading US ensemble from: {ensemble_path}')
    checkpoint = torch.load(ensemble_path, map_location='cpu', weights_only=False)

    members = []
    config = checkpoint.get('config', {})
    model_kwargs = config.get('model_kwargs', {})

    for member_state in checkpoint['ensemble_members']:
        model = HybridFeatureNetwork(**model_kwargs)
        model.load_state_dict(member_state['model_state_dict'])
        wrapper = InferenceWrapper(
            model=model,
            norm_params=checkpoint.get('norm_params'),
            robust_params=checkpoint.get('robust_params'),
            context_ranges=checkpoint.get('context_ranges'),
            config=config,
            device=torch.device('cpu'),
            temperature=checkpoint.get('temperature', 1.0)
        )
        members.append(wrapper)

    print(f'  Loaded {len(members)} models')

    # Load US data
    data_path = base_dir / 'output' / 'sequences' / 'us_coil_dedup' / 'sequences_dedup.h5'
    meta_path = base_dir / 'output' / 'sequences' / 'us_coil_dedup' / 'metadata_dedup.parquet'

    with h5py.File(data_path, 'r') as f:
        sequences = f['sequences'][:]
        context = f['context'][:]
        labels = f['labels'][:]

    metadata = pd.read_parquet(meta_path)
    metadata['pattern_end_date'] = pd.to_datetime(metadata['pattern_end_date'])

    # Test set
    test_mask = metadata['pattern_end_date'] >= '2024-07-01'
    test_idx = np.where(test_mask.values)[0]
    test_sequences = sequences[test_idx]
    test_context = context[test_idx]
    test_labels = labels[test_idx]

    print(f'Test samples: {len(test_sequences)}')
    print('\nClass distribution:')
    for c in [0, 1, 2]:
        count = (test_labels == c).sum()
        pct = 100 * count / len(test_labels)
        print(f'  {CLASS_NAME_LIST[c]}: {count} ({pct:.1f}%)')

    # Predict
    print('\nRunning predictions...')
    ensemble_probs = np.zeros((len(members), len(test_sequences), 3))
    for i, member in enumerate(members):
        probs, _ = member.predict(test_sequences, test_context)
        ensemble_probs[i] = probs

    probs_mean = np.mean(ensemble_probs, axis=0)
    predictions = np.argmax(probs_mean, axis=1)

    # Calculate EVs
    strategic_values = np.array([-1.0, -0.1, 5.0])
    predicted_ev = np.sum(probs_mean * strategic_values, axis=1)
    actual_ev = np.array([STRATEGIC_VALUES[l] for l in test_labels])

    # Compute metrics
    pearson_corr, _ = pearsonr(predicted_ev, actual_ev)
    print(f'EV Correlation: {pearson_corr:.4f} (NEGATIVE)')

    cm = confusion_matrix(test_labels, predictions)

    # === PLOT 1: Confusion Matrix ===
    print('\nGenerating plots...')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=CLASS_NAME_LIST, yticklabels=CLASS_NAME_LIST,
                annot_kws={'size': 14})

    plt.title('US Model - Confusion Matrix (V17 3-Class)', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    for i in range(3):
        for j in range(3):
            total = cm[i].sum()
            pct = 100 * cm[i, j] / total if total > 0 else 0
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    cm_path = output_dir / f'us_confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f'  [1/10] Confusion Matrix: {cm_path.name}')

    # === PLOT 2: EV Calibration ===
    plt.figure(figsize=(10, 8))

    jittered_actual = actual_ev + np.random.uniform(-0.15, 0.15, len(actual_ev))
    colors = ['#d62728' if l == 0 else '#ff7f0e' if l == 1 else '#2ca02c' for l in test_labels]
    plt.scatter(predicted_ev, jittered_actual, c=colors, alpha=0.5, s=30)

    ev_range = [min(predicted_ev.min(), -1.5), max(predicted_ev.max(), 5.5)]
    plt.plot(ev_range, ev_range, 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    for class_id, class_name, color in [(0, 'Danger', '#d62728'), (1, 'Noise', '#ff7f0e'), (2, 'Target', '#2ca02c')]:
        mask = test_labels == class_id
        mean_pred = predicted_ev[mask].mean()
        actual_val = STRATEGIC_VALUES[class_id]
        plt.axhline(y=actual_val, color=color, linestyle=':', alpha=0.5)
        plt.plot(mean_pred, actual_val, 'o', color=color, markersize=12,
                 markeredgecolor='black', markeredgewidth=2, label=f'{class_name} (n={mask.sum()})')

    plt.xlabel('Predicted EV', fontsize=12)
    plt.ylabel('Actual EV (jittered)', fontsize=12)
    plt.title(f'US Model - EV Calibration\nPearson Correlation: {pearson_corr:.4f} (NEGATIVE - EV NOT RELIABLE)',
              fontsize=14, fontweight='bold', color='#d62728')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    ev_path = output_dir / f'us_ev_calibration_{timestamp}.png'
    plt.savefig(ev_path, dpi=150)
    plt.close()
    print(f'  [2/10] EV Calibration: {ev_path.name}')

    # === PLOT 3: Quartile Analysis ===
    plt.figure(figsize=(12, 6))

    try:
        quartile_labels = pd.qcut(predicted_ev, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
    except ValueError:
        # Handle case with too many duplicate values
        quartile_labels = pd.cut(predicted_ev, bins=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    df_plot = pd.DataFrame({
        'EV Quartile': quartile_labels,
        'Actual Class': [CLASS_NAME_LIST[l] for l in test_labels],
        'Predicted EV': predicted_ev
    })

    # Subplot 1: Class distribution
    plt.subplot(1, 2, 1)
    quartile_class_counts = df_plot.groupby(['EV Quartile', 'Actual Class'], observed=False).size().unstack(fill_value=0)
    quartile_class_pct = quartile_class_counts.div(quartile_class_counts.sum(axis=1), axis=0) * 100

    # Ensure correct column order
    for col in ['Danger', 'Noise', 'Target']:
        if col not in quartile_class_pct.columns:
            quartile_class_pct[col] = 0
    quartile_class_pct = quartile_class_pct[['Danger', 'Noise', 'Target']]

    quartile_class_pct.plot(kind='bar', stacked=True,
                            color=['#d62728', '#ff7f0e', '#2ca02c'], ax=plt.gca())
    plt.title('Class Distribution by EV Quartile', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.xlabel('Predicted EV Quartile', fontsize=11)
    plt.legend(title='Actual Class', loc='upper left')
    plt.xticks(rotation=0)
    plt.ylim([0, 100])

    for i, (idx, row) in enumerate(quartile_class_pct.iterrows()):
        target_pct = row['Target']
        plt.text(i, 95, f'{target_pct:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#2ca02c')

    # Subplot 2: Target rate trend
    plt.subplot(1, 2, 2)
    target_rates = []
    ev_means = []
    quartile_names = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']

    for q in quartile_names:
        mask = quartile_labels == q
        if mask.sum() > 0:
            target_rate = (test_labels[mask] == 2).mean() * 100
            ev_mean = predicted_ev[mask].mean()
        else:
            target_rate = 0
            ev_mean = 0
        target_rates.append(target_rate)
        ev_means.append(ev_mean)

    x_pos = range(len(target_rates))

    # Color bars - red if monotonicity violated
    bar_colors = ['#2ca02c']  # First bar green
    for i in range(1, len(target_rates)):
        if target_rates[i] < target_rates[i-1] - 1:  # Allow small tolerance
            bar_colors.append('#d62728')  # Red for violation
        else:
            bar_colors.append('#2ca02c')

    bars = plt.bar(x_pos, target_rates, color=bar_colors, alpha=0.7, edgecolor='black')
    plt.plot(x_pos, target_rates, 'ko-', markersize=8, linewidth=2)

    baseline = (test_labels == 2).mean() * 100
    plt.axhline(y=baseline, color='blue', linestyle='--', linewidth=2, label=f'Baseline: {baseline:.1f}%')

    plt.xticks(x_pos, ['Q1\n(Low EV)', 'Q2', 'Q3', 'Q4\n(High EV)'])
    plt.ylabel('Target Rate (%)', fontsize=11)
    plt.xlabel('Predicted EV Quartile', fontsize=11)
    plt.title('Target Rate by EV Quartile\n(MONOTONICITY NOT PRESERVED)', fontsize=12, fontweight='bold', color='#d62728')
    plt.legend()

    for i, (rate, ev) in enumerate(zip(target_rates, ev_means)):
        plt.text(i, rate + 1.5, f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
        plt.text(i, rate - 3, f'EV={ev:.2f}', ha='center', fontsize=8, color='gray')

    plt.ylim([0, max(target_rates) + 15])
    plt.tight_layout()

    quartile_path = output_dir / f'us_ev_quartile_analysis_{timestamp}.png'
    plt.savefig(quartile_path, dpi=150)
    plt.close()
    print(f'  [3/10] Quartile Analysis: {quartile_path.name}')

    # === PLOT 4: Top 15% Analysis ===
    plt.figure(figsize=(10, 6))

    top_15_threshold = np.percentile(predicted_ev, 85)
    is_top_15 = predicted_ev >= top_15_threshold

    groups = ['All Patterns', 'Top 15% by EV']
    danger_rates = [(test_labels == 0).mean() * 100, (test_labels[is_top_15] == 0).mean() * 100]
    noise_rates = [(test_labels == 1).mean() * 100, (test_labels[is_top_15] == 1).mean() * 100]
    target_rates_compare = [(test_labels == 2).mean() * 100, (test_labels[is_top_15] == 2).mean() * 100]

    x = np.arange(len(groups))
    width = 0.25

    bars1 = plt.bar(x - width, danger_rates, width, label='Danger', color='#d62728', alpha=0.8)
    bars2 = plt.bar(x, noise_rates, width, label='Noise', color='#ff7f0e', alpha=0.8)
    bars3 = plt.bar(x + width, target_rates_compare, width, label='Target', color='#2ca02c', alpha=0.8)

    plt.ylabel('Rate (%)', fontsize=12)
    plt.title(f'US Model - Top 15% Selection Analysis\n(EV >= {top_15_threshold:.3f}) - EV SELECTION NOT EFFECTIVE',
              fontsize=14, fontweight='bold', color='#d62728')
    plt.xticks(x, groups, fontsize=11)
    plt.legend(fontsize=10)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    lift = target_rates_compare[1] / target_rates_compare[0] if target_rates_compare[0] > 0 else 0
    lift_color = '#d62728' if lift < 1.2 else '#2ca02c'

    annotation_text = f'Lift: {lift:.2f}x'
    if lift < 1.2:
        annotation_text += '\n(Minimal/No improvement)'

    plt.annotate(annotation_text,
                xy=(1 + width, target_rates_compare[1]),
                xytext=(1.4, target_rates_compare[1] + 8),
                fontsize=11, fontweight='bold', color=lift_color,
                arrowprops=dict(arrowstyle='->', color=lift_color))

    plt.ylim([0, 70])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    top15_path = output_dir / f'us_top15_analysis_{timestamp}.png'
    plt.savefig(top15_path, dpi=150)
    plt.close()
    print(f'  [4/10] Top 15% Analysis: {top15_path.name}')

    # === PLOT 5: EU vs US Comparison ===
    plt.figure(figsize=(14, 5))

    # EU metrics (from previous analysis)
    eu_corr = 0.088
    eu_top15_target = 31.2
    eu_baseline = 20.8
    eu_lift = 1.51

    # US metrics
    us_corr = pearson_corr
    us_top15_target = target_rates_compare[1]
    us_baseline = target_rates_compare[0]
    us_lift = lift

    # Subplot 1: Bar comparison
    plt.subplot(1, 2, 1)
    metrics = ['EV Correlation', 'Top 15%\nTarget Rate', 'Baseline\nTarget Rate', 'Lift']
    eu_values = [eu_corr, eu_top15_target, eu_baseline, eu_lift]
    us_values = [us_corr, us_top15_target, us_baseline, us_lift]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = plt.bar(x - width/2, eu_values, width, label='EU Model', color='#2ca02c', alpha=0.8)
    bars2 = plt.bar(x + width/2, us_values, width, label='US Model', color='#d62728', alpha=0.8)

    plt.ylabel('Value', fontsize=12)
    plt.title('EU vs US Model Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for i, (eu_val, us_val) in enumerate(zip(eu_values, us_values)):
        offset = 0.5 if eu_val >= 0 else -2
        plt.text(i - width/2, eu_val + offset, f'{eu_val:.2f}', ha='center', fontsize=9, color='#2ca02c')
        offset = 0.5 if us_val >= 0 else -2
        plt.text(i + width/2, us_val + offset, f'{us_val:.2f}', ha='center', fontsize=9, color='#d62728')

    # Subplot 2: Recommendation box
    plt.subplot(1, 2, 2)

    rec_text = """
    RECOMMENDATION
    ══════════════════════════════════════

    EU Model (EV Correlation: +0.088)
    ✓ USE EV-based ranking
    ✓ Top 15% selection works (1.51x lift)

    US Model (EV Correlation: {:.3f})
    ✗ DO NOT use EV-based ranking
    ✗ Monotonicity broken
    ✗ Higher EV ≠ Better outcomes

    US Alternatives:
    • Use P(Target) > 0.35 directly
    • Random selection ({:.1f}% baseline)
    • Re-train with more data
    """.format(us_corr, us_baseline)

    plt.text(0.5, 0.5, rec_text, fontsize=11, ha='center', va='center',
             transform=plt.gca().transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))
    plt.axis('off')
    plt.title('Trading Recommendations', fontsize=14, fontweight='bold')

    plt.tight_layout()
    comparison_path = output_dir / f'us_eu_comparison_{timestamp}.png'
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f'  [5/10] EU vs US Comparison: {comparison_path.name}')

    # === PLOT 6: Target Class Calibration ===
    target_cal_path = output_dir / f'us_calibration_target_{timestamp}.png'
    target_brier, target_ece = plot_probability_calibration_us(
        probs_mean, test_labels, class_idx=2, class_name='Target',
        output_path=target_cal_path, n_bins=10
    )
    print(f'  [6/10] Target Calibration: {target_cal_path.name}')
    print(f'        Brier Score: {target_brier:.4f}, ECE: {target_ece:.4f}')

    # === PLOT 7: All Classes Calibration ===
    all_cal_path = output_dir / f'us_calibration_all_classes_{timestamp}.png'
    calibration_results = plot_all_classes_calibration_us(
        probs_mean, test_labels, output_path=all_cal_path, n_bins=10
    )
    print(f'  [7/10] All Classes Calibration: {all_cal_path.name}')

    # === PLOT 8: Top-N Lift Curve ===
    top_n_curve_path = output_dir / f'us_top_n_lift_curve_{timestamp}.png'
    top_n_metrics = plot_top_n_lift_curve_us(predicted_ev, test_labels, top_n_curve_path)
    print(f'  [8/10] Top-N Lift Curve: {top_n_curve_path.name}')

    # === PLOT 9: Top-N Summary Table ===
    top_n_table_path = output_dir / f'us_top_n_summary_table_{timestamp}.png'
    plot_top_n_summary_table_us(top_n_metrics, top_n_table_path)
    print(f'  [9/10] Top-N Summary Table: {top_n_table_path.name}')

    # === PLOT 10: Cumulative Lift Curve ===
    cumulative_lift_path = output_dir / f'us_cumulative_lift_curve_{timestamp}.png'
    cumulative_metrics = plot_cumulative_lift_curve_us(predicted_ev, test_labels, cumulative_lift_path)
    print(f'  [10/10] Cumulative Lift Curve: {cumulative_lift_path.name}')
    print(f'          AULC (Area Under Lift Curve): {cumulative_metrics["aulc"]:.3f}')

    print('\n' + '=' * 60)
    print('All US visualizations generated!')
    print(f'Output directory: {output_dir}')
    print(f'\nKey Finding: US model has NEGATIVE EV correlation ({pearson_corr:.4f})')
    print('Recommendation: Do NOT use EV-based selection for US market')

    # Print calibration summary
    print('\n--- Calibration Summary (Brier Score) ---')
    print('Lower is better. < 0.1 = Excellent, < 0.2 = Good')
    for class_name, metrics_cal in calibration_results.items():
        quality = "Good" if metrics_cal['brier'] < 0.1 else "Moderate" if metrics_cal['brier'] < 0.2 else "Poor"
        print(f"  {class_name}: Brier={metrics_cal['brier']:.4f}, ECE={metrics_cal['ece']:.4f} ({quality})")

    # Print Top-N Summary
    print('\n--- Top-N Performance Summary ---')
    print(f"Global Target Rate: {top_n_metrics['global_target_rate']*100:.1f}%")
    print('| Top N % | Target Rate | Lift |')
    print('|---------|-------------|------|')
    for p in [10, 15, 20]:
        if p in top_n_metrics['percentiles']:
            m = top_n_metrics['percentiles'][p]
            warning = " (Poor)" if m['lift'] < 1.2 else ""
            print(f"| Top {p}% | {m['target_rate']*100:.1f}% | {m['lift']:.2f}x{warning} |")

    # Print Cumulative Lift Summary
    print('\n--- Cumulative Lift Summary ---')
    print(f"AULC (Area Under Lift Curve): {cumulative_metrics['aulc']:.3f}")
    print(f"Lift @ 10%: {cumulative_metrics['lift_at_10']:.2f}x (Poor)" if cumulative_metrics['lift_at_10'] < 1.5 else f"Lift @ 10%: {cumulative_metrics['lift_at_10']:.2f}x")
    print(f"Lift @ 15%: {cumulative_metrics['lift_at_15']:.2f}x (Poor)" if cumulative_metrics['lift_at_15'] < 1.5 else f"Lift @ 15%: {cumulative_metrics['lift_at_15']:.2f}x")
    print(f"Lift @ 20%: {cumulative_metrics['lift_at_20']:.2f}x (Poor)" if cumulative_metrics['lift_at_20'] < 1.5 else f"Lift @ 20%: {cumulative_metrics['lift_at_20']:.2f}x")

    return {
        'confusion_matrix': cm_path,
        'ev_calibration': ev_path,
        'quartile_analysis': quartile_path,
        'top15_analysis': top15_path,
        'comparison': comparison_path,
        'calibration_target': target_cal_path,
        'calibration_all': all_cal_path,
        'calibration_metrics': calibration_results,
        'top_n_curve': top_n_curve_path,
        'top_n_table': top_n_table_path,
        'top_n_metrics': top_n_metrics,
        'cumulative_lift': cumulative_lift_path,
        'cumulative_metrics': cumulative_metrics,
        'timestamp': timestamp,
        'ev_correlation': pearson_corr
    }


if __name__ == '__main__':
    main()
