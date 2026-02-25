"""
Generate EU Model Visualizations
================================

Creates publication-quality plots for the EU model analysis:
1. Confusion Matrix Heatmap
2. EV Calibration Scatter Plot
3. EV Quartile Analysis (Monotonicity)
4. Top 15% Selection Analysis

Output: PNG files in output/visualizations/
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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

# Constants
CLASS_NAME_LIST = ['Danger', 'Noise', 'Target']
STRATEGIC_VALUES = {0: -1.0, 1: -0.1, 2: 5.0}


def load_ensemble(ensemble_path: Path, device='cpu'):
    """Load EU ensemble model."""
    checkpoint = torch.load(ensemble_path, map_location=device, weights_only=False)

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
            device=torch.device(device),
            temperature=checkpoint.get('temperature', 1.0)
        )
        members.append(wrapper)

    return members


def predict_ensemble(members, sequences, context):
    """Get ensemble predictions."""
    ensemble_probs = np.zeros((len(members), len(sequences), 3))
    for i, member in enumerate(members):
        probs, _ = member.predict(sequences, context)
        ensemble_probs[i] = probs
    return np.mean(ensemble_probs, axis=0)


def plot_confusion_matrix(cm, output_path):
    """Create confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAME_LIST, yticklabels=CLASS_NAME_LIST,
                annot_kws={'size': 14})

    plt.title('EU Model - Confusion Matrix (V17 3-Class)', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    # Add percentages
    for i in range(3):
        for j in range(3):
            total = cm[i].sum()
            pct = 100 * cm[i, j] / total if total > 0 else 0
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ev_calibration(predicted_ev, actual_ev, test_labels, output_path):
    """Create EV calibration scatter plot."""
    pearson_corr, _ = pearsonr(predicted_ev, actual_ev)

    plt.figure(figsize=(10, 8))

    # Add jitter to actual_ev for visibility
    jittered_actual = actual_ev + np.random.uniform(-0.15, 0.15, len(actual_ev))

    # Color by actual class
    colors = ['#d62728' if l == 0 else '#ff7f0e' if l == 1 else '#2ca02c' for l in test_labels]
    plt.scatter(predicted_ev, jittered_actual, c=colors, alpha=0.5, s=30)

    # Reference line
    ev_range = [min(predicted_ev.min(), -1.5), max(predicted_ev.max(), 5.5)]
    plt.plot(ev_range, ev_range, 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    # Mean markers per class
    for class_id, class_name, color in [(0, 'Danger', '#d62728'), (1, 'Noise', '#ff7f0e'), (2, 'Target', '#2ca02c')]:
        mask = test_labels == class_id
        mean_pred = predicted_ev[mask].mean()
        actual_val = STRATEGIC_VALUES[class_id]
        plt.axhline(y=actual_val, color=color, linestyle=':', alpha=0.5)
        plt.plot(mean_pred, actual_val, 'o', color=color, markersize=12,
                 markeredgecolor='black', markeredgewidth=2, label=f'{class_name} (n={mask.sum()})')

    plt.xlabel('Predicted EV', fontsize=12)
    plt.ylabel('Actual EV (jittered for visibility)', fontsize=12)
    plt.title(f'EU Model - EV Calibration\nPearson Correlation: {pearson_corr:.4f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 2.5])
    plt.ylim([-1.5, 5.5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_quartile_analysis(predicted_ev, test_labels, output_path):
    """Create EV quartile analysis plot."""
    plt.figure(figsize=(12, 6))

    # Create quartiles
    quartile_labels = pd.qcut(predicted_ev, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    df_plot = pd.DataFrame({
        'EV Quartile': quartile_labels,
        'Actual Class': [CLASS_NAME_LIST[l] for l in test_labels],
        'Predicted EV': predicted_ev
    })

    # Subplot 1: Class distribution per quartile
    plt.subplot(1, 2, 1)
    quartile_class_counts = df_plot.groupby(['EV Quartile', 'Actual Class']).size().unstack(fill_value=0)
    quartile_class_pct = quartile_class_counts.div(quartile_class_counts.sum(axis=1), axis=0) * 100
    quartile_class_pct = quartile_class_pct[['Danger', 'Noise', 'Target']]

    quartile_class_pct.plot(kind='bar', stacked=True,
                            color=['#d62728', '#ff7f0e', '#2ca02c'], ax=plt.gca())
    plt.title('Class Distribution by EV Quartile', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.xlabel('Predicted EV Quartile', fontsize=11)
    plt.legend(title='Actual Class', loc='upper left')
    plt.xticks(rotation=0)
    plt.ylim([0, 100])

    # Add Target % labels
    for i, (idx, row) in enumerate(quartile_class_pct.iterrows()):
        target_pct = row['Target']
        plt.text(i, 95, f'{target_pct:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#2ca02c')

    # Subplot 2: Target rate trend
    plt.subplot(1, 2, 2)
    target_rates = []
    ev_means = []
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        mask = quartile_labels == q
        target_rate = (test_labels[mask] == 2).mean() * 100
        ev_mean = predicted_ev[mask].mean()
        target_rates.append(target_rate)
        ev_means.append(ev_mean)

    x_pos = range(4)
    bars = plt.bar(x_pos, target_rates, color='#2ca02c', alpha=0.7, edgecolor='black')
    plt.plot(x_pos, target_rates, 'ko-', markersize=8, linewidth=2)

    # Baseline
    baseline = (test_labels == 2).mean() * 100
    plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline:.1f}%')

    plt.xticks(x_pos, ['Q1\n(Low EV)', 'Q2', 'Q3', 'Q4\n(High EV)'])
    plt.ylabel('Target Rate (%)', fontsize=11)
    plt.xlabel('Predicted EV Quartile', fontsize=11)
    plt.title('Target Rate by EV Quartile (Monotonicity)', fontsize=12, fontweight='bold')
    plt.legend()

    # Value labels
    for i, (rate, ev) in enumerate(zip(target_rates, ev_means)):
        plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
        plt.text(i, rate - 3, f'EV={ev:.2f}', ha='center', fontsize=8, color='gray')

    plt.ylim([0, 35])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top15_analysis(predicted_ev, test_labels, output_path):
    """Create Top 15% selection analysis plot."""
    plt.figure(figsize=(10, 6))

    top_15_threshold = np.percentile(predicted_ev, 85)
    is_top_15 = predicted_ev >= top_15_threshold

    # Create comparison data
    groups = ['All Patterns', 'Top 15% by EV']
    danger_rates = [(test_labels == 0).mean() * 100, (test_labels[is_top_15] == 0).mean() * 100]
    noise_rates = [(test_labels == 1).mean() * 100, (test_labels[is_top_15] == 1).mean() * 100]
    target_rates = [(test_labels == 2).mean() * 100, (test_labels[is_top_15] == 2).mean() * 100]

    x = np.arange(len(groups))
    width = 0.25

    bars1 = plt.bar(x - width, danger_rates, width, label='Danger', color='#d62728', alpha=0.8)
    bars2 = plt.bar(x, noise_rates, width, label='Noise', color='#ff7f0e', alpha=0.8)
    bars3 = plt.bar(x + width, target_rates, width, label='Target', color='#2ca02c', alpha=0.8)

    plt.ylabel('Rate (%)', fontsize=12)
    plt.title(f'EU Model - Top 15% Selection Analysis\n(EV >= {top_15_threshold:.3f})',
              fontsize=14, fontweight='bold')
    plt.xticks(x, groups, fontsize=11)
    plt.legend(fontsize=10)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Lift annotation
    lift = target_rates[1] / target_rates[0]
    plt.annotate(f'Lift: {lift:.2f}x',
                xy=(1 + width, target_rates[1]),
                xytext=(1.5, target_rates[1] + 5),
                fontsize=12, fontweight='bold', color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    plt.ylim([0, 65])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return top_15_threshold, lift


def calculate_top_n_metrics(predicted_ev, test_labels, percentiles=[5, 10, 15, 20, 25, 30]):
    """
    Calculate Target rate and Lift for various Top-N percentiles.

    Args:
        predicted_ev: Array of predicted expected values
        test_labels: Array of true labels
        percentiles: List of percentile thresholds to analyze

    Returns:
        dict with metrics for each percentile and global stats
    """
    # Sort by EV descending
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
        danger_lift = danger_rate / global_danger_rate if global_danger_rate > 0 else 0

        # EV threshold at this percentile
        ev_threshold = top_ev[-1] if len(top_ev) > 0 else 0

        metrics['percentiles'][p] = {
            'n_samples': n_samples,
            'target_rate': target_rate,
            'danger_rate': danger_rate,
            'noise_rate': noise_rate,
            'lift': lift,
            'danger_lift': danger_lift,
            'ev_threshold': ev_threshold
        }

    return metrics


def plot_cumulative_lift_curve(predicted_ev, test_labels, output_path, model_name='EU'):
    """
    Create a cumulative lift curve showing model efficiency across the entire population.

    This shows how lift changes as you include more samples sorted by EV descending.
    A good model maintains high lift in the top percentiles and gradually decreases to 1.0.
    """
    # Sort by EV descending
    sorted_indices = np.argsort(predicted_ev)[::-1]
    sorted_labels = test_labels[sorted_indices]

    global_target_rate = (test_labels == 2).mean()

    # Calculate cumulative target rate at each point
    cumulative_targets = np.cumsum(sorted_labels == 2)
    cumulative_counts = np.arange(1, len(sorted_labels) + 1)
    cumulative_target_rate = cumulative_targets / cumulative_counts
    cumulative_lift = cumulative_target_rate / global_target_rate

    # X-axis: percentage of population
    x_axis = np.linspace(0, 100, len(cumulative_lift))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme based on model
    if model_name == 'EU':
        color = '#2ca02c'  # Green for EU
        title_color = 'black'
    else:
        color = '#d62728'  # Red for US
        title_color = '#d62728'

    # === Left Plot: Cumulative Lift Curve ===
    ax1 = axes[0]
    ax1.plot(x_axis, cumulative_lift, label='Model Lift', color=color, linewidth=2)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Random (Lift=1.0)')

    # Fill area above random
    ax1.fill_between(x_axis, 1.0, cumulative_lift,
                     where=cumulative_lift >= 1.0,
                     alpha=0.3, color=color, label='Above Random')
    ax1.fill_between(x_axis, 1.0, cumulative_lift,
                     where=cumulative_lift < 1.0,
                     alpha=0.3, color='red', label='Below Random')

    # Add vertical lines for key percentiles
    for pct, style in [(10, ':'), (15, '--'), (20, ':')]:
        idx = int(len(cumulative_lift) * pct / 100)
        lift_at_pct = cumulative_lift[idx]
        ax1.axvline(x=pct, color='gray', linestyle=style, alpha=0.5)
        ax1.annotate(f'{pct}%: {lift_at_pct:.2f}x',
                    xy=(pct, lift_at_pct),
                    xytext=(pct + 3, lift_at_pct + 0.1),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax1.set_xlabel('% of Population (sorted by EV descending)', fontsize=11)
    ax1.set_ylabel('Cumulative Lift', fontsize=11)
    ax1.set_title(f'{model_name} Model - Cumulative Lift Curve\n(Target Sorting Efficiency)',
                  fontsize=12, fontweight='bold', color=title_color)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0.5, max(cumulative_lift) * 1.1])

    # === Right Plot: Cumulative Target Rate ===
    ax2 = axes[1]
    ax2.plot(x_axis, cumulative_target_rate * 100, label='Cumulative Target Rate',
             color=color, linewidth=2)
    ax2.axhline(y=global_target_rate * 100, color='red', linestyle='--', linewidth=2,
                label=f'Baseline: {global_target_rate*100:.1f}%')

    # Fill area
    ax2.fill_between(x_axis, global_target_rate * 100, cumulative_target_rate * 100,
                     where=cumulative_target_rate >= global_target_rate,
                     alpha=0.3, color=color)

    # Add annotations for key percentiles
    for pct in [10, 15, 20]:
        idx = int(len(cumulative_target_rate) * pct / 100)
        rate_at_pct = cumulative_target_rate[idx] * 100
        ax2.plot(pct, rate_at_pct, 'o', color=color, markersize=8)
        ax2.annotate(f'{rate_at_pct:.1f}%',
                    xy=(pct, rate_at_pct),
                    xytext=(pct + 3, rate_at_pct + 2),
                    fontsize=9, fontweight='bold')

    ax2.set_xlabel('% of Population (sorted by EV descending)', fontsize=11)
    ax2.set_ylabel('Cumulative Target Rate (%)', fontsize=11)
    ax2.set_title(f'{model_name} Model - Cumulative Target Rate\n(vs. Baseline)',
                  fontsize=12, fontweight='bold', color=title_color)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Calculate area under lift curve (AULC) as a summary metric
    aulc = np.trapz(cumulative_lift, x_axis) / 100  # Normalized to [0, 100] scale

    return {
        'aulc': aulc,
        'lift_at_10': cumulative_lift[int(len(cumulative_lift) * 0.10)],
        'lift_at_15': cumulative_lift[int(len(cumulative_lift) * 0.15)],
        'lift_at_20': cumulative_lift[int(len(cumulative_lift) * 0.20)]
    }


def plot_top_n_lift_curve(predicted_ev, test_labels, output_path):
    """
    Create a comprehensive Top-N lift curve analysis plot.

    Shows how Target rate and Lift change as we select more samples by EV.
    """
    # Calculate metrics for many percentiles
    percentiles = list(range(5, 101, 5))  # 5%, 10%, 15%, ..., 100%
    metrics = calculate_top_n_metrics(predicted_ev, test_labels, percentiles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract data for plotting
    pcts = list(metrics['percentiles'].keys())
    target_rates = [metrics['percentiles'][p]['target_rate'] * 100 for p in pcts]
    danger_rates = [metrics['percentiles'][p]['danger_rate'] * 100 for p in pcts]
    lifts = [metrics['percentiles'][p]['lift'] for p in pcts]
    ev_thresholds = [metrics['percentiles'][p]['ev_threshold'] for p in pcts]

    global_target = metrics['global_target_rate'] * 100
    global_danger = metrics['global_danger_rate'] * 100

    # Plot 1: Target Rate by Percentile
    ax1 = axes[0, 0]
    ax1.plot(pcts, target_rates, 'o-', color='#2ca02c', linewidth=2, markersize=6, label='Target Rate')
    ax1.axhline(y=global_target, color='red', linestyle='--', linewidth=2, label=f'Baseline: {global_target:.1f}%')
    ax1.fill_between(pcts, global_target, target_rates, where=[t > global_target for t in target_rates],
                     alpha=0.3, color='#2ca02c', label='Above Baseline')
    ax1.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax1.set_ylabel('Target Rate (%)', fontsize=11)
    ax1.set_title('Target Rate by Selection Depth', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 105])

    # Plot 2: Lift Curve
    ax2 = axes[0, 1]
    ax2.plot(pcts, lifts, 'o-', color='#1f77b4', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Lift (1.0x)')
    ax2.fill_between(pcts, 1.0, lifts, where=[l > 1.0 for l in lifts],
                     alpha=0.3, color='#1f77b4', label='Positive Lift')
    ax2.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax2.set_ylabel('Lift (vs. Global)', fontsize=11)
    ax2.set_title('Lift Curve (Model Value)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 105])

    # Add key lift annotations
    for key_pct in [10, 15, 20]:
        if key_pct in metrics['percentiles']:
            lift_val = metrics['percentiles'][key_pct]['lift']
            ax2.annotate(f'{lift_val:.2f}x', xy=(key_pct, lift_val),
                        xytext=(key_pct + 5, lift_val + 0.1),
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Plot 3: Danger Rate by Percentile
    ax3 = axes[1, 0]
    ax3.plot(pcts, danger_rates, 'o-', color='#d62728', linewidth=2, markersize=6, label='Danger Rate')
    ax3.axhline(y=global_danger, color='gray', linestyle='--', linewidth=2, label=f'Baseline: {global_danger:.1f}%')
    ax3.fill_between(pcts, global_danger, danger_rates, where=[d < global_danger for d in danger_rates],
                     alpha=0.3, color='#2ca02c', label='Below Baseline (Good)')
    ax3.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax3.set_ylabel('Danger Rate (%)', fontsize=11)
    ax3.set_title('Danger Rate by Selection Depth', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 105])

    # Plot 4: EV Threshold Curve
    ax4 = axes[1, 1]
    ax4.plot(pcts, ev_thresholds, 'o-', color='#9467bd', linewidth=2, markersize=6)
    ax4.set_xlabel('Top N Percentile (%)', fontsize=11)
    ax4.set_ylabel('EV Threshold', fontsize=11)
    ax4.set_title('EV Threshold by Selection Depth', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 105])

    # Add EV threshold annotations for key percentiles
    for key_pct in [10, 15, 20]:
        if key_pct in metrics['percentiles']:
            ev_val = metrics['percentiles'][key_pct]['ev_threshold']
            ax4.annotate(f'EV >= {ev_val:.2f}', xy=(key_pct, ev_val),
                        xytext=(key_pct + 8, ev_val + 0.05),
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.suptitle('EU Model - Top-N Performance Analysis (Signal Quality)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def plot_top_n_summary_table(metrics, output_path):
    """
    Create a visual table summarizing Top-N performance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Prepare table data
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

    # Add baseline row
    table_data.append([
        'Baseline (100%)',
        f"{metrics['total_samples']}",
        f"{metrics['global_target_rate']*100:.1f}%",
        '1.00x',
        f"{metrics['global_danger_rate']*100:.1f}%",
        '-'
    ])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(headers)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header text white
    for j in range(len(headers)):
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best lift rows
    for i, p in enumerate(key_percentiles):
        if p in metrics['percentiles']:
            lift = metrics['percentiles'][p]['lift']
            if lift >= 1.5:
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor('#c6efce')  # Light green
            elif lift >= 1.2:
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor('#ffeb9c')  # Light yellow

    # Baseline row styling
    for j in range(len(headers)):
        table[(len(key_percentiles) + 1, j)].set_facecolor('#f2f2f2')

    ax.set_title('EU Model - Signal Quality Summary (Top-N Performance)\n'
                 f'Green: Lift >= 1.5x | Yellow: Lift >= 1.2x',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_probability_calibration(probs, labels, class_idx, class_name, output_path, n_bins=10):
    """
    Create Reliability Diagram (Calibration Curve) for a specific class.

    Args:
        probs: Probability matrix (N, 3)
        labels: True labels (N,)
        class_idx: Index of the class to calibrate (0=Danger, 1=Noise, 2=Target)
        class_name: Name of the class for labeling
        output_path: Path to save the plot
        n_bins: Number of bins for calibration curve

    Returns:
        brier_score: Brier score for this class
        ece: Expected Calibration Error
    """
    # Binary labels for this class
    y_true = (labels == class_idx).astype(int)
    y_prob = probs[:, class_idx]

    # Calculate Brier Score
    brier = brier_score_loss(y_true, y_prob)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Calculate Expected Calibration Error (ECE)
    # ECE = sum(|accuracy - confidence| * bin_weight)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_counts = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = mask.sum()
        bin_counts.append(bin_count)
        if bin_count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * (bin_count / len(y_prob))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define colors based on class
    colors = {'Danger': '#d62728', 'Noise': '#ff7f0e', 'Target': '#2ca02c'}
    class_color = colors.get(class_name, '#1f77b4')

    # Subplot 1: Reliability Diagram
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    ax1.plot(prob_pred, prob_true, 'o-', color=class_color, linewidth=2, markersize=8,
             label=f'{class_name} (n={y_true.sum()})')

    # Fill the gap between perfect calibration and actual
    ax1.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color=class_color)

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency (Fraction of Positives)', fontsize=12)
    ax1.set_title(f'EU Model - {class_name} Probability Calibration\n'
                  f'Brier Score: {brier:.4f} | ECE: {ece:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Add calibration quality annotation
    if brier < 0.1:
        quality = "Excellent"
        quality_color = '#2ca02c'
    elif brier < 0.2:
        quality = "Good"
        quality_color = '#ff7f0e'
    else:
        quality = "Needs Improvement"
        quality_color = '#d62728'

    ax1.text(0.95, 0.05, f'Calibration: {quality}', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', color=quality_color,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Probability Distribution Histogram
    ax2 = axes[1]

    # Histogram of predicted probabilities, split by actual class
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

    # Add vertical line at mean probability for actual class
    mean_prob_actual = y_prob[y_true == 1].mean() if y_true.sum() > 0 else 0
    ax2.axvline(x=mean_prob_actual, color=class_color, linestyle='--', linewidth=2,
                label=f'Mean (Actual): {mean_prob_actual:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return brier, ece


def plot_all_classes_calibration(probs, labels, output_path, n_bins=10):
    """
    Create a combined calibration plot showing all three classes.

    Returns:
        dict with brier scores and ECE for each class
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    results = {}

    for idx, (class_idx, class_name) in enumerate([(0, 'Danger'), (1, 'Noise'), (2, 'Target')]):
        ax = axes[idx]

        # Binary labels for this class
        y_true = (labels == class_idx).astype(int)
        y_prob = probs[:, class_idx]

        # Calculate metrics
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

        # Plot
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

        # Sample count annotation
        ax.text(0.95, 0.05, f'n={y_true.sum()}', transform=ax.transAxes,
                fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('EU Model - Probability Calibration (Reliability Diagrams)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return results


def main():
    print('Generating EU Model Visualizations...')
    print('=' * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ensemble
    ensemble_path = base_dir / 'output' / 'models' / 'ensemble' / 'eu_ensemble_combined.pt'
    print(f'Loading ensemble from: {ensemble_path}')
    members = load_ensemble(ensemble_path)
    print(f'  Loaded {len(members)} models')

    # Load data
    data_path = base_dir / 'output' / 'sequences' / 'eu_dedup' / 'sequences_dedup.h5'
    meta_path = base_dir / 'output' / 'sequences' / 'eu_dedup' / 'metadata_dedup.parquet'

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

    # Predict
    print('Running predictions...')
    probs_mean = predict_ensemble(members, test_sequences, test_context)
    predictions = np.argmax(probs_mean, axis=1)

    # Calculate EVs
    strategic_values = np.array([-1.0, -0.1, 5.0])
    predicted_ev = np.sum(probs_mean * strategic_values, axis=1)
    actual_ev = np.array([STRATEGIC_VALUES[l] for l in test_labels])

    # Generate plots
    print('\nGenerating plots...')

    # 1. Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    cm_path = output_dir / f'eu_confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(cm, cm_path)
    print(f'  [1/9] Confusion Matrix: {cm_path.name}')

    # 2. EV Calibration
    ev_path = output_dir / f'eu_ev_calibration_{timestamp}.png'
    plot_ev_calibration(predicted_ev, actual_ev, test_labels, ev_path)
    print(f'  [2/9] EV Calibration: {ev_path.name}')

    # 3. Quartile Analysis
    quartile_path = output_dir / f'eu_ev_quartile_analysis_{timestamp}.png'
    plot_quartile_analysis(predicted_ev, test_labels, quartile_path)
    print(f'  [3/9] Quartile Analysis: {quartile_path.name}')

    # 4. Top 15% Analysis
    top15_path = output_dir / f'eu_top15_analysis_{timestamp}.png'
    threshold, lift = plot_top15_analysis(predicted_ev, test_labels, top15_path)
    print(f'  [4/9] Top 15% Analysis: {top15_path.name}')

    # 5. Target Class Calibration (Primary - most critical for EV decisions)
    target_cal_path = output_dir / f'eu_calibration_target_{timestamp}.png'
    target_brier, target_ece = plot_probability_calibration(
        probs_mean, test_labels, class_idx=2, class_name='Target',
        output_path=target_cal_path, n_bins=10
    )
    print(f'  [5/9] Target Calibration: {target_cal_path.name}')
    print(f'        Brier Score: {target_brier:.4f}, ECE: {target_ece:.4f}')

    # 6. All Classes Calibration Summary
    all_cal_path = output_dir / f'eu_calibration_all_classes_{timestamp}.png'
    calibration_results = plot_all_classes_calibration(
        probs_mean, test_labels, output_path=all_cal_path, n_bins=10
    )
    print(f'  [6/9] All Classes Calibration: {all_cal_path.name}')

    # 7. Top-N Lift Curve Analysis
    top_n_curve_path = output_dir / f'eu_top_n_lift_curve_{timestamp}.png'
    top_n_metrics = plot_top_n_lift_curve(predicted_ev, test_labels, top_n_curve_path)
    print(f'  [7/9] Top-N Lift Curve: {top_n_curve_path.name}')

    # 8. Top-N Summary Table
    top_n_table_path = output_dir / f'eu_top_n_summary_table_{timestamp}.png'
    plot_top_n_summary_table(top_n_metrics, top_n_table_path)
    print(f'  [8/9] Top-N Summary Table: {top_n_table_path.name}')

    # 9. Cumulative Lift Curve
    cumulative_lift_path = output_dir / f'eu_cumulative_lift_curve_{timestamp}.png'
    cumulative_metrics = plot_cumulative_lift_curve(predicted_ev, test_labels, cumulative_lift_path, model_name='EU')
    print(f'  [9/9] Cumulative Lift Curve: {cumulative_lift_path.name}')
    print(f'        AULC (Area Under Lift Curve): {cumulative_metrics["aulc"]:.3f}')

    print('\n' + '=' * 60)
    print('All visualizations generated successfully!')
    print(f'Output directory: {output_dir}')
    print(f'\nTop 15% Threshold: {threshold:.3f}')
    print(f'Lift: {lift:.2f}x')

    # Print calibration summary
    print('\n--- Calibration Summary (Brier Score) ---')
    print('Lower is better. < 0.1 = Excellent, < 0.2 = Good')
    for class_name, metrics in calibration_results.items():
        quality = "Excellent" if metrics['brier'] < 0.1 else "Good" if metrics['brier'] < 0.2 else "Needs Work"
        print(f"  {class_name}: Brier={metrics['brier']:.4f}, ECE={metrics['ece']:.4f} ({quality})")

    # Print Top-N Performance Summary
    print('\n--- Top-N Performance Summary ---')
    print(f"Global Target Rate: {top_n_metrics['global_target_rate']*100:.1f}%")
    print('| Top N % | Target Rate | Lift |')
    print('|---------|-------------|------|')
    for p in [10, 15, 20]:
        if p in top_n_metrics['percentiles']:
            m = top_n_metrics['percentiles'][p]
            print(f"| Top {p}% | {m['target_rate']*100:.1f}% | {m['lift']:.2f}x |")

    # Print Cumulative Lift Summary
    print('\n--- Cumulative Lift Summary ---')
    print(f"AULC (Area Under Lift Curve): {cumulative_metrics['aulc']:.3f}")
    print(f"Lift @ 10%: {cumulative_metrics['lift_at_10']:.2f}x")
    print(f"Lift @ 15%: {cumulative_metrics['lift_at_15']:.2f}x")
    print(f"Lift @ 20%: {cumulative_metrics['lift_at_20']:.2f}x")

    # Return paths for report integration
    return {
        'confusion_matrix': cm_path,
        'ev_calibration': ev_path,
        'quartile_analysis': quartile_path,
        'top15_analysis': top15_path,
        'calibration_target': target_cal_path,
        'calibration_all': all_cal_path,
        'calibration_metrics': calibration_results,
        'top_n_curve': top_n_curve_path,
        'top_n_table': top_n_table_path,
        'top_n_metrics': top_n_metrics,
        'cumulative_lift': cumulative_lift_path,
        'cumulative_metrics': cumulative_metrics,
        'timestamp': timestamp
    }


if __name__ == '__main__':
    main()
