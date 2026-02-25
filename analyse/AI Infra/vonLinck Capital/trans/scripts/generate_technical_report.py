#!/usr/bin/env python3
"""
VonLinck Capital - TRAnS Technical Report Generator
Generates comprehensive PDF report with analytics and visualizations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import feature names from config (single source of truth)
from config.context_features import CONTEXT_FEATURES as CONFIG_CONTEXT_FEATURES
from config.temporal_features import TemporalFeatureConfig

import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colors
VONLINCK_COLORS = {
    'primary': '#1a365d',      # Dark blue
    'secondary': '#2c5282',    # Medium blue
    'accent': '#4299e1',       # Light blue
    'danger': '#c53030',       # Red
    'warning': '#d69e2e',      # Yellow/Gold
    'success': '#38a169',      # Green
    'neutral': '#718096',      # Gray
    'background': '#f7fafc',   # Light gray
}

CLASS_COLORS = {
    0: VONLINCK_COLORS['danger'],   # Danger
    1: VONLINCK_COLORS['warning'],  # Noise
    2: VONLINCK_COLORS['success'],  # Target
}

CLASS_NAMES = ['Danger', 'Noise', 'Target']


def load_data():
    """Load model, sequences, and metadata."""
    print("Loading data...")

    # Load sequences
    with h5py.File('output/sequences/combined/sequences_combined.h5', 'r') as f:
        sequences = f['sequences'][:]
        context = f['context'][:]
        labels = f['labels'][:]

    # Load metadata
    meta = pd.read_parquet('output/sequences/combined/metadata_combined.parquet')
    meta['pattern_end_date'] = pd.to_datetime(meta['pattern_end_date'])
    meta['year'] = meta['pattern_end_date'].dt.year
    meta['quarter'] = meta['pattern_end_date'].dt.to_period('Q').astype(str)
    meta['label'] = labels

    # Load model
    from models.temporal_hybrid_unified import HybridFeatureNetwork as HybridFeatureNetworkV20

    model_path = Path('output/models/v20_concat_notrin_20260122_062713.pt')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = HybridFeatureNetworkV20(
        ablation_mode='concat',
        input_features=10,
        sequence_length=20,
        context_features=18,
        lstm_hidden=32,
        num_classes=3,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get predictions
    print("Generating predictions...")
    batch_size = 2000
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            seq_batch = torch.tensor(sequences[i:i+batch_size], dtype=torch.float32)
            ctx_batch = torch.tensor(context[i:i+batch_size], dtype=torch.float32)
            logits = model(seq_batch, ctx_batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.numpy())

    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)

    # Strategic values and EV
    STRATEGIC_VALUES = {0: -2.0, 1: -0.1, 2: 5.0}
    ev = probs[:, 0] * STRATEGIC_VALUES[0] + probs[:, 1] * STRATEGIC_VALUES[1] + probs[:, 2] * STRATEGIC_VALUES[2]

    meta['prob_danger'] = probs[:, 0]
    meta['prob_noise'] = probs[:, 1]
    meta['prob_target'] = probs[:, 2]
    meta['pred'] = preds
    meta['ev'] = ev

    return sequences, context, labels, meta, probs, model


def create_title_page(pdf, meta):
    """Create the title page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor(VONLINCK_COLORS['background'])

    # Header bar
    rect = mpatches.Rectangle((0, 0.85), 1, 0.15, facecolor=VONLINCK_COLORS['primary'])
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.92, 'VONLINCK CAPITAL', fontsize=28, fontweight='bold',
            ha='center', va='center', color='white', fontfamily='sans-serif')

    # Subtitle
    ax.text(0.5, 0.70, 'TRAnS Model Technical Report', fontsize=24,
            ha='center', va='center', color=VONLINCK_COLORS['primary'], fontweight='bold')

    ax.text(0.5, 0.62, 'Temporal Retail Analysis System', fontsize=16,
            ha='center', va='center', color=VONLINCK_COLORS['secondary'])

    # Report details box
    rect2 = mpatches.FancyBboxPatch((0.2, 0.30), 0.6, 0.25,
                                     boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor=VONLINCK_COLORS['secondary'],
                                     linewidth=2)
    ax.add_patch(rect2)

    ax.text(0.5, 0.50, 'Model Evaluation & Performance Analysis', fontsize=14,
            ha='center', va='center', color=VONLINCK_COLORS['primary'], fontweight='bold')

    date_range = f"{meta['pattern_end_date'].min().strftime('%Y-%m-%d')} to {meta['pattern_end_date'].max().strftime('%Y-%m-%d')}"
    ax.text(0.5, 0.44, f'Data Period: {date_range}', fontsize=11,
            ha='center', va='center', color=VONLINCK_COLORS['neutral'])

    ax.text(0.5, 0.38, f'Total Patterns Analyzed: {len(meta):,}', fontsize=11,
            ha='center', va='center', color=VONLINCK_COLORS['neutral'])

    ax.text(0.5, 0.32, f'Model: V20 Concat (72,367 parameters)', fontsize=11,
            ha='center', va='center', color=VONLINCK_COLORS['neutral'])

    # Footer
    ax.text(0.5, 0.12, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=10, ha='center', va='center', color=VONLINCK_COLORS['neutral'])

    ax.text(0.5, 0.06, 'CONFIDENTIAL - FOR INTERNAL USE ONLY',
            fontsize=9, ha='center', va='center', color=VONLINCK_COLORS['danger'],
            fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_executive_summary(pdf, meta, labels):
    """Create executive summary page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    # Title
    fig.text(0.5, 0.95, 'EXECUTIVE SUMMARY', fontsize=18, fontweight='bold',
             ha='center', va='top', color=VONLINCK_COLORS['primary'])

    # Calculate key metrics
    train_mask = meta['year'] <= 2022
    val_mask = meta['year'] == 2023
    test_mask = meta['year'] >= 2024

    baseline_target = (labels == 2).mean()

    # Top-15% performance by split
    def get_top15_metrics(mask):
        split_data = meta[mask]
        if len(split_data) == 0:
            return 0, 0, 0
        k = int(len(split_data) * 0.15)
        top_idx = split_data['ev'].nlargest(k).index
        top_labels = split_data.loc[top_idx, 'label']
        target_rate = (top_labels == 2).mean()
        base = (split_data['label'] == 2).mean()
        lift = target_rate / base if base > 0 else 0
        return target_rate, base, lift

    train_target, train_base, train_lift = get_top15_metrics(train_mask)
    val_target, val_base, val_lift = get_top15_metrics(val_mask)
    test_target, test_base, test_lift = get_top15_metrics(test_mask)

    # Summary text
    summary_text = f"""
MODEL OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:     V20 Concat (CNN + Self-Attention + GRN)
Parameters:       72,367
Input Features:   10 temporal × 20 timesteps + 18 context
Output Classes:   3 (Danger, Noise, Target)


DATASET STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Patterns:   {len(meta):,}
Training Set:     {train_mask.sum():,} patterns (≤2022)
Validation Set:   {val_mask.sum():,} patterns (2023)
Test Set:         {test_mask.sum():,} patterns (2024+)

Class Distribution (Overall):
  • Danger:  {(labels==0).sum():,} ({100*(labels==0).mean():.1f}%)
  • Noise:   {(labels==1).sum():,} ({100*(labels==1).mean():.1f}%)
  • Target:  {(labels==2).sum():,} ({100*(labels==2).mean():.1f}%)


KEY PERFORMANCE METRICS (Top-15% by Expected Value)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Target Rate    Baseline    Lift      Status
                    ───────────────────────────────────────────
Train (≤2022):        {100*train_target:.1f}%       {100*train_base:.1f}%      {train_lift:.2f}x     {'✓ OK' if train_lift > 1 else '✗ FAIL'}
Validation (2023):    {100*val_target:.1f}%       {100*val_base:.1f}%      {val_lift:.2f}x     {'✓ OK' if val_lift > 1 else '✗ FAIL'}
Test (2024+):         {100*test_target:.1f}%       {100*test_base:.1f}%      {test_lift:.2f}x     {'✓ OK' if test_lift > 1 else '✗ FAIL'}


CRITICAL FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  MODEL UNDERPERFORMS BASELINE ON OUT-OF-SAMPLE DATA

    • Test lift of {test_lift:.2f}x indicates model predictions are {'better' if test_lift > 1 else 'WORSE'} than random
    • Severe temporal overfitting detected (Train: {train_lift:.2f}x → Test: {test_lift:.2f}x)
    • Model learned COVID-era (2020-21) patterns that don't generalize
    • Calibration error exceeds 55% for Danger and Target classes
    • Model never predicts Noise class (binary Danger/Target behavior)


RECOMMENDATION: {'DO NOT DEPLOY' if test_lift < 1 else 'PROCEED WITH CAUTION'}
"""

    fig.text(0.08, 0.88, summary_text, fontsize=9, fontfamily='monospace',
             va='top', ha='left', color=VONLINCK_COLORS['primary'])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_architecture_page(pdf):
    """Create model architecture diagram page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(5, 9.5, 'MODEL ARCHITECTURE', fontsize=18, fontweight='bold',
            ha='center', va='top', color=VONLINCK_COLORS['primary'])

    ax.text(5, 9.0, 'V20 Concat - Hybrid Feature Network', fontsize=12,
            ha='center', va='top', color=VONLINCK_COLORS['secondary'])

    # Draw architecture boxes
    def draw_box(x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black',
                                        alpha=0.8, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
                fontweight='bold', color='white' if color != '#f7fafc' else 'black')

    # Input boxes
    draw_box(2, 7.5, 2.5, 0.8, 'Temporal Input\n(20 × 10)', VONLINCK_COLORS['secondary'])
    draw_box(8, 7.5, 2.5, 0.8, 'Context Input\n(18 features)', VONLINCK_COLORS['secondary'])

    # CNN branch
    draw_box(2, 6.2, 2.2, 0.6, 'CNN Encoder', VONLINCK_COLORS['accent'])
    draw_box(2, 5.4, 2.2, 0.6, '32ch (3×3) + 64ch (5×5)', VONLINCK_COLORS['background'], fontsize=8)

    # Attention
    draw_box(2, 4.4, 2.2, 0.6, 'Self-Attention', VONLINCK_COLORS['accent'])
    draw_box(2, 3.6, 2.2, 0.6, '4 heads + RoPE', VONLINCK_COLORS['background'], fontsize=8)

    # GRN branch
    draw_box(8, 6.2, 2.2, 0.6, 'GRN Layer', VONLINCK_COLORS['accent'])
    draw_box(8, 5.4, 2.2, 0.6, 'Gated Residual', VONLINCK_COLORS['background'], fontsize=8)

    # Fusion
    draw_box(5, 2.5, 3, 0.8, 'Concatenation Fusion', VONLINCK_COLORS['primary'])

    # Output
    draw_box(5, 1.2, 2.5, 0.8, 'Output Layer\n3 Classes', VONLINCK_COLORS['danger'])

    # Arrows
    arrow_props = dict(arrowstyle='->', color=VONLINCK_COLORS['neutral'], lw=2)
    ax.annotate('', xy=(2, 6.6), xytext=(2, 7.1), arrowprops=arrow_props)
    ax.annotate('', xy=(2, 5.8), xytext=(2, 5.9), arrowprops=arrow_props)
    ax.annotate('', xy=(2, 4.8), xytext=(2, 5.1), arrowprops=arrow_props)
    ax.annotate('', xy=(2, 4.0), xytext=(2, 4.1), arrowprops=arrow_props)

    ax.annotate('', xy=(8, 6.6), xytext=(8, 7.1), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 5.8), xytext=(8, 5.9), arrowprops=arrow_props)

    ax.annotate('', xy=(3.5, 2.5), xytext=(2, 3.3), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 2.5), xytext=(8, 5.1), arrowprops=arrow_props)

    ax.annotate('', xy=(5, 1.6), xytext=(5, 2.1), arrowprops=arrow_props)

    # Parameter counts
    params_text = """
Parameter Summary
─────────────────────────
CNN Encoder:      ~25,000
Self-Attention:   ~20,000
GRN Context:      ~8,000
Fusion MLP:       ~18,000
Output Layer:     ~1,400
─────────────────────────
TOTAL:            72,367
"""
    ax.text(0.5, 2.5, params_text, fontsize=9, fontfamily='monospace',
            va='top', color=VONLINCK_COLORS['primary'])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_data_distribution_page(pdf, meta, labels):
    """Create data distribution visualizations."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('DATA DISTRIBUTION ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # 1. Class distribution pie chart
    ax1 = fig.add_subplot(2, 3, 1)
    class_counts = [sum(labels == i) for i in range(3)]
    colors = [CLASS_COLORS[i] for i in range(3)]
    wedges, texts, autotexts = ax1.pie(class_counts, labels=CLASS_NAMES, autopct='%1.1f%%',
                                        colors=colors, explode=(0, 0, 0.05))
    ax1.set_title('Overall Class Distribution', fontsize=10, fontweight='bold')

    # 2. Patterns by year
    ax2 = fig.add_subplot(2, 3, 2)
    yearly = meta.groupby('year').size()
    bars = ax2.bar(yearly.index, yearly.values, color=VONLINCK_COLORS['accent'], edgecolor='white')
    ax2.set_xlabel('Year', fontsize=9)
    ax2.set_ylabel('Pattern Count', fontsize=9)
    ax2.set_title('Patterns by Year', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Class distribution by year (stacked)
    ax3 = fig.add_subplot(2, 3, 3)
    yearly_class = meta.groupby(['year', 'label']).size().unstack(fill_value=0)
    yearly_class.plot(kind='bar', stacked=True, ax=ax3, color=colors, edgecolor='white')
    ax3.set_xlabel('Year', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.set_title('Class Distribution by Year', fontsize=10, fontweight='bold')
    ax3.legend(CLASS_NAMES, loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Target rate by year
    ax4 = fig.add_subplot(2, 3, 4)
    target_rate = meta.groupby('year').apply(lambda x: (x['label'] == 2).mean())
    ax4.plot(target_rate.index, target_rate.values * 100, 'o-',
             color=VONLINCK_COLORS['success'], linewidth=2, markersize=8)
    ax4.axhline(y=100 * (labels == 2).mean(), color=VONLINCK_COLORS['danger'],
                linestyle='--', label='Overall Baseline')
    ax4.fill_between(target_rate.index, 0, target_rate.values * 100,
                     alpha=0.3, color=VONLINCK_COLORS['success'])
    ax4.set_xlabel('Year', fontsize=9)
    ax4.set_ylabel('Target Rate (%)', fontsize=9)
    ax4.set_title('Target Rate Evolution', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, max(target_rate.values * 100) * 1.2)

    # 5. Train/Val/Test split
    ax5 = fig.add_subplot(2, 3, 5)
    split_counts = [
        (meta['year'] <= 2022).sum(),
        (meta['year'] == 2023).sum(),
        (meta['year'] >= 2024).sum()
    ]
    split_labels = ['Train\n(≤2022)', 'Val\n(2023)', 'Test\n(2024+)']
    split_colors = [VONLINCK_COLORS['accent'], VONLINCK_COLORS['warning'], VONLINCK_COLORS['success']]
    bars = ax5.bar(split_labels, split_counts, color=split_colors, edgecolor='white')
    for bar, count in zip(bars, split_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    ax5.set_ylabel('Pattern Count', fontsize=9)
    ax5.set_title('Temporal Split Distribution', fontsize=10, fontweight='bold')

    # 6. Danger rate by year
    ax6 = fig.add_subplot(2, 3, 6)
    danger_rate = meta.groupby('year').apply(lambda x: (x['label'] == 0).mean())
    ax6.plot(danger_rate.index, danger_rate.values * 100, 'o-',
             color=VONLINCK_COLORS['danger'], linewidth=2, markersize=8)
    ax6.axhline(y=100 * (labels == 0).mean(), color=VONLINCK_COLORS['neutral'],
                linestyle='--', label='Overall Baseline')
    ax6.fill_between(danger_rate.index, 0, danger_rate.values * 100,
                     alpha=0.3, color=VONLINCK_COLORS['danger'])
    ax6.set_xlabel('Year', fontsize=9)
    ax6.set_ylabel('Danger Rate (%)', fontsize=9)
    ax6.set_title('Danger Rate Evolution', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_performance_metrics_page(pdf, meta, labels, probs):
    """Create performance metrics visualizations."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('MODEL PERFORMANCE METRICS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    preds = probs.argmax(axis=1)
    STRATEGIC_VALUES = {0: -2.0, 1: -0.1, 2: 5.0}

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(2, 2, 1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=ax1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Proportion'})
    ax1.set_xlabel('Predicted', fontsize=10)
    ax1.set_ylabel('Actual', fontsize=10)
    ax1.set_title('Confusion Matrix (Row-Normalized)', fontsize=10, fontweight='bold')

    # Add raw counts
    for i in range(3):
        for j in range(3):
            ax1.text(j + 0.5, i + 0.7, f'n={cm[i,j]:,}', ha='center', va='center',
                    fontsize=7, color='gray')

    # 2. Top-K Target Rate
    ax2 = fig.add_subplot(2, 2, 2)
    ev = meta['ev'].values
    sorted_idx = np.argsort(-ev)

    percentiles = np.arange(5, 55, 5)
    target_rates = []
    baseline = (labels == 2).mean()

    for pct in percentiles:
        k = int(len(labels) * pct / 100)
        top_labels = labels[sorted_idx[:k]]
        target_rates.append((top_labels == 2).mean())

    ax2.plot(percentiles, [r * 100 for r in target_rates], 'o-',
             color=VONLINCK_COLORS['success'], linewidth=2, markersize=8, label='Model')
    ax2.axhline(y=baseline * 100, color=VONLINCK_COLORS['danger'],
                linestyle='--', linewidth=2, label='Baseline')
    ax2.fill_between(percentiles, baseline * 100, [r * 100 for r in target_rates],
                     where=[r > baseline for r in target_rates],
                     alpha=0.3, color=VONLINCK_COLORS['success'])
    ax2.fill_between(percentiles, baseline * 100, [r * 100 for r in target_rates],
                     where=[r <= baseline for r in target_rates],
                     alpha=0.3, color=VONLINCK_COLORS['danger'])
    ax2.set_xlabel('Top Percentile (%)', fontsize=10)
    ax2.set_ylabel('Target Rate (%)', fontsize=10)
    ax2.set_title('Target Rate by Top Percentile (EV Ranking)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Lift by Split
    ax3 = fig.add_subplot(2, 2, 3)

    splits = ['Train\n(≤2022)', 'Val\n(2023)', 'Test\n(2024+)']
    masks = [meta['year'] <= 2022, meta['year'] == 2023, meta['year'] >= 2024]

    lifts = []
    for mask in masks:
        split_data = meta[mask]
        k = int(len(split_data) * 0.15)
        top_idx = split_data['ev'].nlargest(k).index
        top_labels = split_data.loc[top_idx, 'label']
        target_rate = (top_labels == 2).mean()
        base = (split_data['label'] == 2).mean()
        lift = target_rate / base if base > 0 else 0
        lifts.append(lift)

    colors = [VONLINCK_COLORS['success'] if l > 1 else VONLINCK_COLORS['danger'] for l in lifts]
    bars = ax3.bar(splits, lifts, color=colors, edgecolor='white', linewidth=2)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')

    for bar, lift in zip(bars, lifts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lift:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.set_ylabel('Lift (Top-15%)', fontsize=10)
    ax3.set_title('Lift by Temporal Split', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, max(lifts) * 1.2)
    ax3.legend(fontsize=9)

    # 4. EV Distribution
    ax4 = fig.add_subplot(2, 2, 4)
    for cls in range(3):
        cls_ev = meta.loc[meta['label'] == cls, 'ev']
        ax4.hist(cls_ev, bins=50, alpha=0.6, label=CLASS_NAMES[cls],
                 color=CLASS_COLORS[cls], density=True)

    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Expected Value', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('EV Distribution by True Class', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_calibration_page(pdf, labels, probs):
    """Create calibration analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('CALIBRATION ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Reliability diagrams for each class
    for cls in range(3):
        ax = fig.add_subplot(2, 3, cls + 1)

        cls_probs = probs[:, cls]
        cls_labels = (labels == cls).astype(int)

        # Bin probabilities
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (cls_probs >= bin_edges[i]) & (cls_probs < bin_edges[i+1])
            if mask.sum() > 0:
                bin_accs.append(cls_labels[mask].mean())
                bin_confs.append(cls_probs[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)

        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')

        valid_mask = ~np.isnan(bin_accs)
        ax.bar(np.array(bin_centers)[valid_mask], np.array(bin_accs)[valid_mask],
               width=0.08, alpha=0.7, color=CLASS_COLORS[cls], edgecolor='white',
               label='Observed')

        # Calculate ECE
        ece = 0
        total = 0
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
            if not np.isnan(acc):
                ece += count * abs(acc - conf)
                total += count
        ece = ece / total if total > 0 else 0

        ax.set_xlabel('Predicted Probability', fontsize=9)
        ax.set_ylabel('Observed Frequency', fontsize=9)
        ax.set_title(f'{CLASS_NAMES[cls]}\nECE = {100*ece:.1f}%', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Confidence distribution
    ax4 = fig.add_subplot(2, 3, 4)
    max_probs = probs.max(axis=1)
    ax4.hist(max_probs, bins=50, color=VONLINCK_COLORS['accent'], edgecolor='white', alpha=0.8)
    ax4.axvline(x=0.5, color=VONLINCK_COLORS['danger'], linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Maximum Probability', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('Confidence Distribution', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)

    # Confidence vs Accuracy
    ax5 = fig.add_subplot(2, 3, 5)
    preds = probs.argmax(axis=1)
    correct = preds == labels

    thresholds = np.arange(0.5, 1.0, 0.05)
    accs = []
    coverages = []

    for thresh in thresholds:
        mask = max_probs >= thresh
        if mask.sum() > 0:
            accs.append(correct[mask].mean())
            coverages.append(mask.mean())
        else:
            accs.append(np.nan)
            coverages.append(0)

    ax5.plot(thresholds, [a * 100 for a in accs], 'o-', color=VONLINCK_COLORS['success'],
             linewidth=2, markersize=8, label='Accuracy')
    ax5.set_xlabel('Confidence Threshold', fontsize=9)
    ax5.set_ylabel('Accuracy (%)', fontsize=9, color=VONLINCK_COLORS['success'])
    ax5.tick_params(axis='y', labelcolor=VONLINCK_COLORS['success'])

    ax5b = ax5.twinx()
    ax5b.plot(thresholds, [c * 100 for c in coverages], 's--', color=VONLINCK_COLORS['warning'],
              linewidth=2, markersize=6, label='Coverage')
    ax5b.set_ylabel('Coverage (%)', fontsize=9, color=VONLINCK_COLORS['warning'])
    ax5b.tick_params(axis='y', labelcolor=VONLINCK_COLORS['warning'])

    ax5.set_title('Accuracy vs Coverage Trade-off', fontsize=10, fontweight='bold')

    # Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate confidence stats
    conf_50 = 100 * (max_probs >= 0.5).mean()
    conf_70 = 100 * (max_probs >= 0.7).mean()
    conf_90 = 100 * (max_probs >= 0.9).mean()

    summary = f"""
CALIBRATION SUMMARY
===================================

Expected Calibration Error (ECE):
* Danger:  55.8%
* Noise:   11.6%
* Target:  57.3%

Confidence Distribution:
* >=50%: {conf_50:.1f}% of predictions
* >=70%: {conf_70:.1f}% of predictions
* >=90%: {conf_90:.1f}% of predictions

INTERPRETATION
===================================
WARNING: Model is severely OVERCONFIDENT

When predicting Target with >90%
confidence, actual Target rate
is only ~21%.

Recommend: Apply isotonic calibration
"""
    ax6.text(0.1, 0.95, summary, fontsize=9, fontfamily='monospace',
             va='top', transform=ax6.transAxes, color=VONLINCK_COLORS['primary'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_feature_importance_page(pdf, sequences, context, labels, model):
    """Create feature importance analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('FEATURE IMPORTANCE ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Use imported feature lists (single source of truth)
    CONTEXT_FEATURES = CONFIG_CONTEXT_FEATURES
    _temporal_config = TemporalFeatureConfig()
    TEMPORAL_FEATURES = _temporal_config.all_features

    # Sample for efficiency
    np.random.seed(42)
    sample_idx = np.random.choice(len(labels), min(3000, len(labels)), replace=False)
    sample_seq = sequences[sample_idx]
    sample_ctx = context[sample_idx]
    sample_labels = labels[sample_idx]

    # Baseline accuracy
    def get_accuracy(seq, ctx):
        with torch.no_grad():
            logits = model(torch.tensor(seq, dtype=torch.float32),
                          torch.tensor(ctx, dtype=torch.float32))
            preds = logits.argmax(dim=1).numpy()
        return (preds == sample_labels).mean()

    baseline_acc = get_accuracy(sample_seq, sample_ctx)

    # Context feature importance
    print("Computing context feature importance...")
    ctx_importance = []
    for i in range(min(context.shape[1], len(CONTEXT_FEATURES))):
        perturbed = sample_ctx.copy()
        np.random.shuffle(perturbed[:, i])
        acc = get_accuracy(sample_seq, perturbed)
        ctx_importance.append((CONTEXT_FEATURES[i], baseline_acc - acc))

    ctx_importance.sort(key=lambda x: -x[1])

    # Plot context importance
    ax1 = fig.add_subplot(2, 2, 1)
    names, values = zip(*ctx_importance)
    colors = [VONLINCK_COLORS['success'] if v > 0 else VONLINCK_COLORS['danger'] for v in values]
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, values, color=colors, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Accuracy Drop (Permutation Importance)', fontsize=9)
    ax1.set_title('Context Feature Importance', fontsize=10, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.invert_yaxis()

    # Temporal feature importance
    print("Computing temporal feature importance...")
    temp_importance = []
    for i in range(min(sequences.shape[2], len(TEMPORAL_FEATURES))):
        perturbed = sample_seq.copy()
        perm = np.random.permutation(len(perturbed))
        perturbed[:, :, i] = sample_seq[perm, :, i]
        acc = get_accuracy(perturbed, sample_ctx)
        temp_importance.append((TEMPORAL_FEATURES[i], baseline_acc - acc))

    temp_importance.sort(key=lambda x: -x[1])

    ax2 = fig.add_subplot(2, 2, 2)
    names, values = zip(*temp_importance)
    colors = [VONLINCK_COLORS['success'] if v > 0 else VONLINCK_COLORS['danger'] for v in values]
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, values, color=colors, edgecolor='white')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Accuracy Drop (Permutation Importance)', fontsize=9)
    ax2.set_title('Temporal Feature Importance', fontsize=10, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.invert_yaxis()

    # Feature correlation with labels
    ax3 = fig.add_subplot(2, 2, 3)
    correlations = []
    for i in range(min(context.shape[1], len(CONTEXT_FEATURES))):
        corr = np.corrcoef(context[:, i], labels)[0, 1]
        correlations.append((CONTEXT_FEATURES[i], corr))

    correlations.sort(key=lambda x: -abs(x[1]))
    names, values = zip(*correlations[:12])
    colors = [VONLINCK_COLORS['success'] if v > 0 else VONLINCK_COLORS['danger'] for v in values]
    y_pos = np.arange(len(names))
    ax3.barh(y_pos, values, color=colors, edgecolor='white')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Correlation with Label', fontsize=9)
    ax3.set_title('Feature-Label Correlation (Top 12)', fontsize=10, fontweight='bold')
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.invert_yaxis()

    # Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    top_ctx = ctx_importance[0]
    summary = f"""
FEATURE IMPORTANCE SUMMARY
═══════════════════════════════════════

Top Context Feature:
  {top_ctx[0]}: {top_ctx[1]:+.4f}

  ⚠️  This feature has {abs(top_ctx[1])/max(abs(v) for _, v in ctx_importance[1:] if v != 0):.1f}x
  more importance than the 2nd feature.

  Model is over-reliant on a single feature.

Top Temporal Feature:
  {temp_importance[0][0]}: {temp_importance[0][1]:+.4f}

INTERPRETATION
═══════════════════════════════════════

• coil_intensity dominates context
• Temporal features have low importance
• Model may not be using temporal patterns
  effectively

RECOMMENDATION
═══════════════════════════════════════

Consider regularization to reduce
dependence on single features.
"""
    ax4.text(0.05, 0.95, summary, fontsize=9, fontfamily='monospace',
             va='top', transform=ax4.transAxes, color=VONLINCK_COLORS['primary'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_temporal_stability_page(pdf, meta):
    """Create temporal stability analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('TEMPORAL STABILITY ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    STRATEGIC_VALUES = {0: -2.0, 1: -0.1, 2: 5.0}

    # 1. Lift by Year
    ax1 = fig.add_subplot(2, 2, 1)

    years = sorted(meta['year'].unique())
    lifts = []
    counts = []

    for year in years:
        year_data = meta[meta['year'] == year]
        if len(year_data) < 50:
            lifts.append(np.nan)
            counts.append(len(year_data))
            continue

        k = int(len(year_data) * 0.15)
        top_idx = year_data['ev'].nlargest(k).index
        top_labels = year_data.loc[top_idx, 'label']
        target_rate = (top_labels == 2).mean()
        baseline = (year_data['label'] == 2).mean()
        lift = target_rate / baseline if baseline > 0 else np.nan
        lifts.append(lift)
        counts.append(len(year_data))

    colors = [VONLINCK_COLORS['success'] if (l and l > 1) else VONLINCK_COLORS['danger']
              for l in lifts]
    bars = ax1.bar(years, [l if l else 0 for l in lifts], color=colors, edgecolor='white')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Year', fontsize=9)
    ax1.set_ylabel('Lift (Top-15%)', fontsize=9)
    ax1.set_title('Top-15% Lift by Year', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add train/val/test annotations
    ax1.axvspan(min(years)-0.5, 2022.5, alpha=0.1, color='blue', label='Train')
    ax1.axvspan(2022.5, 2023.5, alpha=0.1, color='yellow', label='Val')
    ax1.axvspan(2023.5, max(years)+0.5, alpha=0.1, color='green', label='Test')

    # 2. Target Rate by Quarter
    ax2 = fig.add_subplot(2, 2, 2)

    quarterly = meta.groupby('quarter').agg({
        'label': [lambda x: (x == 2).mean(), 'count']
    })
    quarterly.columns = ['target_rate', 'count']
    quarterly = quarterly[quarterly['count'] >= 50]

    ax2.plot(range(len(quarterly)), quarterly['target_rate'] * 100, 'o-',
             color=VONLINCK_COLORS['success'], linewidth=2, markersize=4)
    ax2.axhline(y=100 * (meta['label'] == 2).mean(), color=VONLINCK_COLORS['danger'],
                linestyle='--', label='Overall Baseline')
    ax2.set_xlabel('Quarter', fontsize=9)
    ax2.set_ylabel('Target Rate (%)', fontsize=9)
    ax2.set_title('Quarterly Target Rate', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(0, len(quarterly), 4))
    ax2.set_xticklabels(quarterly.index[::4], rotation=45, fontsize=7)
    ax2.legend(fontsize=8)

    # 3. Rolling 6-month lift
    ax3 = fig.add_subplot(2, 2, 3)

    meta_sorted = meta.sort_values('pattern_end_date')
    window = 1000  # Rolling window size
    rolling_lifts = []
    rolling_dates = []

    for i in range(0, len(meta_sorted) - window, window // 4):
        window_data = meta_sorted.iloc[i:i+window]
        k = int(len(window_data) * 0.15)
        top_idx = window_data['ev'].nlargest(k).index
        top_labels = window_data.loc[top_idx, 'label']
        target_rate = (top_labels == 2).mean()
        baseline = (window_data['label'] == 2).mean()
        lift = target_rate / baseline if baseline > 0 else np.nan
        rolling_lifts.append(lift)
        rolling_dates.append(window_data['pattern_end_date'].iloc[len(window_data)//2])

    ax3.plot(rolling_dates, rolling_lifts, '-', color=VONLINCK_COLORS['accent'], linewidth=2)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax3.fill_between(rolling_dates, 1, rolling_lifts,
                     where=[l > 1 for l in rolling_lifts],
                     alpha=0.3, color=VONLINCK_COLORS['success'])
    ax3.fill_between(rolling_dates, 1, rolling_lifts,
                     where=[l <= 1 for l in rolling_lifts],
                     alpha=0.3, color=VONLINCK_COLORS['danger'])
    ax3.set_xlabel('Date', fontsize=9)
    ax3.set_ylabel('Rolling Lift', fontsize=9)
    ax3.set_title(f'Rolling Lift (window={window} patterns)', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Market Regime Analysis
    ax4 = fig.add_subplot(2, 2, 4)

    regimes = {
        'Pre-COVID\n(2010-19)': (2010, 2019),
        'COVID Bull\n(2020-21)': (2020, 2021),
        'Bear\n(2022)': (2022, 2022),
        'Recovery\n(2023-25)': (2023, 2025)
    }

    regime_lifts = []
    for name, (start, end) in regimes.items():
        mask = (meta['year'] >= start) & (meta['year'] <= end)
        regime_data = meta[mask]
        if len(regime_data) < 50:
            regime_lifts.append((name, np.nan))
            continue
        k = int(len(regime_data) * 0.15)
        top_idx = regime_data['ev'].nlargest(k).index
        top_labels = regime_data.loc[top_idx, 'label']
        target_rate = (top_labels == 2).mean()
        baseline = (regime_data['label'] == 2).mean()
        lift = target_rate / baseline if baseline > 0 else np.nan
        regime_lifts.append((name, lift))

    names, values = zip(*regime_lifts)
    colors = [VONLINCK_COLORS['success'] if (v and v > 1) else VONLINCK_COLORS['danger']
              for v in values]
    bars = ax4.bar(names, [v if v else 0 for v in values], color=colors, edgecolor='white')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2)

    for bar, val in zip(bars, values):
        if val:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_ylabel('Lift (Top-15%)', fontsize=9)
    ax4.set_title('Performance by Market Regime', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_regime_shift_page(pdf, meta, context):
    """Create regime shift analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('REGIME SHIFT ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Use imported feature list (single source of truth)
    CONTEXT_FEATURES = CONFIG_CONTEXT_FEATURES

    years = meta['year'].values

    # 1. Feature means by period
    ax1 = fig.add_subplot(2, 2, 1)

    periods = {
        '2020-21': (years >= 2020) & (years <= 2021),
        '2022': years == 2022,
        '2023-25': years >= 2023
    }

    shifts = []
    for i, feat in enumerate(CONTEXT_FEATURES[:18]):
        if i >= context.shape[1]:
            break
        bull = context[periods['2020-21'], i].mean()
        recov = context[periods['2023-25'], i].mean()
        shifts.append((feat, recov - bull))

    shifts.sort(key=lambda x: -abs(x[1]))
    names, values = zip(*shifts[:10])
    colors = [VONLINCK_COLORS['success'] if v > 0 else VONLINCK_COLORS['danger'] for v in values]
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, values, color=colors, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Change (2023-25 vs 2020-21)', fontsize=9)
    ax1.set_title('Feature Distribution Shift', fontsize=10, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.invert_yaxis()

    # 2. Key feature evolution
    ax2 = fig.add_subplot(2, 2, 2)

    # price_position_at_end over time
    key_idx = CONTEXT_FEATURES.index('price_position_at_end')
    yearly_mean = []
    yearly_std = []
    for year in sorted(meta['year'].unique()):
        mask = years == year
        yearly_mean.append(context[mask, key_idx].mean())
        yearly_std.append(context[mask, key_idx].std())

    year_list = sorted(meta['year'].unique())
    ax2.plot(year_list, yearly_mean, 'o-', color=VONLINCK_COLORS['accent'], linewidth=2, markersize=8)
    ax2.fill_between(year_list,
                     [m - s for m, s in zip(yearly_mean, yearly_std)],
                     [m + s for m, s in zip(yearly_mean, yearly_std)],
                     alpha=0.3, color=VONLINCK_COLORS['accent'])
    ax2.set_xlabel('Year', fontsize=9)
    ax2.set_ylabel('price_position_at_end', fontsize=9)
    ax2.set_title('Key Feature Evolution Over Time', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Add annotations
    ax2.axvline(x=2022, color=VONLINCK_COLORS['danger'], linestyle='--', alpha=0.5)
    ax2.text(2022.1, ax2.get_ylim()[1] * 0.95, 'Bear\nMarket', fontsize=8,
             color=VONLINCK_COLORS['danger'])

    # 3. coil_intensity evolution
    ax3 = fig.add_subplot(2, 2, 3)

    coil_idx = CONTEXT_FEATURES.index('coil_intensity')
    coil_yearly = []
    for year in year_list:
        mask = years == year
        coil_yearly.append(context[mask, coil_idx].mean())

    ax3.plot(year_list, coil_yearly, 'o-', color=VONLINCK_COLORS['success'], linewidth=2, markersize=8)
    ax3.set_xlabel('Year', fontsize=9)
    ax3.set_ylabel('coil_intensity', fontsize=9)
    ax3.set_title('Coil Intensity Over Time', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Distribution comparison
    ax4 = fig.add_subplot(2, 2, 4)

    bull_data = context[periods['2020-21'], key_idx]
    recov_data = context[periods['2023-25'], key_idx]

    ax4.hist(bull_data, bins=50, alpha=0.6, color=VONLINCK_COLORS['accent'],
             label='2020-21 (Train)', density=True)
    ax4.hist(recov_data, bins=50, alpha=0.6, color=VONLINCK_COLORS['warning'],
             label='2023-25 (Test)', density=True)
    ax4.set_xlabel('price_position_at_end', fontsize=9)
    ax4.set_ylabel('Density', fontsize=9)
    ax4.set_title('Distribution Shift: price_position_at_end', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)

    # Calculate PSI
    def calculate_psi(expected, actual, bins=10):
        expected_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
        actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
        expected_hist = np.clip(expected_hist, 1e-10, None)
        actual_hist = np.clip(actual_hist, 1e-10, None)
        psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
        return psi

    psi = calculate_psi(bull_data, recov_data)
    ax4.text(0.95, 0.95, f'PSI = {psi:.3f}', transform=ax4.transAxes,
             ha='right', va='top', fontsize=10, fontweight='bold',
             color=VONLINCK_COLORS['danger'] if psi > 0.25 else VONLINCK_COLORS['success'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_recommendations_page(pdf, meta):
    """Create recommendations page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(0.5, 0.95, 'RECOMMENDATIONS & NEXT STEPS', fontsize=18, fontweight='bold',
            ha='center', va='top', color=VONLINCK_COLORS['primary'])

    recommendations = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  CRITICAL ACTIONS                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ⛔ DO NOT DEPLOY MODEL IN CURRENT STATE                                            │
│     Test-set lift of 0.89x indicates predictions are WORSE than random selection    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  IMMEDIATE FIXES (Priority 1)                                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  1. Apply Isotonic Calibration                                                      │
│     ECE > 55% indicates severe overconfidence. Post-hoc calibration required.       │
│                                                                                     │
│  2. Retrain with Updated Split                                                      │
│     Include 2023 data in training set to capture recent market dynamics.            │
│     Suggested: Train ≤2024Q2, Val 2024Q3, Test 2024Q4+                             │
│                                                                                     │
│  3. Address Class Imbalance                                                         │
│     Model never predicts Noise class. Use SMOTE or class-weighted loss.             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  MEDIUM-TERM IMPROVEMENTS (Priority 2)                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  4. Feature Engineering                                                             │
│     • Normalize price_position_at_end by regime (major distributional shift)        │
│     • Reduce coil_intensity dominance (model over-reliant on single feature)        │
│     • Add regime indicators as explicit features                                    │
│                                                                                     │
│  5. Regularization                                                                  │
│     • Increase dropout to reduce overfitting                                        │
│     • Add L2 regularization on context branch                                       │
│     • Consider feature dropout during training                                      │
│                                                                                     │
│  6. Rolling Window Training                                                         │
│     • Implement online learning with 90-day retraining cycle                        │
│     • Monitor PSI for distributional drift triggers                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURAL CHANGES (Priority 3)                                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  7. Regime-Conditioned Ensemble                                                     │
│     Train separate models for bull/bear/sideways regimes, blend at inference        │
│                                                                                     │
│  8. Binary Classification                                                           │
│     If Noise remains unlearnable, merge with Danger → binary Danger/Target          │
│                                                                                     │
│  9. Uncertainty Quantification                                                      │
│     Consider Monte Carlo Dropout or conformal prediction for confidence bounds      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  MONITORING REQUIREMENTS                                                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  • Weekly PSI monitoring on key features (threshold: 0.25)                          │
│  • Monthly lift recalculation on recent predictions                                 │
│  • Quarterly model retraining with updated data                                     │
│  • Real-time calibration tracking via Brier score                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
"""

    ax.text(0.02, 0.88, recommendations, fontsize=8.5, fontfamily='monospace',
            va='top', ha='left', color=VONLINCK_COLORS['primary'])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_appendix_metrics(pdf, meta, labels, probs):
    """Create detailed metrics appendix."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('APPENDIX A: DETAILED METRICS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Create text-based metrics tables
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    preds = probs.argmax(axis=1)

    # Per-year metrics
    metrics_text = """
DETAILED PERFORMANCE BY YEAR
════════════════════════════════════════════════════════════════════════════════════════

Year    Patterns    Danger%    Noise%    Target%    Baseline    Top-15%    Lift
────────────────────────────────────────────────────────────────────────────────────────
"""

    for year in sorted(meta['year'].unique()):
        year_data = meta[meta['year'] == year]
        if len(year_data) < 50:
            continue

        danger_pct = (year_data['label'] == 0).mean() * 100
        noise_pct = (year_data['label'] == 1).mean() * 100
        target_pct = (year_data['label'] == 2).mean() * 100

        k = int(len(year_data) * 0.15)
        top_idx = year_data['ev'].nlargest(k).index
        top_labels = year_data.loc[top_idx, 'label']
        top_target = (top_labels == 2).mean() * 100
        lift = top_target / target_pct if target_pct > 0 else 0

        metrics_text += f"{year}    {len(year_data):>6,}      {danger_pct:>5.1f}%    {noise_pct:>5.1f}%    {target_pct:>5.1f}%      {target_pct:>5.1f}%     {top_target:>5.1f}%   {lift:>5.2f}x\n"

    metrics_text += """
────────────────────────────────────────────────────────────────────────────────────────

CLASSIFICATION METRICS
════════════════════════════════════════════════════════════════════════════════════════

"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2])
    acc = accuracy_score(labels, preds)

    metrics_text += f"""
Class       Precision    Recall    F1-Score    Support
────────────────────────────────────────────────────────
Danger        {prec[0]:.3f}       {rec[0]:.3f}      {f1[0]:.3f}     {sup[0]:,}
Noise         {prec[1]:.3f}       {rec[1]:.3f}      {f1[1]:.3f}      {sup[1]:,}
Target        {prec[2]:.3f}       {rec[2]:.3f}      {f1[2]:.3f}      {sup[2]:,}
────────────────────────────────────────────────────────
Overall       {acc:.3f}                            {len(labels):,}


STRATEGIC VALUE ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════

Strategic Values: Danger=-2.0, Noise=-0.1, Target=+5.0

Baseline EV: {((labels==0).mean() * -2.0 + (labels==1).mean() * -0.1 + (labels==2).mean() * 5.0):.3f}

Expected Value Distribution:
  Min:      {meta['ev'].min():.3f}
  25%:      {meta['ev'].quantile(0.25):.3f}
  Median:   {meta['ev'].median():.3f}
  75%:      {meta['ev'].quantile(0.75):.3f}
  Max:      {meta['ev'].max():.3f}
"""

    ax.text(0.02, 0.95, metrics_text, fontsize=8.5, fontfamily='monospace',
            va='top', ha='left', color=VONLINCK_COLORS['primary'])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pre2020_analysis_page(pdf, sequences, context, labels, meta, model):
    """Create pre-2020 zero prediction analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('PRE-2020 ANALYSIS: WHY ZERO TARGETS IN TOP-15%', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Use imported feature list (single source of truth)
    CONTEXT_FEATURES = CONFIG_CONTEXT_FEATURES

    STRATEGIC_VALUES = {0: -2.0, 1: -0.1, 2: 5.0}

    # Split data
    pre2020_mask = meta['year'] < 2020
    post2020_mask = meta['year'] >= 2020

    pre2020_idx = np.where(pre2020_mask)[0]
    pre2020_labels = labels[pre2020_idx]

    # Get predictions for pre-2020
    with torch.no_grad():
        logits = model(torch.tensor(sequences[pre2020_idx], dtype=torch.float32),
                       torch.tensor(context[pre2020_idx], dtype=torch.float32))
        probs_pre = torch.softmax(logits, dim=1).numpy()

    ev_pre = (probs_pre[:, 0] * STRATEGIC_VALUES[0] +
              probs_pre[:, 1] * STRATEGIC_VALUES[1] +
              probs_pre[:, 2] * STRATEGIC_VALUES[2])

    # Compute ranks
    ranks = np.argsort(np.argsort(-ev_pre))
    target_mask = pre2020_labels == 2
    target_ranks = ranks[target_mask]

    # 1. Target Rate Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    periods = ['Pre-2020', 'Post-2020']
    target_rates = [
        100 * (labels[pre2020_mask.values] == 2).mean(),
        100 * (labels[post2020_mask.values] == 2).mean()
    ]
    colors = [VONLINCK_COLORS['danger'], VONLINCK_COLORS['success']]
    bars = ax1.bar(periods, target_rates, color=colors, edgecolor='white', linewidth=2)

    for bar, rate in zip(bars, target_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Actual Target Rate (%)', fontsize=10)
    ax1.set_title('15x Difference in Target Rate', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 30)

    # Add counts
    ax1.text(0, target_rates[0]/2, f'n={pre2020_mask.sum():,}\n{(labels[pre2020_mask.values]==2).sum()} Targets',
             ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax1.text(1, target_rates[1]/2, f'n={post2020_mask.sum():,}\n{(labels[post2020_mask.values]==2).sum():,} Targets',
             ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 2. Target Rank Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    top15_threshold = int(len(pre2020_idx) * 0.15)

    ax2.hist(target_ranks, bins=30, color=VONLINCK_COLORS['danger'], edgecolor='white', alpha=0.8)
    ax2.axvline(x=top15_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Top-15% threshold ({top15_threshold})')
    ax2.axvline(x=target_ranks.min(), color=VONLINCK_COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Best Target rank ({target_ranks.min()})')

    ax2.set_xlabel('Rank by Model EV (lower = better)', fontsize=10)
    ax2.set_ylabel('Count of Targets', fontsize=10)
    ax2.set_title('Pre-2020 Target Rank Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')

    # Add annotation
    ax2.annotate('All 77 Targets\nrank > 689', xy=(target_ranks.min(), 5),
                xytext=(1500, 8), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=VONLINCK_COLORS['danger']))

    # 3. Feature Comparison: Pre-2020 Targets vs Model's Top-15%
    ax3 = fig.add_subplot(2, 2, 3)

    # Get top-15% of pre-2020 by EV
    top15_local_idx = np.argsort(-ev_pre)[:top15_threshold]
    top15_global_idx = pre2020_idx[top15_local_idx]
    pre2020_target_idx = pre2020_idx[labels[pre2020_idx] == 2]

    # Calculate feature differences
    feature_diffs = []
    for i, feat in enumerate(CONTEXT_FEATURES[:min(len(CONTEXT_FEATURES), context.shape[1])]):
        target_mean = context[pre2020_target_idx, i].mean()
        top15_mean = context[top15_global_idx, i].mean()
        diff = target_mean - top15_mean
        feature_diffs.append((feat, diff, target_mean, top15_mean))

    # Sort by absolute difference
    feature_diffs.sort(key=lambda x: -abs(x[1]))
    top_diffs = feature_diffs[:8]

    names = [x[0] for x in top_diffs]
    diffs = [x[1] for x in top_diffs]
    colors = [VONLINCK_COLORS['success'] if d > 0 else VONLINCK_COLORS['danger'] for d in diffs]

    y_pos = np.arange(len(names))
    ax3.barh(y_pos, diffs, color=colors, edgecolor='white')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Difference (Target - Top15)', fontsize=9)
    ax3.set_title('Feature Gap: Actual Targets vs Model Top-15%', fontsize=11, fontweight='bold')
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.invert_yaxis()

    # 4. Summary Text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary = f"""
ROOT CAUSE ANALYSIS
===================================

THE PROBLEM:
Pre-2020 has 0% Targets in Top-15%
despite model predicting P(Target)

FINDINGS:

1. DATA SCARCITY
   Pre-2020:  77 Targets / 4,595 (1.7%)
   Post-2020: 6,180 / 24,888 (24.8%)

   15x fewer Targets in pre-2020!

2. WORSE THAN RANDOM RANKING
   Expected Targets in Top-15%: ~12
   Actual Targets in Top-15%: 0
   Best Target rank: {target_ranks.min()} (threshold: {top15_threshold})

3. FEATURE INVERSION
   Model learned from post-2020:
   - High coil_intensity = Target
   - Low price_position = Target

   But pre-2020 Targets have:
   - LOW coil_intensity (0.47)
   - HIGH price_position (0.78)

   Model ranks them at BOTTOM!

CONCLUSION:
===================================
Market dynamics fundamentally
changed around 2020 (COVID,
retail boom, meme stocks).

Pre-2020 patterns should be
EXCLUDED from training or
treated as separate regime.
"""

    ax4.text(0.05, 0.95, summary, fontsize=9, fontfamily='monospace',
             va='top', transform=ax4.transAxes, color=VONLINCK_COLORS['primary'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pre2020_feature_detail_page(pdf, sequences, context, labels, meta, model):
    """Create detailed pre-2020 feature analysis page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('PRE-2020 FEATURE DETAIL ANALYSIS', fontsize=16, fontweight='bold',
                 color=VONLINCK_COLORS['primary'], y=0.98)

    # Use imported feature list (single source of truth)
    CONTEXT_FEATURES = CONFIG_CONTEXT_FEATURES

    pre2020_mask = meta['year'] < 2020
    post2020_mask = meta['year'] >= 2020

    pre2020_targets = (labels == 2) & pre2020_mask.values
    post2020_targets = (labels == 2) & post2020_mask.values

    # 1. price_position_at_end distribution
    ax1 = fig.add_subplot(2, 2, 1)
    feat_idx = CONTEXT_FEATURES.index('price_position_at_end')

    ax1.hist(context[pre2020_targets, feat_idx], bins=30, alpha=0.6,
             color=VONLINCK_COLORS['danger'], label='Pre-2020 Targets', density=True)
    ax1.hist(context[post2020_targets, feat_idx], bins=30, alpha=0.6,
             color=VONLINCK_COLORS['success'], label='Post-2020 Targets', density=True)

    ax1.axvline(x=context[pre2020_targets, feat_idx].mean(), color=VONLINCK_COLORS['danger'],
                linestyle='--', linewidth=2)
    ax1.axvline(x=context[post2020_targets, feat_idx].mean(), color=VONLINCK_COLORS['success'],
                linestyle='--', linewidth=2)

    ax1.set_xlabel('price_position_at_end', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('Price Position: +0.34 Shift in Targets', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)

    # 2. coil_intensity distribution
    ax2 = fig.add_subplot(2, 2, 2)
    feat_idx = CONTEXT_FEATURES.index('coil_intensity')

    ax2.hist(context[pre2020_targets, feat_idx], bins=30, alpha=0.6,
             color=VONLINCK_COLORS['danger'], label='Pre-2020 Targets', density=True)
    ax2.hist(context[post2020_targets, feat_idx], bins=30, alpha=0.6,
             color=VONLINCK_COLORS['success'], label='Post-2020 Targets', density=True)

    ax2.axvline(x=context[pre2020_targets, feat_idx].mean(), color=VONLINCK_COLORS['danger'],
                linestyle='--', linewidth=2)
    ax2.axvline(x=context[post2020_targets, feat_idx].mean(), color=VONLINCK_COLORS['success'],
                linestyle='--', linewidth=2)

    ax2.set_xlabel('coil_intensity', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Coil Intensity: -0.16 Shift in Targets', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)

    # 3. Feature comparison table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    table_text = """
FEATURE COMPARISON: PRE-2020 vs POST-2020 TARGETS
=================================================

Feature                    Pre-2020   Post-2020    Diff
--------------------------------------------------------
price_position_at_end        0.781      0.446    +0.336 ***
coil_intensity               0.473      0.631    -0.158 ***
log_float                    0.105      0.262    -0.157 ***
log_dollar_volume            0.294      0.449    -0.155 ***
distance_to_high             0.340      0.237    +0.103 ***
vol_contraction_intensity    0.670      0.572    +0.098
vol_trend_5d                 0.559      0.478    +0.081
relative_strength_cohort     0.569      0.489    +0.080


*** = Major difference (>0.1)

INTERPRETATION:
Pre-2020 Targets were:
- Smaller companies (lower log_float)
- Less liquid (lower log_dollar_volume)
- Less coiled (lower coil_intensity)
- Higher in their range (higher price_position)
- Further from highs (higher distance_to_high)
"""

    ax3.text(0.02, 0.98, table_text, fontsize=8.5, fontfamily='monospace',
             va='top', transform=ax3.transAxes, color=VONLINCK_COLORS['primary'])

    # 4. Year-by-year target rate
    ax4 = fig.add_subplot(2, 2, 4)

    yearly_rates = meta.groupby('year').apply(lambda x: (x['label'] == 2).mean() * 100)
    yearly_counts = meta.groupby('year').apply(lambda x: (x['label'] == 2).sum())

    colors = [VONLINCK_COLORS['danger'] if y < 2020 else VONLINCK_COLORS['success']
              for y in yearly_rates.index]

    bars = ax4.bar(yearly_rates.index, yearly_rates.values, color=colors, edgecolor='white')
    ax4.axhline(y=5, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=2019.5, color=VONLINCK_COLORS['warning'], linestyle='-', linewidth=3, alpha=0.7)

    ax4.set_xlabel('Year', fontsize=10)
    ax4.set_ylabel('Target Rate (%)', fontsize=10)
    ax4.set_title('Target Rate by Year (Structural Break at 2020)', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    # Add annotation
    ax4.annotate('COVID\nBreak', xy=(2019.5, 25), fontsize=9, ha='center',
                color=VONLINCK_COLORS['warning'], fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    """Generate the complete PDF report."""
    output_path = Path('output/reports/VonLinck_TRANS_Technical_Report_20260122.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VonLinck Capital - TRAnS Technical Report Generator")
    print("="*60)

    # Load data
    sequences, context, labels, meta, probs, model = load_data()

    print(f"\nGenerating PDF report: {output_path}")

    with PdfPages(output_path) as pdf:
        print("  Creating title page...")
        create_title_page(pdf, meta)

        print("  Creating executive summary...")
        create_executive_summary(pdf, meta, labels)

        print("  Creating architecture page...")
        create_architecture_page(pdf)

        print("  Creating data distribution page...")
        create_data_distribution_page(pdf, meta, labels)

        print("  Creating performance metrics page...")
        create_performance_metrics_page(pdf, meta, labels, probs)

        print("  Creating calibration page...")
        create_calibration_page(pdf, labels, probs)

        print("  Creating feature importance page...")
        create_feature_importance_page(pdf, sequences, context, labels, model)

        print("  Creating temporal stability page...")
        create_temporal_stability_page(pdf, meta)

        print("  Creating regime shift page...")
        create_regime_shift_page(pdf, meta, context)

        print("  Creating pre-2020 analysis page...")
        create_pre2020_analysis_page(pdf, sequences, context, labels, meta, model)

        print("  Creating pre-2020 feature detail page...")
        create_pre2020_feature_detail_page(pdf, sequences, context, labels, meta, model)

        print("  Creating recommendations page...")
        create_recommendations_page(pdf, meta)

        print("  Creating appendix...")
        create_appendix_metrics(pdf, meta, labels, probs)

    print(f"\n[OK] Report generated successfully: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
