"""
Consolidation Pattern Visualization
Shows the typical consolidation pattern structure with all key components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_consolidation_pattern_data():
    """Generate sample data showing a consolidation pattern"""
    np.random.seed(42)

    # Create timeline
    days = 150
    dates = pd.date_range(start='2024-06-01', periods=days, freq='D')

    # Phase 1: Pre-consolidation trend (days 0-30)
    pre_trend = np.linspace(50, 65, 30) + np.random.normal(0, 1.5, 30)

    # Phase 2: Qualification period (days 31-40) - volatility starts contracting
    qualification = np.linspace(65, 63, 10) + np.random.normal(0, 0.8, 10)

    # Phase 3: Active consolidation (days 41-90) - tight range
    consolidation_base = 63
    consolidation = consolidation_base + np.random.normal(0, 0.5, 50)
    # Keep within boundaries
    upper_boundary = 64.5
    lower_boundary = 61.5
    consolidation = np.clip(consolidation, lower_boundary, upper_boundary)

    # Phase 4: Breakout (days 91-100)
    breakout_start = consolidation[-1]
    breakout = np.linspace(breakout_start, 72, 10) + np.random.normal(0, 0.3, 10)

    # Phase 5: Post-breakout move (days 101-150)
    post_breakout = np.linspace(72, 85, 50) + np.random.normal(0, 2, 50)

    # Combine all phases
    prices = np.concatenate([pre_trend, qualification, consolidation, breakout, post_breakout])

    # Create volume data (lower during consolidation)
    volume = np.ones(days) * 1000000
    volume[31:41] = np.linspace(1000000, 400000, 10)  # Declining in qualification
    volume[41:91] = 300000 + np.random.normal(0, 50000, 50)  # Low during consolidation
    volume[91:101] = np.linspace(400000, 1500000, 10)  # Spike on breakout
    volume[101:] = 1200000 + np.random.normal(0, 200000, len(volume[101:]))  # Post-breakout
    volume = np.abs(volume)

    # Calculate Bollinger Bands
    window = 20
    rolling_mean = pd.Series(prices).rolling(window).mean()
    rolling_std = pd.Series(prices).rolling(window).std()
    upper_bb = rolling_mean + 2 * rolling_std
    lower_bb = rolling_mean - 2 * rolling_std

    # Calculate BBW (Bollinger Band Width)
    bbw = ((upper_bb - lower_bb) / rolling_mean * 100).fillna(0)

    return {
        'dates': dates,
        'prices': prices,
        'volume': volume,
        'upper_bb': upper_bb.values,
        'lower_bb': lower_bb.values,
        'bbw': bbw.values,
        'upper_boundary': upper_boundary,
        'lower_boundary': lower_boundary,
        'power_boundary': upper_boundary * 1.005  # 0.5% above upper
    }

def plot_consolidation_pattern():
    """Create comprehensive visualization of consolidation pattern"""
    data = create_consolidation_pattern_data()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('AIv3 Consolidation Pattern Detection - Complete Visualization',
                 fontsize=16, fontweight='bold')

    # Main price chart
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data['dates'], data['prices'], 'b-', linewidth=1.5, label='Price', alpha=0.8)

    # Add Bollinger Bands
    ax1.plot(data['dates'], data['upper_bb'], 'g--', alpha=0.3, label='Upper BB')
    ax1.plot(data['dates'], data['lower_bb'], 'g--', alpha=0.3, label='Lower BB')
    ax1.fill_between(data['dates'], data['upper_bb'], data['lower_bb'],
                     alpha=0.1, color='green')

    # Mark consolidation phases
    # Qualification phase (yellow)
    qualification_rect = patches.Rectangle((data['dates'][31], 50),
                                          timedelta(days=10), 40,
                                          linewidth=0, edgecolor='none',
                                          facecolor='yellow', alpha=0.2)
    ax1.add_patch(qualification_rect)
    ax1.text(data['dates'][36], 88, 'QUALIFICATION\n(10 days)',
             ha='center', fontsize=9, fontweight='bold')

    # Active consolidation phase (green)
    consolidation_rect = patches.Rectangle((data['dates'][41], 50),
                                          timedelta(days=50), 40,
                                          linewidth=0, edgecolor='none',
                                          facecolor='green', alpha=0.2)
    ax1.add_patch(consolidation_rect)
    ax1.text(data['dates'][66], 88, 'ACTIVE CONSOLIDATION\n(50 days)',
             ha='center', fontsize=9, fontweight='bold')

    # Breakout phase (red)
    breakout_rect = patches.Rectangle((data['dates'][91], 50),
                                     timedelta(days=10), 40,
                                     linewidth=0, edgecolor='none',
                                     facecolor='red', alpha=0.2)
    ax1.add_patch(breakout_rect)
    ax1.text(data['dates'][96], 88, 'BREAKOUT',
             ha='center', fontsize=9, fontweight='bold')

    # Draw boundary lines during consolidation
    ax1.axhline(y=data['upper_boundary'], xmin=0.27, xmax=0.61,
                color='red', linestyle='-', linewidth=2, label='Upper Boundary')
    ax1.axhline(y=data['lower_boundary'], xmin=0.27, xmax=0.61,
                color='blue', linestyle='-', linewidth=2, label='Lower Boundary')
    ax1.axhline(y=data['power_boundary'], xmin=0.27, xmax=0.61,
                color='purple', linestyle=':', linewidth=1.5, label='Power Boundary (Upper + 0.5%)')

    # Add annotations
    ax1.annotate('Pattern Start', xy=(data['dates'][41], data['prices'][41]),
                xytext=(data['dates'][35], 70),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax1.annotate('Breakout Point', xy=(data['dates'][91], data['prices'][91]),
                xytext=(data['dates'][85], 75),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax1.annotate(f"Target: {data['prices'][91]*1.4:.1f}\n(+40% from breakout)",
                xy=(data['dates'][120], data['prices'][91]*1.4),
                xytext=(data['dates'][110], 82),
                arrowprops=dict(arrowstyle='->', color='green', lw=1))

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Price Action with Consolidation Pattern Phases', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(data['dates'][0], data['dates'][-1])

    # Volume chart
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    colors = ['red' if i >= 91 and i <= 100 else 'gray' for i in range(len(data['volume']))]
    ax2.bar(data['dates'], data['volume'], color=colors, alpha=0.6)
    ax2.axhline(y=np.mean(data['volume'][:30]), color='blue',
                linestyle='--', alpha=0.5, label='Pre-consolidation avg volume')
    ax2.axhline(y=np.mean(data['volume'][41:91]), color='red',
                linestyle='--', alpha=0.5, label='Consolidation avg volume')

    ax2.set_ylabel('Volume', fontsize=11)
    ax2.set_title('Volume Analysis - Low Volume During Consolidation', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # BBW chart
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(data['dates'], data['bbw'], 'purple', linewidth=1.5, label='BBW')
    ax3.axhline(y=np.percentile(data['bbw'][20:], 30), color='red',
                linestyle='--', alpha=0.5, label='30th percentile (qualification threshold)')
    ax3.fill_between(data['dates'], 0, data['bbw'],
                     where=(data['bbw'] < np.percentile(data['bbw'][20:], 30)),
                     color='green', alpha=0.2, label='Qualified for pattern')

    # Mark phases on BBW
    ax3.axvspan(data['dates'][31], data['dates'][41], alpha=0.2, color='yellow')
    ax3.axvspan(data['dates'][41], data['dates'][91], alpha=0.2, color='green')
    ax3.axvspan(data['dates'][91], data['dates'][101], alpha=0.2, color='red')

    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('BBW (%)', fontsize=11)
    ax3.set_title('Bollinger Band Width - Volatility Contraction Indicator', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add text box with pattern criteria
    textstr = '''PATTERN QUALIFICATION CRITERIA:
    • BBW < 30th percentile
    • ADX < 32 (low trending)
    • Volume < 35% of 20-day avg
    • Daily range < 65% of 20-day avg
    • Minimum 10 days to qualify'''

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.15, 0.95, textstr, fontsize=9, verticalalignment='top',
            bbox=props, transform=fig.transFigure)

    # Add outcome classification box
    outcome_str = '''OUTCOME CLASSES:
    K4: >75% gain (Value: +10)
    K3: 35-75% gain (Value: +3)
    K2: 15-35% gain (Value: +1)
    K1: 5-15% gain (Value: -0.2)
    K0: <5% gain (Value: -2)
    K5: Breakdown (Value: -10)'''

    props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    fig.text(0.82, 0.95, outcome_str, fontsize=9, verticalalignment='top',
            bbox=props2, transform=fig.transFigure)

    return fig

def main():
    """Generate and save the consolidation pattern visualization"""
    print("Generating consolidation pattern visualization...")

    fig = plot_consolidation_pattern()

    # Save the figure
    output_file = 'consolidation_pattern_visualization.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")

    # Also show the plot
    plt.show()

    return output_file

if __name__ == "__main__":
    main()