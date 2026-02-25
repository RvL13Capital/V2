"""
Simple visualization of consolidation patterns from historical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def analyze_and_visualize_patterns():
    """Load and visualize pattern statistics from historical data"""

    # Load historical patterns
    print("Loading historical pattern data...")
    df = pd.read_parquet('historical_patterns.parquet')

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('AIv3 Consolidation Pattern Analysis - Historical Data Overview',
                 fontsize=16, fontweight='bold')

    # 1. Pattern outcome distribution
    ax1 = plt.subplot(2, 3, 1)
    outcome_counts = df['outcome_class'].value_counts()
    colors = ['red' if 'FAIL' in str(x) or '<5%' in str(x) else
              'yellow' if 'WEAK' in str(x) or '5-10%' in str(x) else
              'lightgreen' if 'MODERATE' in str(x) or '10-20%' in str(x) else
              'green' if 'STRONG' in str(x) or '20-40%' in str(x) else
              'darkgreen' for x in outcome_counts.index]

    bars = ax1.bar(range(len(outcome_counts)), outcome_counts.values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(outcome_counts)))
    ax1.set_xticklabels([str(x).replace(' ', '\n') for x in outcome_counts.index],
                        rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Count')
    ax1.set_title('Pattern Outcome Distribution')
    ax1.grid(True, alpha=0.3)

    # Add percentages on bars
    for i, (bar, count) in enumerate(zip(bars, outcome_counts.values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)

    # 2. Gain distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    gains = df['outcome_max_gain'].dropna()
    ax2.hist(gains, bins=50, color='blue', alpha=0.6, edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Explosive (40%+)')
    ax2.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Strong (20%+)')
    ax2.set_xlabel('Maximum Gain (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Maximum Gains')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Pattern duration analysis
    ax3 = plt.subplot(2, 3, 3)
    duration_col = 'pattern_duration_days' if 'pattern_duration_days' in df.columns else 'duration_days'
    if duration_col in df.columns:
        durations = df[duration_col].dropna()
        ax3.hist(durations, bins=30, color='purple', alpha=0.6, edgecolor='black')
        ax3.axvline(x=durations.mean(), color='red', linestyle='--',
                   label=f'Mean: {durations.mean():.1f} days')
        ax3.set_xlabel('Pattern Duration (days)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pattern Duration Distribution')
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Top performing patterns
    ax4 = plt.subplot(2, 3, 4)
    top_patterns = df.nlargest(15, 'outcome_max_gain')[['ticker', 'outcome_max_gain']]
    y_pos = np.arange(len(top_patterns))
    ax4.barh(y_pos, top_patterns['outcome_max_gain'].values, color='green', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_patterns['ticker'].values, fontsize=8)
    ax4.set_xlabel('Maximum Gain (%)')
    ax4.set_title('Top 15 Performing Patterns')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, v in enumerate(top_patterns['outcome_max_gain'].values):
        ax4.text(v, i, f'{v:.1f}%', va='center', fontsize=8)

    # 5. Pattern quality metrics
    ax5 = plt.subplot(2, 3, 5)
    if 'avg_bbw' in df.columns and 'avg_volume_ratio' in df.columns:
        # Scatter plot of BBW vs Volume Ratio colored by outcome
        explosive_mask = df['outcome_max_gain'] > 40
        strong_mask = (df['outcome_max_gain'] > 20) & (df['outcome_max_gain'] <= 40)
        moderate_mask = (df['outcome_max_gain'] > 10) & (df['outcome_max_gain'] <= 20)
        weak_mask = df['outcome_max_gain'] <= 10

        ax5.scatter(df.loc[weak_mask, 'avg_bbw'], df.loc[weak_mask, 'avg_volume_ratio'],
                   alpha=0.3, s=20, c='gray', label='Weak (<10%)')
        ax5.scatter(df.loc[moderate_mask, 'avg_bbw'], df.loc[moderate_mask, 'avg_volume_ratio'],
                   alpha=0.5, s=30, c='yellow', label='Moderate (10-20%)')
        ax5.scatter(df.loc[strong_mask, 'avg_bbw'], df.loc[strong_mask, 'avg_volume_ratio'],
                   alpha=0.6, s=40, c='orange', label='Strong (20-40%)')
        ax5.scatter(df.loc[explosive_mask, 'avg_bbw'], df.loc[explosive_mask, 'avg_volume_ratio'],
                   alpha=0.8, s=50, c='red', label='Explosive (40%+)', edgecolors='black')

        ax5.set_xlabel('Average BBW')
        ax5.set_ylabel('Average Volume Ratio')
        ax5.set_title('Pattern Quality Metrics')
        ax5.legend(fontsize=8, loc='upper right')
        ax5.grid(True, alpha=0.3)

    # 6. Yearly pattern distribution
    ax6 = plt.subplot(2, 3, 6)
    if 'pattern_year' in df.columns:
        yearly = df.groupby('pattern_year').size()
        ax6.bar(yearly.index, yearly.values, color='navy', alpha=0.7)
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Number of Patterns')
        ax6.set_title('Patterns by Year')
        ax6.grid(True, alpha=0.3)
        # Rotate x labels
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Add summary statistics box
    total_patterns = len(df)
    explosive_count = len(df[df['outcome_max_gain'] > 40])
    strong_count = len(df[(df['outcome_max_gain'] > 20) & (df['outcome_max_gain'] <= 40)])
    avg_gain = df['outcome_max_gain'].mean()
    success_rate = len(df[df['outcome_max_gain'] > 10]) / total_patterns * 100

    summary_text = f"""Summary Statistics:
    Total Patterns: {total_patterns}
    Explosive (40%+): {explosive_count} ({explosive_count/total_patterns*100:.1f}%)
    Strong (20-40%): {strong_count} ({strong_count/total_patterns*100:.1f}%)
    Average Gain: {avg_gain:.1f}%
    Success Rate (10%+): {success_rate:.1f}%"""

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, summary_text, fontsize=10,
            bbox=props, transform=fig.transFigure)

    return fig

def main():
    """Generate and save pattern analysis visualization"""
    try:
        print("Creating pattern analysis visualization...")
        fig = analyze_and_visualize_patterns()

        # Save the figure
        output_file = f'pattern_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as: {output_file}")

        # Show the plot
        plt.show()

        # Also print some key statistics
        df = pd.read_parquet('historical_patterns.parquet')
        print("\n" + "="*60)
        print("KEY PATTERN STATISTICS")
        print("="*60)

        # Best performing patterns
        print("\nTop 10 Best Performing Patterns:")
        top10 = df.nlargest(10, 'outcome_max_gain')[['ticker', 'pattern_start_date',
                                                      'outcome_max_gain', 'outcome_class']]
        for idx, row in top10.iterrows():
            print(f"  {row['ticker']:6s} | {row['pattern_start_date']} | "
                  f"{row['outcome_max_gain']:6.1f}% | {row['outcome_class']}")

        # Pattern characteristics by outcome
        print("\n" + "-"*60)
        print("Average Characteristics by Outcome Class:")
        outcome_stats = df.groupby('outcome_class').agg({
            'outcome_max_gain': 'mean',
            'avg_bbw': 'mean' if 'avg_bbw' in df.columns else lambda x: 0,
            'avg_volume_ratio': 'mean' if 'avg_volume_ratio' in df.columns else lambda x: 0
        }).round(2)
        print(outcome_stats)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()