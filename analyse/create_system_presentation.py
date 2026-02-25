"""
Create visual presentation and analysis to explain the consolidation pattern system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SystemPresentation:
    """Generate presentation materials explaining the consolidation pattern system"""
    
    def __init__(self):
        self.output_dir = Path('presentation_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def create_concept_visualization(self):
        """Create visualization explaining the consolidation pattern concept"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Consolidation Pattern Detection System - Concept Overview', fontsize=16, fontweight='bold')
        
        # 1. Consolidation Pattern Example
        ax1 = axes[0, 0]
        days = np.arange(0, 150)
        
        # Create synthetic price data showing consolidation and breakout
        price = np.ones(150) * 10
        # Pre-consolidation volatility
        price[:50] += np.sin(days[:50] * 0.3) * 2 + np.random.normal(0, 0.5, 50)
        # Consolidation phase (tight range)
        price[50:100] = 10 + np.random.normal(0, 0.1, 50)
        # Breakout phase
        price[100:120] = 10 + (days[100:120] - 100) * 0.3 + np.random.normal(0, 0.2, 20)
        # Explosive move
        price[120:] = price[119] + (days[120:] - 120) * 0.5 + np.random.normal(0, 0.3, 30)
        
        ax1.plot(days, price, linewidth=2, color='darkblue')
        
        # Mark consolidation zone
        consolidation_rect = patches.Rectangle((50, 9.5), 50, 1, 
                                              linewidth=2, edgecolor='orange', 
                                              facecolor='orange', alpha=0.2)
        ax1.add_patch(consolidation_rect)
        
        # Mark breakout
        ax1.axvline(x=100, color='green', linestyle='--', alpha=0.7, label='Breakout')
        ax1.axhline(y=10.5, color='red', linestyle=':', alpha=0.5, label='Upper Boundary')
        ax1.axhline(y=9.5, color='red', linestyle=':', alpha=0.5, label='Lower Boundary')
        
        # Annotations
        ax1.annotate('Volatility', xy=(25, 12), fontsize=10, ha='center')
        ax1.annotate('CONSOLIDATION\n(10-20 days)', xy=(75, 8.5), fontsize=10, 
                    ha='center', color='orange', fontweight='bold')
        ax1.annotate('Breakout', xy=(100, 11), fontsize=10, ha='center', color='green')
        ax1.annotate('Explosive Move\n(Target: 75%+)', xy=(135, 20), fontsize=10, 
                    ha='center', color='darkgreen', fontweight='bold')
        
        ax1.set_title('1. Pattern Lifecycle: Consolidation → Breakout → Explosive Move', fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Qualification Criteria
        ax2 = axes[0, 1]
        criteria = ['BBW < 30%ile', 'ADX < 32', 'Volume < 35%', 'Range < 65%']
        values = [28, 25, 30, 60]
        thresholds = [30, 32, 35, 65]
        
        x = np.arange(len(criteria))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, values, width, label='Current Value', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, thresholds, width, label='Threshold', color='red', alpha=0.5)
        
        ax2.set_title('2. Qualification Criteria (Must Meet All)', fontweight='bold')
        ax2.set_xlabel('Criteria')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(criteria, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add pass/fail indicators
        for i, (v, t) in enumerate(zip(values, thresholds)):
            if v < t:
                ax2.text(i, max(v, t) + 2, '✓', fontsize=20, ha='center', color='green')
            else:
                ax2.text(i, max(v, t) + 2, '✗', fontsize=20, ha='center', color='red')
        
        # 3. Strategic Value System
        ax3 = axes[1, 0]
        outcomes = ['K5\nFailed', 'K0\nStagnant', 'K1\nMinimal', 'K2\nQuality', 'K3\nStrong', 'K4\nExceptional']
        values = [-10, -2, -0.2, 1, 3, 10]
        gains = ['Breakdown', '<5%', '5-15%', '15-35%', '35-75%', '>75%']
        
        colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
        bars = ax3.bar(outcomes, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, val, gain in zip(bars, values, gains):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{val:+.1f}\n({gain})', ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('3. Strategic Value System - Risk/Reward Balance', fontweight='bold')
        ax3.set_xlabel('Outcome Class')
        ax3.set_ylabel('Strategic Value')
        ax3.set_ylim(-12, 12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Expected Value Calculation
        ax4 = axes[1, 1]
        
        # Sample distribution
        sample_dist = [0.15, 0.20, 0.15, 0.25, 0.15, 0.10]  # Probabilities for K0-K5
        ev_components = [p * v for p, v in zip(sample_dist, values)]
        expected_value = sum(ev_components)
        
        # Create stacked bar showing EV components
        x_pos = np.arange(len(outcomes))
        bottom = np.zeros(len(outcomes))
        
        for i, (comp, color) in enumerate(zip(ev_components, colors)):
            ax4.bar(i, comp, color=color, alpha=0.7, edgecolor='black')
            if comp != 0:
                ax4.text(i, comp/2, f'{comp:+.2f}', ha='center', va='center', 
                        fontweight='bold', fontsize=9, color='white' if abs(comp) > 0.5 else 'black')
        
        # Show total EV
        ax4.axhline(y=expected_value, color='blue', linestyle='--', linewidth=2, label=f'EV = {expected_value:+.2f}')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add probability labels
        for i, prob in enumerate(sample_dist):
            ax4.text(i, -12, f'{prob:.0%}', ha='center', va='top', fontsize=9)
        
        ax4.set_title('4. Expected Value (EV) Calculation\nEV = Σ(Probability × Strategic Value)', fontweight='bold')
        ax4.set_xlabel('Outcome Class (with probability)')
        ax4.set_ylabel('Contribution to EV')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(outcomes)
        ax4.set_ylim(-12, 5)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'system_concept.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Concept visualization saved to: {output_path}")
        
    def create_performance_analysis(self, patterns_file=None):
        """Create performance analysis from actual patterns"""
        
        # Load patterns if available
        if patterns_file and Path(patterns_file).exists():
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
            df = pd.DataFrame(patterns)
        else:
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_patterns = 1000
            
            # Realistic distribution based on market behavior
            outcome_probs = [0.25, 0.30, 0.20, 0.15, 0.08, 0.02]  # K5, K0, K1, K2, K3, K4
            outcomes = np.random.choice(['K5', 'K0', 'K1', 'K2', 'K3', 'K4'], 
                                      size=n_patterns, p=outcome_probs)
            
            # Generate corresponding gains
            gains = []
            for outcome in outcomes:
                if outcome == 'K5':
                    gains.append(np.random.uniform(-20, -5))
                elif outcome == 'K0':
                    gains.append(np.random.uniform(0, 5))
                elif outcome == 'K1':
                    gains.append(np.random.uniform(5, 15))
                elif outcome == 'K2':
                    gains.append(np.random.uniform(15, 35))
                elif outcome == 'K3':
                    gains.append(np.random.uniform(35, 75))
                else:  # K4
                    gains.append(np.random.uniform(75, 150))
            
            df = pd.DataFrame({
                'outcome_class': outcomes,
                'max_gain': gains,
                'duration': np.random.randint(10, 50, n_patterns),
                'days_to_max': np.random.randint(5, 100, n_patterns)
            })
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Consolidation Pattern Analysis - Performance Insights', fontsize=16, fontweight='bold')
        
        # 1. Outcome Distribution
        ax1 = axes[0, 0]
        outcome_counts = df['outcome_class'].value_counts().sort_index()
        colors_map = {'K5': 'darkred', 'K0': 'red', 'K1': 'orange', 
                     'K2': 'yellow', 'K3': 'lightgreen', 'K4': 'darkgreen'}
        colors = [colors_map.get(x, 'gray') for x in outcome_counts.index]
        
        wedges, texts, autotexts = ax1.pie(outcome_counts.values, labels=outcome_counts.index, 
                                           colors=colors, autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('1. Pattern Outcome Distribution', fontweight='bold')
        
        # 2. Gain Distribution by Outcome
        ax2 = axes[0, 1]
        data_to_plot = [df[df['outcome_class'] == oc]['max_gain'].values 
                       for oc in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']]
        bp = ax2.boxplot(data_to_plot, labels=['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                         patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen', 'darkred']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axhline(y=75, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (75%)')
        ax2.set_title('2. Gain Distribution by Outcome Class', fontweight='bold')
        ax2.set_xlabel('Outcome Class')
        ax2.set_ylabel('Max Gain (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time to Max Gain
        ax3 = axes[0, 2]
        if 'days_to_max' in df.columns:
            for outcome in ['K2', 'K3', 'K4']:
                if outcome in df['outcome_class'].values:
                    data = df[df['outcome_class'] == outcome]['days_to_max'].dropna()
                    if len(data) > 0:
                        ax3.hist(data, alpha=0.5, label=outcome, bins=20)
            
            ax3.set_title('3. Days to Reach Maximum Gain', fontweight='bold')
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Strategic Value Analysis
        ax4 = axes[1, 0]
        value_map = {'K4': 10, 'K3': 3, 'K2': 1, 'K1': -0.2, 'K0': -2, 'K5': -10}
        df['strategic_value'] = df['outcome_class'].map(value_map)
        
        total_value = df['strategic_value'].sum()
        avg_value = df['strategic_value'].mean()
        
        value_by_class = df.groupby('outcome_class')['strategic_value'].sum().sort_index()
        colors = [colors_map.get(x, 'gray') for x in value_by_class.index]
        
        bars = ax4.bar(value_by_class.index, value_by_class.values, color=colors, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_title(f'4. Total Strategic Value by Class\nTotal: {total_value:.1f}, Avg: {avg_value:.2f}', 
                     fontweight='bold')
        ax4.set_xlabel('Outcome Class')
        ax4.set_ylabel('Total Strategic Value')
        ax4.grid(True, alpha=0.3)
        
        # 5. Success Rate Over Time (simulated)
        ax5 = axes[1, 1]
        time_periods = pd.date_range(start='2020-01-01', periods=48, freq='M')
        success_rates = np.random.normal(0.25, 0.05, 48)  # 25% average with 5% std
        success_rates = np.clip(success_rates, 0, 1)
        
        ax5.plot(time_periods, success_rates * 100, linewidth=2, color='blue')
        ax5.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Target (25%)')
        ax5.fill_between(time_periods, 0, success_rates * 100, alpha=0.3)
        
        ax5.set_title('5. High-Value Pattern Success Rate (K3+K4)', fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Success Rate (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Risk-Reward Profile
        ax6 = axes[1, 2]
        
        # Calculate metrics
        positive_patterns = df[df['max_gain'] > 0]
        negative_patterns = df[df['max_gain'] < 0]
        
        metrics = {
            'Win Rate': len(positive_patterns) / len(df) * 100,
            'Avg Win': positive_patterns['max_gain'].mean() if len(positive_patterns) > 0 else 0,
            'Avg Loss': abs(negative_patterns['max_gain'].mean()) if len(negative_patterns) > 0 else 0,
            'Risk/Reward': positive_patterns['max_gain'].mean() / abs(negative_patterns['max_gain'].mean()) 
                          if len(negative_patterns) > 0 else 0
        }
        
        # Create bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax6.bar(range(len(metrics)), metric_values, 
                      color=['blue', 'green', 'red', 'purple'], alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}{"%" if "Rate" in metric_names[bars.index(bar)] else ""}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax6.set_title('6. Risk-Reward Metrics', fontweight='bold')
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(metric_names, rotation=45, ha='right')
        ax6.set_ylabel('Value')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'performance_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved to: {output_path}")
        
        return df
    
    def create_insight_report(self, df=None):
        """Generate text report with key insights"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CONSOLIDATION PATTERN DETECTION SYSTEM - INSIGHTS REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # System Overview
        report_lines.append("SYSTEM OVERVIEW")
        report_lines.append("-"*40)
        report_lines.append("The consolidation pattern detection system identifies stocks in tight trading")
        report_lines.append("ranges (consolidation) that historically precede explosive price movements.")
        report_lines.append("")
        report_lines.append("Key Concept: Low volatility periods often precede high volatility breakouts.")
        report_lines.append("Target: Identify patterns leading to 75%+ gains (K4 - Exceptional moves)")
        report_lines.append("")
        
        # Strategic Approach
        report_lines.append("STRATEGIC VALUE SYSTEM")
        report_lines.append("-"*40)
        report_lines.append("The system uses a risk-weighted value system:")
        report_lines.append("  - K4 (>75% gain):     +10 points  [PRIMARY TARGET]")
        report_lines.append("  - K3 (35-75% gain):    +3 points  [SECONDARY TARGET]")
        report_lines.append("  - K2 (15-35% gain):    +1 point   [ACCEPTABLE]")
        report_lines.append("  - K1 (5-15% gain):    -0.2 points [NOT WORTH RISK]")
        report_lines.append("  - K0 (<5% gain):       -2 points  [AVOID]")
        report_lines.append("  - K5 (Breakdown):     -10 points  [STRONG AVOID]")
        report_lines.append("")
        
        if df is not None and not df.empty:
            # Data Insights
            report_lines.append("DATA INSIGHTS")
            report_lines.append("-"*40)
            
            total_patterns = len(df)
            report_lines.append(f"Total Patterns Analyzed: {total_patterns:,}")
            
            # Outcome distribution
            outcome_dist = df['outcome_class'].value_counts(normalize=True).sort_index()
            report_lines.append("")
            report_lines.append("Pattern Outcome Distribution:")
            for outcome, pct in outcome_dist.items():
                report_lines.append(f"  {outcome}: {pct*100:5.1f}% ({int(pct*total_patterns):,} patterns)")
            
            # Strategic value analysis
            if 'strategic_value' in df.columns:
                total_value = df['strategic_value'].sum()
                avg_value = df['strategic_value'].mean()
                expected_value = avg_value  # Simplified EV
                
                report_lines.append("")
                report_lines.append(f"Total Strategic Value: {total_value:+.1f}")
                report_lines.append(f"Average Strategic Value: {avg_value:+.2f}")
                report_lines.append(f"Expected Value (EV): {expected_value:+.3f}")
                
                if expected_value > 0:
                    report_lines.append("  -> POSITIVE EV: System shows profitable edge")
                else:
                    report_lines.append("  -> NEGATIVE EV: System needs refinement")
            
            # Success metrics
            high_value = df[df['outcome_class'].isin(['K3', 'K4'])]
            success_rate = len(high_value) / len(df) * 100
            
            report_lines.append("")
            report_lines.append(f"High-Value Pattern Rate (K3+K4): {success_rate:.1f}%")
            
            if success_rate > 15:
                report_lines.append("  -> GOOD: Above 15% threshold for profitability")
            else:
                report_lines.append("  -> NEEDS IMPROVEMENT: Below 15% threshold")
            
            # Risk analysis
            failed = df[df['outcome_class'] == 'K5']
            failure_rate = len(failed) / len(df) * 100
            
            report_lines.append(f"Failure Rate (K5): {failure_rate:.1f}%")
            if failure_rate < 30:
                report_lines.append("  -> ACCEPTABLE: Below 30% risk threshold")
            else:
                report_lines.append("  -> HIGH RISK: Above 30% threshold")
        
        # Key Recommendations
        report_lines.append("")
        report_lines.append("KEY RECOMMENDATIONS")
        report_lines.append("-"*40)
        report_lines.append("1. Focus on patterns with positive Expected Value (EV > 1.0)")
        report_lines.append("2. Prioritize K3 and K4 outcomes (35%+ gains)")
        report_lines.append("3. Avoid patterns with high K5 probability (>30%)")
        report_lines.append("4. Use $0.01 minimum price filter to exclude penny stocks")
        report_lines.append("5. Monitor for 10-20 day consolidation periods")
        report_lines.append("6. Look for volume contraction during consolidation")
        report_lines.append("")
        
        # Trading Strategy
        report_lines.append("SUGGESTED TRADING APPROACH")
        report_lines.append("-"*40)
        report_lines.append("1. ENTRY: When pattern breaks above upper boundary")
        report_lines.append("2. STOP LOSS: 5% below lower boundary (K5 threshold)")
        report_lines.append("3. TARGET 1: 15% gain (K2 level)")
        report_lines.append("4. TARGET 2: 35% gain (K3 level)")
        report_lines.append("5. TARGET 3: 75% gain (K4 level)")
        report_lines.append("6. POSITION SIZE: Based on EV and confidence level")
        report_lines.append("")
        
        # Risk Warning
        report_lines.append("RISK DISCLAIMER")
        report_lines.append("-"*40)
        report_lines.append("This system is for pattern detection and analysis only.")
        report_lines.append("Past performance does not guarantee future results.")
        report_lines.append("Always use proper risk management and position sizing.")
        report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # Save report
        report_path = self.output_dir / 'insights_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))
        print(f"\nReport saved to: {report_path}")
        
        return report_lines

def main():
    """Generate complete presentation"""
    
    print("="*80)
    print("GENERATING SYSTEM PRESENTATION AND ANALYSIS")
    print("="*80)
    
    presenter = SystemPresentation()
    
    # 1. Create concept visualization
    print("\n1. Creating concept visualization...")
    presenter.create_concept_visualization()
    
    # 2. Create performance analysis
    print("\n2. Creating performance analysis...")
    df = presenter.create_performance_analysis()
    
    # 3. Generate insights report
    print("\n3. Generating insights report...")
    presenter.create_insight_report(df)
    
    print("\n" + "="*80)
    print("PRESENTATION COMPLETE")
    print("="*80)
    print(f"Output directory: {presenter.output_dir}")
    print("\nGenerated files:")
    print("  - system_concept.png: Visual explanation of the system")
    print("  - performance_analysis.png: Performance metrics and analysis")
    print("  - insights_report.txt: Detailed insights and recommendations")

if __name__ == "__main__":
    main()