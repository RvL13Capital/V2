"""
Quick Analysis of Detected Patterns
Analyzes patterns without heavy visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_patterns(pattern_file: str):
    """Quick analysis of patterns"""
    
    print("="*60)
    print("PATTERN ANALYSIS")
    print("="*60)
    
    # Load patterns
    with open(pattern_file, 'r') as f:
        patterns = json.load(f)
    
    print(f"\nLoaded {len(patterns)} patterns from {pattern_file}")
    
    # Convert to DataFrame
    df = pd.DataFrame(patterns)
    
    # Basic statistics
    print("\n" + "-"*40)
    print("BASIC STATISTICS")
    print("-"*40)
    print(f"Total Patterns: {len(df)}")
    print(f"Unique Tickers: {df['ticker'].nunique()}")
    print(f"Date Range: {df['start_date'].min()} to {df['end_date'].max()}")
    
    # Duration analysis
    print("\n" + "-"*40)
    print("DURATION ANALYSIS")
    print("-"*40)
    print(f"Average Duration: {df['duration'].mean():.1f} days")
    print(f"Median Duration: {df['duration'].median():.1f} days")
    print(f"Min Duration: {df['duration'].min()} days")
    print(f"Max Duration: {df['duration'].max()} days")
    
    # Outcome distribution
    print("\n" + "-"*40)
    print("OUTCOME DISTRIBUTION")
    print("-"*40)
    outcome_counts = df['outcome_class'].value_counts()
    outcome_pcts = df['outcome_class'].value_counts(normalize=True) * 100
    
    for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
        count = outcome_counts.get(outcome, 0)
        pct = outcome_pcts.get(outcome, 0)
        
        if outcome == 'K0':
            desc = "Stagnant (<5% gain)"
        elif outcome == 'K1':
            desc = "Minimal (5-15% gain)"
        elif outcome == 'K2':
            desc = "Quality (15-35% gain)"
        elif outcome == 'K3':
            desc = "Strong (35-75% gain)"
        elif outcome == 'K4':
            desc = "Exceptional (>75% gain)"
        else:
            desc = "Failed (breakdown)"
        
        print(f"{outcome} {desc:30} : {count:4} ({pct:5.1f}%)")
    
    # Success metrics
    print("\n" + "-"*40)
    print("SUCCESS METRICS")
    print("-"*40)
    
    successful = df[df['outcome_class'].isin(['K2', 'K3', 'K4'])]
    exceptional = df[df['outcome_class'] == 'K4']
    failed = df[df['outcome_class'] == 'K5']
    
    success_rate = len(successful) / len(df) * 100
    exceptional_rate = len(exceptional) / len(df) * 100
    failure_rate = len(failed) / len(df) * 100
    
    print(f"Success Rate (K2+K3+K4): {success_rate:.1f}%")
    print(f"Exceptional Rate (K4): {exceptional_rate:.1f}%")
    print(f"Failure Rate (K5): {failure_rate:.1f}%")
    
    # Performance metrics
    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
    print("-"*40)
    print(f"Average Max Gain: {df['max_gain'].mean():.2f}%")
    print(f"Median Max Gain: {df['max_gain'].median():.2f}%")
    
    # Positive gains only
    positive_gains = df[df['max_gain'] > 0]['max_gain']
    if len(positive_gains) > 0:
        print(f"Average Positive Gain: {positive_gains.mean():.2f}%")
        print(f"Max Gain Achieved: {positive_gains.max():.2f}%")
    
    # Losses
    losses = df[df['max_gain'] < 0]['max_gain']
    if len(losses) > 0:
        print(f"Average Loss: {losses.mean():.2f}%")
        print(f"Worst Loss: {losses.min():.2f}%")
    
    # Method comparison if multiple methods
    if 'detection_method' in df.columns:
        print("\n" + "-"*40)
        print("DETECTION METHOD COMPARISON")
        print("-"*40)
        
        for method in df['detection_method'].unique():
            method_df = df[df['detection_method'] == method]
            method_successful = method_df[method_df['outcome_class'].isin(['K2', 'K3', 'K4'])]
            method_success_rate = len(method_successful) / len(method_df) * 100 if len(method_df) > 0 else 0
            
            print(f"{method:20} : {len(method_df):4} patterns, {method_success_rate:5.1f}% success rate")
    
    # Top performing tickers
    print("\n" + "-"*40)
    print("TOP PERFORMING TICKERS")
    print("-"*40)
    
    ticker_stats = df.groupby('ticker').agg({
        'outcome_class': 'count',
        'max_gain': 'mean'
    }).rename(columns={'outcome_class': 'pattern_count', 'max_gain': 'avg_gain'})
    
    # Add success rate
    ticker_success = df[df['outcome_class'].isin(['K2', 'K3', 'K4'])].groupby('ticker').size()
    ticker_stats['success_rate'] = (ticker_success / ticker_stats['pattern_count'] * 100).fillna(0)
    
    # Sort by average gain
    ticker_stats = ticker_stats.sort_values('avg_gain', ascending=False)
    
    print(f"{'Ticker':<10} {'Patterns':<10} {'Avg Gain':<12} {'Success Rate':<12}")
    print("-" * 44)
    
    for ticker, row in ticker_stats.head(10).iterrows():
        print(f"{ticker:<10} {row['pattern_count']:<10.0f} {row['avg_gain']:<12.2f}% {row['success_rate']:<12.1f}%")
    
    # Optimal characteristics
    if successful.shape[0] > 0:
        print("\n" + "-"*40)
        print("OPTIMAL PATTERN CHARACTERISTICS")
        print("-"*40)
        print("(Based on successful patterns K2+K3+K4)")
        
        print(f"Average Duration: {successful['duration'].mean():.1f} days")
        print(f"Average Boundary Width: {successful['boundary_width_pct'].mean():.2f}%")
        
        # Extract metrics if available
        if 'qualification_metrics' in successful.columns:
            try:
                # Extract avg_range from qualification_metrics
                avg_ranges = []
                for metrics in successful['qualification_metrics']:
                    if isinstance(metrics, dict) and 'avg_range' in metrics:
                        avg_ranges.append(metrics['avg_range'])
                
                if avg_ranges:
                    print(f"Average Range: {np.mean(avg_ranges):.2f}%")
            except:
                pass
    
    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"analysis_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("PATTERN ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Total Patterns: {len(df)}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Exceptional Rate: {exceptional_rate:.1f}%\n")
        f.write(f"Failure Rate: {failure_rate:.1f}%\n")
        f.write(f"Average Max Gain: {df['max_gain'].mean():.2f}%\n")
        f.write("\nOutcome Distribution:\n")
        for outcome, count in outcome_counts.items():
            f.write(f"  {outcome}: {count}\n")
    
    print(f"\n" + "="*60)
    print(f"Summary saved to: {summary_file}")
    print("="*60)
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick pattern analysis')
    parser.add_argument('--patterns-file', type=str, 
                       default='detected_patterns_20250910_185652.json',
                       help='Pattern file to analyze')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.patterns_file).exists():
        print(f"Error: Pattern file not found: {args.patterns_file}")
        print("\nAvailable pattern files:")
        for f in Path('.').glob('*patterns*.json'):
            print(f"  - {f}")
        return
    
    # Analyze patterns
    df = analyze_patterns(args.patterns_file)


if __name__ == "__main__":
    main()