"""
Analyze real patterns from GCS data and generate insights
"""

import sys
import os
from datetime import datetime
from run_complete_analysis_filtered import FilteredAnalysisPipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    """Analyze real patterns from GCS"""
    
    print("="*80)
    print("REAL DATA ANALYSIS - INSIGHTS GENERATION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create pipeline
    pipeline = FilteredAnalysisPipeline()
    
    # Process limited set for quick insights (100 tickers)
    ticker_files = pipeline.get_all_ticker_files()[:100]
    
    if not ticker_files:
        print("No ticker files found")
        return False
    
    print(f"\nAnalyzing {len(ticker_files)} tickers for insights...")
    
    # Process tickers
    pipeline.process_all_tickers_parallel(ticker_files, max_workers=10)
    
    if not pipeline.all_patterns:
        print("No patterns found")
        return False
    
    # Create DataFrame for analysis
    df = pd.DataFrame(pipeline.all_patterns)
    
    print(f"\nPatterns found: {len(df)}")
    
    # Generate insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM REAL DATA")
    print("="*80)
    
    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS")
    print("-"*40)
    print(f"Total Patterns: {len(df):,}")
    print(f"Unique Tickers: {df['ticker'].nunique()}")
    print(f"Date Range: {df['start_date'].min()} to {df['end_date'].max()}")
    print(f"Average Pattern Duration: {df['duration'].mean():.1f} days")
    
    # 2. Outcome Distribution
    print("\n2. OUTCOME DISTRIBUTION")
    print("-"*40)
    outcome_dist = df['outcome_class'].value_counts().sort_index()
    total = len(df)
    
    for outcome, count in outcome_dist.items():
        pct = count / total * 100
        value = df[df['outcome_class'] == outcome]['strategic_value'].iloc[0]
        print(f"{outcome}: {count:4d} patterns ({pct:5.1f}%) | Value: {value:+6.1f}")
    
    # 3. Strategic Value Analysis
    print("\n3. STRATEGIC VALUE METRICS")
    print("-"*40)
    total_value = df['strategic_value'].sum()
    avg_value = df['strategic_value'].mean()
    
    print(f"Total Strategic Value: {total_value:+.1f}")
    print(f"Average Value per Pattern: {avg_value:+.3f}")
    
    # Calculate Expected Value
    ev = 0
    value_map = {'K4': 10, 'K3': 3, 'K2': 1, 'K1': -0.2, 'K0': -2, 'K5': -10}
    for outcome, value in value_map.items():
        prob = len(df[df['outcome_class'] == outcome]) / len(df) if len(df) > 0 else 0
        ev += prob * value
    
    print(f"Expected Value (EV): {ev:+.3f}")
    
    if ev > 0:
        print("  -> POSITIVE EV: Profitable edge detected!")
    elif ev > -1:
        print("  -> NEUTRAL EV: Marginal profitability")
    else:
        print("  -> NEGATIVE EV: Needs refinement")
    
    # 4. Success Analysis
    print("\n4. SUCCESS METRICS")
    print("-"*40)
    
    # High-value patterns
    high_value = df[df['outcome_class'].isin(['K3', 'K4'])]
    exceptional = df[df['outcome_class'] == 'K4']
    
    success_rate = len(high_value) / len(df) * 100
    exceptional_rate = len(exceptional) / len(df) * 100
    
    print(f"High-Value Rate (K3+K4): {success_rate:.1f}%")
    print(f"Exceptional Rate (K4): {exceptional_rate:.1f}%")
    
    if len(high_value) > 0:
        print(f"Avg Gain for High-Value: {high_value['max_gain'].mean():.1f}%")
    
    # 5. Risk Analysis
    print("\n5. RISK METRICS")
    print("-"*40)
    
    failed = df[df['outcome_class'] == 'K5']
    failure_rate = len(failed) / len(df) * 100
    
    print(f"Failure Rate (K5): {failure_rate:.1f}%")
    
    if len(failed) > 0:
        print(f"Avg Loss on Failure: {failed['max_loss'].mean() if 'max_loss' in failed.columns else 'N/A':.1f}%")
    
    # Win/Loss ratio
    winners = df[df['max_gain'] > 0]
    losers = df[df['max_gain'] <= 0]
    
    win_rate = len(winners) / len(df) * 100
    print(f"Win Rate: {win_rate:.1f}%")
    
    if len(winners) > 0 and len(losers) > 0:
        avg_win = winners['max_gain'].mean()
        avg_loss = abs(losers['max_gain'].mean())
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"Risk/Reward Ratio: {risk_reward:.2f}")
    
    # 6. Time Analysis
    print("\n6. TIME TO TARGET ANALYSIS")
    print("-"*40)
    
    # Check for days_to_target data
    target_cols = ['days_to_5pct', 'days_to_10pct', 'days_to_15pct', 
                   'days_to_25pct', 'days_to_50pct', 'days_to_max']
    
    for col in target_cols:
        if col in df.columns:
            valid_data = df[df[col].notna()][col]
            if len(valid_data) > 0:
                target = col.replace('days_to_', '').replace('pct', '%')
                print(f"{target:8s}: {len(valid_data):3d} reached, avg {valid_data.mean():.1f} days")
    
    # 7. Top Performing Tickers
    print("\n7. TOP PERFORMING TICKERS")
    print("-"*40)
    
    ticker_value = df.groupby('ticker')['strategic_value'].sum().sort_values(ascending=False)
    top_tickers = ticker_value.head(10)
    
    print("Top 10 Tickers by Strategic Value:")
    for ticker, value in top_tickers.items():
        pattern_count = len(df[df['ticker'] == ticker])
        print(f"  {ticker:6s}: Value {value:+6.1f} ({pattern_count} patterns)")
    
    # 8. Price Filter Impact
    print("\n8. PRICE FILTER ANALYSIS")
    print("-"*40)
    
    if 'min_price_in_pattern' in df.columns:
        avg_min_price = df['min_price_in_pattern'].mean()
        print(f"Avg Min Price in Patterns: ${avg_min_price:.3f}")
        
        # Distribution above certain thresholds
        above_1 = len(df[df['min_price_in_pattern'] >= 1.0]) / len(df) * 100
        above_5 = len(df[df['min_price_in_pattern'] >= 5.0]) / len(df) * 100
        
        print(f"Patterns with min price >= $1.00: {above_1:.1f}%")
        print(f"Patterns with min price >= $5.00: {above_5:.1f}%")
    
    # 9. Pattern Quality
    print("\n9. PATTERN QUALITY INDICATORS")
    print("-"*40)
    
    if 'boundary_width_pct' in df.columns:
        print(f"Avg Boundary Width: {df['boundary_width_pct'].mean():.2f}%")
    
    if 'volume_contraction' in df.columns:
        print(f"Avg Volume Contraction: {df['volume_contraction'].mean():.2f}")
    
    # Consolidation duration by outcome
    for outcome in ['K4', 'K3', 'K2']:
        outcome_data = df[df['outcome_class'] == outcome]
        if len(outcome_data) > 0:
            avg_duration = outcome_data['duration'].mean()
            print(f"{outcome} Avg Duration: {avg_duration:.1f} days")
    
    # 10. Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS BASED ON REAL DATA")
    print("="*80)
    
    if ev > 0:
        print("✓ System shows positive expected value - PROFITABLE")
    else:
        print("⚠ System shows negative expected value - NEEDS OPTIMIZATION")
    
    if success_rate > 15:
        print("✓ High-value pattern rate is good (>15%)")
    else:
        print("⚠ High-value pattern rate needs improvement (<15%)")
    
    if failure_rate < 30:
        print("✓ Failure rate is acceptable (<30%)")
    else:
        print("⚠ Failure rate is high (>30%) - adjust criteria")
    
    # Specific recommendations
    print("\nSPECIFIC ACTIONS:")
    
    if failure_rate > 30:
        print("1. Tighten consolidation criteria to reduce K5 patterns")
    
    if exceptional_rate < 2:
        print("2. Focus on identifying characteristics of K4 patterns")
    
    if 'volume_contraction' in df.columns:
        successful = df[df['outcome_class'].isin(['K3', 'K4'])]
        failed = df[df['outcome_class'] == 'K5']
        
        if len(successful) > 0 and len(failed) > 0:
            success_vol = successful['volume_contraction'].mean()
            fail_vol = failed['volume_contraction'].mean()
            
            if success_vol < fail_vol:
                print("3. Successful patterns show stronger volume contraction")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    output_dir = Path('real_data_insights')
    output_dir.mkdir(exist_ok=True)
    
    # Save pattern data
    df.to_csv(output_dir / 'analyzed_patterns.csv', index=False)
    print(f"\nPattern data saved to: {output_dir / 'analyzed_patterns.csv'}")
    
    # Create summary JSON
    import json
    summary = {
        'total_patterns': len(df),
        'unique_tickers': int(df['ticker'].nunique()),
        'expected_value': float(ev),
        'success_rate': float(success_rate),
        'failure_rate': float(failure_rate),
        'outcome_distribution': outcome_dist.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_dir / 'analysis_summary.json'}")

if __name__ == "__main__":
    main()