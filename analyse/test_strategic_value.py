"""
Test strategic value calculation with the K0-K5 classification system
"""

import sys
import os
from datetime import datetime
from run_complete_analysis_filtered import FilteredAnalysisPipeline
import pandas as pd

def main():
    """Test strategic value calculation"""
    
    print("="*80)
    print("STRATEGIC VALUE TEST")
    print("="*80)
    print("\nPattern Classification & Strategic Values:")
    print("K4 (Exceptional): >75% gain -> Value: +10")
    print("K3 (Strong): 35-75% gain -> Value: +3")
    print("K2 (Quality): 15-35% gain -> Value: +1")
    print("K1 (Minimal): 5-15% gain -> Value: -0.2")
    print("K0 (Stagnant): <5% gain -> Value: -2")
    print("K5 (Failed): Breakdown -> Value: -10")
    print("="*80)
    
    # Create pipeline
    pipeline = FilteredAnalysisPipeline()
    
    # Get limited ticker files for testing
    ticker_files = pipeline.get_all_ticker_files()[:5]  # Only process first 5
    
    if not ticker_files:
        print("No ticker files found")
        return False
    
    print(f"\nProcessing {len(ticker_files)} test tickers...")
    
    # Process the limited set
    pipeline.process_all_tickers_parallel(ticker_files, max_workers=3)
    
    if not pipeline.all_patterns:
        print("No patterns found")
        return False
    
    # Create DataFrame for analysis
    df = pd.DataFrame(pipeline.all_patterns)
    
    print(f"\nTotal patterns detected: {len(df)}")
    
    # Show outcome distribution
    print("\n" + "="*80)
    print("OUTCOME DISTRIBUTION")
    print("="*80)
    outcome_counts = df['outcome_class'].value_counts().sort_index()
    for outcome, count in outcome_counts.items():
        pct = count / len(df) * 100
        value = df[df['outcome_class'] == outcome]['strategic_value'].iloc[0] if count > 0 else 0
        total_value = value * count
        print(f"{outcome}: {count:3d} patterns ({pct:5.1f}%) | Unit Value: {value:+6.1f} | Total: {total_value:+7.1f}")
    
    # Calculate expected value
    print("\n" + "="*80)
    print("STRATEGIC VALUE ANALYSIS")
    print("="*80)
    
    total_strategic_value = df['strategic_value'].sum()
    avg_strategic_value = df['strategic_value'].mean()
    positive_value = df[df['strategic_value'] > 0]
    negative_value = df[df['strategic_value'] < 0]
    
    print(f"Total Strategic Value: {total_strategic_value:+.1f}")
    print(f"Average Strategic Value: {avg_strategic_value:+.2f}")
    print(f"Positive Value Patterns: {len(positive_value)} ({len(positive_value)/len(df)*100:.1f}%)")
    print(f"Negative Value Patterns: {len(negative_value)} ({len(negative_value)/len(df)*100:.1f}%)")
    
    # Expected value calculation
    expected_value = 0
    for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
        prob = len(df[df['outcome_class'] == outcome]) / len(df)
        value_map = {'K4': 10, 'K3': 3, 'K2': 1, 'K1': -0.2, 'K0': -2, 'K5': -10}
        expected_value += prob * value_map.get(outcome, 0)
    
    print(f"Expected Value (EV): {expected_value:+.3f}")
    
    # High-value patterns (K3 and K4)
    high_value = df[df['outcome_class'].isin(['K3', 'K4'])]
    if not high_value.empty:
        print("\n" + "="*80)
        print("HIGH-VALUE PATTERNS (K3 & K4)")
        print("="*80)
        print(f"Count: {len(high_value)} patterns")
        print(f"Percentage: {len(high_value)/len(df)*100:.1f}%")
        print(f"Average Gain: {high_value['max_gain'].mean():.1f}%")
        print(f"Total Strategic Value: {high_value['strategic_value'].sum():+.1f}")
        
        # Top tickers
        top_tickers = high_value['ticker'].value_counts().head(5)
        if not top_tickers.empty:
            print("\nTop Tickers with High-Value Patterns:")
            for ticker, count in top_tickers.items():
                ticker_value = high_value[high_value['ticker'] == ticker]['strategic_value'].sum()
                print(f"  {ticker}: {count} patterns (value: {ticker_value:+.1f})")
    
    # Failed patterns (K5)
    failed = df[df['outcome_class'] == 'K5']
    if not failed.empty:
        print("\n" + "="*80)
        print("FAILURE ANALYSIS (K5)")
        print("="*80)
        print(f"Failed Patterns: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")
        print(f"Total Negative Value: {failed['strategic_value'].sum():.1f}")
        print(f"Average Loss: {failed['max_loss'].mean() if 'max_loss' in failed.columns else 'N/A':.1f}%")
    
    # Sample patterns with strategic values
    print("\n" + "="*80)
    print("SAMPLE PATTERNS WITH STRATEGIC VALUES")
    print("="*80)
    
    samples = df.sample(n=min(5, len(df)))
    for idx, pattern in samples.iterrows():
        print(f"\n{pattern['ticker']} - {pattern['start_date']} to {pattern['end_date']}")
        print(f"  Outcome: {pattern['outcome_class']}")
        print(f"  Max Gain: {pattern['max_gain']:.1f}%")
        print(f"  Strategic Value: {pattern['strategic_value']:+.1f}")
        if pattern.get('days_to_max'):
            print(f"  Days to Max: {pattern['days_to_max']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()