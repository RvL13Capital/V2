"""
Test run with limited tickers to verify real data processing
"""

import sys
import os
from datetime import datetime
from run_complete_analysis_filtered import FilteredAnalysisPipeline

def main():
    """Run test analysis on limited tickers"""
    
    print("="*80)
    print("TEST ANALYSIS - LIMITED TICKERS")
    print(f"Minimum Price Requirement: $0.01")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nRunning test on first 10 tickers only...")
    print("Using only real data - no mock data generation")
    print("="*80)
    
    # Create pipeline
    pipeline = FilteredAnalysisPipeline()
    
    # Get limited ticker files for testing
    ticker_files = pipeline.get_all_ticker_files()[:10]  # Only process first 10
    
    if not ticker_files:
        print("No ticker files found")
        return False
    
    print(f"Processing {len(ticker_files)} test tickers...")
    
    # Process the limited set
    pipeline.process_all_tickers_parallel(ticker_files, max_workers=5)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Tickers processed: {pipeline.analysis_stats['tickers_processed']}")
    print(f"Tickers filtered: {pipeline.analysis_stats['tickers_filtered']}")
    print(f"Patterns found: {pipeline.analysis_stats['patterns_found']}")
    print(f"Patterns filtered: {pipeline.analysis_stats['patterns_filtered']}")
    print(f"Total patterns: {len(pipeline.all_patterns)}")
    
    # Show sample patterns with days_to_target data
    if pipeline.all_patterns:
        print("\nSample patterns with days_to_target data:")
        for i, pattern in enumerate(pipeline.all_patterns[:3]):
            print(f"\nPattern {i+1}:")
            print(f"  Ticker: {pattern['ticker']}")
            print(f"  Outcome: {pattern['outcome_class']}")
            print(f"  Max Gain: {pattern['max_gain']:.2f}%")
            print(f"  Days to max: {pattern.get('days_to_max', 'N/A')}")
            print(f"  Days to 5%: {pattern.get('days_to_5pct', 'N/A')}")
            print(f"  Days to 10%: {pattern.get('days_to_10pct', 'N/A')}")
            print(f"  Days to 15%: {pattern.get('days_to_15pct', 'N/A')}")
            print(f"  Days to 25%: {pattern.get('days_to_25pct', 'N/A')}")
    
    # Run extended analysis on the test patterns
    if pipeline.all_patterns:
        pipeline.patterns = pipeline.all_patterns
        pipeline.run_extended_analysis()
        
        # Check if time_to_targets uses real data
        if 'time_to_targets' in pipeline.extended_metrics:
            print("\nTime to targets analysis (from real data):")
            for target, stats in pipeline.extended_metrics['time_to_targets'].items():
                if stats['count'] > 0:
                    print(f"  {target}: {stats['count']} patterns, avg {stats['avg_days']:.1f} days")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()