"""
Run Complete Analysis with Full Historical Data from GCS
=========================================================

This script runs the comprehensive analysis using ALL available historical data
from both GCS paths:
- ignition-ki-csv-data-2025-user123/market_data/
- ignition-ki-csv-data-2025-user123/tickers/

Generates a detailed PDF report with all 5 analyses.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import our analysis modules
from advanced_analysis_with_gcs import (
    load_real_market_data_for_analysis,
    RobustnessSensitivityAnalyzer,
    GCSDataLoader
)
from comprehensive_pdf_report import ComprehensivePDFReportGenerator

# Import additional analyzers
from advanced_analysis import (
    PostBreakoutAnalyzer,
    RegressionAnalyzer,
    ClusterAnalyzer,
    CorrelationHeatmapAnalyzer,
    PatternMetrics
)


def setup_credentials():
    """Setup GCS credentials"""
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        cred_paths = [
            'gcs-key.json',
            'credentials.json',
            '../gcs-key.json',
            'keys/gcs-key.json'
        ]
        for path in cred_paths:
            if os.path.exists(path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
                print(f"Found credentials at: {path}")
                return True
        print("Warning: No GCS credentials found!")
        return False
    return True


def analyze_data_coverage(data: pd.DataFrame):
    """Analyze the historical data coverage"""
    print("\n" + "=" * 60)
    print("DATA COVERAGE ANALYSIS")
    print("=" * 60)
    
    if data.empty:
        print("No data to analyze")
        return
    
    # Date range analysis
    if 'start_date' in data.columns and 'end_date' in data.columns:
        all_dates = pd.concat([data['start_date'], data['end_date']])
        min_date = all_dates.min()
        max_date = all_dates.max()
        
        print(f"\nHistorical Date Range:")
        print(f"  Earliest: {min_date}")
        print(f"  Latest: {max_date}")
        print(f"  Span: {(max_date - min_date).days / 365.25:.1f} years")
    
    # Patterns by year
    if 'start_date' in data.columns:
        data['year'] = pd.to_datetime(data['start_date']).dt.year
        yearly_counts = data['year'].value_counts().sort_index()
        
        print(f"\nPatterns by Year:")
        for year, count in yearly_counts.items():
            print(f"  {year}: {count:,} patterns")
    
    # Ticker coverage
    if 'ticker' in data.columns:
        unique_tickers = data['ticker'].nunique()
        patterns_per_ticker = data.groupby('ticker').size()
        
        print(f"\nTicker Coverage:")
        print(f"  Unique Tickers: {unique_tickers}")
        print(f"  Avg Patterns per Ticker: {patterns_per_ticker.mean():.1f}")
        print(f"  Max Patterns (single ticker): {patterns_per_ticker.max()}")
        print(f"  Min Patterns (single ticker): {patterns_per_ticker.min()}")
    
    # Outcome distribution
    if 'outcome_class' in data.columns:
        print(f"\nOutcome Distribution:")
        outcome_dist = data['outcome_class'].value_counts()
        for outcome, count in outcome_dist.items():
            pct = count / len(data) * 100
            print(f"  {outcome}: {count:,} ({pct:.1f}%)")
    
    # Pattern characteristics
    print(f"\nPattern Characteristics (Averages):")
    if 'duration' in data.columns:
        print(f"  Duration: {data['duration'].mean():.1f} days")
    if 'boundary_width' in data.columns:
        print(f"  Boundary Width: {data['boundary_width'].mean():.1f}%")
    if 'volume_contraction' in data.columns:
        print(f"  Volume Contraction: {data['volume_contraction'].mean():.2f}")
    if 'max_gain' in data.columns:
        print(f"  Max Gain: {data['max_gain'].mean():.1f}%")


def run_complete_analysis():
    """Run the complete analysis pipeline with full historical data"""
    
    print("=" * 70)
    print("COMPLETE ANALYSIS WITH FULL HISTORICAL DATA")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup credentials
    if not setup_credentials():
        print("Please set up GCS credentials to continue")
        return
    
    # Initialize GCS loader to check available data
    print("\n" + "=" * 60)
    print("CHECKING AVAILABLE DATA IN GCS")
    print("=" * 60)
    
    loader = GCSDataLoader()
    available_tickers = loader.get_available_tickers(limit=1000)
    
    if not available_tickers:
        print("No tickers found in GCS!")
        return
    
    print(f"\nTotal available tickers: {len(available_tickers)}")
    print(f"Sample tickers: {', '.join(available_tickers[:10])}")
    
    # Load data with full history
    print("\n" + "=" * 60)
    print("LOADING FULL HISTORICAL DATA")
    print("=" * 60)
    
    # You can adjust num_tickers based on processing capacity
    NUM_TICKERS = 200  # Increase for more comprehensive analysis
    
    real_data = load_real_market_data_for_analysis(
        num_tickers=NUM_TICKERS,
        use_full_history=True  # This is the key - uses ALL available history
    )
    
    if real_data.empty:
        print("\nNo patterns detected in the data!")
        return
    
    # Analyze data coverage
    analyze_data_coverage(real_data)
    
    # Run all 5 analyses
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE ANALYSIS SUITE")
    print("=" * 60)
    
    # 1. Robustness & Sensitivity Analysis
    print("\n1. ROBUSTNESS & SENSITIVITY ANALYSIS")
    print("-" * 40)
    robustness_analyzer = RobustnessSensitivityAnalyzer(real_data)
    
    boundary_results = robustness_analyzer.vary_boundary_width()
    print("\nBoundary Width Sensitivity:")
    for width, metrics in sorted(boundary_results.items()):
        if metrics['count'] > 0:
            print(f"  ≤{width}%: n={metrics['count']:,}, "
                  f"Success={metrics['success_rate']:.1%}, "
                  f"Expectancy={metrics['expectancy']:.2f}")
    
    duration_results = robustness_analyzer.vary_duration()
    print("\nDuration Range Analysis:")
    for duration_range, metrics in duration_results.items():
        if metrics['count'] > 0:
            print(f"  {duration_range}: n={metrics['count']:,}, "
                  f"Success={metrics['success_rate']:.1%}, "
                  f"Expectancy={metrics['expectancy']:.2f}")
    
    # 2. Regression Analysis
    print("\n2. REGRESSION ANALYSIS")
    print("-" * 40)
    regression_analyzer = RegressionAnalyzer(real_data)
    
    linear_results = regression_analyzer.run_linear_regression()
    print(f"Linear Regression R²: {linear_results['r_squared']:.3f}")
    print(f"Adjusted R²: {linear_results['adjusted_r_squared']:.3f}")
    
    if linear_results['significant_features']:
        print("Significant Features (p < 0.05):")
        for feature in linear_results['significant_features']:
            print(f"  - {feature}")
    
    logistic_results = regression_analyzer.run_logistic_regression()
    print(f"\nLogistic Regression McFadden R²: {logistic_results['mcfadden_r2']:.3f}")
    
    # 3. Cluster Analysis
    print("\n3. CLUSTER ANALYSIS")
    print("-" * 40)
    cluster_analyzer = ClusterAnalyzer(real_data)
    
    optimal_k, silhouette_scores = cluster_analyzer.find_optimal_clusters()
    print(f"Optimal number of clusters: {optimal_k}")
    
    cluster_profiles = cluster_analyzer.perform_clustering(n_clusters=optimal_k)
    print(f"\nIdentified {len(cluster_profiles)} distinct pattern types:")
    
    for cluster_id, profile in sorted(cluster_profiles.items(), 
                                     key=lambda x: x[1]['performance']['success_rate'], 
                                     reverse=True):
        print(f"\n  {profile['name']}:")
        print(f"    Size: {profile['size']:,} patterns ({profile['percentage']:.1f}%)")
        print(f"    Success Rate: {profile['performance']['success_rate']:.1%}")
        print(f"    Avg Gain: {profile['performance']['avg_gain']:.1f}%")
        print(f"    K4 Rate: {profile['performance']['k4_rate']:.1%}")
    
    # 4. Correlation Analysis
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 40)
    heatmap_analyzer = CorrelationHeatmapAnalyzer(real_data)
    
    # Calculate key correlations
    numeric_cols = ['duration', 'boundary_width', 'volume_contraction', 
                   'avg_daily_volatility', 'max_gain']
    if all(col in real_data.columns for col in numeric_cols):
        corr_matrix = real_data[numeric_cols].corr()
        
        print("Key Correlations with Max Gain:")
        for col in numeric_cols[:-1]:
            corr_value = corr_matrix.loc['max_gain', col]
            print(f"  {col}: {corr_value:.3f}")
    
    # Generate comprehensive PDF report
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE PDF REPORT")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"full_history_analysis_report_{timestamp}.pdf"
    
    report_generator = ComprehensivePDFReportGenerator(real_data, report_filename)
    report_generator.generate_report()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 60)
    
    total_patterns = len(real_data)
    success_patterns = len(real_data[real_data['outcome_class'].isin(['K2', 'K3', 'K4'])])
    exceptional_patterns = len(real_data[real_data['outcome_class'] == 'K4'])
    
    print(f"\nTotal Patterns Analyzed: {total_patterns:,}")
    print(f"Success Rate (K2-K4): {success_patterns/total_patterns:.1%}")
    print(f"Exceptional Rate (K4): {exceptional_patterns/total_patterns:.1%}")
    
    if 'start_date' in real_data.columns and 'end_date' in real_data.columns:
        date_range = (real_data['end_date'].max() - real_data['start_date'].min()).days / 365.25
        print(f"Historical Data Span: {date_range:.1f} years")
    
    print(f"\nPDF Report: {report_filename}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return real_data, report_filename


if __name__ == "__main__":
    # Run the complete analysis
    data, report = run_complete_analysis()
    
    print("\n" + "=" * 70)
    print("All analyses completed successfully!")
    print("Full historical data from both GCS paths has been analyzed.")
    print("=" * 70)