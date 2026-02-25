"""
Consolidation Pattern Analyzer for Parquet Files
Analyzes actual consolidation data from BigQuery results
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class ConsolidationParquetAnalyzer:
    """Analyzes consolidation patterns from parquet files"""

    def __init__(self):
        self.df = None
        self.results = {}

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load parquet file handling date32 type properly"""
        # Read parquet with specific dtype for date column
        try:
            # First attempt: Read normally
            df = pd.read_parquet(file_path)
        except:
            # If that fails, read with pyarrow and handle date manually
            table = pq.read_table(file_path)

            # Get the data as dict, converting date column specially
            data_dict = {}
            for col_name in table.column_names:
                col = table.column(col_name)
                if 'date' in str(col.type).lower():
                    # Convert date32 to Python dates
                    data_dict[col_name] = col.to_pylist()
                else:
                    # Convert normally
                    data_dict[col_name] = col.to_pandas()

            df = pd.DataFrame(data_dict)

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        self.df = df
        print(f"Loaded {len(df)} rows from {Path(file_path).name}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Tickers: {df['ticker'].unique().tolist()}")

        return df

    def analyze_consolidation_methods(self) -> Dict[str, Any]:
        """Analyze effectiveness of each consolidation detection method"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        methods = {
            'method1_bollinger': 'Bollinger Band Width',
            'method2_range_based': 'Range-Based',
            'method3_volume_weighted': 'Volume-Weighted',
            'method4_atr_based': 'ATR-Based'
        }

        method_stats = {}

        for method_col, method_name in methods.items():
            # Get rows where this method detected consolidation
            consolidated = self.df[self.df[method_col] == True].copy()

            if len(consolidated) == 0:
                method_stats[method_name] = {
                    'total_signals': 0,
                    'success_rate': 0,
                    'avg_gain_20d': 0,
                    'avg_gain_40d': 0,
                    'avg_max_loss_20d': 0,
                    'avg_max_loss_40d': 0
                }
                continue

            # Calculate statistics
            stats = {
                'total_signals': len(consolidated),
                'signal_rate': len(consolidated) / len(self.df) * 100,

                # 20-day performance
                'avg_gain_20d': consolidated['max_gain_20d'].mean() * 100,
                'median_gain_20d': consolidated['max_gain_20d'].median() * 100,
                'avg_loss_20d': consolidated['max_loss_20d'].mean() * 100,
                'success_rate_20d': (consolidated['max_gain_20d'] > 0.1).mean() * 100,  # >10% gain

                # 40-day performance
                'avg_gain_40d': consolidated['max_gain_40d'].mean() * 100,
                'median_gain_40d': consolidated['max_gain_40d'].median() * 100,
                'avg_loss_40d': consolidated['max_loss_40d'].mean() * 100,
                'success_rate_40d': (consolidated['max_gain_40d'] > 0.15).mean() * 100,  # >15% gain

                # Risk metrics
                'risk_reward_20d': abs(consolidated['max_gain_20d'].mean() /
                                      consolidated['max_loss_20d'].mean()) if consolidated['max_loss_20d'].mean() != 0 else 0,
                'risk_reward_40d': abs(consolidated['max_gain_40d'].mean() /
                                      consolidated['max_loss_40d'].mean()) if consolidated['max_loss_40d'].mean() != 0 else 0,

                # Volatility at signal
                'avg_bbw': consolidated['bbw'].mean() if 'bbw' in consolidated.columns else None,
                'avg_volume_ratio': consolidated['volume_ratio'].mean() if 'volume_ratio' in consolidated.columns else None,
                'avg_atr_pct': consolidated['atr_pct'].mean() if 'atr_pct' in consolidated.columns else None
            }

            method_stats[method_name] = stats

        return method_stats

    def analyze_method_agreement(self) -> Dict[str, Any]:
        """Analyze performance when multiple methods agree"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        # Count how many methods detected consolidation for each row
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']

        self.df['methods_agreed'] = self.df[method_cols].sum(axis=1)

        agreement_stats = {}

        for num_methods in range(1, 5):
            agreed_df = self.df[self.df['methods_agreed'] == num_methods]

            if len(agreed_df) == 0:
                continue

            agreement_stats[f'{num_methods}_methods'] = {
                'count': len(agreed_df),
                'percentage': len(agreed_df) / len(self.df) * 100,
                'avg_gain_20d': agreed_df['max_gain_20d'].mean() * 100,
                'avg_gain_40d': agreed_df['max_gain_40d'].mean() * 100,
                'avg_loss_20d': agreed_df['max_loss_20d'].mean() * 100,
                'avg_loss_40d': agreed_df['max_loss_40d'].mean() * 100,
                'success_rate_20d': (agreed_df['max_gain_20d'] > 0.1).mean() * 100,
                'success_rate_40d': (agreed_df['max_gain_40d'] > 0.15).mean() * 100
            }

        return agreement_stats

    def find_best_patterns(self, min_gain_20d: float = 0.2, min_gain_40d: float = 0.3) -> pd.DataFrame:
        """Find the best performing consolidation patterns"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        # Find patterns with strong performance
        best_patterns = self.df[
            (self.df['max_gain_20d'] >= min_gain_20d) |
            (self.df['max_gain_40d'] >= min_gain_40d)
        ].copy()

        # Add methods agreed column if not already present
        if 'methods_agreed' not in best_patterns.columns:
            method_cols = ['method1_bollinger', 'method2_range_based',
                          'method3_volume_weighted', 'method4_atr_based']
            best_patterns['methods_agreed'] = best_patterns[method_cols].sum(axis=1)

        # Sort by 40-day gain
        best_patterns = best_patterns.sort_values('max_gain_40d', ascending=False)

        return best_patterns

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns over time"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        df = self.df.copy()

        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek

        temporal_stats = {}

        # Monthly analysis
        monthly_stats = []
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                monthly_stats.append({
                    'month': month,
                    'avg_gain_20d': month_data['max_gain_20d'].mean() * 100,
                    'avg_gain_40d': month_data['max_gain_40d'].mean() * 100,
                    'consolidation_rate': (month_data[['method1_bollinger', 'method2_range_based',
                                                      'method3_volume_weighted', 'method4_atr_based']].any(axis=1)).mean() * 100
                })

        temporal_stats['monthly'] = monthly_stats

        # Quarterly analysis
        quarterly_stats = []
        for quarter in range(1, 5):
            quarter_data = df[df['quarter'] == quarter]
            if len(quarter_data) > 0:
                quarterly_stats.append({
                    'quarter': quarter,
                    'avg_gain_20d': quarter_data['max_gain_20d'].mean() * 100,
                    'avg_gain_40d': quarter_data['max_gain_40d'].mean() * 100,
                    'consolidation_rate': (quarter_data[['method1_bollinger', 'method2_range_based',
                                                        'method3_volume_weighted', 'method4_atr_based']].any(axis=1)).mean() * 100
                })

        temporal_stats['quarterly'] = quarterly_stats

        return temporal_stats

    def analyze_volatility_patterns(self) -> Dict[str, Any]:
        """Analyze relationship between volatility metrics and outcomes"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        # Analyze BBW ranges
        bbw_analysis = {}
        if 'bbw' in self.df.columns:
            # Create BBW bins
            bbw_percentiles = [0, 25, 50, 75, 100]
            bbw_bins = np.percentile(self.df['bbw'].dropna(), bbw_percentiles)

            for i in range(len(bbw_bins)-1):
                mask = (self.df['bbw'] >= bbw_bins[i]) & (self.df['bbw'] < bbw_bins[i+1])
                bin_data = self.df[mask]

                bbw_analysis[f'bbw_{bbw_percentiles[i]}_{bbw_percentiles[i+1]}_percentile'] = {
                    'range': f"{bbw_bins[i]:.4f} - {bbw_bins[i+1]:.4f}",
                    'count': len(bin_data),
                    'avg_gain_20d': bin_data['max_gain_20d'].mean() * 100,
                    'avg_gain_40d': bin_data['max_gain_40d'].mean() * 100
                }

        # Analyze ATR patterns
        atr_analysis = {}
        if 'atr_pct' in self.df.columns:
            atr_percentiles = [0, 25, 50, 75, 100]
            atr_bins = np.percentile(self.df['atr_pct'].dropna(), atr_percentiles)

            for i in range(len(atr_bins)-1):
                mask = (self.df['atr_pct'] >= atr_bins[i]) & (self.df['atr_pct'] < atr_bins[i+1])
                bin_data = self.df[mask]

                atr_analysis[f'atr_{atr_percentiles[i]}_{atr_percentiles[i+1]}_percentile'] = {
                    'range': f"{atr_bins[i]:.4f} - {atr_bins[i+1]:.4f}",
                    'count': len(bin_data),
                    'avg_gain_20d': bin_data['max_gain_20d'].mean() * 100,
                    'avg_gain_40d': bin_data['max_gain_40d'].mean() * 100
                }

        return {
            'bbw_analysis': bbw_analysis,
            'atr_analysis': atr_analysis
        }

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        report = {
            'data_summary': {
                'total_rows': len(self.df),
                'date_range': {
                    'start': str(self.df['date'].min()),
                    'end': str(self.df['date'].max())
                },
                'tickers': self.df['ticker'].unique().tolist(),
                'ticker_count': self.df['ticker'].nunique()
            },
            'method_effectiveness': self.analyze_consolidation_methods(),
            'method_agreement': self.analyze_method_agreement(),
            'temporal_patterns': self.analyze_temporal_patterns(),
            'volatility_patterns': self.analyze_volatility_patterns(),
            'best_patterns': {
                'count': len(self.find_best_patterns()),
                'top_10': self.find_best_patterns().head(10)[
                    ['ticker', 'date', 'max_gain_20d', 'max_gain_40d', 'methods_agreed']
                ].to_dict('records')
            }
        }

        return report

    def save_analysis(self, output_dir: str = "analysis_output"):
        """Save all analysis results"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary report
        report = self.generate_summary_report()
        with open(output_path / f"analysis_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save best patterns as CSV
        best_patterns = self.find_best_patterns()
        if len(best_patterns) > 0:
            best_patterns.to_csv(output_path / f"best_patterns_{timestamp}.csv", index=False)

        # Save method comparison
        method_stats = pd.DataFrame(self.analyze_consolidation_methods()).T
        method_stats.to_csv(output_path / f"method_comparison_{timestamp}.csv")

        print(f"\nAnalysis saved to {output_path}")
        print(f"  - analysis_report_{timestamp}.json")
        print(f"  - best_patterns_{timestamp}.csv")
        print(f"  - method_comparison_{timestamp}.csv")

        return output_path


def main():
    """Run analysis on parquet files"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze consolidation patterns from parquet files')
    parser.add_argument('file', help='Path to parquet file')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory')
    parser.add_argument('--min-gain-20d', type=float, default=0.2,
                       help='Minimum 20-day gain for best patterns (default: 0.2)')
    parser.add_argument('--min-gain-40d', type=float, default=0.3,
                       help='Minimum 40-day gain for best patterns (default: 0.3)')

    args = parser.parse_args()

    # Run analysis
    analyzer = ConsolidationParquetAnalyzer()

    # Load data
    analyzer.load_parquet(args.file)

    # Print summary
    print("\n" + "="*60)
    print("CONSOLIDATION PATTERN ANALYSIS")
    print("="*60)

    # Method effectiveness
    print("\nMethod Effectiveness:")
    method_stats = analyzer.analyze_consolidation_methods()
    for method, stats in method_stats.items():
        print(f"\n{method}:")
        print(f"  Signals: {stats['total_signals']}")
        print(f"  20-day avg gain: {stats.get('avg_gain_20d', 0):.2f}%")
        print(f"  40-day avg gain: {stats.get('avg_gain_40d', 0):.2f}%")
        print(f"  20-day success rate: {stats.get('success_rate_20d', 0):.1f}%")

    # Method agreement
    print("\nMethod Agreement Analysis:")
    agreement = analyzer.analyze_method_agreement()
    for agreement_level, stats in agreement.items():
        print(f"\n{agreement_level}:")
        print(f"  Count: {stats['count']}")
        print(f"  20-day avg gain: {stats['avg_gain_20d']:.2f}%")
        print(f"  40-day avg gain: {stats['avg_gain_40d']:.2f}%")

    # Best patterns
    best = analyzer.find_best_patterns(args.min_gain_20d, args.min_gain_40d)
    print(f"\nBest Patterns Found: {len(best)}")
    if len(best) > 0:
        print("\nTop 5 Patterns:")
        for _, row in best.head(5).iterrows():
            print(f"  {row['ticker']} on {row['date'].date()}: "
                  f"20d={row['max_gain_20d']*100:.1f}%, "
                  f"40d={row['max_gain_40d']*100:.1f}%")

    # Save results
    analyzer.save_analysis(args.output_dir)

    print("\n" + "="*60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()