"""
Configurable Market Analysis Runner
Supports different stock counts and output formats
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from comprehensive_market_consolidation_analysis import (
    GCSDataLoader, 
    MarketWideConsolidationAnalyzer,
    DailyConsolidationMetrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurableMarketAnalyzer:
    """Extended analyzer with configurable output options"""
    
    def __init__(self, gcs_loader, output_format='full', timestamp=None):
        self.analyzer = MarketWideConsolidationAnalyzer(gcs_loader)
        self.output_format = output_format
        self.timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def export_json_complete(self, results_df: pd.DataFrame, filename: str = None):
        """Export complete dataset as JSON with all daily metrics"""
        if filename is None:
            filename = f"market_consolidation_complete_{self.timestamp}.json"

        logger.info(f"Exporting complete JSON dataset to {filename}")

        # Process DataFrame in chunks to avoid memory errors
        chunk_size = 10000  # Process 10k rows at a time
        records = []
        total_rows = len(results_df)

        logger.info(f"Processing {total_rows} rows in chunks of {chunk_size}")

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = results_df.iloc[start_idx:end_idx]

            # Convert chunk to list of dictionaries
            for idx in range(len(chunk)):
                row = chunk.iloc[idx]
                record = {}
                for col in chunk.columns:
                    value = row[col]
                    # Handle NaN and infinite values
                    if pd.isna(value):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        if np.isinf(value):
                            record[col] = str(value)
                        else:
                            record[col] = float(value)
                    elif isinstance(value, bool):
                        record[col] = bool(value)
                    else:
                        record[col] = str(value)
                records.append(record)

            # Log progress every 50k rows
            if (end_idx % 50000 == 0) or (end_idx == total_rows):
                logger.info(f"Processed {end_idx}/{total_rows} rows ({(end_idx/total_rows)*100:.1f}%)")
        
        # Create comprehensive output structure
        output = {
            'metadata': {
                'analysis_timestamp': self.timestamp,
                'total_tickers': int(results_df['ticker'].nunique()),
                'total_records': len(results_df),
                'date_range': {
                    'start': results_df['date'].min(),
                    'end': results_df['date'].max()
                },
                'configuration': {
                    'min_price_filter': 0.01,
                    'min_trading_days': 252,
                    'min_data_completeness': 0.65,
                    'methods': {
                        'method1': 'Bollinger Band Width (BBW < 30th percentile)',
                        'method2': 'Range-Based (Daily Range < 65% of 20-day avg, ADX < 32)',
                        'method3': 'Volume-Weighted (Volume < 35% of 20-day avg)',
                        'method4': 'ATR-Based (ATR < 30th percentile, Volatility < 40%)'
                    }
                }
            },
            'summary_statistics': {
                'consolidation_detection': {
                    'method1_bollinger': {
                        'total_days': int(results_df['method1_bollinger'].sum()),
                        'percentage': float(results_df['method1_bollinger'].mean() * 100),
                        'avg_upper_boundary': float(results_df['method1_upper'].mean()) if 'method1_upper' in results_df else None,
                        'avg_lower_boundary': float(results_df['method1_lower'].mean()) if 'method1_lower' in results_df else None
                    },
                    'method2_range_based': {
                        'total_days': int(results_df['method2_range_based'].sum()),
                        'percentage': float(results_df['method2_range_based'].mean() * 100),
                        'avg_upper_boundary': float(results_df['method2_upper'].mean()) if 'method2_upper' in results_df else None,
                        'avg_lower_boundary': float(results_df['method2_lower'].mean()) if 'method2_lower' in results_df else None
                    },
                    'method3_volume_weighted': {
                        'total_days': int(results_df['method3_volume_weighted'].sum()),
                        'percentage': float(results_df['method3_volume_weighted'].mean() * 100),
                        'avg_upper_boundary': float(results_df['method3_upper'].mean()) if 'method3_upper' in results_df else None,
                        'avg_lower_boundary': float(results_df['method3_lower'].mean()) if 'method3_lower' in results_df else None
                    },
                    'method4_atr_based': {
                        'total_days': int(results_df['method4_atr_based'].sum()),
                        'percentage': float(results_df['method4_atr_based'].mean() * 100),
                        'avg_upper_boundary': float(results_df['method4_upper'].mean()) if 'method4_upper' in results_df else None,
                        'avg_lower_boundary': float(results_df['method4_lower'].mean()) if 'method4_lower' in results_df else None
                    }
                },
                'future_performance': {
                    'gains': {
                        '20_days': {
                            'mean': float(results_df['max_gain_20d'].mean()) if 'max_gain_20d' in results_df else None,
                            'median': float(results_df['max_gain_20d'].median()) if 'max_gain_20d' in results_df else None,
                            'std': float(results_df['max_gain_20d'].std()) if 'max_gain_20d' in results_df else None,
                            'max': float(results_df['max_gain_20d'].max()) if 'max_gain_20d' in results_df else None
                        },
                        '40_days': {
                            'mean': float(results_df['max_gain_40d'].mean()) if 'max_gain_40d' in results_df else None,
                            'median': float(results_df['max_gain_40d'].median()) if 'max_gain_40d' in results_df else None,
                            'std': float(results_df['max_gain_40d'].std()) if 'max_gain_40d' in results_df else None,
                            'max': float(results_df['max_gain_40d'].max()) if 'max_gain_40d' in results_df else None
                        },
                        '50_days': {
                            'mean': float(results_df['max_gain_50d'].mean()) if 'max_gain_50d' in results_df else None,
                            'median': float(results_df['max_gain_50d'].median()) if 'max_gain_50d' in results_df else None
                        },
                        '70_days': {
                            'mean': float(results_df['max_gain_70d'].mean()) if 'max_gain_70d' in results_df else None,
                            'median': float(results_df['max_gain_70d'].median()) if 'max_gain_70d' in results_df else None
                        },
                        '100_days': {
                            'mean': float(results_df['max_gain_100d'].mean()) if 'max_gain_100d' in results_df else None,
                            'median': float(results_df['max_gain_100d'].median()) if 'max_gain_100d' in results_df else None
                        }
                    },
                    'losses': {
                        '20_days': {
                            'mean': float(results_df['max_loss_20d'].mean()) if 'max_loss_20d' in results_df else None,
                            'median': float(results_df['max_loss_20d'].median()) if 'max_loss_20d' in results_df else None,
                            'worst': float(results_df['max_loss_20d'].min()) if 'max_loss_20d' in results_df else None
                        },
                        '40_days': {
                            'mean': float(results_df['max_loss_40d'].mean()) if 'max_loss_40d' in results_df else None,
                            'median': float(results_df['max_loss_40d'].median()) if 'max_loss_40d' in results_df else None,
                            'worst': float(results_df['max_loss_40d'].min()) if 'max_loss_40d' in results_df else None
                        },
                        '100_days': {
                            'mean': float(results_df['max_loss_100d'].mean()) if 'max_loss_100d' in results_df else None,
                            'median': float(results_df['max_loss_100d'].median()) if 'max_loss_100d' in results_df else None,
                            'worst': float(results_df['max_loss_100d'].min()) if 'max_loss_100d' in results_df else None
                        }
                    }
                },
                'method_agreement': self._calculate_method_agreement(results_df),
                'ticker_statistics': self._calculate_ticker_statistics(results_df)
            },
            'daily_data': records
        }
        
        # Save to JSON file using streaming approach for large datasets
        logger.info(f"Writing JSON file with {len(records)} records...")

        try:
            # For very large datasets, write in streaming fashion
            if len(records) > 100000:
                logger.info("Using streaming JSON write for large dataset...")
                with open(filename, 'w') as f:
                    f.write('{\n')
                    f.write('  "metadata": ')
                    json.dump(output['metadata'], f, indent=2, default=str)
                    f.write(',\n')
                    f.write('  "summary_statistics": ')
                    json.dump(output['summary_statistics'], f, indent=2, default=str)
                    f.write(',\n')
                    f.write('  "daily_data": [\n')

                    # Write records in batches
                    for i, record in enumerate(records):
                        if i > 0:
                            f.write(',\n')
                        f.write('    ')
                        json.dump(record, f, default=str)

                        # Flush buffer periodically
                        if i % 10000 == 0:
                            f.flush()

                    f.write('\n  ]\n')
                    f.write('}\n')
            else:
                # For smaller datasets, use regular json.dump
                with open(filename, 'w') as f:
                    json.dump(output, f, indent=2, default=str)

            logger.info(f"Complete JSON dataset saved to {filename}")

            # Print file size
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            logger.info(f"File size: {file_size:.2f} MB")

        except MemoryError as e:
            logger.error(f"Memory error while writing JSON: {e}")
            logger.info("Attempting to write without indentation to save memory...")
            with open(filename, 'w') as f:
                json.dump(output, f, default=str)
            logger.info(f"JSON saved without formatting to {filename}")
        
        return filename
    
    def _calculate_method_agreement(self, results_df):
        """Calculate how often methods agree on consolidation"""
        results_df['methods_agreeing'] = (
            results_df['method1_bollinger'].astype(int) +
            results_df['method2_range_based'].astype(int) +
            results_df['method3_volume_weighted'].astype(int) +
            results_df['method4_atr_based'].astype(int)
        )
        
        agreement_stats = {}
        for i in range(5):
            count = (results_df['methods_agreeing'] == i).sum()
            agreement_stats[f'{i}_methods_agree'] = {
                'count': int(count),
                'percentage': float((count / len(results_df)) * 100)
            }
        
        # Days when all methods agree on consolidation
        all_agree_consolidation = (results_df['methods_agreeing'] == 4).sum()
        agreement_stats['all_methods_agree_consolidation'] = {
            'count': int(all_agree_consolidation),
            'percentage': float((all_agree_consolidation / len(results_df)) * 100)
        }
        
        # Days when at least 2 methods agree
        at_least_two = (results_df['methods_agreeing'] >= 2).sum()
        agreement_stats['at_least_two_methods_agree'] = {
            'count': int(at_least_two),
            'percentage': float((at_least_two / len(results_df)) * 100)
        }
        
        return agreement_stats
    
    def _calculate_ticker_statistics(self, results_df):
        """Calculate per-ticker statistics"""
        ticker_stats = []
        
        for ticker in results_df['ticker'].unique():
            ticker_data = results_df[results_df['ticker'] == ticker]
            
            stats = {
                'ticker': ticker,
                'total_days': len(ticker_data),
                'date_range': f"{ticker_data['date'].min()} to {ticker_data['date'].max()}",
                'price_range': {
                    'min': float(ticker_data['price'].min()),
                    'max': float(ticker_data['price'].max()),
                    'mean': float(ticker_data['price'].mean())
                },
                'consolidation_days': {
                    'method1': int(ticker_data['method1_bollinger'].sum()),
                    'method2': int(ticker_data['method2_range_based'].sum()),
                    'method3': int(ticker_data['method3_volume_weighted'].sum()),
                    'method4': int(ticker_data['method4_atr_based'].sum())
                },
                'consolidation_percentage': {
                    'method1': float(ticker_data['method1_bollinger'].mean() * 100),
                    'method2': float(ticker_data['method2_range_based'].mean() * 100),
                    'method3': float(ticker_data['method3_volume_weighted'].mean() * 100),
                    'method4': float(ticker_data['method4_atr_based'].mean() * 100)
                }
            }
            
            # Add average future gains/losses if available
            if 'max_gain_20d' in ticker_data.columns:
                stats['avg_future_gain_20d'] = float(ticker_data['max_gain_20d'].mean())
            if 'max_loss_20d' in ticker_data.columns:
                stats['avg_future_loss_20d'] = float(ticker_data['max_loss_20d'].mean())
            
            ticker_stats.append(stats)
        
        return ticker_stats
    
    def run_analysis(self, num_stocks=None, complete=False):
        """Run the analysis with specified parameters"""
        if complete:
            logger.info("Running COMPLETE market analysis (all available tickers)")
            results_df = self.analyzer.analyze_market(max_tickers=None, parallel_workers=10)
        else:
            logger.info(f"Running analysis for {num_stocks} stocks")
            results_df = self.analyzer.analyze_market(max_tickers=num_stocks, parallel_workers=5)
        
        if results_df.empty:
            logger.error("No data available for analysis")
            return None
        
        # Export based on format
        if self.output_format == 'json_only':
            # Export only JSON with complete data
            json_file = self.export_json_complete(results_df)
            return {'json': json_file}
        else:
            # Export full format (JSON + CSV + Excel)
            json_file = self.export_json_complete(results_df)
            
            # Also export CSV and Excel
            csv_file, excel_file, summary_json = self.analyzer.export_comprehensive_results(
                results_df, 
                output_prefix=f"market_consolidation_{self.timestamp}"
            )
            
            return {
                'json': json_file,
                'csv': csv_file,
                'excel': excel_file,
                'summary': summary_json
            }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run configurable market consolidation analysis')
    parser.add_argument('--num-stocks', type=int, help='Number of stocks to analyze')
    parser.add_argument('--complete', action='store_true', help='Analyze all available stocks')
    parser.add_argument('--output-format', choices=['json_only', 'full'], default='full',
                       help='Output format (json_only or full)')
    parser.add_argument('--timestamp', help='Timestamp for output files')
    
    args = parser.parse_args()
    
    if not args.complete and not args.num_stocks:
        logger.error("Please specify either --num-stocks or --complete")
        sys.exit(1)
    
    # Set up GCS credentials
    credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
    
    try:
        # Initialize GCS loader
        logger.info("Initializing GCS connection...")
        gcs_loader = GCSDataLoader(credentials_path)
        
        # Initialize configurable analyzer
        analyzer = ConfigurableMarketAnalyzer(
            gcs_loader, 
            output_format=args.output_format,
            timestamp=args.timestamp
        )
        
        # Run analysis
        results = analyzer.run_analysis(
            num_stocks=args.num_stocks,
            complete=args.complete
        )
        
        if results:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print("\nGenerated files:")
            for file_type, filename in results.items():
                if filename:
                    print(f"  {file_type.upper()}: {filename}")
            print("="*60)
            sys.exit(0)
        else:
            logger.error("Analysis failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()