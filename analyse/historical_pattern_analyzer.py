"""
Historical Consolidation Pattern Analyzer
Analyzes ENTIRE history to find ALL consolidation patterns and their outcomes
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from google.cloud import storage
import io
import time
from typing import List, Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalPatternAnalyzer:
    """Analyze ALL historical consolidation patterns and their outcomes"""

    def __init__(self):
        self.storage_client = storage.Client(project='ignition-ki-csv-storage')
        self.bucket_name = 'ignition-ki-csv-data-2025-user123'
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.all_patterns = []

    def detect_consolidation_patterns(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """
        Detect ALL consolidation patterns in the entire price history
        """
        patterns = []

        # Calculate indicators
        df['ma20'] = df['close'].rolling(window=20, min_periods=20).mean()
        df['std20'] = df['close'].rolling(window=20, min_periods=20).std()
        df['bbw'] = (2 * df['std20']) / df['ma20']
        df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['range'] = df['high'] - df['low']
        df['range_ma20'] = df['range'].rolling(window=20, min_periods=20).mean()
        df['range_ratio'] = df['range'] / df['range_ma20']

        # Calculate percentiles for dynamic thresholds
        bbw_30 = df['bbw'].quantile(0.3)

        # Identify consolidation days
        df['is_consolidation'] = (
            (df['bbw'] < bbw_30) &
            (df['volume_ratio'] < 0.5) &
            (df['range_ratio'] < 0.65)
        )

        # Find consolidation periods (consecutive days)
        df['pattern_group'] = (df['is_consolidation'] != df['is_consolidation'].shift()).cumsum()

        # Group consecutive consolidation days
        consolidation_groups = df[df['is_consolidation']].groupby('pattern_group')

        for group_id, group in consolidation_groups:
            if len(group) >= 5:  # Minimum 5 days for valid pattern

                pattern_start = group.index[0]
                pattern_end = group.index[-1]

                # Get pattern metrics
                pattern_data = {
                    'ticker': ticker,
                    'pattern_start_date': df.loc[pattern_start, 'date'],
                    'pattern_end_date': df.loc[pattern_end, 'date'],
                    'pattern_duration_days': len(group),
                    'pattern_start_price': df.loc[pattern_start, 'close'],
                    'pattern_end_price': df.loc[pattern_end, 'close'],
                    'avg_volume_ratio': group['volume_ratio'].mean(),
                    'avg_bbw': group['bbw'].mean(),
                    'avg_range_ratio': group['range_ratio'].mean(),
                    'price_range_pct': ((group['high'].max() - group['low'].min()) / group['close'].mean()) * 100
                }

                # Calculate OUTCOME (what happened AFTER the pattern)
                future_window = 30  # Look 30 days into future

                if pattern_end + future_window < len(df):
                    future_data = df.iloc[pattern_end+1:pattern_end+future_window+1]

                    if len(future_data) > 0:
                        # Calculate various outcome metrics
                        pattern_data['outcome_max_gain'] = ((future_data['high'].max() - df.loc[pattern_end, 'close']) / df.loc[pattern_end, 'close']) * 100
                        pattern_data['outcome_max_loss'] = ((future_data['low'].min() - df.loc[pattern_end, 'close']) / df.loc[pattern_end, 'close']) * 100
                        pattern_data['outcome_end_gain'] = ((future_data['close'].iloc[-1] - df.loc[pattern_end, 'close']) / df.loc[pattern_end, 'close']) * 100

                        # Find breakout day (if any)
                        breakout_threshold = df.loc[pattern_end, 'close'] * 1.05  # 5% move
                        breakout_days = future_data[future_data['high'] > breakout_threshold]

                        if len(breakout_days) > 0:
                            pattern_data['breakout_occurred'] = True
                            pattern_data['days_to_breakout'] = breakout_days.index[0] - pattern_end
                            pattern_data['breakout_date'] = breakout_days.iloc[0]['date']
                            pattern_data['breakout_volume_spike'] = breakout_days.iloc[0]['volume'] / df.loc[pattern_end, 'volume_ma20']
                        else:
                            pattern_data['breakout_occurred'] = False
                            pattern_data['days_to_breakout'] = None
                            pattern_data['breakout_date'] = None
                            pattern_data['breakout_volume_spike'] = None

                        # Classify pattern outcome
                        if pattern_data['outcome_max_gain'] > 40:
                            pattern_data['outcome_class'] = 'EXPLOSIVE (40%+)'
                        elif pattern_data['outcome_max_gain'] > 20:
                            pattern_data['outcome_class'] = 'STRONG (20-40%)'
                        elif pattern_data['outcome_max_gain'] > 10:
                            pattern_data['outcome_class'] = 'MODERATE (10-20%)'
                        elif pattern_data['outcome_max_gain'] > 5:
                            pattern_data['outcome_class'] = 'WEAK (5-10%)'
                        else:
                            pattern_data['outcome_class'] = 'FAILED (<5%)'
                    else:
                        # No future data available
                        pattern_data['outcome_class'] = 'UNKNOWN (no data)'
                else:
                    # Too recent to evaluate
                    pattern_data['outcome_class'] = 'PENDING (too recent)'

                patterns.append(pattern_data)

        return patterns

    def analyze_ticker(self, ticker: str) -> List[Dict]:
        """Analyze all patterns for a single ticker"""
        try:
            # Find ticker file
            possible_paths = [
                f"{ticker}_full_history.csv",
                f"tickers/{ticker}_full_history.csv",
                f"tickers/{ticker}.csv"
            ]

            blob = None
            for path in possible_paths:
                test_blob = self.bucket.blob(path)
                if test_blob.exists():
                    blob = test_blob
                    break

            if not blob:
                return []

            # Read data
            csv_data = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_data))

            # Standardize
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

            # Need minimum data
            if len(df) < 50:
                return []

            # Detect patterns
            patterns = self.detect_consolidation_patterns(df, ticker)

            return patterns

        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return []

    def analyze_all_tickers(self, limit: int = None) -> pd.DataFrame:
        """Analyze all tickers and find ALL historical patterns"""

        # Get tickers
        logger.info("Getting ticker list...")
        tickers = []

        for blob in self.storage_client.list_blobs(self.bucket_name, prefix='tickers/'):
            if blob.name.endswith('.csv'):
                ticker = blob.name.replace('tickers/', '').replace('.csv', '').split('_')[0]
                if ticker.upper() == ticker and ticker.isalpha():
                    tickers.append(ticker)

        tickers = sorted(list(set(tickers)))[:limit] if limit else sorted(list(set(tickers)))

        logger.info(f"Analyzing {len(tickers)} tickers for historical patterns...")

        all_patterns = []

        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} tickers analyzed...")

            patterns = self.analyze_ticker(ticker)
            all_patterns.extend(patterns)

        # Create DataFrame
        df = pd.DataFrame(all_patterns)

        if not df.empty:
            # Add analysis columns (handle timezone-aware dates)
            try:
                df['pattern_year'] = pd.to_datetime(df['pattern_start_date'], utc=True).dt.year
                df['pattern_month'] = pd.to_datetime(df['pattern_start_date'], utc=True).dt.month
            except:
                df['pattern_year'] = pd.to_datetime(df['pattern_start_date']).dt.year
                df['pattern_month'] = pd.to_datetime(df['pattern_start_date']).dt.month

            # Calculate success rate
            successful = df[df['outcome_max_gain'] > 10]['ticker'].count() if 'outcome_max_gain' in df else 0
            total = len(df)

            logger.info(f"\nFound {total} consolidation patterns")
            logger.info(f"Success rate (>10% gain): {successful/total*100:.1f}%" if total > 0 else "N/A")

        return df

    def generate_report(self, df: pd.DataFrame):
        """Generate analysis report"""

        print("\n" + "="*80)
        print("HISTORICAL CONSOLIDATION PATTERN ANALYSIS")
        print("="*80)

        print(f"\nTotal patterns found: {len(df)}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"Date range: {df['pattern_start_date'].min()} to {df['pattern_start_date'].max()}")

        # Outcome distribution
        print("\n--- OUTCOME DISTRIBUTION ---")
        if 'outcome_class' in df:
            for outcome, count in df['outcome_class'].value_counts().items():
                pct = count/len(df)*100
                print(f"{outcome:20s}: {count:4d} ({pct:5.1f}%)")

        # Best performers
        print("\n--- TOP EXPLOSIVE PATTERNS (40%+ gains) ---")
        explosive = df[df['outcome_class'] == 'EXPLOSIVE (40%+)'].sort_values('outcome_max_gain', ascending=False).head(10)

        if not explosive.empty:
            for _, row in explosive.iterrows():
                print(f"{row['ticker']:6s} | {str(row['pattern_start_date'])[:10]} | "
                      f"Duration: {row['pattern_duration_days']:2d} days | "
                      f"Gain: {row['outcome_max_gain']:.1f}%")

        # Pattern characteristics of winners
        print("\n--- WINNING PATTERN CHARACTERISTICS ---")
        winners = df[df['outcome_max_gain'] > 20] if 'outcome_max_gain' in df else pd.DataFrame()

        if not winners.empty:
            print(f"Average duration: {winners['pattern_duration_days'].mean():.1f} days")
            print(f"Average BBW: {winners['avg_bbw'].mean():.3f}")
            print(f"Average volume ratio: {winners['avg_volume_ratio'].mean():.2f}")
            print(f"Average price range: {winners['price_range_pct'].mean():.1f}%")

        # Current/Recent patterns
        print("\n--- RECENT PATTERNS (Last 30 days) ---")
        recent_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent = df[pd.to_datetime(df['pattern_end_date']) > recent_date]

        if not recent.empty:
            for _, row in recent.head(10).iterrows():
                status = row.get('outcome_class', 'PENDING')
                print(f"{row['ticker']:6s} | {str(row['pattern_end_date'])[:10]} | "
                      f"Duration: {row['pattern_duration_days']:2d} days | "
                      f"Status: {status}")

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of tickers')
    parser.add_argument('--output', default='historical_patterns.parquet')
    parser.add_argument('--json', action='store_true', help='Also save as JSON')

    args = parser.parse_args()

    # Run analysis
    analyzer = HistoricalPatternAnalyzer()

    start_time = time.time()
    results = analyzer.analyze_all_tickers(limit=args.limit)
    elapsed = time.time() - start_time

    if not results.empty:
        # Save results
        results.to_parquet(args.output)
        logger.info(f"Results saved to {args.output}")

        if args.json:
            results.to_json(args.output.replace('.parquet', '.json'),
                          orient='records', indent=2, default_handler=str)

        # Generate report
        analyzer.generate_report(results)

        print(f"\nAnalysis completed in {elapsed:.1f} seconds")

        # Save summary statistics
        summary = {
            'total_patterns': len(results),
            'unique_tickers': results['ticker'].nunique(),
            'explosive_patterns': len(results[results['outcome_class'] == 'EXPLOSIVE (40%+)']) if 'outcome_class' in results else 0,
            'success_rate': len(results[results['outcome_max_gain'] > 10]) / len(results) * 100 if 'outcome_max_gain' in results and len(results) > 0 else 0,
            'analysis_date': datetime.now().isoformat()
        }

        with open('pattern_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        print("No patterns found")


if __name__ == "__main__":
    main()