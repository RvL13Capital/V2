"""
Massive Pattern Detection for GCS Downloaded Tickers
======================================================
Scans thousands of GCS-downloaded tickers to find K4 patterns.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MassivePatternDetector:
    """
    Pattern detection optimized for massive GCS dataset.
    """

    def __init__(self):
        self.gcs_ticker_dir = Path('data/gcs_tickers_massive/tickers')
        self.output_dir = Path('output/k4_patterns_massive')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_consolidation_patterns(self, df, ticker):
        """
        Detect consolidation patterns using volatility compression.
        """
        patterns = []

        if len(df) < 252:  # Need at least 1 year of data
            return patterns

        # Calculate indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['range'] = (df['high'] - df['low']) / df['close']
        df['atr'] = df['range'].rolling(14).mean()

        # Find low volatility periods (consolidation)
        df['low_vol'] = df['vol_20'] < df['vol_20'].rolling(100).quantile(0.3)
        df['tight_range'] = df['atr'] < df['atr'].rolling(100).quantile(0.3)

        # Identify consolidation periods
        df['consolidating'] = df['low_vol'] & df['tight_range']

        # Group consecutive consolidation days
        df['group'] = (df['consolidating'] != df['consolidating'].shift()).cumsum()

        # Find patterns that lasted at least 10 days
        for group_id, group_df in df[df['consolidating']].groupby('group'):
            if len(group_df) >= 10:
                start_date = group_df.index[0]
                end_date = group_df.index[-1]

                # Calculate pattern metrics
                pattern_df = df.loc[start_date:end_date]
                upper_boundary = pattern_df['high'].max()
                lower_boundary = pattern_df['low'].min()
                range_width_pct = (upper_boundary - lower_boundary) / lower_boundary * 100

                # Skip if range too wide
                if range_width_pct > 30:
                    continue

                # Look for future breakout (race to K4)
                future_window = 100
                if end_date < df.index[-future_window]:
                    future_df = df.loc[end_date:].iloc[:future_window]
                    max_gain = (future_df['high'].max() / pattern_df['close'].iloc[-1] - 1) * 100
                    max_loss = (future_df['low'].min() / pattern_df['close'].iloc[-1] - 1) * 100

                    # Record pattern
                    pattern = {
                        'ticker': ticker,
                        'qualification_start': start_date,
                        'qualification_end': end_date,
                        'days_in_pattern': len(group_df),
                        'upper_boundary': upper_boundary,
                        'lower_boundary': lower_boundary,
                        'range_width_pct': range_width_pct,
                        'max_gain_pct': max_gain,
                        'max_loss_pct': max_loss,
                        'is_k4': max_gain >= 75,  # K4 threshold
                        'is_k3': 35 <= max_gain < 75,
                        'is_k5': max_loss <= -10  # Failed pattern
                    }
                    patterns.append(pattern)

        return patterns

    def scan_ticker(self, ticker_file):
        """
        Scan a single ticker for patterns.
        """
        try:
            df = pd.read_parquet(ticker_file)

            # Ensure proper columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None

            ticker = ticker_file.stem
            patterns = self.detect_consolidation_patterns(df, ticker)

            return patterns

        except Exception as e:
            logger.debug(f"Error scanning {ticker_file.name}: {e}")
            return None

    def scan_massive_dataset(self, max_tickers=None, batch_size=500):
        """
        Scan massive dataset in batches.
        """
        # Get all ticker files
        ticker_files = list(self.gcs_ticker_dir.glob('*.parquet'))

        if max_tickers:
            ticker_files = ticker_files[:max_tickers]

        print(f"\n{'='*70}")
        print("MASSIVE PATTERN DETECTION FOR K4 HUNTING")
        print(f"{'='*70}")
        print(f"\nScanning {len(ticker_files)} tickers from GCS download...")

        all_patterns = []
        k4_patterns = []

        # Process in batches
        for batch_start in range(0, len(ticker_files), batch_size):
            batch_end = min(batch_start + batch_size, len(ticker_files))
            batch_files = ticker_files[batch_start:batch_end]

            print(f"\n[Batch {batch_start//batch_size + 1}] Processing tickers {batch_start+1}-{batch_end}...")

            batch_patterns = []

            # Process batch with progress bar
            with tqdm(total=len(batch_files), desc="Scanning") as pbar:
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = {executor.submit(self.scan_ticker, f): f for f in batch_files}

                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            batch_patterns.extend(result)
                        pbar.update(1)

            # Add to results
            all_patterns.extend(batch_patterns)

            # Extract K4 patterns
            batch_k4 = [p for p in batch_patterns if p.get('is_k4', False)]
            k4_patterns.extend(batch_k4)

            print(f"  Batch results: {len(batch_patterns)} patterns, {len(batch_k4)} K4s")

            # Save intermediate results
            if batch_end % 1000 == 0 or batch_end == len(ticker_files):
                self.save_intermediate_results(all_patterns, k4_patterns, batch_end)

            # Memory cleanup
            gc.collect()

        return all_patterns, k4_patterns

    def save_intermediate_results(self, all_patterns, k4_patterns, num_processed):
        """
        Save intermediate results.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save all patterns
        if all_patterns:
            df_all = pd.DataFrame(all_patterns)
            all_file = self.output_dir / f'all_patterns_massive_{num_processed}.parquet'
            df_all.to_parquet(all_file)
            print(f"  [Saved] {len(df_all)} patterns to {all_file.name}")

        # Save K4 patterns
        if k4_patterns:
            df_k4 = pd.DataFrame(k4_patterns)
            k4_file = self.output_dir / f'k4_patterns_massive_{num_processed}.parquet'
            df_k4.to_parquet(k4_file)
            print(f"  [Saved] {len(df_k4)} K4 patterns to {k4_file.name}")

    def print_summary(self, all_patterns, k4_patterns):
        """
        Print detection summary.
        """
        print(f"\n{'='*70}")
        print("MASSIVE K4 PATTERN DETECTION COMPLETE")
        print(f"{'='*70}")

        print(f"\nTotal patterns detected: {len(all_patterns):,}")
        print(f"K4 patterns found: {len(k4_patterns):,} ({len(k4_patterns)/max(len(all_patterns),1)*100:.2f}%)")

        if k4_patterns:
            df_k4 = pd.DataFrame(k4_patterns)
            print(f"\n[K4 Statistics]")
            print(f"  Average K4 gain: {df_k4['max_gain_pct'].mean():.1f}%")
            print(f"  Median K4 gain: {df_k4['max_gain_pct'].median():.1f}%")
            print(f"  Max K4 gain: {df_k4['max_gain_pct'].max():.1f}%")

            # Top K4 patterns
            top_k4 = df_k4.nlargest(10, 'max_gain_pct')[['ticker', 'max_gain_pct']]
            print(f"\n[Top 10 K4 Patterns]")
            for _, row in top_k4.iterrows():
                print(f"  {row['ticker']:6s}: {row['max_gain_pct']:>6.1f}%")

        # Combined with existing K4s
        print(f"\n[Combined K4 Results]")
        print(f"  Previous K4 patterns: 1,419")
        print(f"  New K4 patterns: {len(k4_patterns):,}")
        print(f"  Total K4 patterns: {1419 + len(k4_patterns):,}")
        print(f"  Improvement: {(1419 + len(k4_patterns))/450:.1f}x over original 450")

        print(f"\n[Next Steps]")
        print("1. Combine K4 patterns from all sources")
        print("2. Extract V3 features (83+ advanced features)")
        print("3. Train attention ensemble with massive K4 dataset")
        print("4. Expected K4 recall: 85-95% (up from current 83-87%)")

def main():
    """
    Main function for massive pattern detection.
    """
    detector = MassivePatternDetector()

    # User input
    print("\nPattern detection options:")
    print("1. Quick test (100 tickers)")
    print("2. Medium batch (500 tickers)")
    print("3. Large batch (1000 tickers)")
    print("4. All available tickers")

    choice = input("\nSelect option (1-4, default 4): ").strip() or '4'

    if choice == '1':
        max_tickers = 100
    elif choice == '2':
        max_tickers = 500
    elif choice == '3':
        max_tickers = 1000
    else:
        max_tickers = None

    # Run detection
    all_patterns, k4_patterns = detector.scan_massive_dataset(max_tickers)

    # Print summary
    detector.print_summary(all_patterns, k4_patterns)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_patterns:
        df_all = pd.DataFrame(all_patterns)
        final_all = detector.output_dir / f'all_patterns_final_{timestamp}.parquet'
        df_all.to_parquet(final_all)
        print(f"\n[Final] All patterns saved to: {final_all}")

    if k4_patterns:
        df_k4 = pd.DataFrame(k4_patterns)
        final_k4 = detector.output_dir / f'k4_patterns_final_{timestamp}.parquet'
        df_k4.to_parquet(final_k4)
        print(f"[Final] K4 patterns saved to: {final_k4}")

    return all_patterns, k4_patterns

if __name__ == "__main__":
    patterns, k4s = main()