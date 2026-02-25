"""
Massive Pattern Scanner for Expanded GCS Dataset
================================================
Efficiently scans thousands of tickers for patterns to massively increase training data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Import from the module without the numeric prefix
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'pipeline'))
import importlib.util
spec = importlib.util.spec_from_file_location("scan_patterns_batch",
                                               str(Path(__file__).parent / 'src' / 'pipeline' / '01_scan_patterns_batch.py'))
scan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_module)
BatchPatternScanner = scan_module.BatchPatternScanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MassivePatternScanner:
    """
    Optimized pattern scanner for massive datasets.
    """

    def __init__(self):
        self.gcs_cache_dir = Path('data/massive_gcs_cache/tickers')
        self.local_cache_dir = Path('data_acquisition/storage')
        self.output_dir = Path('output/massive_patterns')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base scanner
        self.scanner = BatchPatternScanner(cache_dir=str(self.gcs_cache_dir))

    def get_available_tickers(self):
        """
        Get all available tickers from both GCS cache and local storage.
        """
        tickers = []

        # Get from GCS cache
        if self.gcs_cache_dir.exists():
            gcs_files = list(self.gcs_cache_dir.glob('*.parquet'))
            tickers.extend([f.stem for f in gcs_files])
            logger.info(f"Found {len(gcs_files)} tickers in GCS cache")

        # Get from local storage
        if self.local_cache_dir.exists():
            local_files = list(self.local_cache_dir.glob('*.parquet'))
            new_tickers = [f.stem for f in local_files if f.stem not in tickers]
            tickers.extend(new_tickers)
            logger.info(f"Found {len(new_tickers)} additional tickers in local storage")

        return list(set(tickers))  # Remove duplicates

    def scan_ticker_optimized(self, ticker):
        """
        Optimized single ticker scanning.
        """
        try:
            # Try GCS cache first
            ticker_file = self.gcs_cache_dir / f"{ticker}.parquet"
            if not ticker_file.exists():
                # Try local storage
                ticker_file = self.local_cache_dir / f"{ticker}.parquet"

            if not ticker_file.exists():
                return None

            # Load data
            df = pd.read_parquet(ticker_file)

            # Skip if insufficient data
            if len(df) < 252:  # Less than 1 year
                return None

            # Use optimized scanner
            patterns = self.scanner.scan_ticker(ticker)
            return patterns

        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            return None

    def parallel_scan(self, tickers, max_workers=None):
        """
        Scan tickers in parallel for maximum efficiency.
        """
        if max_workers is None:
            max_workers = min(cpu_count() * 2, 30)

        all_patterns = []
        failed_tickers = []

        print(f"\nScanning {len(tickers)} tickers with {max_workers} workers...")

        # Process in chunks to manage memory
        chunk_size = 100
        for chunk_start in range(0, len(tickers), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(tickers))
            chunk = tickers[chunk_start:chunk_end]

            chunk_patterns = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.scan_ticker_optimized, ticker): ticker
                          for ticker in chunk}

                with tqdm(total=len(futures),
                         desc=f"Chunk {chunk_start//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}") as pbar:
                    for future in as_completed(futures):
                        ticker = futures[future]
                        try:
                            result = future.result(timeout=60)
                            if result is not None:
                                chunk_patterns.append(result)
                        except Exception as e:
                            logger.debug(f"Failed to scan {ticker}: {e}")
                            failed_tickers.append(ticker)
                        pbar.update(1)

            # Add chunk results
            all_patterns.extend(chunk_patterns)

            # Memory cleanup
            gc.collect()

            # Save intermediate results every 500 tickers
            if len(all_patterns) > 0 and (chunk_end % 500 == 0 or chunk_end == len(tickers)):
                self.save_intermediate_results(all_patterns, chunk_end)

        return all_patterns, failed_tickers

    def save_intermediate_results(self, patterns, num_processed):
        """
        Save intermediate results to avoid data loss.
        """
        if patterns:
            df = pd.concat(patterns, ignore_index=True)
            intermediate_file = self.output_dir / f'patterns_intermediate_{num_processed}.parquet'
            df.to_parquet(intermediate_file)
            logger.info(f"Saved intermediate results: {len(df)} patterns from {num_processed} tickers")

    def analyze_pattern_distribution(self, df_patterns):
        """
        Analyze pattern distribution for K4 potential.
        """
        print("\n" + "="*80)
        print("PATTERN DISTRIBUTION ANALYSIS")
        print("="*80)

        # Basic statistics
        print(f"Total pattern snapshots: {len(df_patterns):,}")
        print(f"Unique patterns: {df_patterns['pattern_id'].nunique():,}")
        print(f"Unique tickers: {df_patterns['ticker'].nunique():,}")

        # Quality metrics
        quality_metrics = {
            'tight_range': (df_patterns['range_width_pct'] < 15).sum(),
            'moderate_range': ((df_patterns['range_width_pct'] >= 15) &
                              (df_patterns['range_width_pct'] < 25)).sum(),
            'mature_patterns': (df_patterns['days_in_pattern'] >= 20).sum(),
            'near_breakout': (df_patterns['distance_to_upper'] < 5).sum()
        }

        print("\nQuality Distribution:")
        for metric, count in quality_metrics.items():
            pct = count / len(df_patterns) * 100
            print(f"  {metric}: {count:,} ({pct:.1f}%)")

        # Estimate K4 patterns
        estimated_k4 = len(df_patterns) * 0.004  # Historical rate
        estimated_k3 = len(df_patterns) * 0.01

        print("\nExpected After Labeling:")
        print(f"  K4 patterns: ~{estimated_k4:,.0f}")
        print(f"  K3 patterns: ~{estimated_k3:,.0f}")
        print(f"  Improvement: {estimated_k4/450:.1f}x more K4s than current")

        return quality_metrics


def main():
    """
    Main function for massive pattern scanning.
    """
    print("\n" + "="*80)
    print("MASSIVE PATTERN SCANNING")
    print("="*80)

    scanner = MassivePatternScanner()

    # Get available tickers
    tickers = scanner.get_available_tickers()
    print(f"\nTotal tickers available: {len(tickers):,}")

    if len(tickers) == 0:
        print("\nNo tickers found! Please run massive_gcs_download.py first")
        return

    # User input
    print("\nScan options:")
    print(f"1. Quick test (100 tickers)")
    print(f"2. Small batch (500 tickers)")
    print(f"3. Medium batch (1000 tickers)")
    print(f"4. Large batch (2500 tickers)")
    print(f"5. All available ({len(tickers)} tickers)")
    print(f"6. Custom number")

    choice = input("\nSelect option (1-6, default 3): ").strip() or '3'

    if choice == '1':
        num_tickers = 100
    elif choice == '2':
        num_tickers = 500
    elif choice == '3':
        num_tickers = 1000
    elif choice == '4':
        num_tickers = 2500
    elif choice == '5':
        num_tickers = len(tickers)
    else:
        num_tickers = int(input(f"Enter number of tickers (max {len(tickers)}): "))

    tickers_to_scan = tickers[:num_tickers]
    print(f"\nWill scan {len(tickers_to_scan)} tickers")

    # Estimate time
    est_time = len(tickers_to_scan) * 2 / 30  # 2 seconds per ticker with 30 workers
    print(f"Estimated time: {est_time:.0f} minutes")

    # Start scanning
    start_time = datetime.now()
    print(f"\nStarting scan at {start_time:%H:%M:%S}...")

    all_patterns, failed = scanner.parallel_scan(tickers_to_scan)

    # Combine results
    if all_patterns:
        df_patterns = pd.concat(all_patterns, ignore_index=True)

        # Post-process
        quality_patterns = scanner.scanner.post_process_patterns(df_patterns)

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = scanner.output_dir / f'massive_patterns_{timestamp}.parquet'
        quality_patterns.to_parquet(output_file)

        # Analyze distribution
        scanner.analyze_pattern_distribution(quality_patterns)

        elapsed = datetime.now() - start_time
        print(f"\nScan completed in {elapsed}")
        print(f"Failed tickers: {len(failed)}")
        print(f"\nPatterns saved to: {output_file}")

        # Create race-labeling ready file
        label_file = scanner.output_dir / f'patterns_for_labeling_{timestamp}.parquet'
        quality_patterns.to_parquet(label_file)

        print("\n" + "="*80)
        print("NEXT STEPS FOR MASSIVE TRAINING")
        print("="*80)
        print("1. Label patterns with race logic:")
        print(f"   python src/pipeline/02_label_snapshots_race.py --input {label_file}")
        print("\n2. Train with massive dataset:")
        print("   python train_attention_ensemble.py")
        print("\n3. Expected results:")
        print(f"   - {quality_patterns['pattern_id'].nunique():,} unique patterns")
        print(f"   - {len(quality_patterns) * 0.004:,.0f}+ K4 patterns")
        print(f"   - {len(quality_patterns) * 0.004 / 450:.1f}x improvement in K4 detection")

        return quality_patterns
    else:
        print("No patterns found!")
        return None


if __name__ == "__main__":
    patterns = main()