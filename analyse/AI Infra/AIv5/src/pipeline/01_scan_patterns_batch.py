"""
Batch Pattern Scanning for Expanded Dataset
===========================================
Efficiently scans patterns across large ticker datasets from GCS.
Optimized for finding more K4 patterns.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_detection.state_machine.modern_tracker import ModernPatternTracker
from shared.config.settings import get_settings
from shared.utils.data_loader import DataLoader
from shared.utils.memory_utils import optimize_memory
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchPatternScanner:
    """
    Batch scanner for detecting patterns across many tickers.
    """

    def __init__(self, cache_dir: str = 'data/expanded_gcs_cache/tickers'):
        self.cache_dir = Path(cache_dir)
        self.settings = get_settings()
        self.output_dir = Path('output/batch_patterns')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Scan single ticker for patterns.
        """
        try:
            # Load ticker data
            ticker_file = self.cache_dir / f"{ticker}.parquet"
            if not ticker_file.exists():
                return None

            df = pd.read_parquet(ticker_file)

            # Ensure proper columns
            if 'adj_close' in df.columns and 'close' not in df.columns:
                df['close'] = df['adj_close']

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None

            # Initialize tracker
            tracker = ModernPatternTracker(
                ticker=ticker,
                bbw_threshold=self.settings.consolidation.bbw_percentile_threshold,
                adx_threshold=self.settings.consolidation.adx_threshold,
                volume_threshold=self.settings.consolidation.volume_ratio_threshold,
                range_threshold=self.settings.consolidation.range_ratio_threshold
            )

            # Process data day by day
            patterns = []
            for date, row in df.iterrows():
                state_change, pattern = tracker.update(
                    date=date,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    df_history=df[:date]
                )

                # Capture snapshots of active patterns
                if tracker.state == 'ACTIVE' and pattern:
                    # Take snapshots at different stages
                    days_active = (date - pattern.pattern_start).days

                    # Snapshot every 5 days and at key points
                    if days_active in [0, 5, 10, 15, 20, 30, 40, 50, 60]:
                        snapshot = {
                            'ticker': ticker,
                            'pattern_id': f"{ticker}_{pattern.pattern_start:%Y%m%d}",
                            'start_date': pattern.pattern_start,
                            'qualification_start': pattern.qualification_start,
                            'qualification_end': pattern.qualification_end,
                            'snapshot_date': date,
                            'days_in_pattern': days_active,
                            'days_qualifying': 10,
                            'upper_boundary': pattern.upper_boundary,
                            'lower_boundary': pattern.lower_boundary,
                            'power_boundary': pattern.power_boundary,
                            'range_width': pattern.upper_boundary - pattern.lower_boundary,
                            'range_width_pct': ((pattern.upper_boundary - pattern.lower_boundary) /
                                              pattern.lower_boundary * 100),
                            'current_price': row['close'],
                            'current_volume': row['volume'],
                            'distance_to_upper': ((pattern.upper_boundary - row['close']) /
                                                pattern.upper_boundary * 100),
                            'distance_to_lower': ((row['close'] - pattern.lower_boundary) /
                                                pattern.lower_boundary * 100)
                        }
                        patterns.append(snapshot)

            if patterns:
                return pd.DataFrame(patterns)
            return None

        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            return None

    def scan_batch(self, tickers: List[str], max_workers: int = 4) -> pd.DataFrame:
        """
        Scan multiple tickers in parallel.
        """
        all_patterns = []
        batch_size = 50  # Process in smaller batches to manage memory

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

            batch_patterns = []
            for ticker in batch:
                result = self.scan_ticker(ticker)
                if result is not None:
                    batch_patterns.append(result)

            if batch_patterns:
                all_patterns.extend(batch_patterns)

            # Memory cleanup
            gc.collect()

        if all_patterns:
            return pd.concat(all_patterns, ignore_index=True)
        return pd.DataFrame()

    def post_process_patterns(self, df_patterns: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process detected patterns for quality.
        """
        if df_patterns.empty:
            return df_patterns

        # Calculate additional metrics
        df_patterns['price_position'] = ((df_patterns['current_price'] - df_patterns['lower_boundary']) /
                                        (df_patterns['upper_boundary'] - df_patterns['lower_boundary']))

        # Filter quality patterns
        quality_mask = (
            (df_patterns['range_width_pct'] < 30) &  # Tight consolidation
            (df_patterns['days_in_pattern'] >= 10) &  # Mature pattern
            (df_patterns['days_in_pattern'] <= 90)    # Not too old
        )

        quality_patterns = df_patterns[quality_mask].copy()

        # Add pattern quality score
        quality_patterns['quality_score'] = (
            (1 - quality_patterns['range_width_pct'] / 100) * 0.4 +  # Tightness
            np.clip(quality_patterns['days_in_pattern'] / 30, 0, 1) * 0.3 +  # Maturity
            quality_patterns['price_position'] * 0.3  # Position in range
        )

        return quality_patterns


def main():
    """
    Main batch scanning function.
    """
    logger.info("="*70)
    logger.info("BATCH PATTERN SCANNING FOR EXPANDED DATASET")
    logger.info("="*70)

    # Initialize scanner
    scanner = BatchPatternScanner()

    # Get ticker list
    ticker_files = list(scanner.cache_dir.glob("*.parquet"))

    if not ticker_files:
        logger.error("No ticker data found in cache directory!")
        logger.info("Please run download_expanded_gcs_data.py first")
        return

    tickers = [f.stem for f in ticker_files]
    logger.info(f"Found {len(tickers)} tickers to scan")

    # User input
    max_tickers = input(f"\nHow many tickers to scan? (max {len(tickers)}, default all): ").strip()
    if max_tickers:
        tickers = tickers[:int(max_tickers)]

    logger.info(f"\nScanning {len(tickers)} tickers for patterns...")

    # Scan patterns
    start_time = datetime.now()
    df_patterns = scanner.scan_batch(tickers)
    elapsed = datetime.now() - start_time

    if df_patterns.empty:
        logger.warning("No patterns detected!")
        return

    logger.info(f"\nDetected {len(df_patterns)} pattern snapshots in {elapsed}")

    # Post-process
    quality_patterns = scanner.post_process_patterns(df_patterns)
    logger.info(f"Quality patterns after filtering: {len(quality_patterns)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = scanner.output_dir / f'patterns_batch_{timestamp}.parquet'
    quality_patterns.to_parquet(output_file, index=False)

    # Display statistics
    logger.info("\n" + "="*70)
    logger.info("PATTERN STATISTICS")
    logger.info("="*70)
    logger.info(f"Total snapshots: {len(quality_patterns)}")
    logger.info(f"Unique patterns: {quality_patterns['pattern_id'].nunique()}")
    logger.info(f"Unique tickers: {quality_patterns['ticker'].nunique()}")

    # Pattern age distribution
    age_dist = quality_patterns['days_in_pattern'].value_counts(bins=5).sort_index()
    logger.info("\nPattern age distribution:")
    for interval in age_dist.index:
        logger.info(f"  {interval}: {age_dist[interval]} patterns")

    # Range width distribution
    logger.info(f"\nRange width statistics:")
    logger.info(f"  Mean: {quality_patterns['range_width_pct'].mean():.2f}%")
    logger.info(f"  Median: {quality_patterns['range_width_pct'].median():.2f}%")
    logger.info(f"  Min: {quality_patterns['range_width_pct'].min():.2f}%")
    logger.info(f"  Max: {quality_patterns['range_width_pct'].max():.2f}%")

    # Top quality patterns
    top_patterns = quality_patterns.nlargest(20, 'quality_score')[
        ['ticker', 'snapshot_date', 'days_in_pattern', 'range_width_pct', 'quality_score']
    ]
    logger.info("\nTop 20 quality patterns:")
    for _, row in top_patterns.iterrows():
        logger.info(f"  {row['ticker']:6s} | Day {row['days_in_pattern']:2.0f} | "
                   f"Range: {row['range_width_pct']:5.2f}% | Score: {row['quality_score']:.3f}")

    logger.info(f"\nPatterns saved to: {output_file}")

    # Create ready-to-label file
    label_ready_file = scanner.output_dir / f'patterns_ready_to_label_{timestamp}.parquet'
    quality_patterns.to_parquet(label_ready_file, index=False)

    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Label patterns with race logic:")
    logger.info(f"   python src/pipeline/02_label_snapshots_race.py --input {label_ready_file}")
    logger.info("\n2. Extract features and train models:")
    logger.info("   python src/pipeline/03_train_enhanced.py")
    logger.info("\n3. Or use attention ensemble:")
    logger.info("   python train_attention_ensemble.py")

    return quality_patterns


if __name__ == "__main__":
    patterns = main()