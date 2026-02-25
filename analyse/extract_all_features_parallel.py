"""
OPTIMIZED Production Feature Extraction - Parallel Processing
5-10x faster than sequential version using ThreadPoolExecutor
Expected runtime: 30-90 minutes (vs 4-8 hours sequential)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Add AI Infra to path
sys.path.insert(0, str(Path(__file__).parent / "AI Infra"))

# Import from analyse core
from core import get_data_loader, UnifiedPatternDetector

# Import from AI Infra
try:
    from final_volume_pattern_system import VolumeFeatureEngine
    print("[OK] Imported VolumeFeatureEngine from AI Infra")
except ImportError as e:
    print(f"[ERROR] Failed to import VolumeFeatureEngine: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'extract_features_parallel_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_pattern_features(pattern_row, data_loader, feature_engine):
    """Extract features for a single pattern (thread-safe)."""
    ticker = pattern_row['ticker']
    start_date = pd.to_datetime(pattern_row['pattern_start_date'])
    end_date = pd.to_datetime(pattern_row['pattern_end_date'])

    try:
        buffer_days = 100
        load_start = start_date - timedelta(days=buffer_days)

        df = data_loader.load_ticker_data(
            ticker=ticker,
            start_date=load_start,
            end_date=end_date + timedelta(days=5)
        )

        if df.empty or len(df) < 30:
            return None

        df = df.reset_index()
        df['symbol'] = ticker
        df['timestamp'] = df['date'] if 'date' in df.columns else df.index

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        df_features, feature_cols = feature_engine.calculate_features(df)

        pattern_data = df_features[
            (df_features['timestamp'] >= start_date) &
            (df_features['timestamp'] <= end_date)
        ]

        if len(pattern_data) == 0:
            return None

        feature_dict = {
            'ticker': ticker,
            'pattern_start_date': start_date,
            'pattern_end_date': end_date,
            'duration_days': pattern_row['duration_days'],
            'outcome_class': pattern_row['outcome_class'],
            'outcome_max_gain': pattern_row['outcome_max_gain'],
            'breakout_occurred': pattern_row['breakout_occurred'],
        }

        for col in ['avg_bbw', 'avg_volume_ratio', 'avg_range_ratio', 'price_range_pct']:
            if col in pattern_row.index:
                feature_dict[col] = pattern_row[col]

        for col in feature_cols:
            if col in pattern_data.columns:
                if 'consec' in col or 'surge' in col or 'explosive' in col:
                    feature_dict[f'{col}'] = pattern_data[col].max()
                else:
                    feature_dict[f'{col}'] = pattern_data[col].mean()

        key_features = ['vol_ratio_5d', 'vol_strength_5d', 'accum_score_5d', 'obv_trend']
        for feat in key_features:
            if feat in pattern_data.columns:
                values = pattern_data[feat].values
                if len(values) > 1:
                    trend = (values[-1] - values[0]) / len(values)
                    feature_dict[f'{feat}_trend'] = trend

        return feature_dict

    except Exception as e:
        logger.debug(f"Error processing {ticker}: {e}")
        return None


def detect_patterns_for_ticker(ticker, detector):
    """Detect patterns for a single ticker (thread-safe)."""
    try:
        patterns = detector.detect_patterns(ticker)
        return ticker, patterns
    except Exception as e:
        logger.debug(f"Pattern detection failed for {ticker}: {e}")
        return ticker, None


def run_parallel_extraction(ticker_limit=None, min_target_patterns=10000, max_workers=None):
    """
    Run optimized parallel feature extraction.

    Args:
        ticker_limit: Max tickers to process (None = ALL)
        min_target_patterns: Pattern goal for logging (informational only, no early stopping)
        max_workers: Number of parallel workers (None = auto-detect)
    """

    if max_workers is None:
        max_workers = min(20, multiprocessing.cpu_count() * 2)
        logger.info(f"Auto-detected {max_workers} workers")

    logger.info("="*70)
    logger.info(f"OPTIMIZED PARALLEL EXTRACTION - {max_workers} Workers")
    logger.info("="*70)

    # Initialize components
    data_loader = get_data_loader()
    detector = UnifiedPatternDetector()
    feature_engine = VolumeFeatureEngine()

    # Get tickers
    logger.info("\nFetching available tickers from GCS...")
    all_tickers = data_loader.list_available_tickers()

    if ticker_limit:
        tickers_to_process = all_tickers[:ticker_limit]
    else:
        tickers_to_process = all_tickers

    logger.info(f"[OK] Found {len(all_tickers)} tickers in GCS")
    logger.info(f"[OK] Will process {len(tickers_to_process)} tickers")
    logger.info(f"[OK] Using {max_workers} parallel workers")

    # Step 1: PARALLEL Pattern Detection
    logger.info("\n" + "="*70)
    logger.info("STEP 1: PARALLEL PATTERN DETECTION")
    logger.info("="*70)

    all_patterns = []
    start_time = time.time()
    processed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all detection jobs
        future_to_ticker = {
            executor.submit(detect_patterns_for_ticker, ticker, detector): ticker
            for ticker in tickers_to_process
        }

        # Process results as they complete
        for future in as_completed(future_to_ticker):
            ticker, patterns = future.result()

            if patterns:
                all_patterns.extend(patterns)
                processed += 1

                if len(patterns) > 50:
                    logger.info(f"  {ticker}: {len(patterns)} patterns (high count)")
            else:
                failed += 1

            # Progress update every 50 tickers
            if (processed + failed) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (processed + failed) / elapsed
                remaining = (len(tickers_to_process) - processed - failed) / rate if rate > 0 else 0

                logger.info(f"\nProgress: {processed + failed}/{len(tickers_to_process)} "
                          f"({(processed + failed)/len(tickers_to_process)*100:.1f}%) | "
                          f"Patterns: {len(all_patterns)} | "
                          f"Rate: {rate:.1f} tickers/sec | "
                          f"ETA: {remaining/60:.1f} min")

            # Continue processing all tickers (no early stopping)

    elapsed_time = time.time() - start_time

    logger.info("\n" + "="*70)
    logger.info("PATTERN DETECTION RESULTS")
    logger.info("="*70)
    logger.info(f"Tickers processed: {processed}")
    logger.info(f"Tickers failed: {failed}")
    logger.info(f"Total patterns: {len(all_patterns)}")
    logger.info(f"Time: {elapsed_time/60:.1f} min ({elapsed_time/3600:.2f} hours)")
    logger.info(f"Rate: {processed/elapsed_time:.2f} tickers/sec")

    if len(all_patterns) == 0:
        logger.error("[ERROR] No patterns found!")
        return False

    # Convert patterns to DataFrame
    patterns_df = detector.patterns_to_dataframe(all_patterns)

    # Save intermediate patterns
    patterns_file = Path(f"output/patterns_parallel_{datetime.now():%Y%m%d_%H%M%S}.parquet")
    patterns_file.parent.mkdir(exist_ok=True)
    patterns_df.to_parquet(patterns_file)
    logger.info(f"[OK] Saved patterns: {patterns_file}")

    # Step 2: PARALLEL Feature Extraction
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PARALLEL FEATURE EXTRACTION")
    logger.info("="*70)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all feature extraction jobs
        futures = [
            executor.submit(extract_pattern_features, row, data_loader, feature_engine)
            for _, row in patterns_df.iterrows()
        ]

        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            features = future.result()

            if features is not None:
                results.append(features)

            # Progress update every 100 patterns
            if i > 0 and i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(patterns_df) - i) / rate if rate > 0 else 0
                success_rate = len(results) / i * 100

                logger.info(f"\nProgress: {i}/{len(patterns_df)} "
                          f"({i/len(patterns_df)*100:.1f}%) | "
                          f"Success: {success_rate:.1f}% | "
                          f"Rate: {rate:.1f} patterns/sec | "
                          f"ETA: {remaining/60:.1f} min")

    elapsed_time = time.time() - start_time

    logger.info("\n" + "="*70)
    logger.info("FEATURE EXTRACTION RESULTS")
    logger.info("="*70)
    logger.info(f"Patterns processed: {len(patterns_df)}")
    logger.info(f"Features extracted: {len(results)}")
    logger.info(f"Success rate: {len(results)/len(patterns_df)*100:.1f}%")
    logger.info(f"Time: {elapsed_time/60:.1f} min ({elapsed_time/3600:.2f} hours)")
    logger.info(f"Rate: {len(patterns_df)/elapsed_time:.2f} patterns/sec")

    if len(results) == 0:
        logger.error("[ERROR] No features extracted!")
        return False

    # Create features dataframe
    features_df = pd.DataFrame(results)

    # Validation
    logger.info("\n" + "="*70)
    logger.info("VALIDATION")
    logger.info("="*70)
    logger.info(f"Total columns: {len(features_df.columns)}")
    logger.info(f"Total rows: {len(features_df)}")

    nan_counts = features_df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    if len(features_with_nan) > 0:
        logger.warning(f"{len(features_with_nan)} features have NaN values")
    else:
        logger.info("[OK] No NaN values")

    # Save output
    output_file = Path(f"AI Infra/data/features/parallel_pattern_features_{datetime.now():%Y%m%d_%H%M%S}.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_file, index=False)

    logger.info("\n" + "="*70)
    logger.info("PARALLEL EXTRACTION COMPLETE")
    logger.info("="*70)
    logger.info(f"[OK] {len(results)} patterns with features")
    logger.info(f"[OK] Saved to: {output_file}")
    logger.info(f"[OK] File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    # Class distribution
    class_dist = features_df['outcome_class'].value_counts()
    logger.info("\n" + "="*70)
    logger.info("OUTCOME CLASS DISTRIBUTION")
    logger.info("="*70)
    for cls, count in class_dist.items():
        pct = count / len(features_df) * 100
        logger.info(f"{cls}: {count} ({pct:.1f}%)")

    # Performance summary
    total_time = (time.time() - start_time) / 3600
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Total time: {total_time:.2f} hours")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Throughput: {len(results)/total_time:.0f} patterns/hour")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Parallel feature extraction (OPTIMIZED)')
    parser.add_argument('--limit', type=int, default=None, help='Limit tickers (None = ALL)')
    parser.add_argument('--min-patterns', type=int, default=10000, help='Pattern goal for logging (informational only)')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (None = auto)')

    args = parser.parse_args()

    success = run_parallel_extraction(
        ticker_limit=args.limit,
        min_target_patterns=args.min_patterns,
        max_workers=args.workers
    )

    sys.exit(0 if success else 1)
