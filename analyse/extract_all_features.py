"""
Production Feature Extraction - Full GCS Dataset
Scales up test pipeline to process ALL available tickers from GCS
Target: 10,000+ patterns from 3,548+ tickers
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging

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
        logging.FileHandler(f'extract_features_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_pattern_features(pattern_row, data_loader, feature_engine):
    """
    Extract features for a single pattern.
    Returns: Dict of features or None if failed
    """
    ticker = pattern_row['ticker']
    start_date = pd.to_datetime(pattern_row['pattern_start_date'])
    end_date = pd.to_datetime(pattern_row['pattern_end_date'])

    try:
        # Load ticker data with buffer for indicator calculation
        buffer_days = 100
        load_start = start_date - timedelta(days=buffer_days)

        # Get data from GCS/cache
        df = data_loader.load_ticker_data(
            ticker=ticker,
            start_date=load_start,
            end_date=end_date + timedelta(days=5)
        )

        if df.empty or len(df) < 30:
            return None

        # Prepare data for feature engine
        df = df.reset_index()
        df['symbol'] = ticker
        df['timestamp'] = df['date'] if 'date' in df.columns else df.index

        # Ensure required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # Calculate features using VolumeFeatureEngine
        df_features, feature_cols = feature_engine.calculate_features(df)

        # Filter to pattern period only
        pattern_data = df_features[
            (df_features['timestamp'] >= start_date) &
            (df_features['timestamp'] <= end_date)
        ]

        if len(pattern_data) == 0:
            return None

        # Aggregate features over pattern duration
        feature_dict = {
            'ticker': ticker,
            'pattern_start_date': start_date,
            'pattern_end_date': end_date,
            'duration_days': pattern_row['duration_days'],
            'outcome_class': pattern_row['outcome_class'],
            'outcome_max_gain': pattern_row['outcome_max_gain'],
            'breakout_occurred': pattern_row['breakout_occurred'],
        }

        # Add pattern metadata
        for col in ['avg_bbw', 'avg_volume_ratio', 'avg_range_ratio', 'price_range_pct']:
            if col in pattern_row.index:
                feature_dict[col] = pattern_row[col]

        # Aggregate volume features
        for col in feature_cols:
            if col in pattern_data.columns:
                if 'consec' in col or 'surge' in col or 'explosive' in col:
                    feature_dict[f'{col}'] = pattern_data[col].max()
                else:
                    feature_dict[f'{col}'] = pattern_data[col].mean()

        # Add feature trends
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


def run_full_extraction(ticker_limit=None, min_target_patterns=10000):
    """
    Run full production feature extraction.

    Args:
        ticker_limit: Max number of tickers to process (None = ALL)
        min_target_patterns: Pattern goal for logging (informational only, no early stopping)
    """

    logger.info("="*70)
    logger.info("PRODUCTION FEATURE EXTRACTION - Full GCS Dataset")
    logger.info("="*70)

    # Initialize components
    data_loader = get_data_loader()
    detector = UnifiedPatternDetector()
    feature_engine = VolumeFeatureEngine()

    # Get ALL available tickers from GCS
    logger.info("\nFetching available tickers from GCS...")
    all_tickers = data_loader.list_available_tickers()

    if ticker_limit:
        tickers_to_process = all_tickers[:ticker_limit]
    else:
        tickers_to_process = all_tickers

    logger.info(f"[OK] Found {len(all_tickers)} tickers in GCS")
    logger.info(f"[OK] Will process {len(tickers_to_process)} tickers")

    # Step 1: Detect patterns for all tickers
    logger.info("\n" + "="*70)
    logger.info("STEP 1: PATTERN DETECTION")
    logger.info("="*70)

    all_patterns = []
    processed_count = 0
    failed_count = 0
    start_time = time.time()

    for i, ticker in enumerate(tickers_to_process):
        try:
            # Progress update every 50 tickers
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(tickers_to_process) - i) / rate if rate > 0 else 0
                logger.info(f"\nProgress: {i}/{len(tickers_to_process)} tickers "
                          f"({i/len(tickers_to_process)*100:.1f}%) | "
                          f"{len(all_patterns)} patterns found | "
                          f"Rate: {rate:.1f} tickers/sec | "
                          f"ETA: {remaining/60:.1f} min")

            # Detect patterns for this ticker
            patterns = detector.detect_patterns(ticker)

            if patterns:
                all_patterns.extend(patterns)
                processed_count += 1

                # Log high-pattern tickers
                if len(patterns) > 50:
                    logger.info(f"  {ticker}: {len(patterns)} patterns (high count)")

        except Exception as e:
            failed_count += 1
            logger.debug(f"Failed to process {ticker}: {e}")
            continue

        # Continue processing all tickers (no early stopping)

    elapsed_time = time.time() - start_time

    logger.info("\n" + "="*70)
    logger.info("PATTERN DETECTION RESULTS")
    logger.info("="*70)
    logger.info(f"Tickers processed: {processed_count}")
    logger.info(f"Tickers failed: {failed_count}")
    logger.info(f"Total patterns found: {len(all_patterns)}")
    logger.info(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    logger.info(f"Processing rate: {processed_count/elapsed_time:.2f} tickers/sec")

    if len(all_patterns) == 0:
        logger.error("[ERROR] No patterns found!")
        return False

    # Convert patterns to DataFrame
    patterns_df = detector.patterns_to_dataframe(all_patterns)

    # Save intermediate patterns file
    patterns_file = Path(f"output/patterns_full_gcs_{datetime.now():%Y%m%d_%H%M%S}.parquet")
    patterns_file.parent.mkdir(exist_ok=True)
    patterns_df.to_parquet(patterns_file)
    logger.info(f"[OK] Saved patterns to {patterns_file}")

    # Step 2: Extract features for all patterns
    logger.info("\n" + "="*70)
    logger.info("STEP 2: FEATURE EXTRACTION")
    logger.info("="*70)

    results = []
    start_time = time.time()

    for idx, row in patterns_df.iterrows():
        # Progress update every 100 patterns
        if idx > 0 and idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (len(patterns_df) - idx) / rate if rate > 0 else 0
            success_rate = len(results) / idx * 100
            logger.info(f"\nProgress: {idx}/{len(patterns_df)} patterns "
                      f"({idx/len(patterns_df)*100:.1f}%) | "
                      f"Success: {success_rate:.1f}% | "
                      f"ETA: {remaining/60:.1f} min")

        features = extract_pattern_features(row, data_loader, feature_engine)
        if features is not None:
            results.append(features)

    elapsed_time = time.time() - start_time

    logger.info("\n" + "="*70)
    logger.info("FEATURE EXTRACTION RESULTS")
    logger.info("="*70)
    logger.info(f"Patterns processed: {len(patterns_df)}")
    logger.info(f"Features extracted: {len(results)}")
    logger.info(f"Success rate: {len(results)/len(patterns_df)*100:.1f}%")
    logger.info(f"Time elapsed: {elapsed_time/60:.1f} minutes")

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

    # Check for NaN values
    nan_counts = features_df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    if len(features_with_nan) > 0:
        logger.warning(f"{len(features_with_nan)} features have NaN values")
    else:
        logger.info("[OK] No NaN values in features")

    # Save output
    output_file = Path(f"AI Infra/data/features/full_pattern_features_{datetime.now():%Y%m%d_%H%M%S}.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_file, index=False)

    logger.info("\n" + "="*70)
    logger.info("PRODUCTION EXTRACTION COMPLETE")
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

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from full GCS dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers (default: process ALL)')
    parser.add_argument('--min-patterns', type=int, default=10000, help='Pattern goal for logging (informational only)')

    args = parser.parse_args()

    success = run_full_extraction(
        ticker_limit=args.limit,
        min_target_patterns=args.min_patterns
    )

    sys.exit(0 if success else 1)
