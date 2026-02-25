"""
Improved Feature Extraction with Better Error Handling
========================================================

This improved version handles date-related errors more gracefully and provides
better progress tracking. It addresses issues with international tickers that
may have sparse or missing data.

Improvements:
- Better error handling for NaT (Not a Time) errors
- Progress tracking with success/failure counts
- Continues processing even when some patterns fail
- More informative error messages
- Saves partial results periodically

Expected Runtime: 30-45 minutes for all snapshots
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

from pattern_detection.features.canonical_feature_extractor import CanonicalFeatureExtractor
from pattern_detection.models.pattern import ConsolidationPattern, PatternPhase

# =============================================================================
# CONFIGURATION
# =============================================================================

HISTORICAL_PATTERNS_FILE = Path('output/unused_patterns/pattern_features_historical.parquet')
CACHE_DIR = Path('data/unused_tickers_cache')
OUTPUT_DIR = Path('output/unused_patterns')
OUTPUT_FILE = OUTPUT_DIR / 'pattern_features_canonical_69.parquet'
CHECKPOINT_FILE = OUTPUT_DIR / 'extraction_checkpoint.parquet'

# Get canonical feature list
extractor = CanonicalFeatureExtractor()
MODEL_EXPECTED_FEATURES = extractor.get_feature_names()  # Exactly 69 features
NUM_FEATURES = len(MODEL_EXPECTED_FEATURES)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load cached OHLCV data for a ticker."""
    cache_file = CACHE_DIR / f"{ticker}.parquet"

    if not cache_file.exists():
        return None

    try:
        df = pd.read_parquet(cache_file)

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif df.index.name == 'date':
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()

        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['date', 'close'])

        return df

    except Exception as e:
        logger.error(f"  Error loading {ticker}: {str(e)[:100]}")
        return None


def safe_extract_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    pattern: ConsolidationPattern,
    snapshot_idx: int
) -> dict:
    """
    Safely extract features with NaT handling.

    This wraps the canonical feature extractor to handle edge cases
    where certain calculations may fail due to sparse data.
    """
    try:
        # Create a modified extractor that handles NaT
        extractor = CanonicalFeatureExtractor()

        # Extract features
        features = extractor.extract_snapshot_features(
            df,
            snapshot_date,
            pattern,
            snapshot_idx
        )

        # Fill any NaN values with appropriate defaults
        for feature_name in MODEL_EXPECTED_FEATURES:
            if feature_name not in features:
                # Add missing feature with default value
                if 'ratio' in feature_name or 'pct' in feature_name:
                    features[feature_name] = 1.0
                elif 'days' in feature_name or 'count' in feature_name:
                    features[feature_name] = 0
                else:
                    features[feature_name] = 0.0
            elif pd.isna(features[feature_name]):
                # Replace NaN with appropriate default
                if 'ratio' in feature_name or 'pct' in feature_name:
                    features[feature_name] = 1.0
                elif 'days' in feature_name or 'count' in feature_name:
                    features[feature_name] = 0
                else:
                    features[feature_name] = 0.0

        return features

    except Exception as e:
        # Log the specific error for debugging
        error_msg = str(e)
        if 'NaT' in error_msg:
            logger.debug(f"    NaT error in feature extraction: {error_msg[:100]}")
        else:
            logger.debug(f"    Feature extraction error: {error_msg[:100]}")

        # Return None to indicate failure
        return None


def calculate_pattern_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    active_start: pd.Timestamp,
    upper_boundary: float,
    lower_boundary: float,
    power_boundary: float,
    start_date: pd.Timestamp,
    start_price: float
) -> dict:
    """
    Calculate all 69 canonical features for a pattern snapshot.

    Improved version with better error handling.
    """
    # Create a ConsolidationPattern object with the boundaries
    pattern = ConsolidationPattern(
        ticker='VALIDATION',
        start_date=start_date,
        start_idx=0,
        start_price=start_price,
        phase=PatternPhase.ACTIVE,
        activation_date=active_start,
        activation_idx=10  # Approximate
    )

    # Add boundaries as attributes
    pattern.upper_boundary = upper_boundary
    pattern.lower_boundary = lower_boundary
    pattern.power_boundary = power_boundary

    # Set up the DataFrame with proper index
    if 'date' in df.columns:
        df = df.set_index('date')

    # Calculate indicators
    df = calculate_indicators(df)

    # Find snapshot index
    try:
        if snapshot_date not in df.index:
            # Find closest date
            closest_idx = df.index.get_indexer([snapshot_date], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(df):
                actual_date = df.index[closest_idx]
                # Only use if within 5 days
                if abs((actual_date - snapshot_date).days) <= 5:
                    snapshot_idx = closest_idx
                    logger.debug(f"    Using closest date {actual_date} for snapshot {snapshot_date}")
                else:
                    logger.debug(f"    Snapshot date {snapshot_date} too far from data")
                    return None
            else:
                return None
        else:
            snapshot_idx = df.index.get_loc(snapshot_date)
    except Exception as e:
        logger.debug(f"    Error finding snapshot date: {str(e)[:100]}")
        return None

    # Extract features with improved error handling
    features = safe_extract_features(df, snapshot_date, pattern, snapshot_idx)

    return features


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculate technical indicators needed by the canonical extractor.

    Improved version with NaN handling.
    """
    from shared.indicators.technical import (
        calculate_bbw,
        calculate_adx,
        calculate_volume_ratio,
        get_bbw_percentile
    )

    # BBW and percentile
    df['bbw_20'] = calculate_bbw(df, period=20)
    df['bbw_percentile'] = get_bbw_percentile(df, window=100)

    # ADX
    df['adx'] = calculate_adx(df, period=14)

    # Volume ratio
    if 'volume' in df.columns:
        df['volume_ratio_20'] = calculate_volume_ratio(df, period=20)
    else:
        df['volume_ratio_20'] = 1.0

    # Range ratio
    if all(col in df.columns for col in ['high', 'low']):
        df['daily_range'] = df['high'] - df['low']
        avg_range_20 = df['daily_range'].rolling(20, min_periods=10).mean()
        df['range_ratio'] = df['daily_range'] / (avg_range_20 + 1e-10)
    else:
        df['range_ratio'] = 1.0

    # Fill NaN values with appropriate defaults
    df = df.fillna(method='ffill').fillna(method='bfill')

    # For any remaining NaNs, use column-specific defaults
    for col in df.columns:
        if df[col].isna().any():
            if 'ratio' in col or 'pct' in col:
                df[col] = df[col].fillna(1.0)
            elif 'volume' in col:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("IMPROVED CANONICAL FEATURE EXTRACTION")
    print("With Enhanced Error Handling")
    print("=" * 80)

    # Check if historical patterns file exists
    if not HISTORICAL_PATTERNS_FILE.exists():
        print(f"\n[ERROR] Historical patterns file not found: {HISTORICAL_PATTERNS_FILE}")
        exit(1)

    # Load historical patterns
    print(f"\n[STEP 1] Loading historical pattern snapshots...")
    patterns_df = pd.read_parquet(HISTORICAL_PATTERNS_FILE)
    total_patterns = len(patterns_df)
    print(f"  Total snapshots: {total_patterns:,}")
    print(f"  Unique tickers: {patterns_df['ticker'].nunique()}")

    # Convert date columns to datetime
    date_cols = ['qualification_start', 'active_start', 'end_date', 'snapshot_date']
    for col in date_cols:
        if col in patterns_df.columns:
            patterns_df[col] = pd.to_datetime(patterns_df[col])

    # Check for checkpoint
    start_idx = 0
    if CHECKPOINT_FILE.exists():
        print(f"\n[INFO] Found checkpoint file, loading previous results...")
        checkpoint_df = pd.read_parquet(CHECKPOINT_FILE)
        start_idx = len(checkpoint_df)
        print(f"  Resuming from pattern {start_idx}/{total_patterns}")
        results = checkpoint_df.to_dict('records')
    else:
        results = []

    # Initialize counters
    success_count = start_idx
    fail_count = 0
    error_types = {}

    # Process each snapshot
    print(f"\n[STEP 2] Extracting {NUM_FEATURES} canonical features...")
    print(f"  Processing patterns {start_idx+1} to {total_patterns}")
    print(f"  Estimated time: {(total_patterns - start_idx) * 3 / 60:.0f} minutes")
    print("")

    for idx in range(start_idx, total_patterns):
        row = patterns_df.iloc[idx]

        # Progress indicator
        if (idx + 1) % 10 == 0:
            success_rate = success_count / (idx + 1) * 100 if idx > 0 else 0
            print(f"  Progress: {idx+1}/{total_patterns} ({(idx+1)/total_patterns*100:.1f}%) "
                  f"| Success: {success_count} ({success_rate:.1f}%) | Failed: {fail_count}")

        ticker = row['ticker']
        snapshot_date = row['snapshot_date']
        active_start = row.get('active_start', row.get('qualification_start'))
        start_date = row.get('qualification_start', active_start)
        upper_boundary = row['upper_boundary']
        lower_boundary = row['lower_boundary']
        power_boundary = row['power_boundary']

        # Load ticker data
        df = load_ticker_data(ticker)

        if df is None:
            fail_count += 1
            error_types['no_data'] = error_types.get('no_data', 0) + 1
            continue

        # Filter to temporal-safe data (only dates <= snapshot_date)
        df = df[df['date'] <= snapshot_date].copy()

        if len(df) < 100:
            fail_count += 1
            error_types['insufficient_data'] = error_types.get('insufficient_data', 0) + 1
            continue

        # Get start price
        start_price_candidates = df[df['date'] == start_date]['close']
        if len(start_price_candidates) > 0:
            start_price = start_price_candidates.iloc[0]
        else:
            # Fallback: use close price from first available date
            start_price = df['close'].iloc[0]

        # Calculate features
        features = calculate_pattern_features(
            df=df,
            snapshot_date=snapshot_date,
            active_start=active_start,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            power_boundary=power_boundary,
            start_date=start_date,
            start_price=start_price
        )

        if features is None:
            fail_count += 1
            error_types['extraction_failed'] = error_types.get('extraction_failed', 0) + 1
            continue

        # Add metadata
        features['ticker'] = ticker
        features['snapshot_date'] = snapshot_date
        features['actual_class'] = row.get('actual_class', 'UNKNOWN')
        features['max_gain_pct'] = row.get('max_gain_pct', np.nan)
        features['days_to_peak'] = row.get('days_to_peak', np.nan)

        # Append to results
        results.append(features)
        success_count += 1

        # Save checkpoint every 100 patterns
        if len(results) % 100 == 0 and len(results) > 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_parquet(CHECKPOINT_FILE, index=False)
            print(f"    [Checkpoint saved: {len(results)} patterns]")

    # Create final results DataFrame
    print(f"\n[STEP 3] Creating results DataFrame...")
    results_df = pd.DataFrame(results)
    print(f"  Successfully extracted features: {len(results_df):,}/{total_patterns:,} ({len(results_df)/total_patterns*100:.1f}%)")
    print(f"  Failed extractions: {fail_count:,}")

    if fail_count > 0:
        print(f"\n  Error breakdown:")
        for error_type, count in error_types.items():
            print(f"    {error_type}: {count}")

    # Verify feature count
    feature_cols = [col for col in MODEL_EXPECTED_FEATURES if col in results_df.columns]
    print(f"\n  Features extracted: {len(feature_cols)}/{NUM_FEATURES}")

    if len(feature_cols) != NUM_FEATURES:
        missing = set(MODEL_EXPECTED_FEATURES) - set(feature_cols)
        print(f"  WARNING: Missing features: {list(missing)[:5]}...")

    # Save to Parquet
    print(f"\n[STEP 4] Saving to Parquet...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"  Checkpoint file removed")

    # Display summary statistics
    print(f"\n[STEP 5] Summary Statistics:")
    print("-" * 50)

    if 'actual_class' in results_df.columns:
        print("\nActual class distribution:")
        class_dist = results_df['actual_class'].value_counts()
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} ({count/len(results_df)*100:.1f}%)")

    if 'max_gain_pct' in results_df.columns and not results_df['max_gain_pct'].isna().all():
        print(f"\nGain statistics:")
        print(f"  Mean: {results_df['max_gain_pct'].mean():.1f}%")
        print(f"  Median: {results_df['max_gain_pct'].median():.1f}%")
        print(f"  Max: {results_df['max_gain_pct'].max():.1f}%")
        print(f"  Min: {results_df['max_gain_pct'].min():.1f}%")

    # Success rate by ticker type
    if len(results_df) > 0:
        results_df['ticker_suffix'] = results_df['ticker'].str.extract(r'\.([A-Z]+)$', expand=False).fillna('US')
        suffix_success = results_df.groupby('ticker_suffix').size()
        print(f"\nSuccess by ticker type:")
        for suffix, count in suffix_success.head(5).items():
            total_suffix = patterns_df[patterns_df['ticker'].str.contains(f'\\.{suffix}$', na=False)].shape[0]
            if suffix == 'US':
                total_suffix = patterns_df[~patterns_df['ticker'].str.contains('\\.', na=False)].shape[0]
            if total_suffix > 0:
                print(f"  {suffix}: {count}/{total_suffix} ({count/total_suffix*100:.1f}%)")

    print("\n" + "=" * 80)
    print("IMPROVED FEATURE EXTRACTION COMPLETE!")
    print(f"Success Rate: {success_count/total_patterns*100:.1f}%")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Next step: Run predictions using these features")
    print("=" * 80)