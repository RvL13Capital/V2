"""
Extract 69 Canonical Features for Historical Validation Patterns
=====================================================================

This script extracts all 69 canonical features from historical validation
snapshots using the SAME canonical feature extractor used for training.

Purpose:
- Fix feature pipeline mismatch between training and validation
- Training used 69 enhanced features
- Old validation used different 31 features with only 3 overlapping
- This ensures validation uses exact same 69 features as training

Process:
1. Load historical snapshots (ticker, snapshot_date, boundaries)
2. For each snapshot:
   - Load cached OHLCV data from data/unused_tickers_cache/
   - Filter to only dates <= snapshot_date (temporal-safe)
   - Calculate all 69 canonical features using CanonicalFeatureExtractor
   - Extract features at snapshot_date
3. Save to output/unused_patterns/pattern_features_canonical_69.parquet

Expected Runtime: 30-45 minutes for all snapshots
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

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
        logger.warning(f"  Cache file not found for {ticker}")
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
            logger.warning(f"  Missing required columns for {ticker}")
            return None

        return df

    except Exception as e:
        logger.error(f"  Error loading {ticker}: {str(e)}")
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

    Args:
        df: OHLCV data (temporal-safe: only dates <= snapshot_date)
        snapshot_date: Date of snapshot
        active_start: Pattern activation date
        upper_boundary: Upper consolidation boundary
        lower_boundary: Lower consolidation boundary
        power_boundary: Power boundary (breakout threshold)
        start_date: Pattern start date
        start_price: Price at pattern start

    Returns:
        Dictionary with 69 canonical feature values
    """
    # Create a ConsolidationPattern object with the boundaries
    pattern = ConsolidationPattern(
        ticker='VALIDATION',  # Ticker name doesn't matter for feature extraction
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

    # Initialize canonical feature extractor
    extractor = CanonicalFeatureExtractor()

    # Set up the DataFrame with proper index
    if 'date' in df.columns:
        df = df.set_index('date')

    # Calculate indicators (these should be pre-calculated in df for efficiency)
    df = calculate_indicators(df)

    # Extract features at snapshot_date
    try:
        snapshot_idx = df.index.get_loc(snapshot_date)
        features = extractor.extract_snapshot_features(
            df,
            snapshot_date,
            pattern,
            snapshot_idx
        )
        return features
    except Exception as e:
        logger.error(f"    Feature extraction failed: {str(e)}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculate technical indicators needed by the canonical extractor.
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

    # Pre-calculate enhanced features for efficiency
    # Recent window statistics
    df['avg_range_20d'] = df['daily_range'].rolling(20, min_periods=10).mean()
    df['bbw_std_20d'] = df['bbw_20'].rolling(20, min_periods=10).std()
    df['price_volatility_20d'] = df['close'].rolling(20, min_periods=10).std()

    # Baseline statistics
    df['baseline_bbw_avg'] = df['bbw_20'].rolling(30, min_periods=10).mean().shift(20)
    df['baseline_volume_avg'] = df['volume_ratio_20'].rolling(30, min_periods=10).mean().shift(20)
    df['baseline_volatility'] = df['close'].rolling(30, min_periods=10).std().shift(20)

    # Compression ratios
    df['bbw_compression_ratio'] = df['bbw_20'] / (df['baseline_bbw_avg'] + 1e-10)
    df['volume_compression_ratio'] = df['volume_ratio_20'] / (df['baseline_volume_avg'] + 1e-10)
    df['volatility_compression_ratio'] = df['price_volatility_20d'] / (df['baseline_volatility'] + 1e-10)

    # Slopes (simplified for speed)
    df['bbw_slope_20d'] = df['bbw_20'].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else 0)
    df['adx_slope_20d'] = df['adx'].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else 0)

    # Quality score
    df['consolidation_quality_score'] = (
        (1.0 - df['bbw_compression_ratio'].clip(0, 1)) * 0.33 +
        (1.0 - (df['volume_compression_ratio'] / 2.0).clip(0, 1)) * 0.33 +
        (1.0 - (df['adx'] / 50.0).clip(0, 1)) * 0.33
    )

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HISTORICAL VALIDATION - 69 CANONICAL FEATURE EXTRACTION")
    print("=" * 80)

    # Check if historical patterns file exists
    if not HISTORICAL_PATTERNS_FILE.exists():
        print(f"\n❌ ERROR: Historical patterns file not found: {HISTORICAL_PATTERNS_FILE}")
        print("Please ensure the validation patterns are available.")
        exit(1)

    # Load historical patterns
    print(f"\n[STEP 1] Loading historical pattern snapshots...")
    patterns_df = pd.read_parquet(HISTORICAL_PATTERNS_FILE)
    print(f"  Total snapshots: {len(patterns_df):,}")
    print(f"  Unique tickers: {patterns_df['ticker'].nunique()}")

    # Convert date columns to datetime
    date_cols = ['qualification_start', 'active_start', 'end_date', 'snapshot_date']
    for col in date_cols:
        if col in patterns_df.columns:
            patterns_df[col] = pd.to_datetime(patterns_df[col])

    # Prepare results storage
    results = []
    failed_count = 0

    # Process each snapshot
    print(f"\n[STEP 2] Extracting {NUM_FEATURES} canonical features...")
    print(f"  This will take ~30-45 minutes for {len(patterns_df)} snapshots")

    for idx, row in patterns_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(patterns_df)} ({(idx+1)/len(patterns_df)*100:.1f}%)")

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
            logger.warning(f"  [{idx+1}] {ticker} - No data available")
            failed_count += 1
            continue

        # Filter to temporal-safe data (only dates <= snapshot_date)
        df = df[df['date'] <= snapshot_date].copy()

        if len(df) < 100:
            logger.warning(f"  [{idx+1}] {ticker} - Insufficient data ({len(df)} days)")
            failed_count += 1
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
            logger.warning(f"  [{idx+1}] {ticker} - Feature extraction failed")
            failed_count += 1
            continue

        # Add metadata
        features['ticker'] = ticker
        features['snapshot_date'] = snapshot_date
        features['actual_class'] = row.get('actual_class', 'UNKNOWN')
        features['max_gain_pct'] = row.get('max_gain_pct', np.nan)
        features['days_to_peak'] = row.get('days_to_peak', np.nan)

        # Append to results
        results.append(features)

    # Create results DataFrame
    print(f"\n[STEP 3] Creating results DataFrame...")
    results_df = pd.DataFrame(results)
    print(f"  Successfully extracted features: {len(results_df):,}/{len(patterns_df):,}")
    print(f"  Failed extractions: {failed_count:,}")

    # Verify feature count
    feature_cols = [col for col in MODEL_EXPECTED_FEATURES if col in results_df.columns]
    print(f"  Features extracted: {len(feature_cols)}/{NUM_FEATURES}")

    if len(feature_cols) != NUM_FEATURES:
        missing = set(MODEL_EXPECTED_FEATURES) - set(feature_cols)
        print(f"  WARNING: Missing features: {missing}")

    # Save to Parquet
    print(f"\n[STEP 4] Saving to Parquet...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    # Display summary statistics
    print(f"\n[STEP 5] Summary Statistics:")
    print("-" * 50)
    if 'actual_class' in results_df.columns:
        print("\nActual class distribution:")
        print(results_df['actual_class'].value_counts())

    if 'max_gain_pct' in results_df.columns:
        print(f"\nGain statistics:")
        print(f"  Mean: {results_df['max_gain_pct'].mean():.1f}%")
        print(f"  Median: {results_df['max_gain_pct'].median():.1f}%")
        print(f"  Max: {results_df['max_gain_pct'].max():.1f}%")
        print(f"  Min: {results_df['max_gain_pct'].min():.1f}%")

    print("\n" + "=" * 80)
    print("CANONICAL FEATURE EXTRACTION COMPLETE!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Next step: Run predictions using these features")
    print("=" * 80)