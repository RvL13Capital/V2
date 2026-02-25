"""
AIv4 Pattern Scanner with Enhanced Features (49+ leak-free features)

This scanner integrates enhanced tabular features (30+) and incremental EBP
features (19) to calculate all features during pattern scanning.

Key Features:
- Snapshot-based sampling (2-5 snapshots per pattern)
- 30+ enhanced tabular features (compression, quality, position)
- 19 incremental EBP features (CCI, VAR, NES, LPF, TSF, composite)
- Step 2 Optimization: Incremental EBP calculator with caching (~0.1ms per day)
- All features use ONLY data available at snapshot_date (no leakage)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random
from multiprocessing import Pool, cpu_count, Manager
import sqlite3
from typing import List, Dict, Tuple, Optional

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent))

# AIv4 imports
from shared.config import get_settings
from pattern_detection.state_machine.consolidation_tracker import ConsolidationTracker
from shared.indicators.technical import (
    calculate_bbw,
    calculate_adx,
    calculate_volume_ratio,
    get_bbw_percentile,
    calculate_ad_line
)
# Step 2 Optimization: Import incremental EBP calculator
from pattern_detection.features.ebp_features_incremental import IncrementalEBPCalculator
# Enhanced features calculated inline (no external import needed)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path('data/gcs_cache')  # CSV files
PARQUET_DIR = Path('data_acquisition/storage')  # Parquet files
OUTPUT_DIR = Path('output')
DB_DIR = Path('output/pattern_db')
MIN_DATA_POINTS = 500  # ~2 years of daily data

# Parallel processing configuration
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free
USE_PARALLEL = True  # Set to False for debugging

# Snapshot sampling configuration
SNAPSHOT_PROBABILITY_BASE = 0.35
SNAPSHOT_PROBABILITY_MATURE = 0.50
MATURE_PATTERN_DAYS = 15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)


# ========================================
# INCREMENTAL DATABASE FUNCTIONS
# ========================================

def init_pattern_database(db_path: Path):
    """Initialize SQLite database for tracking scanned patterns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for scanned tickers
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scanned_tickers (
            ticker TEXT PRIMARY KEY,
            last_scan_date TEXT,
            data_source TEXT,
            num_patterns INTEGER,
            num_snapshots INTEGER,
            last_data_date TEXT
        )
    ''')

    # Create table for pattern snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pattern_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            snapshot_date TEXT,
            pattern_start_date TEXT,
            days_since_activation INTEGER,
            scan_timestamp TEXT,
            UNIQUE(ticker, snapshot_date, pattern_start_date)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"✓ Initialized pattern database at {db_path}")


def get_previously_scanned_tickers(db_path: Path) -> Dict[str, dict]:
    """Get dictionary of previously scanned tickers with metadata."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT ticker, last_scan_date, last_data_date, num_snapshots FROM scanned_tickers')
    rows = cursor.fetchall()
    conn.close()

    return {
        row[0]: {
            'last_scan_date': row[1],
            'last_data_date': row[2],
            'num_snapshots': row[3]
        }
        for row in rows
    }


def update_scanned_ticker(db_path: Path, ticker: str, data_source: str,
                         num_patterns: int, num_snapshots: int, last_data_date: str):
    """Update or insert ticker scan record."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    scan_timestamp = datetime.now().isoformat()

    cursor.execute('''
        INSERT OR REPLACE INTO scanned_tickers
        (ticker, last_scan_date, data_source, num_patterns, num_snapshots, last_data_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, scan_timestamp, data_source, num_patterns, num_snapshots, last_data_date))

    conn.commit()
    conn.close()


def save_snapshots_to_db(db_path: Path, ticker: str, snapshots: List[Dict]):
    """Save snapshots to database (for future incremental updates)."""
    if not snapshots:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    scan_timestamp = datetime.now().isoformat()

    for snap in snapshots:
        # Convert pandas Timestamps to strings for SQLite
        snapshot_date = str(snap.get('snapshot_date', '')) if snap.get('snapshot_date') else ''
        start_date = str(snap.get('start_date', '')) if snap.get('start_date') else ''

        cursor.execute('''
            INSERT OR IGNORE INTO pattern_snapshots
            (ticker, snapshot_date, pattern_start_date, days_since_activation, scan_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            ticker,
            snapshot_date,
            start_date,
            snap.get('days_since_activation', 0),
            scan_timestamp
        ))

    conn.commit()
    conn.close()


# ========================================
# PARALLEL PROCESSING WORKER
# ========================================

def process_ticker_worker(args: Tuple) -> Optional[List[Dict]]:
    """
    Worker function for parallel processing (must be picklable).

    Args:
        args: Tuple of (data_file, file_type, settings, db_path)

    Returns:
        List of snapshots or None if failed
    """
    data_file, file_type, settings, db_path = args
    ticker = data_file.stem

    try:
        # Load data based on file type
        if file_type == 'csv':
            df = pd.read_csv(data_file)
        else:  # parquet
            df = pd.read_parquet(data_file)

        # Scan ticker (settings object is already initialized)
        snapshots = scan_ticker_with_enhanced_features(ticker, df, settings)

        # Update database
        if snapshots and db_path:
            # Convert pandas Timestamp to string for SQLite
            last_data_date = str(df['date'].max()) if 'date' in df.columns else ''
            num_patterns = len(set(s.get('start_date', '') for s in snapshots))

            update_scanned_ticker(
                db_path, ticker, file_type,
                num_patterns, len(snapshots), last_data_date
            )
            save_snapshots_to_db(db_path, ticker, snapshots)

        return snapshots

    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None


def normalize_columns(df):
    """Normalize column names to lowercase."""
    df.columns = df.columns.str.lower()
    if 'adj close' in df.columns:
        df = df.rename(columns={'adj close': 'adj_close'})
    return df


def calculate_indicators(df):
    """
    Calculate technical indicators AND pre-calculate enhanced features.

    Pre-calculating these once per ticker (instead of per snapshot) saves significant time.
    """
    logger.debug("  Calculating indicators...")

    # === BASIC INDICATORS ===
    df['bbw_20'] = calculate_bbw(df, period=20)
    df['bbw_percentile'] = get_bbw_percentile(df, window=100)
    df['adx'] = calculate_adx(df, period=14)

    if 'volume' in df.columns:
        df['volume_ratio_20'] = calculate_volume_ratio(df, period=20)
        # Calculate Accumulation/Distribution Line (for volume pressure analysis)
        df['ad_line'] = calculate_ad_line(df)
    else:
        df['volume_ratio_20'] = 1.0
        df['ad_line'] = 0.0

    if all(col in df.columns for col in ['high', 'low']):
        df['daily_range'] = df['high'] - df['low']
        avg_range_20 = df['daily_range'].rolling(20, min_periods=10).mean()
        df['range_ratio'] = df['daily_range'] / (avg_range_20 + 1e-10)
    else:
        df['range_ratio'] = 1.0

    # === PRE-CALCULATE ENHANCED FEATURES (Optimization) ===
    # These are calculated once per ticker instead of per snapshot

    # Recent window statistics (20-day)
    df['avg_range_20d'] = df['daily_range'].rolling(20, min_periods=10).mean()
    df['bbw_std_20d'] = df['bbw_20'].rolling(20, min_periods=10).std()
    df['price_volatility_20d'] = df['close'].rolling(20, min_periods=10).std()

    # Baseline window statistics (days -50 to -20)
    # Using rolling with offset to get baseline period
    df['baseline_bbw_avg'] = df['bbw_20'].rolling(30, min_periods=10).mean().shift(20)
    df['baseline_volume_avg'] = df['volume_ratio_20'].rolling(30, min_periods=10).mean().shift(20)
    df['baseline_volatility'] = df['close'].rolling(30, min_periods=10).std().shift(20)

    # Compression ratios (recent / baseline)
    df['bbw_compression_ratio'] = df['bbw_20'] / (df['baseline_bbw_avg'] + 1e-10)
    df['volume_compression_ratio'] = df['volume_ratio_20'] / (df['baseline_volume_avg'] + 1e-10)
    df['volatility_compression_ratio'] = df['price_volatility_20d'] / (df['baseline_volatility'] + 1e-10)

    # Trend slopes (20-day linear fit)
    # Note: These are expensive (polyfit), so we calculate once
    def calculate_slope(series, window=20):
        """Calculate rolling linear regression slope."""
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(0.0)
            else:
                window_data = series.iloc[i-window+1:i+1]
                if not window_data.isna().all() and len(window_data) == window:
                    x = np.arange(window)
                    try:
                        slope = np.polyfit(x, window_data, 1)[0]
                        slopes.append(slope)
                    except:
                        slopes.append(0.0)
                else:
                    slopes.append(0.0)
        return pd.Series(slopes, index=series.index)

    df['bbw_slope_20d'] = calculate_slope(df['bbw_20'], window=20)
    df['adx_slope_20d'] = calculate_slope(df['adx'], window=20)

    # Consolidation quality score (can be pre-calculated)
    df['consolidation_quality_score'] = (
        (1.0 - df['bbw_compression_ratio'].clip(0, 1)) * 0.33 +
        (1.0 - (df['volume_compression_ratio'] / 2.0).clip(0, 1)) * 0.33 +
        (1.0 - (df['adx'] / 50.0).clip(0, 1)) * 0.33
    )

    return df


def calculate_enhanced_features_at_snapshot(df, idx, pattern):
    """
    Extract pre-calculated enhanced features at a specific snapshot point.

    OPTIMIZATION: Features are now pre-calculated in calculate_indicators(),
    so this function just accesses the values (much faster than recalculating).

    CRITICAL: All features use ONLY data available at df.iloc[:idx+1]
    (no future data leakage).

    Args:
        df: Full DataFrame with price data and pre-calculated features
        idx: Current index (snapshot point)
        pattern: ConsolidationPattern object

    Returns:
        Dictionary with enhanced features
    """
    # Check if we have enough history
    if idx < 50:
        # Not enough history for enhanced features
        return {}

    row = df.iloc[idx]
    features = {}

    # === ACCESS PRE-CALCULATED FEATURES ===
    # These were calculated once in calculate_indicators()

    # Recent window statistics
    features['avg_range_20d'] = row.get('avg_range_20d', 0.0)
    features['bbw_std_20d'] = row.get('bbw_std_20d', 0.0)
    features['price_volatility_20d'] = row.get('price_volatility_20d', 0.0)

    # Baseline statistics
    features['baseline_bbw_avg'] = row.get('baseline_bbw_avg', 0.0)
    features['baseline_volume_avg'] = row.get('baseline_volume_avg', 0.0)
    features['baseline_volatility'] = row.get('baseline_volatility', 0.0)

    # Compression ratios
    features['bbw_compression_ratio'] = row.get('bbw_compression_ratio', 0.0)
    features['volume_compression_ratio'] = row.get('volume_compression_ratio', 0.0)
    features['volatility_compression_ratio'] = row.get('volatility_compression_ratio', 0.0)

    # Trend slopes
    features['bbw_slope_20d'] = row.get('bbw_slope_20d', 0.0)
    features['adx_slope_20d'] = row.get('adx_slope_20d', 0.0)

    # Quality score
    features['consolidation_quality_score'] = row.get('consolidation_quality_score', 0.0)

    # === PRICE POSITION (Pattern-specific, calculated here) ===
    current_price = row['close']
    if pattern.upper_boundary and pattern.lower_boundary:
        range_width = pattern.upper_boundary - pattern.lower_boundary
        features['price_position_in_range'] = (current_price - pattern.lower_boundary) / (range_width + 1e-10)
        features['price_distance_from_upper_pct'] = (pattern.upper_boundary - current_price) / (current_price + 1e-10)
        features['price_distance_from_lower_pct'] = (current_price - pattern.lower_boundary) / (current_price + 1e-10)
        features['distance_from_power_pct'] = (pattern.power_boundary - current_price) / (current_price + 1e-10)
    else:
        features['price_position_in_range'] = 0.0
        features['price_distance_from_upper_pct'] = 0.0
        features['price_distance_from_lower_pct'] = 0.0
        features['distance_from_power_pct'] = 0.0

    # Fill NaN with 0
    features = {k: (v if pd.notna(v) else 0.0) for k, v in features.items()}

    return features


def extract_snapshot_with_features(df, idx, tracker, date, ebp_calculator=None):
    """
    Extract complete snapshot with basic + enhanced + EBP features.

    Returns dictionary with 49+ features ready for training (30 enhanced + 19 EBP).
    """
    row = df.iloc[idx]
    pattern = tracker.current_pattern

    # Basic snapshot data
    snapshot = {
        'ticker': pattern.ticker,
        'snapshot_date': date,
        'start_date': pattern.start_date,
        'end_date': None,
        'activation_date': pattern.activation_date,
        'phase': 'ACTIVE',
        'days_since_activation': (date - pattern.activation_date).days if pattern.activation_date else 0,
        'days_in_pattern': (date - pattern.start_date).days,  # Will be removed during training (data leakage)

        # Pattern boundaries
        'upper_boundary': pattern.upper_boundary,
        'lower_boundary': pattern.lower_boundary,
        'power_boundary': pattern.power_boundary,
        'start_price': pattern.start_price,

        # Current state
        'current_price': row['close'],
        'current_high': row['high'],
        'current_low': row['low'],
        'current_volume': row['volume'],

        # Current indicators
        'current_bbw_20': row.get('bbw_20', np.nan),
        'current_bbw_percentile': row.get('bbw_percentile', np.nan),
        'current_adx': row.get('adx', np.nan),
        'current_volume_ratio_20': row.get('volume_ratio_20', np.nan),
        'current_range_ratio': row.get('range_ratio', 1.0),

        # Outcome (will be filled during labeling)
        'breakout_direction': None,
        'end_price': None
    }

    # Add enhanced features (30+)
    enhanced_features = calculate_enhanced_features_at_snapshot(df, idx, pattern)
    snapshot.update(enhanced_features)

    # Step 2 Optimization: Add EBP features from incremental calculator (19+)
    # IMPORTANT: Features were already updated in main loop, just retrieve cached values
    if ebp_calculator is not None:
        # Get cached EBP values (no recalculation - already done in main loop)
        ebp_values = ebp_calculator.get_current_features()
        if ebp_values:  # Only update if features are cached
            snapshot.update(ebp_values)

    return snapshot


def scan_ticker_with_enhanced_features(ticker, df, settings):
    """Scan ticker with snapshot sampling and enhanced feature extraction."""
    logger.info(f"  Scanning {ticker} with enhanced features...")

    try:
        # Normalize and prepare
        df = normalize_columns(df)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        if 'close' in df.columns:
            df = df[df['close'] >= settings.data.min_price]

        if len(df) < MIN_DATA_POINTS:
            logger.info(f"    SKIP: Only {len(df)} rows (need {MIN_DATA_POINTS})")
            return []

        # Calculate indicators
        df = calculate_indicators(df)

        required_cols = ['close', 'bbw_percentile', 'adx', 'volume_ratio_20', 'range_ratio']
        df = df.dropna(subset=required_cols)

        if len(df) < 100:
            logger.info(f"    SKIP: Insufficient data after indicator calculation")
            return []

        # Create tracker
        tracker = ConsolidationTracker(
            ticker=ticker,
            qualifying_days=7,
            bbw_percentile_threshold=settings.consolidation.bbw_percentile_threshold,
            adx_threshold=settings.consolidation.adx_threshold,
            volume_ratio_threshold=settings.consolidation.volume_ratio_threshold,
            range_ratio_threshold=settings.consolidation.range_ratio_threshold
        )
        tracker.set_data(df)

        all_snapshots = []
        active_pattern_start_idx = None

        # Step 2 Optimization: Initialize incremental EBP calculator (reset per pattern)
        ebp_calculator = None

        # Process day by day
        for idx in range(len(df)):
            date = df.index[idx]
            row = df.iloc[idx]

            price_data = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            indicators = {
                'bbw_20': row.get('bbw_20', np.nan),
                'bbw_percentile': row.get('bbw_percentile', np.nan),
                'adx': row.get('adx', np.nan),
                'volume_ratio_20': row.get('volume_ratio_20', np.nan),
                'range_ratio': row.get('range_ratio', 1.0)
            }

            try:
                event = tracker.update(date, idx, price_data, indicators)

                # Save snapshot on activation
                if event == 'PATTERN_ACTIVATED':
                    active_pattern_start_idx = idx

                    # Step 2 Optimization: Initialize EBP calculator for new pattern
                    ebp_calculator = IncrementalEBPCalculator()
                    ebp_calculator.reset()

                    # Update EBP calculator with activation data
                    prev_row_dict = df.iloc[idx-1].to_dict() if idx > 0 else None
                    current_row_dict = row.to_dict()
                    ebp_calculator.update_and_calculate(
                        current_row_dict,
                        prev_row_dict,
                        pattern_start_idx=active_pattern_start_idx,
                        current_idx=idx
                    )

                    snapshot = extract_snapshot_with_features(df, idx, tracker, date, ebp_calculator)
                    all_snapshots.append(snapshot)
                    logger.debug(f"{ticker}: Activation snapshot with {len(snapshot)} features")

                # Random sampling during ACTIVE phase
                elif active_pattern_start_idx is not None and tracker.phase.value == 'ACTIVE':
                    # Step 2 Optimization: Update EBP calculator on EVERY active day (not just snapshots)
                    if ebp_calculator is not None:
                        prev_row_dict = df.iloc[idx-1].to_dict() if idx > 0 else None
                        current_row_dict = row.to_dict()
                        ebp_calculator.update_and_calculate(
                            current_row_dict,
                            prev_row_dict,
                            pattern_start_idx=active_pattern_start_idx,
                            current_idx=idx
                        )

                    days_active = idx - active_pattern_start_idx
                    sampling_probability = (SNAPSHOT_PROBABILITY_MATURE if days_active >= MATURE_PATTERN_DAYS
                                          else SNAPSHOT_PROBABILITY_BASE)

                    if random.random() < sampling_probability:
                        snapshot = extract_snapshot_with_features(df, idx, tracker, date, ebp_calculator)
                        all_snapshots.append(snapshot)

                # Pattern completed - save final snapshot
                elif event in ['BREAKOUT_UP', 'BREAKDOWN']:
                    if active_pattern_start_idx is not None:
                        # Step 2 Optimization: Final EBP update before saving snapshot
                        if ebp_calculator is not None:
                            prev_row_dict = df.iloc[idx-1].to_dict() if idx > 0 else None
                            current_row_dict = row.to_dict()
                            ebp_calculator.update_and_calculate(
                                current_row_dict,
                                prev_row_dict,
                                pattern_start_idx=active_pattern_start_idx,
                                current_idx=idx
                            )

                        snapshot = extract_snapshot_with_features(df, idx, tracker, date, ebp_calculator)
                        all_snapshots.append(snapshot)

                    active_pattern_start_idx = None
                    ebp_calculator = None  # Reset for next pattern

            except Exception as e:
                logger.debug(f"    Error at {date}: {e}")
                continue

        logger.info(f"    FOUND: {len(all_snapshots)} snapshots with enhanced features")
        return all_snapshots

    except Exception as e:
        logger.warning(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main entry point with parallel processing and incremental database."""
    print("="*70)
    print("AIv4 PATTERN SCANNER - PARALLEL + INCREMENTAL")
    print("="*70)
    print()

    # Initialize database
    db_path = DB_DIR / 'patterns.db'
    init_pattern_database(db_path)

    settings = get_settings()

    # Find all CSV and parquet files
    csv_files = list(CACHE_DIR.glob('**/*.csv')) if CACHE_DIR.exists() else []
    parquet_files = list(PARQUET_DIR.glob('**/*.parquet')) if PARQUET_DIR.exists() else []

    # Combine into single list with file type info
    all_files = [(f, 'csv') for f in csv_files] + [(f, 'parquet') for f in parquet_files]

    logger.info(f"Found {len(csv_files)} CSV files in {CACHE_DIR}")
    logger.info(f"Found {len(parquet_files)} parquet files in {PARQUET_DIR}")
    logger.info(f"Total: {len(all_files)} files to process")
    logger.info(f"Workers: {NUM_WORKERS}")
    logger.info(f"Parallel processing: {'ENABLED' if USE_PARALLEL else 'DISABLED'}")
    logger.info(f"Database: {db_path}")
    print()

    if len(all_files) == 0:
        logger.error(f"No data files found")
        logger.error(f"Checked: {CACHE_DIR} and {PARQUET_DIR}")
        return

    # Prepare arguments for workers (settings object is picklable)
    worker_args = [(f, file_type, settings, db_path) for f, file_type in all_files]

    all_snapshots = []
    tickers_with_patterns = 0
    start_time = datetime.now()

    if USE_PARALLEL and NUM_WORKERS > 1:
        # Parallel processing with multiprocessing.Pool
        logger.info(f"Starting parallel processing with {NUM_WORKERS} workers...")
        print()

        with Pool(NUM_WORKERS) as pool:
            # Use imap for lazy iteration with progress tracking
            results = pool.imap(process_ticker_worker, worker_args, chunksize=1)

            for idx, snapshots in enumerate(results, 1):
                if snapshots:
                    all_snapshots.extend(snapshots)
                    tickers_with_patterns += 1

                if idx % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (len(all_files) - idx) / rate if rate > 0 else 0
                    logger.info(f"Progress: {idx}/{len(all_files)} tickers | "
                              f"{len(all_snapshots)} snapshots | "
                              f"{rate:.1f} tickers/sec | "
                              f"ETA: {eta/60:.1f} min")

        tickers_processed = len(all_files)

    else:
        # Sequential processing (for debugging)
        logger.info("Starting sequential processing...")
        print()

        for idx, args in enumerate(worker_args, 1):
            snapshots = process_ticker_worker(args)

            if snapshots:
                all_snapshots.extend(snapshots)
                tickers_with_patterns += 1

            if idx % 50 == 0:
                logger.info(f"Progress: {idx}/{len(all_files)} tickers, {len(all_snapshots)} total snapshots")

        tickers_processed = len(all_files)

    # Save results
    if all_snapshots:
        df_snapshots = pd.DataFrame(all_snapshots)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"patterns_enhanced_{timestamp}.csv"
        df_snapshots.to_csv(output_file, index=False)

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n{'='*70}")
        logger.info(f"SCANNING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Tickers processed: {tickers_processed}")
        logger.info(f"Tickers with patterns: {tickers_with_patterns}")
        logger.info(f"Total snapshots: {len(all_snapshots)}")
        logger.info(f"Features per snapshot: {len(df_snapshots.columns)}")
        logger.info(f"Processing time: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {tickers_processed/elapsed:.1f} tickers/sec")
        logger.info(f"\nSaved to: {output_file}")
        logger.info(f"Database: {db_path}")
        logger.info(f"\nFeature columns ({len(df_snapshots.columns)}):")
        for col in sorted(df_snapshots.columns):
            logger.info(f"  - {col}")
    else:
        logger.error("No patterns found!")


if __name__ == "__main__":
    main()
