"""
Test Feature Extraction - Validate pipeline on 10 patterns
Extracts volume features from pattern metadata using AI Infra's VolumeFeatureEngine
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add AI Infra to path
sys.path.insert(0, str(Path(__file__).parent / "AI Infra"))

# Import from analyse core
from core import get_data_loader

# Import from AI Infra
try:
    from final_volume_pattern_system import VolumeFeatureEngine
    print("[OK] Imported VolumeFeatureEngine from AI Infra")
except ImportError as e:
    print(f"[ERROR] Failed to import VolumeFeatureEngine: {e}")
    sys.exit(1)


def extract_pattern_features(pattern_row, data_loader, feature_engine):
    """
    Extract features for a single pattern.

    Args:
        pattern_row: Row from patterns dataframe
        data_loader: Data loader instance
        feature_engine: VolumeFeatureEngine instance

    Returns:
        Dict of features or None if failed
    """
    ticker = pattern_row['ticker']
    start_date = pd.to_datetime(pattern_row['pattern_start_date'])
    end_date = pd.to_datetime(pattern_row['pattern_end_date'])

    print(f"\n  Processing {ticker}: {start_date.date()} to {end_date.date()}")

    try:
        # Load ticker data with buffer for indicator calculation
        buffer_days = 100
        load_start = start_date - timedelta(days=buffer_days)

        # Get data from GCS/cache
        df = data_loader.load_ticker_data(
            ticker=ticker,
            start_date=load_start,
            end_date=end_date + timedelta(days=5)  # Small buffer after
        )

        if df.empty or len(df) < 30:
            print(f"    [SKIP] Insufficient data: {len(df)} rows")
            return None

        # Prepare data for feature engine (needs 'symbol' and 'timestamp' columns)
        df = df.reset_index()
        df['symbol'] = ticker
        df['timestamp'] = df['date'] if 'date' in df.columns else df.index

        # Ensure required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"    [SKIP] Missing OHLCV columns")
            return None

        # Calculate features using VolumeFeatureEngine
        df_features, feature_cols = feature_engine.calculate_features(df)

        # Filter to pattern period only
        pattern_data = df_features[
            (df_features['timestamp'] >= start_date) &
            (df_features['timestamp'] <= end_date)
        ]

        if len(pattern_data) == 0:
            print(f"    [SKIP] No data in pattern period")
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

        # Add pattern metadata (already calculated)
        for col in ['avg_bbw', 'avg_volume_ratio', 'avg_range_ratio', 'price_range_pct']:
            if col in pattern_row.index:
                feature_dict[col] = pattern_row[col]

        # Aggregate volume features (mean over pattern period)
        for col in feature_cols:
            if col in pattern_data.columns:
                # Use mean for most features
                if 'consec' in col or 'surge' in col or 'explosive' in col:
                    # Use max for consecutive/binary features
                    feature_dict[f'{col}'] = pattern_data[col].max()
                else:
                    # Use mean for ratio/strength features
                    feature_dict[f'{col}'] = pattern_data[col].mean()

        # Add feature trends (slope during pattern)
        key_features = ['vol_ratio_5d', 'vol_strength_5d', 'accum_score_5d', 'obv_trend']
        for feat in key_features:
            if feat in pattern_data.columns:
                values = pattern_data[feat].values
                if len(values) > 1:
                    # Simple linear trend (last - first) / duration
                    trend = (values[-1] - values[0]) / len(values)
                    feature_dict[f'{feat}_trend'] = trend

        print(f"    [OK] Extracted {len(feature_dict)} features")
        return feature_dict

    except Exception as e:
        print(f"    [ERROR] Error: {e}")
        return None


def main():
    """Main test function."""

    print("="*70)
    print("TEST FEATURE EXTRACTION - 10 Patterns")
    print("="*70)

    # Load patterns
    patterns_file = Path("output/patterns_historical.parquet")
    if not patterns_file.exists():
        print(f"[ERROR] Patterns file not found: {patterns_file}")
        return False

    patterns_df = pd.read_parquet(patterns_file)
    print(f"\n[OK] Loaded {len(patterns_df)} total patterns")

    # Take first 10 for testing
    test_patterns = patterns_df.head(10)
    print(f"[OK] Selected {len(test_patterns)} patterns for testing")
    print(f"\nTickers: {test_patterns['ticker'].unique().tolist()}")

    # Initialize components
    data_loader = get_data_loader()
    feature_engine = VolumeFeatureEngine()

    print(f"\n{'='*70}")
    print("EXTRACTING FEATURES")
    print("="*70)

    # Extract features for each pattern
    results = []
    for idx, row in test_patterns.iterrows():
        features = extract_pattern_features(row, data_loader, feature_engine)
        if features is not None:
            results.append(features)

    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Patterns processed: {len(test_patterns)}")
    print(f"Features extracted: {len(results)}")
    print(f"Success rate: {len(results)/len(test_patterns)*100:.1f}%")

    if len(results) == 0:
        print("\n[ERROR] No features extracted - pipeline failed")
        return False

    # Create features dataframe
    features_df = pd.DataFrame(results)

    # Validate features
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("="*70)
    print(f"Total columns: {len(features_df.columns)}")
    print(f"Total rows: {len(features_df)}")

    # Check for NaN values
    nan_counts = features_df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]

    if len(features_with_nan) > 0:
        print(f"\n[WARN] Warning: {len(features_with_nan)} features have NaN values:")
        for col, count in features_with_nan.head(5).items():
            print(f"  - {col}: {count} NaN")
    else:
        print("[OK] No NaN values in features")

    # Check for infinities
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    inf_counts = features_df[numeric_cols].apply(lambda x: np.isinf(x).sum())
    features_with_inf = inf_counts[inf_counts > 0]

    if len(features_with_inf) > 0:
        print(f"\n[WARN] Warning: {len(features_with_inf)} features have infinite values")
    else:
        print("[OK] No infinite values in features")

    # Sample feature values
    print(f"\n{'='*70}")
    print("SAMPLE FEATURE VALUES")
    print("="*70)

    volume_features = [col for col in features_df.columns if 'vol_' in col]
    if len(volume_features) > 0:
        print(f"\nVolume features ({len(volume_features)} total):")
        for col in volume_features[:5]:
            print(f"  {col}: {features_df[col].mean():.4f} (mean)")

    # Save output
    output_dir = Path("AI Infra/data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_pattern_features.parquet"

    features_df.to_parquet(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"[OK] Saved features to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("="*70)

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print("="*70)
    print(f"[OK] Pipeline validated successfully")
    print(f"[OK] {len(results)}/{len(test_patterns)} patterns processed")
    print(f"[OK] {len(volume_features)} volume features extracted")
    print(f"[OK] Output saved to {output_file}")
    print("\nReady to scale up to full GCS dataset!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
