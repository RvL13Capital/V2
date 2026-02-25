"""
Prepare Training Data from Race-Labeled Snapshots
==================================================
Adapts the race-labeled snapshots for feature extraction and training.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Input/Output paths
LABELED_FILE = Path('output/snapshots_labeled_race_20251104_192443.parquet')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("PREPARING TRAINING DATA FROM RACE-LABELED SNAPSHOTS")
    print("=" * 70)

    # Load race-labeled snapshots
    print(f"\nLoading: {LABELED_FILE}")
    df = pd.read_parquet(LABELED_FILE)
    print(f"Loaded {len(df):,} labeled snapshots")

    # Display class distribution
    print("\nClass Distribution:")
    class_counts = df['outcome_class'].value_counts()
    for cls in ['K4_EXCEPTIONAL', 'K3_STRONG', 'K2_QUALITY', 'K1_MINIMAL', 'K0_STAGNANT', 'K5_FAILED']:
        if cls in class_counts.index:
            count = class_counts[cls]
            pct = count / len(df) * 100
            print(f"  {cls:15s}: {count:7,} ({pct:5.2f}%)")

    # Ensure required columns exist for feature extraction
    required_cols = [
        'ticker', 'snapshot_date', 'upper_boundary', 'lower_boundary',
        'power_boundary', 'outcome_class', 'max_gain_from_upper_pct'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWARNING: Missing columns: {missing_cols}")
        # Add default values for missing columns
        if 'power_boundary' not in df.columns and 'upper_boundary' in df.columns:
            df['power_boundary'] = df['upper_boundary'] * 1.005
            print("  Added power_boundary as upper_boundary * 1.005")

    # Add pattern metadata if missing
    if 'pattern_id' not in df.columns:
        # Create synthetic pattern IDs based on ticker and start date
        df['pattern_id'] = df['ticker'] + '_' + df['start_date'].astype(str)
        print("  Added synthetic pattern_id")

    if 'days_in_pattern' not in df.columns:
        # Use breach_day as proxy
        df['days_in_pattern'] = df['breach_day'].fillna(50)
        print("  Added days_in_pattern from breach_day")

    # Convert date columns to datetime if needed
    date_cols = ['snapshot_date', 'start_date', 'qualification_start', 'qualification_end']
    for col in date_cols:
        if col in df.columns:
            try:
                # Try different parsing strategies
                if df[col].dtype == 'object':
                    # Handle mixed formats
                    df[col] = pd.to_datetime(df[col], format='mixed', utc=True)
                else:
                    df[col] = pd.to_datetime(df[col])
            except:
                print(f"  WARNING: Could not convert {col} to datetime")

    # Save prepared data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f'patterns_race_labeled_{timestamp}.parquet'
    df.to_parquet(output_file, index=False)

    print(f"\n[SAVED] Prepared data: {output_file}")
    print(f"  Total snapshots: {len(df):,}")
    print(f"  K4 patterns: {(df['outcome_class'] == 'K4_EXCEPTIONAL').sum():,}")
    print(f"  K3 patterns: {(df['outcome_class'] == 'K3_STRONG').sum():,}")

    # Also save a version matching the expected format for training
    # Rename to match expected naming convention
    training_file = OUTPUT_DIR / f'patterns_labeled_enhanced_{timestamp}.parquet'
    df.to_parquet(training_file, index=False)
    print(f"\n[SAVED] Training-ready file: {training_file}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Extract features:")
    print(f"   python extract_features_historical_enhanced.py --input {output_file}")
    print("\n2. Train models:")
    print(f"   python src/pipeline/03_train_enhanced.py --input {training_file}")
    print("\nOr run the full pipeline:")
    print("   python run_training_pipeline.py")

    return output_file, training_file

if __name__ == "__main__":
    output_file, training_file = main()