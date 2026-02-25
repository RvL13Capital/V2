"""
Test feature extraction with converted Parquet files
"""

import pandas as pd
from pathlib import Path
import sys

# Add path for imports
sys.path.append(str(Path(__file__).parent))

from final_volume_pattern_system import VolumeFeatureEngine, SystemConfig

def test_feature_extraction():
    """Test feature extraction on converted data."""

    # Load converted parquet data
    data_path = Path('data/raw/pattern_sample.parquet')

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return False

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

    # Initialize feature engine
    feature_engine = VolumeFeatureEngine()

    # Calculate features
    print("\nCalculating volume features...")
    df_with_features, feature_cols = feature_engine.calculate_features(df)

    # Check what features were added
    original_cols = set(df.columns)
    new_cols = set(df_with_features.columns) - original_cols

    print(f"\nOriginal columns: {len(original_cols)}")
    print(f"Total columns after feature extraction: {len(df_with_features.columns)}")
    print(f"New features added: {len(new_cols)}")

    if new_cols:
        print("\nSample of new features:")
        for col in sorted(list(new_cols))[:10]:
            print(f"  - {col}")

        # Check for NaN values in features
        nan_counts = df_with_features[list(new_cols)].isna().sum()
        features_with_nan = nan_counts[nan_counts > 0]

        if len(features_with_nan) > 0:
            print(f"\nWarning: {len(features_with_nan)} features have NaN values")
        else:
            print("\nAll features calculated successfully without NaN values")

        return True
    else:
        print("\nError: No features were extracted!")
        return False

if __name__ == "__main__":
    success = test_feature_extraction()

    if success:
        print("\n✓ Feature extraction test PASSED")
    else:
        print("\n✗ Feature extraction test FAILED")