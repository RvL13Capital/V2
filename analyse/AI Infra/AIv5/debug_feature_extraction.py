"""
Debug Feature Extraction
Test if CanonicalFeatureExtractor works with cached data
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pattern_detection.features.canonical_feature_extractor import CanonicalFeatureExtractor

@dataclass
class ConsolidationPattern:
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    upper_boundary: float
    lower_boundary: float
    power_boundary: float
    qualification_days: int = 10
    phase: str = 'ACTIVE'
    activation_date: pd.Timestamp = None
    start_price: float = None
    days_qualifying: int = 10

# Load cache data
print("Loading AAP cache data...")
df = pd.read_parquet('data/historical_cache/AAP.parquet')
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Rename columns to match expected format
df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
df = df.sort_values('date')

# Remove timezone for comparison
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)

print(f"\nRenamed columns: {df.columns.tolist()}")

# Create a test pattern
snapshot_date = pd.Timestamp('2022-06-08')
pattern = ConsolidationPattern(
    ticker='AAP',
    start_date=pd.Timestamp('2022-05-27'),
    end_date=snapshot_date,
    upper_boundary=174.0,
    lower_boundary=168.0,
    power_boundary=174.87,
    qualification_days=10,
    phase='ACTIVE',
    activation_date=pd.Timestamp('2022-06-08'),
    start_price=168.5,
    days_qualifying=10
)

# Filter data up to snapshot date
df_upto_snapshot = df[df['date'] <= snapshot_date].copy()

# CRITICAL: Set 'date' as index (required by CanonicalFeatureExtractor)
df_upto_snapshot = df_upto_snapshot.set_index('date')

print(f"\nData up to {snapshot_date}: {len(df_upto_snapshot)} rows")
print(f"Date range: {df_upto_snapshot.index.min()} to {df_upto_snapshot.index.max()}")
print(f"Snapshot date in index: {snapshot_date in df_upto_snapshot.index}")
print(f"\nFirst 3 rows:")
print(df_upto_snapshot[['open', 'high', 'low', 'close', 'volume']].head(3))

# Try feature extraction
print("\nTesting CanonicalFeatureExtractor...")
try:
    extractor = CanonicalFeatureExtractor()
    print(f"Extractor initialized with {len(extractor.FEATURE_NAMES)} features")

    features = extractor.extract_snapshot_features(
        df=df_upto_snapshot,
        snapshot_date=snapshot_date,
        pattern=pattern
    )

    print("\n✅ Feature extraction SUCCESSFUL!")
    print(f"Extracted {len(features)} features")
    print("\nFirst 10 features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name}: {value}")

    # Check if features are non-zero
    feature_array = np.array(list(features.values()))
    print(f"\nFeature statistics:")
    print(f"  Non-zero features: {(feature_array != 0).sum()} / {len(feature_array)}")
    print(f"  Mean: {feature_array.mean():.4f}")
    print(f"  Std: {feature_array.std():.4f}")
    print(f"  Min: {feature_array.min():.4f}")
    print(f"  Max: {feature_array.max():.4f}")

except Exception as e:
    print(f"\n❌ Feature extraction FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
