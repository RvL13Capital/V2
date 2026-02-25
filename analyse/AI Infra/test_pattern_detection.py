"""
Check why no signals are being generated.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_path = Path(r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\Duales AI\dual\AI Infra\data\raw')
df = pd.read_parquet(data_path / '60day_constrained_patterns.parquet')

print("Data Analysis:")
print(f"Total rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Check pattern detection
if 'pattern_count' in df.columns:
    print(f"\nPattern count distribution:")
    print(df['pattern_count'].value_counts().head())
    print(f"Patterns with 2+ detections: {(df['pattern_count'] >= 2).sum()}")

# Check BBW
if 'bbw' in df.columns:
    print(f"\nBBW statistics:")
    print(f"  Min: {df['bbw'].min():.4f}")
    print(f"  25%: {df['bbw'].quantile(0.25):.4f}")
    print(f"  50%: {df['bbw'].median():.4f}")
    print(f"  75%: {df['bbw'].quantile(0.75):.4f}")
    print(f"  Max: {df['bbw'].max():.4f}")
    print(f"BBW < 0.05: {(df['bbw'] < 0.05).sum()}")

# Check volume ratio
if 'volume_ratio' in df.columns:
    print(f"\nVolume ratio statistics:")
    print(f"  Min: {df['volume_ratio'].min():.4f}")
    print(f"  25%: {df['volume_ratio'].quantile(0.25):.4f}")
    print(f"  50%: {df['volume_ratio'].median():.4f}")
    print(f"  75%: {df['volume_ratio'].quantile(0.75):.4f}")
    print(f"Volume between 0.5-2.0: {df['volume_ratio'].between(0.5, 2.0).sum()}")

# Check combined conditions
if all(col in df.columns for col in ['pattern_count', 'bbw', 'volume_ratio']):
    condition1 = df['pattern_count'] >= 2
    condition2 = df['bbw'] < 0.05
    condition3 = df['volume_ratio'].between(0.5, 2.0)

    print(f"\nCondition breakdown:")
    print(f"  pattern_count >= 2: {condition1.sum()}")
    print(f"  bbw < 0.05: {condition2.sum()}")
    print(f"  volume 0.5-2.0: {condition3.sum()}")
    print(f"  All conditions met: {(condition1 & condition2 & condition3).sum()}")

    # Show samples that meet criteria
    qualified = df[condition1 & condition2 & condition3]
    if len(qualified) > 0:
        print(f"\nExample qualified patterns:")
        print(qualified[['symbol', 'bbw', 'volume_ratio', 'pattern_count', 'breakout_class']].head())