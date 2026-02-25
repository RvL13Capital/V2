"""
Verify K4 Patterns from Race Logic Labeling
============================================
Quick verification that K4 patterns truly reached 75% gain from upper_boundary
BEFORE any K5 breach (10% below lower_boundary).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load labeled snapshots
labeled_file = Path('output/snapshots_labeled_race_20251104_192443.parquet')
df = pd.read_parquet(labeled_file)

print("=" * 70)
print("K4 PATTERN VERIFICATION")
print("=" * 70)

# Filter for K4 patterns
k4_patterns = df[df['outcome_class'] == 'K4_EXCEPTIONAL']
print(f"\nTotal K4 patterns found: {len(k4_patterns)}")

# Sample some K4 patterns for verification
sample_k4 = k4_patterns.sample(min(10, len(k4_patterns)), random_state=42)

print("\nSampling 10 K4 patterns for verification:")
print("-" * 50)

for idx, row in sample_k4.iterrows():
    ticker = row['ticker']
    snapshot_date = row['snapshot_date']
    upper_boundary = row['upper_boundary']
    lower_boundary = row['lower_boundary']

    # Calculate gain from upper_boundary
    max_gain = row['max_gain_from_upper_pct']
    max_loss = row['max_loss_from_lower_pct']
    breach_day = row['breach_day']
    k5_occurred = row['k5_occurred']
    k5_day = row.get('k5_day', None)

    print(f"\n{ticker} - Snapshot: {snapshot_date}")
    print(f"  Upper boundary: ${upper_boundary:.2f}")
    print(f"  Lower boundary: ${lower_boundary:.2f}")
    print(f"  Max gain from upper: {max_gain:.1f}% (should be >=75%)")
    print(f"  Max loss from lower: {max_loss:.1f}%")
    print(f"  K4 achieved on day: {breach_day}")
    print(f"  K5 occurred: {k5_occurred}")
    if k5_occurred:
        print(f"  K5 day: {k5_day} (should be AFTER K4 day {breach_day})")
        if k5_day <= breach_day:
            print("  WARNING: K5 occurred before or same day as K4!")

    # Verify K4 criteria
    if max_gain < 75:
        print("  WARNING: Max gain less than 75%!")
    if k5_occurred and k5_day is not None and k5_day < breach_day:
        print("  RACE LOGIC VIOLATION: K5 happened before K4!")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Overall statistics
print(f"\nAll K4 patterns (n={len(k4_patterns)}):")
print(f"  Average max gain: {k4_patterns['max_gain_from_upper_pct'].mean():.1f}%")
print(f"  Median max gain: {k4_patterns['max_gain_from_upper_pct'].median():.1f}%")
print(f"  Min max gain: {k4_patterns['max_gain_from_upper_pct'].min():.1f}%")
print(f"  Max max gain: {k4_patterns['max_gain_from_upper_pct'].max():.1f}%")

# Check for any violations
violations = k4_patterns[k4_patterns['max_gain_from_upper_pct'] < 75]
if len(violations) > 0:
    print(f"\nWARNING: Found {len(violations)} K4 patterns with gain < 75%!")

# Check race logic
k4_with_k5 = k4_patterns[k4_patterns['k5_occurred'] == True]
print(f"\nK4 patterns where K5 also occurred: {len(k4_with_k5)}")
print(f"  (These should have K4 happening BEFORE K5)")

# Distribution of days to K4 breach
print(f"\nDays to K4 breach:")
print(f"  Mean: {k4_patterns['breach_day'].mean():.1f} days")
print(f"  Median: {k4_patterns['breach_day'].median():.1f} days")
print(f"  Min: {k4_patterns['breach_day'].min()} days")
print(f"  Max: {k4_patterns['breach_day'].max()} days")

print("\nVerification complete!")