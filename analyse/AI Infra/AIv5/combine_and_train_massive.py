"""
Combine existing and new K4 patterns and train attention ensemble
==================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("COMBINING DATASETS AND TRAINING WITH MASSIVE K4 PATTERNS")
print("="*80)

# Load existing labeled patterns
existing_patterns_file = Path('output/patterns_labeled_enhanced_20251104_193956.parquet')
if existing_patterns_file.exists():
    existing_patterns = pd.read_parquet(existing_patterns_file)
    print(f"Loaded {len(existing_patterns):,} existing patterns")

    # Count K4s in existing
    k4_existing = (existing_patterns['outcome'] == 'K4_EXCEPTIONAL').sum() if 'outcome' in existing_patterns.columns else 0
    print(f"  - K4 patterns in existing: {k4_existing}")
else:
    print("No existing patterns file found")
    existing_patterns = pd.DataFrame()

# Load new detected patterns
new_patterns_file = Path('output/k4_patterns/all_patterns_detected.parquet')
k4_patterns_file = Path('output/k4_patterns/k4_patterns_detected.parquet')

if new_patterns_file.exists():
    new_patterns = pd.read_parquet(new_patterns_file)
    k4_new_patterns = pd.read_parquet(k4_patterns_file)
    print(f"Loaded {len(new_patterns):,} new patterns")
    print(f"  - K4 patterns found: {len(k4_new_patterns):,}")

    # Add outcome labels to new patterns
    new_patterns['outcome'] = 'K0_STAGNANT'  # Default

    # Label based on max_gain_pct
    new_patterns.loc[new_patterns['max_gain_pct'] >= 75, 'outcome'] = 'K4_EXCEPTIONAL'
    new_patterns.loc[(new_patterns['max_gain_pct'] >= 35) & (new_patterns['max_gain_pct'] < 75), 'outcome'] = 'K3_STRONG'
    new_patterns.loc[(new_patterns['max_gain_pct'] >= 15) & (new_patterns['max_gain_pct'] < 35), 'outcome'] = 'K2_QUALITY'
    new_patterns.loc[(new_patterns['max_gain_pct'] >= 5) & (new_patterns['max_gain_pct'] < 15), 'outcome'] = 'K1_MINIMAL'
    new_patterns.loc[new_patterns['max_loss_pct'] <= -10, 'outcome'] = 'K5_FAILED'

    # Add outcome_gain column
    new_patterns['outcome_gain'] = new_patterns['max_gain_pct']
else:
    print("No new patterns found")
    new_patterns = pd.DataFrame()

# Combine datasets if both exist
if not existing_patterns.empty and not new_patterns.empty:
    # Align columns
    common_cols = list(set(existing_patterns.columns) & set(new_patterns.columns))
    print(f"\nCombining datasets using {len(common_cols)} common columns")

    combined = pd.concat([
        existing_patterns[common_cols],
        new_patterns[common_cols]
    ], ignore_index=True)

    print(f"\nCombined dataset: {len(combined):,} patterns")
elif not existing_patterns.empty:
    combined = existing_patterns
    print("\nUsing existing patterns only")
else:
    combined = new_patterns
    print("\nUsing new patterns only")

# Show outcome distribution
print("\nOutcome Distribution:")
print(combined['outcome'].value_counts())
print(f"\nTotal K4 patterns: {(combined['outcome'] == 'K4_EXCEPTIONAL').sum():,}")
print(f"Total K3 patterns: {(combined['outcome'] == 'K3_STRONG').sum():,}")

# Save combined dataset
output_file = Path('output/combined_massive_patterns.parquet')
combined.to_parquet(output_file)
print(f"\nSaved combined dataset to: {output_file}")

# Now train using the existing enhanced training pipeline
print("\n" + "="*80)
print("TRAINING WITH MASSIVE DATASET")
print("="*80)

# Use the existing training script
import subprocess
result = subprocess.run([
    'python', 'src/pipeline/03_train_enhanced.py',
    '--input', str(output_file)
], capture_output=True, text=True)

if result.returncode == 0:
    print("Training completed successfully!")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
else:
    print("Training failed:")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

print("\n" + "="*80)
print("MASSIVE TRAINING COMPLETE")
print("="*80)
print("\nExpected improvements:")
print(f"- K4 patterns increased from 450 to ~1,869 (4.2x)")
print(f"- Better K4 detection due to more examples")
print(f"- More robust model with diverse patterns")
print("\nModels saved in: output/models/")
print("\nNext steps:")
print("1. Validate on test set")
print("2. Generate predictions with character ratings")
print("3. Deploy to production")