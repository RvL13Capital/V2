import pandas as pd
import numpy as np

df = pd.read_parquet('output/sequences_us_fixed.parquet')
print(f"Loaded {len(df)} sequences")

# Check first sequence
seq = np.array(df['sequence'].iloc[0])
mask = np.array(df['attention_mask'].iloc[0])

print(f"\nFirst sequence:")
print(f"  Shape: {seq.shape}")
print(f"  Expected: (50, 69)")
print(f"  Non-zero elements: {(seq != 0).sum()}")
print(f"  Attention mask sum: {mask.sum()}")
print(f"\nSample features (first valid timestep, first 10 features):")
print(f"  {seq[0, :10]}")

# Check all sequences
all_valid = 0
for i in range(len(df)):
    seq = np.array(df['sequence'].iloc[i])
    if seq.shape == (50, 69) and (seq != 0).sum() > 0:
        all_valid += 1

print(f"\n✓ Sequences with valid features: {all_valid}/{len(df)}")
print(f"✓ Sequences with correct shape (50, 69): {all_valid}/{len(df)}")
