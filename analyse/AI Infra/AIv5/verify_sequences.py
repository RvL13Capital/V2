"""
Quick Verification of Built Sequences
======================================
Check if sequences have valid features (not all zeros)
"""
import pandas as pd
import numpy as np

# Load sequences
print("Loading sequences from output/sequences_us_fixed.parquet...")
try:
    df = pd.read_parquet('output/sequences_us_fixed.parquet')
    print(f"SUCCESS: Loaded {len(df)} sequences")
    print(f"Columns: {df.columns.tolist()}")

    # Check attention masks
    print("\n=== ATTENTION MASK ANALYSIS ===")
    if 'attention_mask' in df.columns:
        masks = np.stack(df['attention_mask'].values)
        mask_sums = masks.sum(axis=1)

        print(f"Total sequences: {len(masks)}")
        print(f"Sequences with all zeros (BAD): {(mask_sums == 0).sum()}")
        print(f"Sequences with valid data (GOOD): {(mask_sums > 0).sum()}")
        print(f"Mean valid timesteps per sequence: {mask_sums.mean():.1f}")
        print(f"Min valid timesteps: {mask_sums.min():.0f}")
        print(f"Max valid timesteps: {mask_sums.max():.0f}")

        if (mask_sums == 0).sum() > 0:
            print("\nERROR: Some sequences still have all-zero masks!")
        else:
            print("\nSUCCESS: All sequences have valid data!")

    # Check sequence features
    print("\n=== SEQUENCE FEATURES ANALYSIS ===")
    if 'sequence' in df.columns:
        sequences = np.stack(df['sequence'].values)
        print(f"Sequence shape: {sequences.shape} (should be (N, 50, 69))")

        # Check for all-zero sequences
        seq_sums = sequences.sum(axis=(1, 2))
        print(f"Sequences with all-zero features (BAD): {(seq_sums == 0).sum()}")
        print(f"Sequences with valid features (GOOD): {(seq_sums != 0).sum()}")

        if (seq_sums == 0).sum() > 0:
            print("\nERROR: Some sequences still have all-zero features!")
        else:
            print("\nSUCCESS: All sequences have non-zero features!")

        # Sample feature statistics
        print(f"\nFeature statistics (across all sequences):")
        print(f"Mean: {sequences.mean():.4f}")
        print(f"Std: {sequences.std():.4f}")
        print(f"Min: {sequences.min():.4f}")
        print(f"Max: {sequences.max():.4f}")

    # Check outcome distribution
    print("\n=== OUTCOME DISTRIBUTION ===")
    if 'outcome_class' in df.columns:
        print(df['outcome_class'].value_counts().sort_index())
        k4_count = (df['outcome_class'] == 4).sum()
        print(f"\nK4 (EXCEPTIONAL) sequences: {k4_count} ({k4_count/len(df)*100:.1f}%)")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE!")
    print("="*50)

except FileNotFoundError:
    print("ERROR: File not found - sequences not built yet or failed")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
