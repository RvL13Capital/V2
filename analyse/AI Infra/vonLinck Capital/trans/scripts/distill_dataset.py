import numpy as np
import pandas as pd
from pathlib import Path
import os

def distill_dataset():
    print("Starting Dataset Distillation (Zero-RAM Mode)...")

    # 1. Configuration
    input_dir = Path("output/sequences/eu")
    seq_path = input_dir / "sequences_20251228_184303.npy"
    lbl_path = input_dir / "labels_20251228_184303.npy"
    meta_path = input_dir / "metadata_20251228_184303.parquet"

    # Output directory
    output_dir = input_dir / "distilled"
    output_dir.mkdir(exist_ok=True)

    # 2. Load Metadata (Lightweight)
    print(f"   Loading metadata index from {meta_path}...")
    df = pd.read_parquet(meta_path)

    # Create explicit index to map back to numpy arrays
    if '_seq_idx' not in df.columns:
        df['_seq_idx'] = np.arange(len(df))

    # 3. The "Golden Frame" Logic
    # Group by pattern_id and take the LAST entry (most mature window)
    # This removes the 5-10 overlapping windows preceding the breakout
    print("   Selecting 'Golden Frames' (Last window per pattern)...")

    # Create base pattern key from ticker + pattern_start_date
    df['base_pattern'] = df['ticker'] + '_' + df['pattern_start_date'].astype(str)

    if 'pattern_end_date' in df.columns:
        df['pattern_end_date'] = pd.to_datetime(df['pattern_end_date'])
        df = df.sort_values('pattern_end_date')

    df_golden = df.groupby('base_pattern').tail(1).sort_index()

    golden_indices = df_golden['_seq_idx'].values
    print(f"   Reduction: {len(df):,} -> {len(golden_indices):,} sequences")
    print(f"   Efficiency Gain: {100 * (1 - len(golden_indices)/len(df)):.1f}%")

    # 4. Memory-Mapped Extraction (Zero-RAM loading)
    print("   Slicing data arrays (Memory Mapped)...")
    X_all = np.load(seq_path, mmap_mode='r')
    y_all = np.load(lbl_path, mmap_mode='r')

    # Materialize only the clean subset into RAM
    X_golden = X_all[golden_indices].copy()
    y_golden = y_all[golden_indices].copy()

    # 5. Verify label distribution
    from collections import Counter
    orig_dist = Counter(y_all)
    gold_dist = Counter(y_golden)
    print(f"   Original labels: D={orig_dist[0]}, N={orig_dist[1]}, T={orig_dist[2]}")
    print(f"   Golden labels:   D={gold_dist[0]}, N={gold_dist[1]}, T={gold_dist[2]}")

    # Calculate percentages
    orig_total = sum(orig_dist.values())
    gold_total = sum(gold_dist.values())
    print(f"   Original %: D={orig_dist[0]/orig_total*100:.1f}%, N={orig_dist[1]/orig_total*100:.1f}%, T={orig_dist[2]/orig_total*100:.1f}%")
    print(f"   Golden %:   D={gold_dist[0]/gold_total*100:.1f}%, N={gold_dist[1]/gold_total*100:.1f}%, T={gold_dist[2]/gold_total*100:.1f}%")

    # 6. Save Clean Artifacts
    print("   Saving distilled artifacts...")
    np.save(output_dir / "sequences.npy", X_golden)
    np.save(output_dir / "labels.npy", y_golden)
    np.save(output_dir / "pattern_ids.npy", df_golden['pattern_id'].values)

    # Save metadata with split column for temporal integrity
    df_golden_out = df_golden[['ticker', 'label', 'pattern_start_date', 'pattern_end_date', 'pattern_id', 'market_phase']].copy()
    df_golden_out['_seq_idx'] = np.arange(len(df_golden_out))
    df_golden_out.to_parquet(output_dir / "metadata.parquet", index=False)

    print(f"\n   Saved to {output_dir}:")
    print(f"   - sequences.npy: {X_golden.shape} ({X_golden.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   - labels.npy: {y_golden.shape}")
    print(f"   - pattern_ids.npy: {len(df_golden)} patterns")
    print(f"   - metadata.parquet: {len(df_golden_out)} rows")

    return output_dir

if __name__ == "__main__":
    distill_dataset()
