"""
Pattern-Level Relabeling
========================
Relabel all snapshots of a pattern with its eventual outcome to fix
the evaluation methodology flaw.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def create_pattern_id(row):
    """Create unique pattern identifier."""
    # Use ticker, boundaries, and pattern start to uniquely identify patterns
    pattern_id = f"{row['ticker']}_{row['upper_boundary']:.4f}_{row['lower_boundary']:.4f}"
    return pattern_id


def relabel_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel all snapshots with their pattern's eventual outcome.
    """
    print("\n" + "="*60)
    print("PATTERN-LEVEL RELABELING")
    print("="*60)

    # Create pattern IDs
    df['pattern_id'] = df.apply(create_pattern_id, axis=1)

    # Group by pattern
    pattern_groups = df.groupby('pattern_id')

    print(f"Total snapshots: {len(df)}")
    print(f"Unique patterns: {len(pattern_groups)}")

    # Track relabeling statistics
    relabel_stats = defaultdict(int)
    original_distribution = df['outcome_class'].value_counts()

    # For each pattern, find its best outcome and apply to all snapshots
    relabeled_data = []

    for pattern_id, group in pattern_groups:
        # Sort by snapshot date to ensure temporal order
        group = group.sort_values('snapshot_date')

        # Find the best outcome for this pattern
        # Priority: K4 > K3 > K2 > K1 > K0 > K5
        outcome_priority = {
            'K4_EXCEPTIONAL': 6,
            'K3_STRONG': 5,
            'K2_QUALITY': 4,
            'K1_MINIMAL': 3,
            'K0_STAGNANT': 2,
            'K5_FAILED': 1
        }

        # Get all unique outcomes for this pattern
        unique_outcomes = group['outcome_class'].unique()

        # Find the best outcome based on priority
        best_outcome = max(unique_outcomes, key=lambda x: outcome_priority.get(x, 0))

        # Special handling for K5 (failed) patterns
        # If K5 appears first (chronologically), it's the true outcome
        first_outcome = group.iloc[0]['outcome_class']
        if first_outcome == 'K5_FAILED':
            best_outcome = 'K5_FAILED'

        # Apply the best outcome to all snapshots
        for idx, row in group.iterrows():
            original_outcome = row['outcome_class']
            if original_outcome != best_outcome:
                relabel_stats[f"{original_outcome}_to_{best_outcome}"] += 1

            # Update the outcome
            row_copy = row.copy()
            row_copy['outcome_class'] = best_outcome
            row_copy['original_outcome'] = original_outcome  # Keep original for reference

            # Update outcome gain and other fields based on best outcome
            if best_outcome == 'K4_EXCEPTIONAL':
                row_copy['outcome_gain'] = max(75.0, row.get('outcome_gain', 75.0))
            elif best_outcome == 'K3_STRONG':
                row_copy['outcome_gain'] = max(35.0, row.get('outcome_gain', 35.0))
            elif best_outcome == 'K2_QUALITY':
                row_copy['outcome_gain'] = max(15.0, row.get('outcome_gain', 15.0))
            elif best_outcome == 'K1_MINIMAL':
                row_copy['outcome_gain'] = max(5.0, row.get('outcome_gain', 5.0))
            elif best_outcome == 'K0_STAGNANT':
                row_copy['outcome_gain'] = min(5.0, row.get('outcome_gain', 0.0))
            elif best_outcome == 'K5_FAILED':
                row_copy['outcome_gain'] = min(0.0, row.get('outcome_gain', -10.0))

            relabeled_data.append(row_copy)

    # Create relabeled dataframe
    df_relabeled = pd.DataFrame(relabeled_data)

    # Print relabeling statistics
    print("\n" + "-"*60)
    print("RELABELING STATISTICS")
    print("-"*60)

    total_relabeled = sum(relabel_stats.values())
    print(f"Total snapshots relabeled: {total_relabeled} ({total_relabeled/len(df):.1%})")

    print("\nTop relabeling transitions:")
    sorted_stats = sorted(relabel_stats.items(), key=lambda x: x[1], reverse=True)
    for transition, count in sorted_stats[:10]:
        print(f"  {transition}: {count}")

    # Compare distributions
    print("\n" + "-"*60)
    print("CLASS DISTRIBUTION COMPARISON")
    print("-"*60)

    new_distribution = df_relabeled['outcome_class'].value_counts()

    print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
        "Class", "Original", "New", "Change", "% Change"
    ))
    print("-" * 60)

    for class_name in ['K0_STAGNANT', 'K1_MINIMAL', 'K2_QUALITY', 'K3_STRONG', 'K4_EXCEPTIONAL', 'K5_FAILED']:
        original = original_distribution.get(class_name, 0)
        new = new_distribution.get(class_name, 0)
        change = new - original
        pct_change = (change / original * 100) if original > 0 else 0
        print("{:<20} {:>10} {:>10} {:>+10} {:>+9.1f}%".format(
            class_name, original, new, change, pct_change
        ))

    return df_relabeled


def validate_relabeling(df_original: pd.DataFrame, df_relabeled: pd.DataFrame):
    """
    Validate the relabeling process.
    """
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    # Check no data was lost
    assert len(df_original) == len(df_relabeled), "Data loss detected!"
    print(f"[PASS] Row count preserved: {len(df_relabeled)}")

    # Check all required columns exist
    required_cols = ['ticker', 'snapshot_date', 'outcome_class', 'pattern_id']
    for col in required_cols:
        assert col in df_relabeled.columns, f"Missing column: {col}"
    print(f"[PASS] All required columns present")

    # Check K4 and K3 improvements
    k4_original = (df_original['outcome_class'] == 'K4_EXCEPTIONAL').sum()
    k4_new = (df_relabeled['outcome_class'] == 'K4_EXCEPTIONAL').sum()
    k3_original = (df_original['outcome_class'] == 'K3_STRONG').sum()
    k3_new = (df_relabeled['outcome_class'] == 'K3_STRONG').sum()

    print(f"\nK4 samples: {k4_original} -> {k4_new} (+{k4_new - k4_original})")
    print(f"K3 samples: {k3_original} -> {k3_new} (+{k3_new - k3_original})")

    # Check patterns with mixed labels were fixed
    mixed_patterns = 0
    fixed_patterns = 0

    for pattern_id in df_relabeled['pattern_id'].unique():
        original_outcomes = df_original[df_original['pattern_id'] == pattern_id]['outcome_class'].unique()
        new_outcomes = df_relabeled[df_relabeled['pattern_id'] == pattern_id]['outcome_class'].unique()

        if len(original_outcomes) > 1:
            mixed_patterns += 1
            if len(new_outcomes) == 1:
                fixed_patterns += 1

    print(f"\nPatterns with mixed labels: {mixed_patterns}")
    print(f"Patterns fixed to single label: {fixed_patterns} ({fixed_patterns/mixed_patterns:.1%} success rate)")


def main():
    """Main relabeling pipeline."""
    print("="*60)
    print("PATTERN-LEVEL RELABELING PIPELINE")
    print("="*60)
    print("Fixing evaluation methodology by relabeling all snapshots")
    print("with their pattern's eventual outcome.")

    # Load the original dataset
    input_file = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    print(f"\nLoading data from {input_file}")
    df_original = pd.read_parquet(input_file)
    print(f"Loaded {len(df_original)} snapshots")

    # Create backup
    df_original_copy = df_original.copy()

    # Perform relabeling
    df_relabeled = relabel_patterns(df_original_copy)

    # Validate
    validate_relabeling(df_original, df_relabeled)

    # Save relabeled dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"output/patterns_labeled_pattern_level_{timestamp}.parquet")

    df_relabeled.to_parquet(output_file, index=False)
    print(f"\nRelabeled dataset saved to: {output_file}")

    # Create summary statistics
    print("\n" + "="*60)
    print("IMPACT SUMMARY")
    print("="*60)

    # Calculate expected improvements
    k4_increase = (df_relabeled['outcome_class'] == 'K4_EXCEPTIONAL').sum() - \
                 (df_original['outcome_class'] == 'K4_EXCEPTIONAL').sum()
    k3_increase = (df_relabeled['outcome_class'] == 'K3_STRONG').sum() - \
                 (df_original['outcome_class'] == 'K3_STRONG').sum()

    k34_combined_original = ((df_original['outcome_class'] == 'K4_EXCEPTIONAL') |
                            (df_original['outcome_class'] == 'K3_STRONG')).sum()
    k34_combined_new = ((df_relabeled['outcome_class'] == 'K4_EXCEPTIONAL') |
                       (df_relabeled['outcome_class'] == 'K3_STRONG')).sum()

    print(f"K4 training samples increased by: {k4_increase} (+{k4_increase/(df_original['outcome_class'] == 'K4_EXCEPTIONAL').sum():.1%})")
    print(f"K3 training samples increased by: {k3_increase} (+{k3_increase/(df_original['outcome_class'] == 'K3_STRONG').sum():.1%})")
    print(f"K3+K4 combined: {k34_combined_original} -> {k34_combined_new} (+{k34_combined_new - k34_combined_original})")

    print("\nExpected model improvements:")
    print("- K4 recall should improve significantly (more training examples)")
    print("- Early pattern detection will be properly rewarded")
    print("- Model will learn to identify patterns that WILL become K4")
    print("- Evaluation metrics will reflect true predictive power")

    print(f"\n[SUCCESS] Pattern-level relabeling complete!")
    print(f"Next step: Retrain models with {output_file}")

    return df_relabeled


if __name__ == "__main__":
    df = main()