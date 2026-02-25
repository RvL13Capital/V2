"""
Analyze Snapshot vs Pattern-Level Evaluation
============================================
Check if early snapshots of successful patterns are being unfairly penalized
in our evaluation methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_pattern_evolution(df: pd.DataFrame):
    """
    Analyze how patterns evolve over time and check if early snapshots
    of eventually successful patterns are being mislabeled.
    """
    print("\n" + "="*60)
    print("PATTERN EVOLUTION ANALYSIS")
    print("="*60)

    # Group by ticker and pattern_id to track evolution
    pattern_groups = defaultdict(list)

    for idx, row in df.iterrows():
        # Create unique pattern identifier
        pattern_key = f"{row['ticker']}_{row.get('pattern_id', '')}_{row.get('upper_boundary', 0):.2f}"
        pattern_groups[pattern_key].append({
            'snapshot_date': row['snapshot_date'],
            'days_in_pattern': row.get('days_in_pattern', 0),
            'outcome_class': row.get('outcome_class', 'UNKNOWN'),
            'outcome_gain': row.get('outcome_gain', 0),
            'days_to_peak': row.get('days_to_peak', None)
        })

    # Analyze patterns with multiple snapshots
    multi_snapshot_patterns = {k: v for k, v in pattern_groups.items() if len(v) > 1}

    print(f"\nTotal unique patterns: {len(pattern_groups)}")
    print(f"Patterns with multiple snapshots: {len(multi_snapshot_patterns)}")

    # Track misclassification patterns
    evolving_k4_patterns = []
    evolving_k3_patterns = []
    inconsistent_patterns = []

    for pattern_id, snapshots in multi_snapshot_patterns.items():
        # Sort snapshots by date
        snapshots = sorted(snapshots, key=lambda x: x['snapshot_date'])

        # Get all outcomes for this pattern
        outcomes = [s['outcome_class'] for s in snapshots]
        unique_outcomes = set(outcomes)

        # Check if pattern eventually achieves K4 or K3
        final_outcome = snapshots[-1]['outcome_class']

        if 'K4_EXCEPTIONAL' in outcomes:
            # This pattern eventually achieved K4
            non_k4_count = sum(1 for o in outcomes if o != 'K4_EXCEPTIONAL')
            if non_k4_count > 0:
                evolving_k4_patterns.append({
                    'pattern_id': pattern_id,
                    'snapshots': snapshots,
                    'progression': outcomes,
                    'non_k4_snapshots': non_k4_count,
                    'total_snapshots': len(snapshots)
                })

        elif 'K3_STRONG' in outcomes:
            # This pattern eventually achieved K3
            non_k3_count = sum(1 for o in outcomes if o not in ['K3_STRONG', 'K4_EXCEPTIONAL'])
            if non_k3_count > 0:
                evolving_k3_patterns.append({
                    'pattern_id': pattern_id,
                    'snapshots': snapshots,
                    'progression': outcomes,
                    'non_k3_snapshots': non_k3_count,
                    'total_snapshots': len(snapshots)
                })

        # Check for inconsistent labeling
        if len(unique_outcomes) > 1:
            inconsistent_patterns.append({
                'pattern_id': pattern_id,
                'unique_outcomes': unique_outcomes,
                'progression': outcomes
            })

    # Report findings
    print("\n" + "-"*60)
    print("K4 PATTERN EVOLUTION")
    print("-"*60)

    if evolving_k4_patterns:
        total_k4_snapshots = sum(p['total_snapshots'] for p in evolving_k4_patterns)
        misclassified_k4_snapshots = sum(p['non_k4_snapshots'] for p in evolving_k4_patterns)

        print(f"K4 patterns with multiple snapshots: {len(evolving_k4_patterns)}")
        print(f"Total K4 pattern snapshots: {total_k4_snapshots}")
        print(f"Snapshots not labeled as K4: {misclassified_k4_snapshots} ({misclassified_k4_snapshots/total_k4_snapshots:.1%})")

        # Show examples
        print("\nExample K4 patterns with evolving labels:")
        for i, pattern in enumerate(evolving_k4_patterns[:3]):
            print(f"\nPattern {i+1}:")
            print(f"  Progression: {' -> '.join(pattern['progression'])}")
            print(f"  Days in pattern: {[s['days_in_pattern'] for s in pattern['snapshots']]}")

    print("\n" + "-"*60)
    print("K3 PATTERN EVOLUTION")
    print("-"*60)

    if evolving_k3_patterns:
        total_k3_snapshots = sum(p['total_snapshots'] for p in evolving_k3_patterns)
        misclassified_k3_snapshots = sum(p['non_k3_snapshots'] for p in evolving_k3_patterns)

        print(f"K3 patterns with multiple snapshots: {len(evolving_k3_patterns)}")
        print(f"Total K3 pattern snapshots: {total_k3_snapshots}")
        print(f"Snapshots not labeled as K3/K4: {misclassified_k3_snapshots} ({misclassified_k3_snapshots/total_k3_snapshots:.1%})")

        # Show examples
        print("\nExample K3 patterns with evolving labels:")
        for i, pattern in enumerate(evolving_k3_patterns[:3]):
            print(f"\nPattern {i+1}:")
            print(f"  Progression: {' -> '.join(pattern['progression'])}")
            print(f"  Days in pattern: {[s['days_in_pattern'] for s in pattern['snapshots']]}")

    return evolving_k4_patterns, evolving_k3_patterns, inconsistent_patterns


def analyze_temporal_labeling(df: pd.DataFrame):
    """
    Analyze if the labeling considers the eventual outcome of the pattern
    or just the immediate future from each snapshot.
    """
    print("\n" + "="*60)
    print("TEMPORAL LABELING ANALYSIS")
    print("="*60)

    # Check if early snapshots have lower outcome gains
    df['days_in_pattern'] = pd.to_numeric(df.get('days_in_pattern', 0), errors='coerce').fillna(0)

    # Group by days in pattern
    early_snapshots = df[df['days_in_pattern'] <= 15]
    mid_snapshots = df[(df['days_in_pattern'] > 15) & (df['days_in_pattern'] <= 30)]
    late_snapshots = df[df['days_in_pattern'] > 30]

    print("\nOutcome distribution by pattern age:")

    for name, subset in [("Early (<=15 days)", early_snapshots),
                         ("Mid (15-30 days)", mid_snapshots),
                         ("Late (>30 days)", late_snapshots)]:
        if len(subset) > 0:
            print(f"\n{name}: {len(subset)} snapshots")
            outcome_dist = subset['outcome_class'].value_counts()
            for outcome, count in outcome_dist.items():
                pct = count / len(subset) * 100
                print(f"  {outcome}: {count} ({pct:.2f}%)")

    # Check if K4/K3 patterns have different distributions at different ages
    k4_patterns = df[df['outcome_class'] == 'K4_EXCEPTIONAL']
    k3_patterns = df[df['outcome_class'] == 'K3_STRONG']

    if len(k4_patterns) > 0:
        print(f"\nK4 patterns - days in pattern distribution:")
        print(f"  Mean: {k4_patterns['days_in_pattern'].mean():.1f} days")
        print(f"  Median: {k4_patterns['days_in_pattern'].median():.1f} days")
        print(f"  Min: {k4_patterns['days_in_pattern'].min():.1f} days")
        print(f"  Max: {k4_patterns['days_in_pattern'].max():.1f} days")

    if len(k3_patterns) > 0:
        print(f"\nK3 patterns - days in pattern distribution:")
        print(f"  Mean: {k3_patterns['days_in_pattern'].mean():.1f} days")
        print(f"  Median: {k3_patterns['days_in_pattern'].median():.1f} days")
        print(f"  Min: {k3_patterns['days_in_pattern'].min():.1f} days")
        print(f"  Max: {k3_patterns['days_in_pattern'].max():.1f} days")


def propose_pattern_level_evaluation(evolving_k4_patterns, evolving_k3_patterns):
    """
    Propose a pattern-level evaluation approach.
    """
    print("\n" + "="*60)
    print("PROPOSED PATTERN-LEVEL EVALUATION")
    print("="*60)

    print("""
    Current Issue: Each snapshot is evaluated independently, causing:
    - Early snapshots of K4 patterns labeled as K0/K1/K2
    - Model penalized for correctly identifying future K4 patterns early
    - Artificially low K4 recall metrics

    Proposed Solution: Pattern-Level Evaluation
    1. Group all snapshots by unique pattern
    2. Use the pattern's FINAL outcome for ALL its snapshots
    3. Evaluate model's ability to identify patterns that WILL become K4
    4. Give credit for early detection of eventual winners

    Benefits:
    - More realistic assessment of predictive power
    - Rewards early pattern identification
    - Aligns with trading reality (we want early signals)
    """)

    if evolving_k4_patterns:
        total_k4_snapshots = sum(p['total_snapshots'] for p in evolving_k4_patterns)
        misclassified = sum(p['non_k4_snapshots'] for p in evolving_k4_patterns)

        print(f"\nPotential K4 Recall Improvement:")
        print(f"  Current approach: Counts {misclassified}/{total_k4_snapshots} snapshots as failures")
        print(f"  Pattern-level: Would count ALL {total_k4_snapshots} as K4 successes")
        print(f"  Potential improvement: {misclassified/total_k4_snapshots:.1%} more K4 samples recognized")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("SNAPSHOT VS PATTERN-LEVEL EVALUATION ANALYSIS")
    print("="*60)

    # Load the dataset
    data_file = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    print(f"\nLoading data from {data_file}")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} snapshots")

    # Analyze pattern evolution
    evolving_k4, evolving_k3, inconsistent = analyze_pattern_evolution(df)

    # Analyze temporal labeling
    analyze_temporal_labeling(df)

    # Propose solution
    propose_pattern_level_evaluation(evolving_k4, evolving_k3)

    # Calculate impact
    print("\n" + "="*60)
    print("IMPACT ASSESSMENT")
    print("="*60)

    if evolving_k4 or evolving_k3:
        print("\nIf we relabel snapshots based on eventual pattern outcome:")

        current_k4_count = (df['outcome_class'] == 'K4_EXCEPTIONAL').sum()
        current_k3_count = (df['outcome_class'] == 'K3_STRONG').sum()

        additional_k4 = sum(p['non_k4_snapshots'] for p in evolving_k4) if evolving_k4 else 0
        additional_k3 = sum(p['non_k3_snapshots'] for p in evolving_k3) if evolving_k3 else 0

        new_k4_count = current_k4_count + additional_k4
        new_k3_count = current_k3_count + additional_k3

        print(f"\nK4 samples:")
        print(f"  Current: {current_k4_count}")
        print(f"  After relabeling: {new_k4_count} (+{additional_k4}, {additional_k4/current_k4_count:.1%} increase)")

        print(f"\nK3 samples:")
        print(f"  Current: {current_k3_count}")
        print(f"  After relabeling: {new_k3_count} (+{additional_k3}, {additional_k3/current_k3_count:.1%} increase)")

        print(f"\nThis would provide {additional_k4 + additional_k3} more high-value training examples!")

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
    1. Implement pattern-level labeling:
       - Group snapshots by unique pattern ID
       - Apply final outcome to ALL snapshots of that pattern
       - Retrain models with consistent labels

    2. Adjust evaluation metrics:
       - Evaluate at pattern level, not snapshot level
       - Give credit for early detection
       - Track time-to-detection as bonus metric

    3. Consider temporal weighting:
       - Weight later snapshots higher (closer to outcome)
       - But still give credit for early detection

    This approach better reflects the true goal:
    Identifying patterns that WILL become exceptional (K4) as early as possible.
    """)


if __name__ == "__main__":
    main()