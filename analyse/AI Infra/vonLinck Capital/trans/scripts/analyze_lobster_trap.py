"""
Analyze untradeable (gap > 0.5R) pattern outcomes.

Purpose: Determine if 0.5R gap limit is optimal or too conservative.

The "Lobster Trap" refers to patterns where the price gaps up significantly
at market open, making it impossible for retail traders to enter at the
expected trigger price. These patterns are marked untradeable=True but
are still labeled and included in training data.

This script analyzes whether untradeable patterns have different success
rates than tradeable ones, which informs whether the 0.5R threshold should
be adjusted.

Output:
- Outcome distribution: tradeable vs untradeable patterns
- Success rate comparison (Target class = 2)
- Gap size correlation with outcomes
- Recommendation on threshold adjustment

Usage:
    python scripts/analyze_lobster_trap.py --patterns output/labeled_patterns.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Strategic values from CLAUDE.md
STRATEGIC_VALUES = {0: -2.0, 1: -0.1, 2: 5.0}
OUTCOME_NAMES = {0: 'Danger', 1: 'Noise', 2: 'Target'}


def analyze_lobster_trap(labeled_patterns_path: str) -> dict:
    """
    Compare outcomes of tradeable vs untradeable patterns.

    Args:
        labeled_patterns_path: Path to labeled_patterns.parquet

    Returns:
        dict with analysis results including lift, recommendation
    """
    path = Path(labeled_patterns_path)
    if not path.exists():
        logger.error(f"File not found: {labeled_patterns_path}")
        return {'error': 'file_not_found'}

    df = pd.read_parquet(labeled_patterns_path)

    # Check required columns
    if 'outcome_class' not in df.columns:
        logger.error("Missing required column: outcome_class")
        return {'error': 'missing_columns', 'missing': ['outcome_class']}

    # Infer untradeable from gap_size_r if column doesn't exist
    GAP_LIMIT_R = 0.5
    if 'untradeable' not in df.columns:
        if 'gap_size_r' in df.columns:
            logger.info(f"Inferring 'untradeable' from gap_size_r > {GAP_LIMIT_R}")
            df['untradeable'] = df['gap_size_r'] > GAP_LIMIT_R
        else:
            logger.error("Missing both 'untradeable' and 'gap_size_r' columns")
            return {'error': 'missing_columns', 'missing': ['untradeable', 'gap_size_r']}

    # Filter to only labeled patterns (outcome_class is not null)
    df_labeled = df[df['outcome_class'].notna()].copy()

    if len(df_labeled) == 0:
        logger.error("No labeled patterns found")
        return {'error': 'no_labeled_patterns'}

    # Ensure untradeable column is boolean (handle NaN as False)
    df_labeled['untradeable'] = df_labeled['untradeable'].fillna(False).astype(bool)

    # Separate tradeable vs untradeable
    tradeable = df_labeled[df_labeled['untradeable'] == False]
    untradeable = df_labeled[df_labeled['untradeable'] == True]

    results = {
        'total_patterns': len(df_labeled),
        'tradeable_count': len(tradeable),
        'untradeable_count': len(untradeable),
        'tradeable_pct': len(tradeable) / len(df_labeled) * 100,
        'untradeable_pct': len(untradeable) / len(df_labeled) * 100,
    }

    print(f"\n{'='*60}")
    print("LOBSTER TRAP ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal labeled patterns: {len(df_labeled):,}")
    print(f"  Tradeable:   {len(tradeable):,} ({results['tradeable_pct']:.1f}%)")
    print(f"  Untradeable: {len(untradeable):,} ({results['untradeable_pct']:.1f}%)")

    # Outcome distribution
    print(f"\n{'='*60}")
    print("OUTCOME DISTRIBUTION")
    print(f"{'='*60}")

    for name, subset in [("Tradeable", tradeable), ("Untradeable", untradeable)]:
        if len(subset) == 0:
            print(f"\n{name}: No patterns")
            continue

        danger = (subset['outcome_class'] == 0).sum()
        noise = (subset['outcome_class'] == 1).sum()
        target = (subset['outcome_class'] == 2).sum()

        danger_pct = danger / len(subset) * 100
        noise_pct = noise / len(subset) * 100
        target_pct = target / len(subset) * 100

        # Calculate expected value
        ev = (danger_pct/100 * STRATEGIC_VALUES[0] +
              noise_pct/100 * STRATEGIC_VALUES[1] +
              target_pct/100 * STRATEGIC_VALUES[2])

        print(f"\n{name} (n={len(subset):,}):")
        print(f"  Danger (0): {danger:,} ({danger_pct:.1f}%)")
        print(f"  Noise  (1): {noise:,} ({noise_pct:.1f}%)")
        print(f"  Target (2): {target:,} ({target_pct:.1f}%)")
        print(f"  Expected Value: {ev:.2f}")

        # Store results
        key = 'tradeable' if name == 'Tradeable' else 'untradeable'
        results[f'{key}_danger_pct'] = danger_pct
        results[f'{key}_noise_pct'] = noise_pct
        results[f'{key}_target_pct'] = target_pct
        results[f'{key}_ev'] = ev

    # Statistical comparison and recommendation
    if len(untradeable) > 0 and len(tradeable) > 0:
        tradeable_target_rate = (tradeable['outcome_class'] == 2).mean()
        untradeable_target_rate = (untradeable['outcome_class'] == 2).mean()

        lift = untradeable_target_rate / tradeable_target_rate if tradeable_target_rate > 0 else float('inf')

        results['tradeable_target_rate'] = tradeable_target_rate
        results['untradeable_target_rate'] = untradeable_target_rate
        results['lift'] = lift

        print(f"\n{'='*60}")
        print("TARGET RATE COMPARISON")
        print(f"{'='*60}")
        print(f"\nTarget Rate (Class 2):")
        print(f"  Tradeable:   {tradeable_target_rate*100:.1f}%")
        print(f"  Untradeable: {untradeable_target_rate*100:.1f}%")
        print(f"  Lift:        {lift:.2f}x")

        # EV comparison
        tradeable_ev = results.get('tradeable_ev', 0)
        untradeable_ev = results.get('untradeable_ev', 0)
        print(f"\nExpected Value:")
        print(f"  Tradeable:   {tradeable_ev:.2f}")
        print(f"  Untradeable: {untradeable_ev:.2f}")

        print(f"\n{'='*60}")
        print("RECOMMENDATION")
        print(f"{'='*60}")

        if lift > 1.5:
            recommendation = 'widen_threshold'
            print(f"\n[!] ALERT: Untradeable patterns have {lift:.1f}x higher success rate!")
            print("    Consider WIDENING gap limit for STRONG signal tiers.")
            print("    Current: 0.5R -> Suggested: 0.75R for EV > 5.0")
            print("\n    Rationale: These gaps represent strong momentum that")
            print("    often leads to successful breakouts. Retail may be able")
            print("    to catch some of these with limit orders above trigger.")
        elif lift > 1.0:
            recommendation = 'monitor'
            print(f"\n[i] Untradeable patterns slightly outperform ({lift:.2f}x).")
            print("    Current threshold reasonable, minor alpha left on table.")
            print("    No immediate action needed, but worth monitoring.")
        else:
            recommendation = 'maintain'
            print(f"\n[OK] Untradeable patterns underperform ({lift:.2f}x).")
            print("    Current 0.5R threshold is correct.")
            print("    Patterns that gap too much tend to fail - good to avoid.")

        results['recommendation'] = recommendation

    # Gap size correlation analysis
    if 'gap_size_r' in df_labeled.columns and len(untradeable) > 0:
        print(f"\n{'='*60}")
        print("GAP SIZE ANALYSIS (Untradeable Patterns Only)")
        print(f"{'='*60}")

        # Filter to untradeable patterns with valid gap_size_r
        gap_df = untradeable[untradeable['gap_size_r'].notna()].copy()

        if len(gap_df) > 0:
            # Bin by gap size
            bins = [0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
            labels = ['0.5-0.75R', '0.75-1.0R', '1.0-1.5R', '1.5-2.0R', '>2.0R']
            gap_df['gap_bin'] = pd.cut(gap_df['gap_size_r'], bins=bins, labels=labels)

            print(f"\nTarget rate by gap size:")
            gap_analysis = []

            for gap_bin in labels:
                subset = gap_df[gap_df['gap_bin'] == gap_bin]
                if len(subset) > 0:
                    target_rate = (subset['outcome_class'] == 2).mean() * 100
                    danger_rate = (subset['outcome_class'] == 0).mean() * 100
                    print(f"  Gap {gap_bin}: n={len(subset):,}, Target={target_rate:.1f}%, Danger={danger_rate:.1f}%")
                    gap_analysis.append({
                        'bin': gap_bin,
                        'count': len(subset),
                        'target_pct': target_rate,
                        'danger_pct': danger_rate
                    })

            results['gap_analysis'] = gap_analysis

            # Correlation between gap size and outcome
            if len(gap_df) >= 10:
                corr = gap_df['gap_size_r'].corr(gap_df['outcome_class'])
                print(f"\n  Gap size vs outcome correlation: {corr:.3f}")
                results['gap_outcome_correlation'] = corr

                if corr > 0.1:
                    print("  [i] Larger gaps correlate with better outcomes (momentum)")
                elif corr < -0.1:
                    print("  [i] Larger gaps correlate with worse outcomes (exhaustion)")
                else:
                    print("  [i] No significant correlation between gap size and outcome")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nKey findings:")
    print(f"  - {results['untradeable_pct']:.1f}% of patterns gap beyond 0.5R (untradeable)")

    if 'lift' in results:
        if results['lift'] > 1.0:
            print(f"  - Untradeable patterns have {results['lift']:.2f}x higher Target rate")
            alpha_lost = (results['untradeable_count'] *
                         (results['untradeable_target_rate'] - results['tradeable_target_rate']))
            print(f"  - Estimated extra Targets missed: ~{int(alpha_lost):,}")
        else:
            print(f"  - Tradeable patterns have {1/results['lift']:.2f}x higher Target rate")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze outcomes of untradeable (gap > 0.5R) patterns'
    )
    parser.add_argument(
        '--patterns',
        default='output/labeled_patterns.parquet',
        help='Path to labeled_patterns.parquet (default: output/labeled_patterns.parquet)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional path to save results as JSON'
    )
    args = parser.parse_args()

    results = analyze_lobster_trap(args.patterns)

    if args.output and 'error' not in results:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = {k: convert(v) for k, v in results.items()}

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
