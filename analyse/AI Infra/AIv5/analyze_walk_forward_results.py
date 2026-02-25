"""
Analyze Walk-Forward Validation Results
========================================
Load and analyze the complete walk-forward validation results.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(results_file: Path):
    """Load the walk-forward validation results."""
    print(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    model_type = data['model_type']
    features = data['features']

    print(f"\nModel type: {model_type}")
    print(f"Features used: {len(features)}")
    print(f"Total validation windows: {len(results)}")

    return results, model_type, features


def analyze_results(results, model_type):
    """Analyze walk-forward validation results in detail."""
    print("\n" + "="*60)
    print("DETAILED WALK-FORWARD ANALYSIS")
    print("="*60)

    # Extract all metrics
    all_windows = []

    for r in results:
        window_data = {
            'window': r['window'],
            'samples': r['samples'],
            'accuracy': r['accuracy']
        }

        if 'winner_recall' in r:
            window_data['winner_recall'] = r['winner_recall']
            window_data['winner_precision'] = r['winner_precision']
            window_data['winner_count'] = r['winner_count']

        all_windows.append(window_data)

    df = pd.DataFrame(all_windows)

    # BIG_WINNER statistics
    winner_windows = df[df['winner_recall'].notna()]

    print(f"\n{'-'*60}")
    print("BIG_WINNER DETECTION STATISTICS")
    print(f"{'-'*60}")

    if len(winner_windows) > 0:
        print(f"Windows with BIG_WINNER samples: {len(winner_windows)}")
        print(f"Total BIG_WINNER samples across all windows: {winner_windows['winner_count'].sum():.0f}")

        # Recall statistics
        print(f"\nBIG_WINNER Recall Statistics:")
        print(f"  Mean: {winner_windows['winner_recall'].mean():.1%}")
        print(f"  Median: {winner_windows['winner_recall'].median():.1%}")
        print(f"  Std Dev: {winner_windows['winner_recall'].std():.1%}")
        print(f"  Min: {winner_windows['winner_recall'].min():.1%}")
        print(f"  Max: {winner_windows['winner_recall'].max():.1%}")
        print(f"  25th percentile: {winner_windows['winner_recall'].quantile(0.25):.1%}")
        print(f"  75th percentile: {winner_windows['winner_recall'].quantile(0.75):.1%}")

        # Precision statistics
        precision_windows = winner_windows[winner_windows['winner_precision'] > 0]
        if len(precision_windows) > 0:
            print(f"\nBIG_WINNER Precision Statistics (when detected):")
            print(f"  Mean: {precision_windows['winner_precision'].mean():.1%}")
            print(f"  Median: {precision_windows['winner_precision'].median():.1%}")

        # Detection rate
        detected_windows = winner_windows[winner_windows['winner_recall'] > 0]
        detection_rate = len(detected_windows) / len(winner_windows)
        print(f"\nDetection Rate: {detection_rate:.1%} ({len(detected_windows)}/{len(winner_windows)} windows)")

        # Zero recall windows
        zero_recall_windows = winner_windows[winner_windows['winner_recall'] == 0]
        print(f"Zero Recall Windows: {len(zero_recall_windows)} ({len(zero_recall_windows)/len(winner_windows):.1%})")

    # Overall accuracy
    print(f"\n{'-'*60}")
    print("OVERALL ACCURACY STATISTICS")
    print(f"{'-'*60}")

    print(f"Mean Accuracy: {df['accuracy'].mean():.1%}")
    print(f"Median Accuracy: {df['accuracy'].median():.1%}")
    print(f"Std Dev: {df['accuracy'].std():.1%}")

    # Temporal analysis
    print(f"\n{'-'*60}")
    print("TEMPORAL PERFORMANCE ANALYSIS")
    print(f"{'-'*60}")

    # Extract dates from window names
    if len(winner_windows) > 0:
        winner_windows['date'] = winner_windows['window'].str.extract(r'Window_\d+_(\d{8})')[0]
        winner_windows['year'] = winner_windows['date'].str[:4]

        yearly_stats = winner_windows.groupby('year').agg({
            'winner_recall': ['mean', 'count'],
            'winner_count': 'sum'
        })

        print("\nBIG_WINNER Recall by Year:")
        for year, row in yearly_stats.iterrows():
            mean_recall = row['winner_recall']['mean']
            num_windows = row['winner_recall']['count']
            total_winners = row['winner_count']['sum']
            print(f"  {year}: {mean_recall:.1%} ({num_windows} windows, {total_winners:.0f} BIG_WINNER samples)")

    # Best and worst windows
    print(f"\n{'-'*60}")
    print("BEST PERFORMING WINDOWS")
    print(f"{'-'*60}")

    if len(winner_windows) > 0:
        best_windows = winner_windows.nlargest(5, 'winner_recall')
        for idx, row in best_windows.iterrows():
            print(f"  {row['window']}: {row['winner_recall']:.1%} recall, "
                  f"{row['winner_precision']:.1%} precision ({row['winner_count']:.0f} samples)")

    print(f"\n{'-'*60}")
    print("WORST PERFORMING WINDOWS")
    print(f"{'-'*60}")

    if len(winner_windows) > 0:
        # Only show windows with BIG_WINNER samples
        worst_windows = winner_windows.nsmallest(5, 'winner_recall')
        for idx, row in worst_windows.iterrows():
            print(f"  {row['window']}: {row['winner_recall']:.1%} recall ({row['winner_count']:.0f} samples)")

    # Production readiness assessment
    print(f"\n{'='*60}")
    print("PRODUCTION READINESS ASSESSMENT")
    print(f"{'='*60}")

    if len(winner_windows) > 0:
        avg_recall = winner_windows['winner_recall'].mean()
        std_recall = winner_windows['winner_recall'].std()
        median_recall = winner_windows['winner_recall'].median()

        print(f"\nExpected BIG_WINNER Detection:")
        print(f"  Average Recall: {avg_recall:.1%}")
        print(f"  Median Recall: {median_recall:.1%}")
        print(f"  Consistency (std): {std_recall:.1%}")
        print(f"  Detection Rate: {detection_rate:.1%}")

        # Compare to single hold-out validation
        single_validation_recall = 0.715  # 71.5% from LightGBM Combined
        print(f"\nComparison to Single Hold-Out Validation:")
        print(f"  Single validation: 71.5% recall")
        print(f"  Walk-forward average: {avg_recall:.1%}")
        print(f"  Difference: {avg_recall - single_validation_recall:.1%}")

        if avg_recall > 0.30 and std_recall < 0.20:
            print(f"\n  Status: ✓ ACCEPTABLE FOR PRODUCTION")
            print(f"  Moderate performance with reasonable consistency")
        elif avg_recall > 0.20:
            print(f"\n  Status: ~ NEEDS MONITORING")
            print(f"  Low performance, use with caution")
        else:
            print(f"\n  Status: X NEEDS IMPROVEMENT")
            print(f"  Performance below acceptable threshold")
            print(f"\nRecommendations:")
            print(f"  1. Model may be overfitting to recent data")
            print(f"  2. Consider regime-specific models (bull/bear/sideways)")
            print(f"  3. Add market context features (VIX, sentiment)")
            print(f"  4. Increase training data diversity")

    return df


def main():
    """Main analysis pipeline."""
    # Find latest results file
    results_files = list(Path("output").glob("walk_forward_results_*.pkl"))

    if not results_files:
        print("No walk-forward results found!")
        return

    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)

    # Load and analyze
    results, model_type, features = load_results(latest_file)
    df = analyze_results(results, model_type)

    # Save detailed analysis
    csv_file = latest_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")


if __name__ == "__main__":
    main()
