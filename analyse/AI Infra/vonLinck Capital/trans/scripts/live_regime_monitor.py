"""
Live Regime Monitor for TRAnS Pattern Trading

Monitors market regime using lagged pattern outcomes (2-month delay).
Only signals BULL regime when it's safe to trade patterns.

Usage:
    python scripts/live_regime_monitor.py --patterns output/labeled_patterns.parquet
    python scripts/live_regime_monitor.py --patterns output/labeled_patterns.parquet --alert-file alerts.json

Regime Definition:
    BULL:     danger_rate < 40% AND target_rate > 12%  -> TRADE
    BEAR:     danger_rate > 55%                        -> NO TRADE
    SIDEWAYS: everything else                          -> NO TRADE
"""

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.regime_detector import Regime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Regime thresholds (from regime_config)
DANGER_THRESHOLD = 0.40      # danger_rate < 40% for BULL
TARGET_THRESHOLD = 0.12      # target_rate > 12% for BULL
BEAR_DANGER_THRESHOLD = 0.55 # danger_rate > 55% = definite BEAR

# Lookback configuration
LOOKBACK_PATTERNS = 100      # Number of recent patterns to consider
OUTCOME_LAG_DAYS = 60        # 2-month lag for outcome knowledge (bias-free)

# Alert configuration
REGIME_CHANGE_COOLDOWN_DAYS = 7  # Don't spam alerts


@dataclass
class RegimeStatus:
    """Current regime status."""
    regime: str
    danger_rate: float
    target_rate: float
    noise_rate: float
    n_patterns: int
    reference_date: str
    outcome_cutoff_date: str
    recommendation: str
    confidence: str
    timestamp: str


def load_labeled_patterns(patterns_path: str) -> pd.DataFrame:
    """
    Load labeled patterns with outcome information.

    Supports multiple formats:
    - Sequence metadata (label column, pattern_end_date)
    - Raw labeled patterns (outcome_class column, end_date)
    """
    df = pd.read_parquet(patterns_path)

    # Normalize column names
    if 'label' in df.columns and 'outcome_class' not in df.columns:
        df['outcome_class'] = df['label']
    if 'end_date' in df.columns and 'pattern_end_date' not in df.columns:
        df['pattern_end_date'] = df['end_date']

    # Verify required columns exist
    if 'outcome_class' not in df.columns:
        raise ValueError(f"Missing outcome_class/label column. Found: {df.columns.tolist()}")
    if 'pattern_end_date' not in df.columns:
        raise ValueError(f"Missing pattern_end_date/end_date column. Found: {df.columns.tolist()}")

    # Parse dates
    df['pattern_end_date'] = pd.to_datetime(df['pattern_end_date'])

    # Print data summary
    n = len(df)
    danger = (df['outcome_class'] == 0).mean() * 100
    target = (df['outcome_class'] == 2).mean() * 100
    date_min = df['pattern_end_date'].min().strftime('%Y-%m-%d')
    date_max = df['pattern_end_date'].max().strftime('%Y-%m-%d')
    print(f"Data summary: n={n:,}, Danger={danger:.1f}%, Target={target:.1f}%")
    print(f"Date range: {date_min} to {date_max}")

    return df


def calculate_lagged_regime(
    patterns_df: pd.DataFrame,
    reference_date: datetime = None,
    lookback_patterns: int = LOOKBACK_PATTERNS,
    outcome_lag_days: int = OUTCOME_LAG_DAYS
) -> RegimeStatus:
    """
    Calculate regime using only outcomes known at reference_date.

    KEY: Only uses patterns where outcome was determined at least
    outcome_lag_days BEFORE reference_date to avoid look-ahead bias.

    Args:
        patterns_df: DataFrame with pattern_end_date and outcome_class
        reference_date: Date to calculate regime for (default: today)
        lookback_patterns: Number of recent patterns to consider
        outcome_lag_days: Days to wait for outcome to be known

    Returns:
        RegimeStatus with current regime classification
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Calculate outcome cutoff: only use patterns whose outcomes are known
    # Pattern outcome is known ~40 days after pattern_end_date (dynamic window)
    # Plus we add safety margin = outcome_lag_days
    outcome_cutoff = reference_date - timedelta(days=outcome_lag_days)

    # Filter to patterns with known outcomes
    # Pattern outcome is known when: pattern_end_date + outcome_window < outcome_cutoff
    # Simplified: pattern_end_date < outcome_cutoff (conservative)
    known_outcomes = patterns_df[
        patterns_df['pattern_end_date'] < outcome_cutoff
    ].copy()

    if len(known_outcomes) < 10:
        return RegimeStatus(
            regime="UNKNOWN",
            danger_rate=0.0,
            target_rate=0.0,
            noise_rate=0.0,
            n_patterns=len(known_outcomes),
            reference_date=reference_date.strftime('%Y-%m-%d'),
            outcome_cutoff_date=outcome_cutoff.strftime('%Y-%m-%d'),
            recommendation="INSUFFICIENT DATA - Cannot determine regime",
            confidence="NONE",
            timestamp=datetime.now().isoformat()
        )

    # Sort by date and take most recent N patterns
    known_outcomes = known_outcomes.sort_values('pattern_end_date', ascending=False)
    recent_patterns = known_outcomes.head(lookback_patterns)

    # Calculate rates
    n = len(recent_patterns)
    danger_rate = (recent_patterns['outcome_class'] == 0).sum() / n
    target_rate = (recent_patterns['outcome_class'] == 2).sum() / n
    noise_rate = (recent_patterns['outcome_class'] == 1).sum() / n

    # Classify regime
    if danger_rate < DANGER_THRESHOLD and target_rate > TARGET_THRESHOLD:
        regime = "BULL"
        recommendation = "TRADE: Bull regime confirmed. Execute top 20% GBM signals."
        confidence = "HIGH" if danger_rate < 0.30 and target_rate > 0.15 else "MEDIUM"
    elif danger_rate > BEAR_DANGER_THRESHOLD:
        regime = "BEAR"
        recommendation = "NO TRADE: Bear regime. Stay flat until danger < 40%."
        confidence = "HIGH" if danger_rate > 0.60 else "MEDIUM"
    else:
        regime = "SIDEWAYS"
        recommendation = "NO TRADE: Sideways regime. Patterns unreliable."
        confidence = "MEDIUM"

    # Date range of patterns used
    date_range_start = recent_patterns['pattern_end_date'].min().strftime('%Y-%m-%d')
    date_range_end = recent_patterns['pattern_end_date'].max().strftime('%Y-%m-%d')

    return RegimeStatus(
        regime=regime,
        danger_rate=round(danger_rate, 3),
        target_rate=round(target_rate, 3),
        noise_rate=round(noise_rate, 3),
        n_patterns=n,
        reference_date=reference_date.strftime('%Y-%m-%d'),
        outcome_cutoff_date=outcome_cutoff.strftime('%Y-%m-%d'),
        recommendation=recommendation,
        confidence=confidence,
        timestamp=datetime.now().isoformat()
    )


def calculate_regime_history(
    patterns_df: pd.DataFrame,
    start_date: datetime = None,
    end_date: datetime = None,
    step_days: int = 7
) -> pd.DataFrame:
    """
    Calculate regime history over time (for validation/charting).

    Args:
        patterns_df: DataFrame with pattern_end_date and outcome_class
        start_date: Start of history (default: 1 year ago)
        end_date: End of history (default: today)
        step_days: Days between regime calculations

    Returns:
        DataFrame with date, regime, danger_rate, target_rate
    """
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    history = []
    current_date = start_date

    while current_date <= end_date:
        status = calculate_lagged_regime(patterns_df, reference_date=current_date)
        history.append({
            'date': current_date,
            'regime': status.regime,
            'danger_rate': status.danger_rate,
            'target_rate': status.target_rate,
            'n_patterns': status.n_patterns
        })
        current_date += timedelta(days=step_days)

    return pd.DataFrame(history)


def check_regime_change(
    current_status: RegimeStatus,
    previous_status_file: str = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if regime has changed since last check.

    Returns:
        (changed, change_description)
    """
    if previous_status_file is None or not Path(previous_status_file).exists():
        return False, None

    with open(previous_status_file, 'r') as f:
        previous = json.load(f)

    prev_regime = previous.get('regime', 'UNKNOWN')
    curr_regime = current_status.regime

    if prev_regime != curr_regime:
        if prev_regime in ['BEAR', 'SIDEWAYS'] and curr_regime == 'BULL':
            return True, f"REGIME CHANGE: {prev_regime} -> BULL (Trading window OPEN)"
        elif prev_regime == 'BULL' and curr_regime in ['BEAR', 'SIDEWAYS']:
            return True, f"REGIME CHANGE: BULL -> {curr_regime} (Trading window CLOSED)"
        else:
            return True, f"REGIME CHANGE: {prev_regime} -> {curr_regime}"

    return False, None


def print_status(status: RegimeStatus, verbose: bool = True):
    """Print regime status to console."""
    print("\n" + "=" * 70)
    print("LIVE REGIME MONITOR - TRAnS Pattern Trading")
    print("=" * 70)

    # Regime indicator
    if status.regime == "BULL":
        indicator = "[BULL]  TRADE"
        color_start = ""  # Could add ANSI colors here
    elif status.regime == "BEAR":
        indicator = "[BEAR]  NO TRADE"
    else:
        indicator = "[SIDEWAYS]  NO TRADE"

    print(f"\nCurrent Regime: {indicator}")
    print(f"Confidence: {status.confidence}")
    print(f"\nReference Date: {status.reference_date}")
    print(f"Outcome Cutoff: {status.outcome_cutoff_date} (2-month lag)")

    if verbose:
        print(f"\n--- Pattern Statistics (last {status.n_patterns} patterns) ---")
        print(f"  Danger Rate: {status.danger_rate:.1%} (threshold: <{DANGER_THRESHOLD:.0%})")
        print(f"  Target Rate: {status.target_rate:.1%} (threshold: >{TARGET_THRESHOLD:.0%})")
        print(f"  Noise Rate:  {status.noise_rate:.1%}")

    print(f"\n>>> {status.recommendation}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Live Regime Monitor for TRAnS Pattern Trading'
    )
    parser.add_argument(
        '--patterns', type=str, required=True,
        help='Path to labeled patterns parquet file'
    )
    parser.add_argument(
        '--reference-date', type=str, default=None,
        help='Reference date for regime calculation (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--lookback', type=int, default=LOOKBACK_PATTERNS,
        help=f'Number of patterns to consider (default: {LOOKBACK_PATTERNS})'
    )
    parser.add_argument(
        '--lag-days', type=int, default=OUTCOME_LAG_DAYS,
        help=f'Outcome lag in days (default: {OUTCOME_LAG_DAYS})'
    )
    parser.add_argument(
        '--status-file', type=str, default=None,
        help='Path to save/load status JSON (for change detection)'
    )
    parser.add_argument(
        '--history', action='store_true',
        help='Calculate and print regime history for last year'
    )
    parser.add_argument(
        '--history-output', type=str, default=None,
        help='Save history to CSV file'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Only print regime (for scripting)'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON (for automation)'
    )

    args = parser.parse_args()

    # Load patterns
    patterns_df = load_labeled_patterns(args.patterns)
    print(f"Loaded {len(patterns_df):,} labeled patterns")

    # Parse reference date
    if args.reference_date:
        reference_date = datetime.strptime(args.reference_date, '%Y-%m-%d')
    else:
        reference_date = datetime.now()

    # Calculate current regime
    status = calculate_lagged_regime(
        patterns_df,
        reference_date=reference_date,
        lookback_patterns=args.lookback,
        outcome_lag_days=args.lag_days
    )

    # Check for regime change
    changed, change_msg = check_regime_change(status, args.status_file)

    # Output
    if args.json:
        output = asdict(status)
        output['regime_changed'] = changed
        output['change_message'] = change_msg
        print(json.dumps(output, indent=2))
    elif args.quiet:
        print(status.regime)
    else:
        if changed and change_msg:
            print("\n" + "*" * 70)
            print(f"*** ALERT: {change_msg} ***")
            print("*" * 70)

        print_status(status, verbose=True)

    # Save status for next comparison
    if args.status_file:
        with open(args.status_file, 'w') as f:
            json.dump(asdict(status), f, indent=2)
        if not args.quiet and not args.json:
            print(f"Status saved to: {args.status_file}")

    # Calculate history if requested
    if args.history or args.history_output:
        print("\nCalculating regime history (last 12 months)...")
        history = calculate_regime_history(
            patterns_df,
            start_date=reference_date - timedelta(days=365),
            end_date=reference_date,
            step_days=7
        )

        if args.history:
            print("\n--- REGIME HISTORY ---")
            print(history.to_string(index=False))

            # Summary
            regime_counts = history['regime'].value_counts()
            print("\nRegime Distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(history) * 100
                print(f"  {regime}: {count} weeks ({pct:.1f}%)")

        if args.history_output:
            history.to_csv(args.history_output, index=False)
            print(f"\nHistory saved to: {args.history_output}")

    # Return exit code based on regime (for automation)
    if status.regime == "BULL":
        return 0  # Success - can trade
    else:
        return 1  # Not trading conditions


if __name__ == '__main__':
    exit(main())
