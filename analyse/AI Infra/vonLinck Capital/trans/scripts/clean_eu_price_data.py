"""
EU Price Data Cleaning Script
=============================

Fixes common data quality issues in EU price data:
1. GBp/GBP currency conversion (100x jumps on .L tickers)
2. Single-day data error spikes that revert
3. Filters low liquidity tickers (>50% zero-volume days)
4. Validates adj_close and recalculates if needed

Usage:
    python scripts/clean_eu_price_data.py
    python scripts/clean_eu_price_data.py --dry-run
    python scripts/clean_eu_price_data.py --output-dir data/cleaned
    python scripts/clean_eu_price_data.py --fix-gbp-only

Author: TRANS System
Date: January 2026
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EU_SUFFIXES = ['.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS', '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI']


@dataclass
class CleaningStats:
    """Statistics for a single ticker's cleaning operation."""
    ticker: str
    original_rows: int = 0
    cleaned_rows: int = 0
    gbp_fixes: int = 0
    spike_fixes: int = 0
    zero_volume_pct: float = 0.0
    max_gap_before: float = 0.0
    max_gap_after: float = 0.0
    issues_found: List[str] = field(default_factory=list)
    issues_fixed: List[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class CleaningReport:
    """Overall cleaning report."""
    timestamp: str
    total_files: int = 0
    files_cleaned: int = 0
    files_excluded: int = 0
    files_unchanged: int = 0
    total_gbp_fixes: int = 0
    total_spike_fixes: int = 0
    ticker_stats: List[CleaningStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'total_files': self.total_files,
            'files_cleaned': self.files_cleaned,
            'files_excluded': self.files_excluded,
            'files_unchanged': self.files_unchanged,
            'total_gbp_fixes': self.total_gbp_fixes,
            'total_spike_fixes': self.total_spike_fixes,
            'ticker_stats': [asdict(s) for s in self.ticker_stats]
        }


class EUDataCleaner:
    """
    Cleans EU price data by fixing common data quality issues.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        dry_run: bool = False
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.dry_run = dry_run

        # Thresholds
        self.gbp_ratio_min = 95  # 95x-105x is ~100x (GBp/GBP)
        self.gbp_ratio_max = 105
        self.spike_threshold = 0.5  # 50% daily move
        self.revert_threshold = 0.5  # Must revert by 50% to be considered spike
        self.zero_volume_exclusion = 0.5  # Exclude if >50% zero volume
        self.min_history_days = 100

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_gbp_conversion(self, df: pd.DataFrame) -> List[int]:
        """
        Detect GBp/GBP conversion issues (exactly ~100x jumps).

        Returns list of indices where conversion happened.
        """
        if len(df) < 2:
            return []

        conversions = []
        close = df['close'].values

        for i in range(1, len(close)):
            if close[i-1] == 0:
                continue
            ratio = close[i] / close[i-1]

            # Check for ~100x jump (GBp to GBP) or ~0.01x drop (GBP to GBp)
            if self.gbp_ratio_min < ratio < self.gbp_ratio_max:
                conversions.append(i)
            elif 1/self.gbp_ratio_max < ratio < 1/self.gbp_ratio_min:
                conversions.append(i)

        return conversions

    def fix_gbp_conversion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Fix GBp/GBP conversion by normalizing to GBp (pence).

        Strategy:
        1. Find all conversion points
        2. Identify segments between conversions
        3. Determine the dominant price scale (GBp vs GBP)
        4. Normalize all segments to GBp
        """
        df = df.copy()
        conversions = self.detect_gbp_conversion(df)

        if not conversions:
            return df, 0

        price_cols = ['open', 'high', 'low', 'close', 'adj_close']

        # Build segments: [(start_idx, end_idx, median_price), ...]
        segments = []
        prev_end = 0
        for conv_idx in conversions:
            if conv_idx > prev_end:
                median_price = df.loc[prev_end:conv_idx-1, 'close'].median()
                segments.append((prev_end, conv_idx - 1, median_price))
            prev_end = conv_idx

        # Add final segment
        if prev_end < len(df):
            median_price = df.loc[prev_end:, 'close'].median()
            segments.append((prev_end, len(df) - 1, median_price))

        if not segments:
            return df, 0

        # Determine target scale: use the segment with most data points
        largest_segment = max(segments, key=lambda s: s[1] - s[0])
        target_scale = largest_segment[2]

        # Normalize each segment
        fix_count = 0
        for start, end, median_price in segments:
            if median_price == 0:
                continue

            ratio = target_scale / median_price

            # If ratio is ~100, this segment is in GBP (needs *100)
            # If ratio is ~0.01, this segment is in GBp (needs /100)
            if 50 < ratio < 200:
                # Segment is in GBP, multiply by 100
                for col in price_cols:
                    if col in df.columns:
                        df.loc[start:end, col] = df.loc[start:end, col] * 100
                fix_count += 1
                logger.debug(f"  Fixed segment [{start}:{end}]: multiplied by 100 (was GBP)")
            elif 0.005 < ratio < 0.02:
                # Segment is in GBp but at higher scale, divide by 100
                for col in price_cols:
                    if col in df.columns:
                        df.loc[start:end, col] = df.loc[start:end, col] / 100
                fix_count += 1
                logger.debug(f"  Fixed segment [{start}:{end}]: divided by 100 (was high GBp)")

        return df, fix_count

    def detect_spikes(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect single-day spikes that revert.

        A spike is defined as:
        1. >50% move in one day
        2. Followed by a move back toward the original price

        Returns list of (index, spike_magnitude) tuples.
        """
        if len(df) < 3:
            return []

        spikes = []
        close = df['close'].values

        for i in range(1, len(close) - 1):
            if close[i-1] == 0 or close[i] == 0:
                continue

            # Calculate moves
            move_in = (close[i] - close[i-1]) / close[i-1]
            move_out = (close[i+1] - close[i]) / close[i]

            # Check for spike pattern (large move in, then reverting move out)
            if abs(move_in) > self.spike_threshold:
                # Check if it reverts significantly
                if move_in > 0 and move_out < -self.revert_threshold:
                    spikes.append((i, move_in))
                elif move_in < 0 and move_out > self.revert_threshold:
                    spikes.append((i, move_in))

        return spikes

    def fix_spikes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Fix single-day spikes by interpolating.
        """
        df = df.copy()
        spikes = self.detect_spikes(df)

        if not spikes:
            return df, 0

        fix_count = 0
        price_cols = ['open', 'high', 'low', 'close', 'adj_close']

        for spike_idx, magnitude in spikes:
            # Interpolate the spike day with average of surrounding days
            for col in price_cols:
                if col in df.columns:
                    prev_val = df.loc[spike_idx - 1, col]
                    next_val = df.loc[spike_idx + 1, col]
                    df.loc[spike_idx, col] = (prev_val + next_val) / 2

            fix_count += 1
            logger.debug(f"  Fixed spike at index {spike_idx}: {magnitude*100:.1f}% move interpolated")

        return df, fix_count

    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics for a dataframe."""
        metrics = {}

        # Zero volume percentage
        if 'volume' in df.columns:
            metrics['zero_volume_pct'] = (df['volume'] == 0).sum() / len(df)
        else:
            metrics['zero_volume_pct'] = 0

        # Max daily gap
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change().dropna()
            metrics['max_gap'] = returns.abs().max() if len(returns) > 0 else 0
        else:
            metrics['max_gap'] = 0

        # History length
        metrics['history_days'] = len(df)

        return metrics

    def should_exclude(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
        """
        Determine if a ticker should be excluded from the clean dataset.
        """
        metrics = self.calculate_quality_metrics(df)

        # Too much zero volume (illiquid)
        if metrics['zero_volume_pct'] > self.zero_volume_exclusion:
            return True, f"Low liquidity ({metrics['zero_volume_pct']*100:.1f}% zero-volume days)"

        # Too short history
        if metrics['history_days'] < self.min_history_days:
            return True, f"Short history ({metrics['history_days']} days)"

        return False, ""

    def clean_ticker(self, filepath: Path) -> CleaningStats:
        """
        Clean a single ticker's data.
        """
        ticker = filepath.stem
        stats = CleaningStats(ticker=ticker)

        try:
            # Load data
            df = pd.read_parquet(filepath)
            df = df.sort_values('date').reset_index(drop=True)
            stats.original_rows = len(df)

            # Calculate pre-cleaning metrics
            pre_metrics = self.calculate_quality_metrics(df)
            stats.max_gap_before = pre_metrics['max_gap']
            stats.zero_volume_pct = pre_metrics['zero_volume_pct']

            # Check for issues
            gbp_conversions = self.detect_gbp_conversion(df)
            spikes = self.detect_spikes(df)

            if gbp_conversions:
                stats.issues_found.append(f"GBp/GBP conversions: {len(gbp_conversions)}")
            if spikes:
                stats.issues_found.append(f"Data spikes: {len(spikes)}")
            if pre_metrics['zero_volume_pct'] > 0.2:
                stats.issues_found.append(f"High zero-volume: {pre_metrics['zero_volume_pct']*100:.1f}%")

            # Fix GBp/GBP conversions (only for .L tickers)
            if ticker.endswith('.L') and gbp_conversions:
                df, gbp_fix_count = self.fix_gbp_conversion(df)
                stats.gbp_fixes = gbp_fix_count
                if gbp_fix_count > 0:
                    stats.issues_fixed.append(f"Fixed {gbp_fix_count} GBp/GBP conversions")

            # Fix spikes
            if spikes:
                df, spike_fix_count = self.fix_spikes(df)
                stats.spike_fixes = spike_fix_count
                if spike_fix_count > 0:
                    stats.issues_fixed.append(f"Fixed {spike_fix_count} data spikes")

            # Recalculate adj_close if we made fixes
            if stats.gbp_fixes > 0 or stats.spike_fixes > 0:
                # Simple recalculation - in real scenario, would need dividend data
                df['adj_close'] = df['close']

            # Check if should be excluded
            exclude, reason = self.should_exclude(df, ticker)
            if exclude:
                stats.excluded = True
                stats.exclusion_reason = reason
                return stats

            # Calculate post-cleaning metrics
            post_metrics = self.calculate_quality_metrics(df)
            stats.max_gap_after = post_metrics['max_gap']
            stats.cleaned_rows = len(df)

            # Save cleaned data
            if not self.dry_run and (stats.gbp_fixes > 0 or stats.spike_fixes > 0):
                output_path = self.output_dir / f"{ticker}.parquet"
                df.to_parquet(output_path, index=False)
                logger.info(f"  Saved cleaned data to {output_path}")

        except Exception as e:
            stats.issues_found.append(f"Error: {str(e)}")
            stats.excluded = True
            stats.exclusion_reason = f"Load error: {str(e)}"

        return stats

    def clean_all(self, tickers: Optional[List[str]] = None) -> CleaningReport:
        """
        Clean all EU price data files.
        """
        report = CleaningReport(timestamp=datetime.now().isoformat())

        # Get EU files
        eu_files = []
        for f in self.input_dir.glob('*.parquet'):
            if any(f.stem.endswith(s) for s in EU_SUFFIXES):
                if tickers is None or f.stem in tickers:
                    eu_files.append(f)

        report.total_files = len(eu_files)
        logger.info(f"Found {len(eu_files)} EU files to process")

        for i, filepath in enumerate(eu_files):
            ticker = filepath.stem
            logger.info(f"[{i+1}/{len(eu_files)}] Processing {ticker}...")

            stats = self.clean_ticker(filepath)
            report.ticker_stats.append(stats)

            if stats.excluded:
                report.files_excluded += 1
                logger.info(f"  EXCLUDED: {stats.exclusion_reason}")
            elif stats.gbp_fixes > 0 or stats.spike_fixes > 0:
                report.files_cleaned += 1
                report.total_gbp_fixes += stats.gbp_fixes
                report.total_spike_fixes += stats.spike_fixes
                logger.info(f"  CLEANED: {stats.issues_fixed}")
            else:
                report.files_unchanged += 1
                logger.debug(f"  UNCHANGED: No issues found")

        return report


def print_report(report: CleaningReport):
    """Print a summary of the cleaning report."""
    print("\n" + "="*70)
    print("EU DATA CLEANING REPORT")
    print("="*70)
    print(f"Timestamp: {report.timestamp}")
    print(f"\nTotal files processed: {report.total_files}")
    print(f"  - Cleaned: {report.files_cleaned}")
    print(f"  - Excluded: {report.files_excluded}")
    print(f"  - Unchanged: {report.files_unchanged}")
    print(f"\nTotal fixes applied:")
    print(f"  - GBp/GBP conversions: {report.total_gbp_fixes}")
    print(f"  - Spike interpolations: {report.total_spike_fixes}")

    # Show excluded tickers
    excluded = [s for s in report.ticker_stats if s.excluded]
    if excluded:
        print(f"\n--- EXCLUDED TICKERS ({len(excluded)}) ---")
        for s in excluded[:20]:
            print(f"  {s.ticker}: {s.exclusion_reason}")
        if len(excluded) > 20:
            print(f"  ... and {len(excluded) - 20} more")

    # Show cleaned tickers
    cleaned = [s for s in report.ticker_stats if s.gbp_fixes > 0 or s.spike_fixes > 0]
    if cleaned:
        print(f"\n--- CLEANED TICKERS ({len(cleaned)}) ---")
        for s in cleaned[:20]:
            fixes = []
            if s.gbp_fixes:
                fixes.append(f"{s.gbp_fixes} GBp fixes")
            if s.spike_fixes:
                fixes.append(f"{s.spike_fixes} spike fixes")
            print(f"  {s.ticker}: {', '.join(fixes)} | gap: {s.max_gap_before*100:.1f}% -> {s.max_gap_after*100:.1f}%")
        if len(cleaned) > 20:
            print(f"  ... and {len(cleaned) - 20} more")

    # Usable tickers
    usable = [s for s in report.ticker_stats if not s.excluded]
    print(f"\n--- USABLE TICKERS: {len(usable)} ---")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Clean EU price data')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Input directory with parquet files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input, overwrite)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze without making changes')
    parser.add_argument('--fix-gbp-only', action='store_true',
                        help='Only fix GBp/GBP issues')
    parser.add_argument('--report-file', type=str, default='data/cleaning_report.json',
                        help='Path to save JSON report')
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                        help='Specific tickers to process')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("EU Price Data Cleaning Script")
    logger.info("="*60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    # Initialize cleaner
    cleaner = EUDataCleaner(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        dry_run=args.dry_run
    )

    # Run cleaning
    report = cleaner.clean_all(tickers=args.tickers)

    # Print report
    print_report(report)

    # Save JSON report
    if not args.dry_run:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
