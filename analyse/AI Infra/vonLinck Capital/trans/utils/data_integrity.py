"""
Data Integrity Module for TRAnS Pipeline

Mandatory checks to prevent:
1. Row duplication (74x duplication bug)
2. Look-ahead bias (temporal leakage)
3. Feature leakage (outcome columns as features)
4. Insufficient statistical power

Usage:
    from utils.data_integrity import DataIntegrityChecker

    checker = DataIntegrityChecker(df)
    checker.run_all_checks()  # Raises DataIntegrityError if checks fail
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
import warnings


class DataIntegrityError(Exception):
    """Raised when data integrity checks fail."""
    pass


class DataIntegrityWarning(UserWarning):
    """Warning for non-critical integrity issues."""
    pass


class DataIntegrityChecker:
    """
    Mandatory data integrity checks for TRAnS pipeline.

    Enforces:
    - Deduplication (max 1.1x duplication ratio)
    - Temporal split integrity (no future data in training)
    - No feature leakage from outcome columns
    - Minimum statistical power (target events)
    - OHLCV data quality (negative prices, impossible relationships, splits)
    """

    # Columns that indicate outcome and should NEVER be features
    LEAKAGE_COLUMNS = frozenset([
        'breakout_class', 'label_3R', 'label_4R', 'max_r_achieved',
        'outcome_class', 'target_hit', 'stop_hit', 'final_r',
        'days_to_target', 'days_to_stop', 'breakout_date'
    ])

    # Maximum allowed duplication ratio
    MAX_DUPLICATION_RATIO = 1.1

    # Minimum target events for statistical power (increased per fic.txt)
    MIN_TARGET_EVENTS = 500
    MIN_TARGET_EVENTS_WARNING = 1000

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'pattern_end_date',
        ticker_col: str = 'ticker',
        label_col: str = 'label',
        target_label: int = 2,
        strict: bool = True
    ):
        """
        Initialize integrity checker.

        Args:
            df: DataFrame to check
            date_col: Column containing pattern date
            ticker_col: Column containing ticker symbol
            label_col: Column containing class label
            target_label: Value indicating target/success class
            strict: If True, raise errors. If False, only warn.
        """
        self.df = df.copy()
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.label_col = label_col
        self.target_label = target_label
        self.strict = strict

        # Normalize date column
        if self.date_col in self.df.columns:
            self.df['_date'] = pd.to_datetime(self.df[self.date_col])
        elif 'end_date' in self.df.columns:
            self.df['_date'] = pd.to_datetime(self.df['end_date'])
            self.date_col = 'end_date'

        self.issues = []
        self.warnings = []

    def _create_signature(self) -> pd.Series:
        """Create unique pattern signature for deduplication."""
        if self.ticker_col not in self.df.columns:
            raise DataIntegrityError(f"Ticker column '{self.ticker_col}' not found")

        if '_date' not in self.df.columns:
            raise DataIntegrityError(f"Date column '{self.date_col}' not found")

        # Include boundary columns if available for more precise dedup
        sig_parts = [
            self.df[self.ticker_col].astype(str),
            self.df['_date'].dt.strftime('%Y-%m-%d')
        ]

        if 'upper_boundary' in self.df.columns:
            sig_parts.append(self.df['upper_boundary'].round(2).astype(str))
        if 'lower_boundary' in self.df.columns:
            sig_parts.append(self.df['lower_boundary'].round(2).astype(str))

        return sig_parts[0].str.cat(sig_parts[1:], sep='_')

    def check_duplication(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for row duplication.

        Returns:
            (passed, details) tuple
        """
        signature = self._create_signature()
        unique_count = signature.nunique()
        total_count = len(self.df)
        dup_ratio = total_count / unique_count if unique_count > 0 else float('inf')

        details = {
            'total_rows': total_count,
            'unique_patterns': unique_count,
            'duplication_ratio': dup_ratio,
            'duplicate_rows': total_count - unique_count
        }

        passed = dup_ratio <= self.MAX_DUPLICATION_RATIO

        if not passed:
            msg = (f"CRITICAL: Data has {dup_ratio:.1f}x duplication! "
                   f"({total_count:,} rows, {unique_count:,} unique patterns). "
                   f"Call deduplicate() before analysis.")
            self.issues.append(msg)

        return passed, details

    def check_temporal_integrity(
        self,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
        train_mask: Optional[pd.Series] = None,
        val_mask: Optional[pd.Series] = None,
        test_mask: Optional[pd.Series] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for temporal leakage in train/val/test splits.

        Args:
            train_end: End date for training data (exclusive)
            val_end: End date for validation data (exclusive)
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            test_mask: Boolean mask for test data

        Returns:
            (passed, details) tuple
        """
        if '_date' not in self.df.columns:
            return True, {'status': 'skipped', 'reason': 'no date column'}

        details = {}
        issues = []

        # Method 1: Date-based splits
        if train_end is not None:
            train_end_dt = pd.to_datetime(train_end)

            if train_mask is None:
                train_mask = self.df['_date'] <= train_end_dt

            train_dates = self.df.loc[train_mask, '_date']

            # Check if any splits exist
            if val_mask is not None:
                val_dates = self.df.loc[val_mask, '_date']

                # Check: val dates should be AFTER train dates
                val_before_train = (val_dates <= train_dates.max()).sum()
                if val_before_train > 0:
                    pct = 100 * val_before_train / len(val_dates)
                    issues.append(f"TEMPORAL LEAKAGE: {val_before_train:,} val samples ({pct:.1f}%) "
                                 f"are before train max date ({train_dates.max().date()})")
                    details['val_leakage_count'] = val_before_train
                    details['val_leakage_pct'] = pct

            if test_mask is not None:
                test_dates = self.df.loc[test_mask, '_date']

                # Check: test dates should be AFTER val dates (or train if no val)
                cutoff = val_dates.max() if val_mask is not None else train_dates.max()
                test_before_cutoff = (test_dates <= cutoff).sum()
                if test_before_cutoff > 0:
                    pct = 100 * test_before_cutoff / len(test_dates)
                    issues.append(f"TEMPORAL LEAKAGE: {test_before_cutoff:,} test samples ({pct:.1f}%) "
                                 f"are before cutoff date ({cutoff.date()})")
                    details['test_leakage_count'] = test_before_cutoff
                    details['test_leakage_pct'] = pct

        # Method 2: Mask-based validation
        elif train_mask is not None:
            train_max = self.df.loc[train_mask, '_date'].max()
            details['train_max_date'] = train_max

            if val_mask is not None:
                val_min = self.df.loc[val_mask, '_date'].min()
                val_before = (self.df.loc[val_mask, '_date'] <= train_max).sum()
                details['val_min_date'] = val_min
                details['val_before_train_max'] = val_before

                if val_before > 0:
                    pct = 100 * val_before / val_mask.sum()
                    issues.append(f"TEMPORAL LEAKAGE: {val_before:,} val samples ({pct:.1f}%) "
                                 f"<= train max ({train_max.date()})")

            if test_mask is not None:
                test_min = self.df.loc[test_mask, '_date'].min()
                cutoff = self.df.loc[val_mask, '_date'].max() if val_mask is not None else train_max
                test_before = (self.df.loc[test_mask, '_date'] <= cutoff).sum()
                details['test_min_date'] = test_min
                details['test_before_cutoff'] = test_before

                if test_before > 0:
                    pct = 100 * test_before / test_mask.sum()
                    issues.append(f"TEMPORAL LEAKAGE: {test_before:,} test samples ({pct:.1f}%) "
                                 f"<= cutoff ({cutoff.date()})")

        passed = len(issues) == 0
        details['issues'] = issues
        self.issues.extend(issues)

        return passed, details

    def check_feature_leakage(
        self,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for outcome columns being used as features.

        Args:
            feature_columns: List of columns used as features

        Returns:
            (passed, details) tuple
        """
        if feature_columns is None:
            # Check all columns
            feature_columns = list(self.df.columns)

        leakage_found = []
        for col in feature_columns:
            if col.lower() in {c.lower() for c in self.LEAKAGE_COLUMNS}:
                leakage_found.append(col)

        details = {
            'checked_columns': len(feature_columns),
            'leakage_columns': leakage_found
        }

        passed = len(leakage_found) == 0

        if not passed:
            msg = (f"FEATURE LEAKAGE: Outcome columns used as features: {leakage_found}. "
                   f"These contain future information and must be excluded.")
            self.issues.append(msg)

        return passed, details

    def check_statistical_power(
        self,
        subset_mask: Optional[pd.Series] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for minimum statistical power (target events).

        Args:
            subset_mask: Optional mask for subset analysis

        Returns:
            (passed, details) tuple
        """
        if self.label_col not in self.df.columns:
            return True, {'status': 'skipped', 'reason': 'no label column'}

        df = self.df if subset_mask is None else self.df[subset_mask]

        target_count = (df[self.label_col] == self.target_label).sum()
        total_count = len(df)
        target_rate = target_count / total_count if total_count > 0 else 0

        details = {
            'total_samples': total_count,
            'target_events': target_count,
            'target_rate': target_rate,
            'min_required': self.MIN_TARGET_EVENTS,
            'recommended': self.MIN_TARGET_EVENTS_WARNING
        }

        if target_count < self.MIN_TARGET_EVENTS:
            msg = (f"INSUFFICIENT POWER: Only {target_count} target events "
                   f"(need {self.MIN_TARGET_EVENTS}+ for reliable analysis)")
            self.issues.append(msg)
            passed = False
        elif target_count < self.MIN_TARGET_EVENTS_WARNING:
            msg = (f"LOW POWER WARNING: {target_count} target events "
                   f"(recommend {self.MIN_TARGET_EVENTS_WARNING}+)")
            self.warnings.append(msg)
            passed = True
        else:
            passed = True

        return passed, details

    def check_class_balance(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check class distribution for severe imbalance.

        Returns:
            (passed, details) tuple
        """
        if self.label_col not in self.df.columns:
            return True, {'status': 'skipped', 'reason': 'no label column'}

        class_counts = self.df[self.label_col].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        details = {
            'class_distribution': class_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }

        passed = True
        if imbalance_ratio > 20:
            msg = f"SEVERE IMBALANCE: {imbalance_ratio:.1f}x ratio between classes"
            self.warnings.append(msg)

        return passed, details

    def check_ohlcv_quality(
        self,
        ohlcv_df: Optional[pd.DataFrame] = None,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume',
        price_col: Optional[str] = None,
        rolling_window: int = 50,
        outlier_std_threshold: float = 10.0,
        split_detection_threshold: float = 0.50
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate OHLCV data quality.

        Checks:
        - No negative volumes
        - No negative prices
        - High >= Low always
        - Close within [Low, High]
        - Open within [Low, High]
        - Detect potential missing splits (>50% single-day price jumps)
        - Flag extreme outliers (>10 std from rolling mean)

        Args:
            ohlcv_df: DataFrame with OHLCV data (uses self.df if None)
            open_col: Column name for open price
            high_col: Column name for high price
            low_col: Column name for low price
            close_col: Column name for close price
            volume_col: Column name for volume
            price_col: Optional column name for price (uses close if None)
            rolling_window: Window for rolling statistics
            outlier_std_threshold: Standard deviations for outlier detection
            split_detection_threshold: Threshold for split detection (0.50 = 50% jump)

        Returns:
            (passed, details) tuple with validation results
        """
        df = ohlcv_df if ohlcv_df is not None else self.df
        details = {
            'total_rows': len(df),
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }

        passed = True

        # Check for required columns
        required_cols = [open_col, high_col, low_col, close_col]
        available_cols = [c for c in required_cols if c in df.columns]

        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            details['status'] = 'partial'
            details['warnings'].append(f"Missing OHLCV columns: {missing}")

        # Use close as default price column
        if price_col is None:
            price_col = close_col if close_col in df.columns else None

        # 1. Check for negative volumes
        if volume_col in df.columns:
            neg_volume = (df[volume_col] < 0).sum()
            if neg_volume > 0:
                passed = False
                msg = f"NEGATIVE_VOLUME: {neg_volume} rows have negative volume"
                self.issues.append(msg)
                details['checks_failed'].append(('negative_volume', neg_volume))
            else:
                details['checks_passed'].append('no_negative_volume')
            details['negative_volume_count'] = neg_volume

        # 2. Check for negative prices
        neg_price_counts = {}
        for col in available_cols:
            neg_count = (df[col] < 0).sum()
            neg_price_counts[col] = neg_count
            if neg_count > 0:
                passed = False
                msg = f"NEGATIVE_PRICE: {neg_count} rows have negative {col}"
                self.issues.append(msg)
                details['checks_failed'].append(('negative_price', col, neg_count))

        if all(v == 0 for v in neg_price_counts.values()):
            details['checks_passed'].append('no_negative_prices')
        details['negative_price_counts'] = neg_price_counts

        # 3. Check High >= Low
        if high_col in df.columns and low_col in df.columns:
            invalid_hl = (df[high_col] < df[low_col]).sum()
            if invalid_hl > 0:
                passed = False
                msg = f"INVALID_HIGH_LOW: {invalid_hl} rows have High < Low"
                self.issues.append(msg)
                details['checks_failed'].append(('high_less_than_low', invalid_hl))
            else:
                details['checks_passed'].append('high_gte_low')
            details['high_less_than_low_count'] = invalid_hl

        # 4. Check Close within [Low, High]
        if all(c in df.columns for c in [close_col, low_col, high_col]):
            close_below_low = (df[close_col] < df[low_col]).sum()
            close_above_high = (df[close_col] > df[high_col]).sum()
            invalid_close = close_below_low + close_above_high

            if invalid_close > 0:
                passed = False
                msg = f"INVALID_CLOSE: {invalid_close} rows have Close outside [Low, High]"
                self.issues.append(msg)
                details['checks_failed'].append(('close_out_of_range', invalid_close))
            else:
                details['checks_passed'].append('close_in_range')
            details['close_out_of_range'] = {
                'below_low': close_below_low,
                'above_high': close_above_high
            }

        # 5. Check Open within [Low, High]
        if all(c in df.columns for c in [open_col, low_col, high_col]):
            open_below_low = (df[open_col] < df[low_col]).sum()
            open_above_high = (df[open_col] > df[high_col]).sum()
            invalid_open = open_below_low + open_above_high

            if invalid_open > 0:
                passed = False
                msg = f"INVALID_OPEN: {invalid_open} rows have Open outside [Low, High]"
                self.issues.append(msg)
                details['checks_failed'].append(('open_out_of_range', invalid_open))
            else:
                details['checks_passed'].append('open_in_range')
            details['open_out_of_range'] = {
                'below_low': open_below_low,
                'above_high': open_above_high
            }

        # 6. Detect potential missing splits (>50% single-day price jumps)
        if price_col and price_col in df.columns:
            pct_change = df[price_col].pct_change().abs()
            potential_splits = pct_change > split_detection_threshold
            split_count = potential_splits.sum()

            if split_count > 0:
                split_dates = df.index[potential_splits].tolist() if hasattr(df.index, 'tolist') else []
                self.warnings.append(
                    f"POTENTIAL_SPLITS: {split_count} potential unadjusted splits detected "
                    f"(>{split_detection_threshold*100:.0f}% single-day price change)"
                )
                details['warnings'].append(('potential_splits', split_count, split_dates[:10]))
            else:
                details['checks_passed'].append('no_potential_splits')
            details['potential_split_count'] = split_count

        # 7. Flag extreme outliers (>10 std from rolling mean)
        if price_col and price_col in df.columns and len(df) >= rolling_window:
            rolling_mean = df[price_col].rolling(rolling_window).mean()
            rolling_std = df[price_col].rolling(rolling_window).std()

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)

            z_scores = (df[price_col] - rolling_mean) / rolling_std
            outlier_mask = z_scores.abs() > outlier_std_threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outlier_dates = df.index[outlier_mask].tolist() if hasattr(df.index, 'tolist') else []
                self.warnings.append(
                    f"EXTREME_OUTLIERS: {outlier_count} price values exceed "
                    f"{outlier_std_threshold} std from rolling mean"
                )
                details['warnings'].append(('extreme_outliers', outlier_count, outlier_dates[:10]))
            else:
                details['checks_passed'].append('no_extreme_outliers')
            details['extreme_outlier_count'] = outlier_count

        # Summary
        details['passed'] = passed
        details['n_checks_passed'] = len(details['checks_passed'])
        details['n_checks_failed'] = len(details['checks_failed'])
        details['n_warnings'] = len(details['warnings'])

        return passed, details

    def check_target_distribution_per_split(
        self,
        train_mask: Optional[pd.Series] = None,
        val_mask: Optional[pd.Series] = None,
        test_mask: Optional[pd.Series] = None,
        min_targets_per_split: int = 50
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ensure each split has adequate Target samples.

        Args:
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            test_mask: Boolean mask for test data
            min_targets_per_split: Minimum Target events per split

        Returns:
            (passed, details) tuple
        """
        if self.label_col not in self.df.columns:
            return True, {'status': 'skipped', 'reason': 'no label column'}

        details = {
            'train': {},
            'val': {},
            'test': {},
            'min_required': min_targets_per_split
        }

        passed = True
        splits = [
            ('train', train_mask),
            ('val', val_mask),
            ('test', test_mask)
        ]

        for split_name, mask in splits:
            if mask is None:
                details[split_name] = {'status': 'not_provided'}
                continue

            split_df = self.df[mask]
            total = len(split_df)
            target_count = (split_df[self.label_col] == self.target_label).sum()
            target_rate = target_count / total if total > 0 else 0

            details[split_name] = {
                'total': total,
                'target_count': target_count,
                'target_rate': target_rate
            }

            if target_count < min_targets_per_split:
                passed = False
                msg = (f"INSUFFICIENT_TARGETS_{split_name.upper()}: "
                       f"Only {target_count} targets in {split_name} "
                       f"(need {min_targets_per_split}+)")
                self.issues.append(msg)
                details[split_name]['passed'] = False
            else:
                details[split_name]['passed'] = True

        details['all_passed'] = passed
        return passed, details

    def run_all_checks(
        self,
        train_mask: Optional[pd.Series] = None,
        val_mask: Optional[pd.Series] = None,
        test_mask: Optional[pd.Series] = None,
        feature_columns: Optional[List[str]] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
        min_targets_per_split: int = 50
    ) -> Dict[str, Any]:
        """
        Run all integrity checks.

        Args:
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            test_mask: Boolean mask for test data
            feature_columns: List of feature columns to check
            ohlcv_df: Optional OHLCV DataFrame for data quality checks
            min_targets_per_split: Minimum Target events per split

        Returns:
            Dictionary with all check results

        Raises:
            DataIntegrityError: If any critical check fails (when strict=True)
        """
        results = {}

        # 1. Duplication check (CRITICAL)
        passed, details = self.check_duplication()
        results['duplication'] = {'passed': passed, **details}

        # 2. Temporal integrity (CRITICAL if masks provided)
        if any(m is not None for m in [train_mask, val_mask, test_mask]):
            passed, details = self.check_temporal_integrity(
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            results['temporal_integrity'] = {'passed': passed, **details}

        # 3. Feature leakage (CRITICAL if features provided)
        if feature_columns is not None:
            passed, details = self.check_feature_leakage(feature_columns)
            results['feature_leakage'] = {'passed': passed, **details}

        # 4. Statistical power
        passed, details = self.check_statistical_power()
        results['statistical_power'] = {'passed': passed, **details}

        # 5. Class balance
        passed, details = self.check_class_balance()
        results['class_balance'] = {'passed': passed, **details}

        # 6. OHLCV data quality (if OHLCV data provided)
        if ohlcv_df is not None:
            passed, details = self.check_ohlcv_quality(ohlcv_df)
            results['ohlcv_quality'] = {'passed': passed, **details}

        # 7. Target distribution per split (if masks provided)
        if any(m is not None for m in [train_mask, val_mask, test_mask]):
            passed, details = self.check_target_distribution_per_split(
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                min_targets_per_split=min_targets_per_split
            )
            results['target_per_split'] = {'passed': passed, **details}

        # Summary
        results['summary'] = {
            'issues': self.issues,
            'warnings': self.warnings,
            'all_passed': len(self.issues) == 0
        }

        # Raise if strict mode and issues found
        if self.strict and self.issues:
            raise DataIntegrityError(
                f"Data integrity check failed with {len(self.issues)} issue(s):\n" +
                "\n".join(f"  - {issue}" for issue in self.issues)
            )

        # Warn for non-critical issues
        for warning in self.warnings:
            warnings.warn(warning, DataIntegrityWarning)

        return results

    def deduplicate(self) -> pd.DataFrame:
        """
        Return deduplicated DataFrame.

        Returns:
            DataFrame with duplicates removed
        """
        signature = self._create_signature()
        self.df['_signature'] = signature
        deduped = self.df.drop_duplicates(subset='_signature').drop(columns=['_signature'])

        if '_date' in deduped.columns:
            deduped = deduped.drop(columns=['_date'])

        return deduped


def enforce_temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = 'pattern_end_date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Enforce strict temporal split with no leakage.

    Args:
        df: DataFrame to split
        train_end: End date for training (inclusive)
        val_end: End date for validation (inclusive)
        date_col: Date column name

    Returns:
        (train_df, val_df, test_df) tuple

    Raises:
        DataIntegrityError: If split would cause leakage
    """
    # Normalize dates
    if date_col not in df.columns and 'end_date' in df.columns:
        date_col = 'end_date'

    df = df.copy()
    df['_split_date'] = pd.to_datetime(df[date_col])

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    # Create masks
    train_mask = df['_split_date'] <= train_end_dt
    val_mask = (df['_split_date'] > train_end_dt) & (df['_split_date'] <= val_end_dt)
    test_mask = df['_split_date'] > val_end_dt

    # Verify no overlap
    assert (train_mask & val_mask).sum() == 0, "Train/Val overlap detected!"
    assert (train_mask & test_mask).sum() == 0, "Train/Test overlap detected!"
    assert (val_mask & test_mask).sum() == 0, "Val/Test overlap detected!"

    # Split
    train_df = df[train_mask].drop(columns=['_split_date'])
    val_df = df[val_mask].drop(columns=['_split_date'])
    test_df = df[test_mask].drop(columns=['_split_date'])

    # Log split info
    print(f"Temporal Split:")
    print(f"  Train: {len(train_df):,} patterns (≤ {train_end})")
    print(f"  Val:   {len(val_df):,} patterns ({train_end} < date ≤ {val_end})")
    print(f"  Test:  {len(test_df):,} patterns (> {val_end})")

    # Verify temporal integrity
    checker = DataIntegrityChecker(df, date_col=date_col)
    checker.check_temporal_integrity(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    if checker.issues:
        raise DataIntegrityError(
            "Temporal split verification failed:\n" +
            "\n".join(f"  - {issue}" for issue in checker.issues)
        )

    return train_df, val_df, test_df


def validate_before_training(
    metadata_path: str,
    sequences_path: Optional[str] = None,
    train_end: str = '2022-12-31',
    val_end: str = '2023-12-31'
) -> bool:
    """
    Validate data integrity before training.

    Call this at the start of training scripts.

    Args:
        metadata_path: Path to metadata parquet
        sequences_path: Optional path to sequences H5 file
        train_end: Training cutoff date
        val_end: Validation cutoff date

    Returns:
        True if all checks pass

    Raises:
        DataIntegrityError: If critical checks fail
    """
    print("=" * 60)
    print("DATA INTEGRITY VALIDATION")
    print("=" * 60)

    # Load metadata
    df = pd.read_parquet(metadata_path)
    print(f"\nLoaded: {len(df):,} rows from {metadata_path}")

    # Run checker
    checker = DataIntegrityChecker(df, strict=True)

    # Check duplication
    passed, details = checker.check_duplication()
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] Duplication Check:")
    print(f"       Rows: {details['total_rows']:,}, Unique: {details['unique_patterns']:,}")
    print(f"       Ratio: {details['duplication_ratio']:.2f}x")

    if not passed:
        raise DataIntegrityError(
            f"Data has {details['duplication_ratio']:.1f}x duplication. "
            f"Deduplicate before training!"
        )

    # Check statistical power
    passed, details = checker.check_statistical_power()
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] Statistical Power:")
    print(f"       Target events: {details['target_events']:,}")
    print(f"       Target rate: {details['target_rate']*100:.1f}%")

    # Temporal split validation
    date_col = 'pattern_end_date' if 'pattern_end_date' in df.columns else 'end_date'
    df['_date'] = pd.to_datetime(df[date_col])

    train_mask = df['_date'] <= pd.to_datetime(train_end)
    val_mask = (df['_date'] > pd.to_datetime(train_end)) & (df['_date'] <= pd.to_datetime(val_end))
    test_mask = df['_date'] > pd.to_datetime(val_end)

    passed, details = checker.check_temporal_integrity(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] Temporal Integrity:")
    print(f"       Train: ≤ {train_end} ({train_mask.sum():,} patterns)")
    print(f"       Val:   {train_end} < date ≤ {val_end} ({val_mask.sum():,} patterns)")
    print(f"       Test:  > {val_end} ({test_mask.sum():,} patterns)")

    if not passed:
        raise DataIntegrityError(
            "Temporal leakage detected! " +
            "\n".join(details.get('issues', []))
        )

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)

    return True


# Convenience functions for pipeline integration
def deduplicate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Quick deduplication of metadata DataFrame."""
    checker = DataIntegrityChecker(df, strict=False)
    return checker.deduplicate()


def deduplicate_cross_ticker(
    df: pd.DataFrame,
    date_col: str = 'pattern_end_date',
    boundary_tolerance: float = 0.02,
    prefer_higher_volume: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Remove duplicate patterns across different tickers (ETF/cross-listing problem).

    ETFs tracking the same underlying (e.g., GLD/GLDG, SPY/VOO) have IDENTICAL
    price patterns with different ticker symbols. This creates data leakage
    as the model sees the "same" pattern multiple times.

    Args:
        df: DataFrame with pattern data
        date_col: Column containing pattern end date
        boundary_tolerance: Tolerance for boundary matching (as fraction, e.g., 0.02 = 2%)
        prefer_higher_volume: If True, keep pattern with higher volume when deduping
        verbose: Print deduplication stats

    Returns:
        DataFrame with cross-ticker duplicates removed
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Normalize date column
    if date_col not in df.columns and 'end_date' in df.columns:
        date_col = 'end_date'

    # Check required columns
    required = [date_col, 'upper_boundary', 'lower_boundary']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if verbose:
            print(f"[WARN] Cross-ticker dedup skipped - missing columns: {missing}")
        return df

    # Create cross-ticker signature (ignoring ticker)
    # Round boundaries to handle minor floating point differences
    df['_date_str'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    df['_upper_rounded'] = (df['upper_boundary'] / boundary_tolerance).round() * boundary_tolerance
    df['_lower_rounded'] = (df['lower_boundary'] / boundary_tolerance).round() * boundary_tolerance

    df['_cross_ticker_sig'] = (
        df['_date_str'] + '_' +
        df['_upper_rounded'].round(2).astype(str) + '_' +
        df['_lower_rounded'].round(2).astype(str)
    )

    original_count = len(df)
    unique_sigs = df['_cross_ticker_sig'].nunique()

    if unique_sigs == original_count:
        # No duplicates
        df = df.drop(columns=['_date_str', '_upper_rounded', '_lower_rounded', '_cross_ticker_sig'])
        if verbose:
            print(f"[INFO] Cross-ticker dedup: No duplicates found in {original_count:,} patterns")
        return df

    # Sort to determine which pattern to keep
    sort_cols = []
    ascending = []

    if prefer_higher_volume and 'avg_dollar_volume' in df.columns:
        sort_cols.append('avg_dollar_volume')
        ascending.append(False)  # Higher volume first
    elif prefer_higher_volume and 'volume' in df.columns:
        sort_cols.append('volume')
        ascending.append(False)

    # Secondary sort by ticker for deterministic results
    if 'ticker' in df.columns:
        sort_cols.append('ticker')
        ascending.append(True)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    # Keep first occurrence of each signature
    deduped = df.drop_duplicates(subset='_cross_ticker_sig', keep='first')

    # Clean up temp columns
    deduped = deduped.drop(columns=['_date_str', '_upper_rounded', '_lower_rounded', '_cross_ticker_sig'])

    removed_count = original_count - len(deduped)

    if verbose:
        print(f"[INFO] Cross-ticker dedup: {original_count:,} -> {len(deduped):,} patterns")
        print(f"       Removed {removed_count:,} cross-ticker duplicates ({100*removed_count/original_count:.1f}%)")

        # Show examples of duplicates removed
        if removed_count > 0:
            dup_sigs = df[df.duplicated(subset='_cross_ticker_sig', keep='first')]['_cross_ticker_sig'].head(5)
            if len(dup_sigs) > 0:
                print(f"       Example duplicate signatures: {dup_sigs.tolist()[:3]}")

    return deduped


def deduplicate_overlapping_windows(
    df: pd.DataFrame,
    date_col: str = 'pattern_end_date',
    ticker_col: str = 'ticker',
    overlap_days: int = 5,
    prefer_col: str = 'coil_intensity',
    prefer_higher: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Remove patterns with overlapping time windows for the same ticker.

    When patterns overlap in time (e.g., pattern ending Jan 10 and Jan 12 for same ticker),
    they represent essentially the same consolidation viewed at different points.
    This causes data leakage as correlated patterns end up in train/test.

    Args:
        df: DataFrame with pattern data
        date_col: Column containing pattern end date
        ticker_col: Column containing ticker symbol
        overlap_days: Days threshold for considering patterns overlapping
        prefer_col: Column to use for selecting which pattern to keep
        prefer_higher: If True, keep pattern with higher prefer_col value
        verbose: Print deduplication stats

    Returns:
        DataFrame with overlapping patterns removed
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Normalize date column
    if date_col not in df.columns and 'end_date' in df.columns:
        date_col = 'end_date'

    if date_col not in df.columns or ticker_col not in df.columns:
        if verbose:
            print(f"[WARN] Overlapping window dedup skipped - missing columns")
        return df

    df['_date'] = pd.to_datetime(df[date_col])

    original_count = len(df)
    keep_mask = pd.Series(True, index=df.index)

    # Process each ticker
    for ticker in df[ticker_col].unique():
        ticker_mask = df[ticker_col] == ticker
        ticker_df = df[ticker_mask].sort_values('_date')

        if len(ticker_df) <= 1:
            continue

        ticker_indices = ticker_df.index.tolist()
        dates = ticker_df['_date'].tolist()

        # Find overlapping patterns using greedy approach
        i = 0
        while i < len(ticker_indices):
            current_date = dates[i]
            current_idx = ticker_indices[i]

            # Find all patterns within overlap_days
            cluster = [current_idx]
            j = i + 1
            while j < len(ticker_indices):
                if (dates[j] - current_date).days <= overlap_days:
                    cluster.append(ticker_indices[j])
                    j += 1
                else:
                    break

            if len(cluster) > 1:
                # Multiple overlapping patterns - keep the best one
                cluster_df = df.loc[cluster]

                if prefer_col in cluster_df.columns:
                    if prefer_higher:
                        best_idx = cluster_df[prefer_col].idxmax()
                    else:
                        best_idx = cluster_df[prefer_col].idxmin()
                else:
                    # Default: keep first (earliest)
                    best_idx = cluster[0]

                # Mark others for removal
                for idx in cluster:
                    if idx != best_idx:
                        keep_mask[idx] = False

            i = j if j > i + 1 else i + 1

    deduped = df[keep_mask].drop(columns=['_date'])
    removed_count = original_count - len(deduped)

    if verbose:
        print(f"[INFO] Overlapping window dedup ({overlap_days}d threshold): {original_count:,} -> {len(deduped):,}")
        print(f"       Removed {removed_count:,} overlapping patterns ({100*removed_count/original_count:.1f}%)")

    return deduped


def full_deduplication(
    df: pd.DataFrame,
    cross_ticker: bool = True,
    overlapping_windows: bool = True,
    overlap_days: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply full deduplication pipeline.

    1. Cross-ticker dedup (ETF/cross-listing problem)
    2. Overlapping window dedup (same ticker, nearby dates)
    3. Standard row dedup (exact duplicates)

    Args:
        df: DataFrame with pattern data
        cross_ticker: Remove cross-ticker duplicates
        overlapping_windows: Remove overlapping window duplicates
        overlap_days: Days threshold for overlapping windows
        verbose: Print stats

    Returns:
        Fully deduplicated DataFrame
    """
    original_count = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print("FULL DEDUPLICATION PIPELINE")
        print(f"{'='*60}")
        print(f"Input: {original_count:,} patterns")

    result = df.copy()

    # Step 1: Cross-ticker dedup
    if cross_ticker:
        result = deduplicate_cross_ticker(result, verbose=verbose)

    # Step 2: Overlapping windows dedup
    if overlapping_windows:
        result = deduplicate_overlapping_windows(result, overlap_days=overlap_days, verbose=verbose)

    # Step 3: Standard row dedup
    checker = DataIntegrityChecker(result, strict=False)
    result = checker.deduplicate()

    final_count = len(result)
    total_removed = original_count - final_count

    if verbose:
        print(f"\n{'='*60}")
        print(f"DEDUP COMPLETE: {original_count:,} -> {final_count:,} patterns")
        print(f"Total removed: {total_removed:,} ({100*total_removed/original_count:.1f}%)")
        print(f"{'='*60}\n")

    return result


def assert_no_duplicates(df: pd.DataFrame, context: str = ""):
    """Assert that DataFrame has no significant duplication."""
    checker = DataIntegrityChecker(df, strict=True)
    passed, details = checker.check_duplication()
    if not passed:
        ctx = f" ({context})" if context else ""
        raise DataIntegrityError(
            f"Duplicate check failed{ctx}: {details['duplication_ratio']:.1f}x duplication"
        )


def assert_temporal_split(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series
):
    """Assert that temporal split has no leakage."""
    checker = DataIntegrityChecker(df, strict=True)
    passed, details = checker.check_temporal_integrity(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    if not passed:
        raise DataIntegrityError(
            "Temporal leakage detected:\n" +
            "\n".join(details.get('issues', []))
        )
