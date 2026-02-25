"""
Feature Service for AIv3

Encapsulates feature extraction logic for patterns.
Provides clean service interface for extracting features from labeled patterns.
"""

from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import logging

from config import SystemConfig
from features.enhanced_tabular_features import EnhancedTabularFeatureExtractor
from utils.memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Service for extracting features from labeled patterns.

    Encapsulates:
    - Feature extraction from price data
    - Pattern-specific feature alignment
    - Memory optimization
    - Feature persistence

    Usage:
        feature_service = FeatureService(config)
        features_df = feature_service.extract_features(
            labeled_df=labeled_df,
            ticker_data=ticker_data
        )

        # Get feature names
        feature_names = feature_service.get_feature_names()

        # Save results
        feature_service.save_features('features.parquet')
    """

    def __init__(
        self,
        config: SystemConfig,
        memory_optimizer: Optional[MemoryOptimizer] = None
    ):
        """
        Initialize feature service.

        Args:
            config: System configuration
            memory_optimizer: Optional memory optimizer
        """
        self.config = config
        self.memory_optimizer = memory_optimizer

        # Initialize feature extractor
        # NOTE: Robust slopes remain disabled - require algorithmic rewrite for performance
        # Current implementation uses rolling().apply() with RANSAC (100 trials per window)
        # which results in 600K+ model fits even with data slicing optimization
        # TODO: Rewrite robust slopes using vectorized numpy operations or numba JIT
        self.feature_extractor = EnhancedTabularFeatureExtractor(
            use_robust_slopes=False,  # Disabled - needs performance optimization
            use_vol_of_vol=True,
            use_advanced_volume=True,
            use_microstructural=config.microstructural.enabled
        )

        # Storage
        self.features_df = None
        self.feature_names = None

        logger.info("FeatureService initialized")

    def extract_features(
        self,
        labeled_df: pd.DataFrame,
        ticker_data: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from labeled patterns.

        Args:
            labeled_df: DataFrame with labeled patterns
            ticker_data: Dict mapping ticker -> price DataFrame
            verbose: Print progress messages

        Returns:
            DataFrame with features (includes pattern metadata + features)
        """
        if labeled_df.empty:
            logger.warning("Empty labeled patterns DataFrame provided")
            return pd.DataFrame()

        if verbose:
            print(f"\nExtracting features for {len(labeled_df)} patterns...")

        unique_tickers = labeled_df['ticker'].unique()
        features_list = []

        for ticker in unique_tickers:
            ticker_patterns = labeled_df[labeled_df['ticker'] == ticker]

            # Get ticker data from pre-loaded dict
            if ticker not in ticker_data:
                if verbose:
                    logger.warning(f"No price data for {ticker}, skipping {len(ticker_patterns)} patterns")
                continue

            ticker_df = ticker_data[ticker]
            if ticker_df.empty:
                continue

            try:
                # OPTIMIZATION: Slice data to ±200 bars around pattern dates
                # This significantly speeds up feature extraction on long histories
                pattern_dates = pd.to_datetime(ticker_patterns['end_date'])
                min_date = pattern_dates.min()
                max_date = pattern_dates.max()

                # Find indices for min/max dates
                ticker_df_idx = ticker_df.index
                try:
                    min_idx = ticker_df_idx.get_loc(min_date)
                    max_idx = ticker_df_idx.get_loc(max_date)
                except KeyError:
                    # If exact dates not found, use searchsorted
                    min_idx = ticker_df_idx.searchsorted(min_date)
                    max_idx = ticker_df_idx.searchsorted(max_date)

                # Add ±200 bar buffer
                lookback_bars = 200
                lookforward_bars = 200
                slice_start = max(0, min_idx - lookback_bars)
                slice_end = min(len(ticker_df), max_idx + lookforward_bars + 1)

                # Slice the data
                sliced_df = ticker_df.iloc[slice_start:slice_end]

                if verbose and len(sliced_df) < len(ticker_df):
                    reduction_pct = (1 - len(sliced_df) / len(ticker_df)) * 100
                    logger.debug(f"  {ticker}: Sliced {len(ticker_df)} → {len(sliced_df)} bars ({reduction_pct:.1f}% reduction)")

                # Extract all features for this ticker (from sliced data)
                ticker_features = self.feature_extractor.extract_all_features(sliced_df)

                # Extract pattern-specific features
                for _, pattern in ticker_patterns.iterrows():
                    pattern_date = pd.to_datetime(pattern['end_date'])

                    if pattern_date not in ticker_features.index:
                        if verbose:
                            logger.warning(f"Pattern date {pattern_date} not in features for {ticker}")
                        continue

                    # Get features at pattern end
                    pattern_features = ticker_features.loc[pattern_date].to_dict()

                    # Add pattern metadata
                    pattern_features['ticker'] = ticker
                    pattern_features['pattern_id'] = pattern.get(
                        'pattern_id',
                        f"{ticker}_{pattern_date.strftime('%Y%m%d')}"
                    )
                    pattern_features['end_date'] = pattern_date
                    pattern_features['outcome_class'] = pattern.get('outcome_class', 'K0')
                    pattern_features['max_gain'] = pattern.get('max_gain', 0.0)
                    pattern_features['max_loss'] = pattern.get('max_loss', 0.0)

                    features_list.append(pattern_features)

                if verbose and len(features_list) % 50 == 0:
                    print(f"  Extracted features for {len(features_list)} patterns...")

            except Exception as e:
                logger.error(f"Error extracting features for {ticker}: {e}")
                continue

        # Convert to DataFrame
        self.features_df = pd.DataFrame(features_list)

        if verbose:
            print(f"\n[OK] Extracted features for {len(self.features_df)} patterns")

        # Optimize memory if optimizer provided
        if self.memory_optimizer and not self.features_df.empty:
            if verbose:
                print("  Optimizing memory usage...")
            self.features_df = self.memory_optimizer.optimize_dataframe(self.features_df)

        # Extract feature names (exclude metadata columns)
        if not self.features_df.empty:
            metadata_cols = ['ticker', 'pattern_id', 'end_date', 'outcome_class', 'max_gain', 'max_loss']
            self.feature_names = [
                col for col in self.features_df.columns
                if col not in metadata_cols
            ]

            if verbose:
                print(f"  Total features: {len(self.feature_names)}")

        return self.features_df

    def get_features(self) -> pd.DataFrame:
        """
        Get features DataFrame.

        Returns:
            DataFrame with features
        """
        if self.features_df is None:
            logger.warning("No features have been extracted yet")
            return pd.DataFrame()

        return self.features_df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names (excludes metadata).

        Returns:
            List of feature names
        """
        return self.feature_names or []

    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Get feature matrix (only feature columns, no metadata).

        Returns:
            DataFrame with only feature columns
        """
        if self.features_df is None or self.features_df.empty:
            return pd.DataFrame()

        if not self.feature_names:
            return pd.DataFrame()

        return self.features_df[self.feature_names]

    def get_metadata(self) -> pd.DataFrame:
        """
        Get metadata columns (ticker, pattern_id, outcome_class, etc.).

        Returns:
            DataFrame with only metadata columns
        """
        if self.features_df is None or self.features_df.empty:
            return pd.DataFrame()

        metadata_cols = ['ticker', 'pattern_id', 'end_date', 'outcome_class', 'max_gain', 'max_loss']
        available_cols = [col for col in metadata_cols if col in self.features_df.columns]

        return self.features_df[available_cols]

    def save_features(self, output_path: str, format: str = 'parquet') -> Path:
        """
        Save features to file.

        Args:
            output_path: Path to save file
            format: File format ('parquet' or 'csv')

        Returns:
            Path to saved file
        """
        if self.features_df is None or self.features_df.empty:
            logger.warning("No features to save")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'parquet':
            self.features_df.to_parquet(output_path, compression='snappy')
        elif format == 'csv':
            self.features_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(self.features_df)} feature rows to {output_path}")

        return output_path

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for extracted features.

        Returns:
            Dict with summary statistics
        """
        if self.features_df is None or self.features_df.empty:
            return {}

        stats = {
            'total_patterns': len(self.features_df),
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'tickers': self.features_df['ticker'].nunique() if 'ticker' in self.features_df else 0,
        }

        # Add outcome distribution
        if 'outcome_class' in self.features_df.columns:
            stats['outcome_distribution'] = self.features_df['outcome_class'].value_counts().to_dict()

        # Add memory usage
        memory_mb = self.features_df.memory_usage(deep=True).sum() / 1024 / 1024
        stats['memory_usage_mb'] = round(memory_mb, 2)

        return stats

    def format_summary_report(self) -> str:
        """
        Format summary report for extracted features.

        Returns:
            Formatted string report
        """
        if self.features_df is None or self.features_df.empty:
            return "No features available"

        stats = self.get_summary_statistics()

        report = []
        report.append("\n" + "="*60)
        report.append("FEATURE EXTRACTION RESULTS")
        report.append("="*60)
        report.append(f"Patterns with Features:  {stats['total_patterns']}")
        report.append(f"Unique Tickers:          {stats['tickers']}")
        report.append(f"Total Features:          {stats['total_features']}")
        report.append(f"Memory Usage:            {stats['memory_usage_mb']:.2f} MB")

        if 'outcome_distribution' in stats:
            report.append("")
            report.append("Outcome Distribution:")
            for outcome, count in sorted(stats['outcome_distribution'].items()):
                report.append(f"  {outcome}: {count}")

        report.append("="*60)

        return "\n".join(report)

    def validate_features(self) -> Dict:
        """
        Validate extracted features for common issues.

        Returns:
            Dict with validation results
        """
        if self.features_df is None or self.features_df.empty:
            return {'valid': False, 'error': 'No features extracted'}

        issues = []

        # Check for NaN values
        if self.feature_names:
            nan_counts = self.features_df[self.feature_names].isna().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if not nan_cols.empty:
                issues.append(f"{len(nan_cols)} features have NaN values")

        # Check for infinite values
        if self.feature_names:
            for col in self.feature_names:
                if pd.api.types.is_numeric_dtype(self.features_df[col]):
                    inf_count = pd.isinf(self.features_df[col]).sum()
                    if inf_count > 0:
                        issues.append(f"Feature '{col}' has {inf_count} infinite values")

        # Check for constant features
        if self.feature_names:
            for col in self.feature_names:
                if self.features_df[col].nunique() == 1:
                    issues.append(f"Feature '{col}' is constant")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'total_patterns': len(self.features_df)
        }
