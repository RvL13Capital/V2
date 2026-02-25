"""
Labeling Service for AIv3

Encapsulates pattern labeling logic with outcomes.
Provides clean service interface for labeling patterns with K0-K5 classifications.
"""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

from config import SystemConfig
from labeling.enhanced_pattern_labeler import EnhancedPatternLabeler

logger = logging.getLogger(__name__)


class LabelingService:
    """
    Service for labeling consolidation patterns with outcomes.

    Encapsulates:
    - Pattern labeling with K0-K5 classifications
    - Outcome calculation (max_gain, max_loss, etc.)
    - Class distribution analysis
    - Labeled pattern persistence

    Usage:
        labeling_service = LabelingService(config)
        labeled_df = labeling_service.label_patterns(
            patterns_df=patterns_df,
            price_data=price_data
        )

        # Get distribution
        distribution = labeling_service.get_class_distribution()

        # Save results
        labeling_service.save_labeled_patterns('labeled_patterns.csv')
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize labeling service.

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize labeler
        self.labeler = EnhancedPatternLabeler(
            lookforward_days=config.outcomes.lookforward_days,
            explosive_days_threshold=30,
            slow_grind_days_threshold=60
        )

        # Storage
        self.labeled_patterns = None
        self.class_distribution = None

        logger.info("LabelingService initialized")

    def label_patterns(
        self,
        patterns_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Label patterns with outcomes using snapshot-specific evaluation.

        Each training sample (snapshot) is evaluated from its snapshot_date + 100 days,
        not from the pattern's end_date. This ensures outcomes reflect what happens
        after each AI decision point.

        Args:
            patterns_df: DataFrame with training samples (may include multiple snapshots per pattern)
            price_data: Dict mapping ticker -> price DataFrame
            verbose: Print progress messages

        Returns:
            DataFrame with labeled samples (includes outcome_class, max_gain, etc.)
        """
        if patterns_df.empty:
            logger.warning("Empty patterns DataFrame provided")
            return pd.DataFrame()

        if verbose:
            print(f"\nLabeling {len(patterns_df)} training samples...")

        labeled_samples = []

        for idx, sample in patterns_df.iterrows():
            ticker = sample['ticker']

            # Check if we have price data for this ticker
            if ticker not in price_data:
                if verbose and idx < 10:  # Only warn for first 10
                    logger.warning(f"No price data for {ticker}, skipping sample")
                continue

            ticker_df = price_data[ticker]

            # CRITICAL: Only use patterns with snapshot_date
            # Old patterns without snapshots are REJECTED for data quality
            if 'snapshot_date' not in sample or pd.isna(sample['snapshot_date']):
                if verbose and idx < 10:
                    logger.warning(f"Skipping sample without snapshot_date: {ticker} at {sample.get('end_date', 'unknown')}")
                continue

            evaluation_date = pd.to_datetime(sample['snapshot_date'])
            evaluation_type = 'snapshot'

            # Find evaluation date in price data
            if evaluation_date not in ticker_df.index:
                if verbose and idx < 10:
                    logger.warning(f"Evaluation date {evaluation_date} not in price data for {ticker}")
                continue

            evaluation_idx = ticker_df.index.get_loc(evaluation_date)

            # Calculate outcome from evaluation_date + 100 days
            try:
                outcome_class, outcome_dict = self.labeler.label_pattern(
                    ticker_df,
                    evaluation_idx  # Start outcome evaluation from here
                )

                # Add outcome to sample
                sample_dict = sample.to_dict()
                # Use base_class (K0-K5) for training, keep refined_class for analysis
                sample_dict['outcome_class'] = outcome_dict.get('base_class', outcome_class.name)
                sample_dict['refined_class'] = outcome_class.name  # K4_EXPLOSIVE, K4_SLOW, etc.
                sample_dict['evaluation_date'] = evaluation_date  # Track which date was used
                sample_dict['evaluation_type'] = evaluation_type  # Track whether snapshot or end_date
                sample_dict.update(outcome_dict)
                labeled_samples.append(sample_dict)

            except Exception as e:
                logger.error(f"Error labeling sample {ticker} at {evaluation_date}: {e}")
                continue

            # Progress reporting
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Labeled {idx + 1}/{len(patterns_df)} samples...")

        # Convert to DataFrame
        self.labeled_patterns = pd.DataFrame(labeled_samples)

        # Calculate class distribution
        if not self.labeled_patterns.empty:
            self.class_distribution = self.labeler.analyze_class_distribution(
                self.labeled_patterns
            )

        if verbose:
            print(f"\n[OK] Labeled {len(self.labeled_patterns)} samples successfully")
            print(f"   All samples use snapshot-specific evaluation (100-day window from snapshot_date)")
            if self.class_distribution:
                print(f"   K4 (Exceptional): {self.class_distribution.get('k4_total', 0)}")
                print(f"   K3 (Strong): {self.class_distribution.get('k3_total', 0)}")
                print(f"   K2 (Quality): {self.class_distribution.get('k2_total', 0)}")

        return self.labeled_patterns

    def get_labeled_patterns(self) -> pd.DataFrame:
        """
        Get labeled patterns DataFrame.

        Returns:
            DataFrame with labeled patterns
        """
        if self.labeled_patterns is None:
            logger.warning("No patterns have been labeled yet")
            return pd.DataFrame()

        return self.labeled_patterns

    def get_class_distribution(self) -> Optional[Dict]:
        """
        Get class distribution statistics.

        Returns:
            Dict with class distribution (k0_total, k1_total, ..., k5_total, etc.)
        """
        return self.class_distribution

    def save_labeled_patterns(self, output_path: str) -> Path:
        """
        Save labeled patterns to CSV.

        Args:
            output_path: Path to save CSV file

        Returns:
            Path to saved file
        """
        if self.labeled_patterns is None or self.labeled_patterns.empty:
            logger.warning("No labeled patterns to save")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.labeled_patterns.to_csv(output_path, index=False)
        logger.info(f"Saved {len(self.labeled_patterns)} labeled patterns to {output_path}")

        return output_path

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for labeled patterns.

        Returns:
            Dict with summary statistics
        """
        if self.labeled_patterns is None or self.labeled_patterns.empty:
            return {}

        stats = {
            'total_patterns': len(self.labeled_patterns),
            'class_distribution': self.class_distribution or {},
            'avg_max_gain': self.labeled_patterns['max_gain'].mean() if 'max_gain' in self.labeled_patterns else None,
            'avg_max_loss': self.labeled_patterns['max_loss'].mean() if 'max_loss' in self.labeled_patterns else None,
        }

        # Add outcome class counts
        if 'outcome_class' in self.labeled_patterns.columns:
            for class_name in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
                count = (self.labeled_patterns['outcome_class'] == class_name).sum()
                stats[f'{class_name.lower()}_count'] = count

        return stats

    def format_summary_report(self) -> str:
        """
        Format summary report for labeled patterns.

        Returns:
            Formatted string report
        """
        if self.labeled_patterns is None or self.labeled_patterns.empty:
            return "No labeled patterns available"

        stats = self.get_summary_statistics()
        dist = self.class_distribution or {}

        report = []
        report.append("\n" + "="*60)
        report.append("PATTERN LABELING RESULTS")
        report.append("="*60)
        report.append(f"Total Patterns Labeled:  {stats['total_patterns']}")
        report.append("")
        report.append("Outcome Distribution:")
        report.append(f"  K4 (Exceptional >75%):  {dist.get('k4_total', 0)}")
        report.append(f"  K3 (Strong 35-75%):     {dist.get('k3_total', 0)}")
        report.append(f"  K2 (Quality 15-35%):    {dist.get('k2_total', 0)}")
        report.append(f"  K1 (Minimal 5-15%):     {dist.get('k1_total', 0)}")
        report.append(f"  K0 (Stagnant <5%):      {dist.get('k0_total', 0)}")
        report.append(f"  K5 (Failed):            {dist.get('k5_total', 0)}")
        report.append("")

        if stats['avg_max_gain'] is not None:
            report.append(f"Average Max Gain:        {stats['avg_max_gain']:.1%}")
        if stats['avg_max_loss'] is not None:
            report.append(f"Average Max Loss:        {stats['avg_max_loss']:.1%}")

        report.append("="*60)

        return "\n".join(report)

    def filter_by_outcome(
        self,
        min_outcome: str = 'K2',
        exclude_k5: bool = True
    ) -> pd.DataFrame:
        """
        Filter labeled patterns by outcome class.

        Args:
            min_outcome: Minimum outcome class (K2, K3, K4)
            exclude_k5: Exclude failed patterns (K5)

        Returns:
            Filtered DataFrame
        """
        if self.labeled_patterns is None or self.labeled_patterns.empty:
            return pd.DataFrame()

        if 'outcome_class' not in self.labeled_patterns.columns:
            logger.warning("No outcome_class column in labeled patterns")
            return self.labeled_patterns

        # Define outcome order
        outcome_order = {'K0': 0, 'K1': 1, 'K2': 2, 'K3': 3, 'K4': 4, 'K5': -1}
        min_value = outcome_order.get(min_outcome, 2)

        # Filter
        filtered = self.labeled_patterns[
            self.labeled_patterns['outcome_class'].map(outcome_order) >= min_value
        ].copy()

        # Exclude K5 if requested
        if exclude_k5:
            filtered = filtered[filtered['outcome_class'] != 'K5']

        logger.info(f"Filtered to {len(filtered)} patterns (min={min_outcome}, exclude_k5={exclude_k5})")

        return filtered
