"""
Drift Monitor - Feature and Prediction Drift Detection
=======================================================

Monitors for distribution shifts between training and production:
- PSI (Population Stability Index) for feature distributions
- KS-Statistic for distribution comparison
- Prediction drift monitoring
- Alert thresholds and reporting

Designed to catch feature mismatch before it impacts predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# PSI thresholds (industry standard)
PSI_THRESHOLDS = {
    'low': 0.1,      # < 0.1: No significant shift
    'medium': 0.2,   # 0.1-0.2: Moderate shift, monitor
    'high': 0.25,    # > 0.25: Significant shift, action required
}

# KS thresholds
KS_THRESHOLDS = {
    'low': 0.1,
    'medium': 0.2,
    'high': 0.3,
}


@dataclass
class FeatureDriftResult:
    """Result of drift analysis for a single feature."""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    mean_shift: float
    std_shift: float
    alert_level: str  # 'ok', 'warning', 'critical'
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'psi': self.psi,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'mean_shift': self.mean_shift,
            'std_shift': self.std_shift,
            'alert_level': self.alert_level,
            'details': self.details,
        }


@dataclass
class DriftReport:
    """Complete drift analysis report."""
    timestamp: datetime
    n_features_analyzed: int
    n_samples_reference: int
    n_samples_current: int

    # Feature-level results
    feature_results: List[FeatureDriftResult]

    # Summary metrics
    features_ok: int = 0
    features_warning: int = 0
    features_critical: int = 0

    # Overall alert
    overall_alert: str = 'ok'
    action_required: bool = False

    # Prediction drift (if available)
    prediction_psi: Optional[float] = None
    prediction_ks: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'n_features_analyzed': self.n_features_analyzed,
            'n_samples_reference': self.n_samples_reference,
            'n_samples_current': self.n_samples_current,
            'features_ok': self.features_ok,
            'features_warning': self.features_warning,
            'features_critical': self.features_critical,
            'overall_alert': self.overall_alert,
            'action_required': self.action_required,
            'prediction_psi': self.prediction_psi,
            'prediction_ks': self.prediction_ks,
            'feature_results': [r.to_dict() for r in self.feature_results],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60]
        lines.append("DRIFT MONITORING REPORT")
        lines.append(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        lines.append(f"\nSamples: {self.n_samples_reference} reference, {self.n_samples_current} current")
        lines.append(f"Features analyzed: {self.n_features_analyzed}")

        lines.append(f"\nStatus Summary:")
        lines.append(f"  OK:       {self.features_ok}")
        lines.append(f"  Warning:  {self.features_warning}")
        lines.append(f"  Critical: {self.features_critical}")

        lines.append(f"\nOverall Alert: {self.overall_alert.upper()}")
        lines.append(f"Action Required: {'YES' if self.action_required else 'No'}")

        if self.features_critical > 0:
            lines.append("\nCRITICAL FEATURES:")
            for r in self.feature_results:
                if r.alert_level == 'critical':
                    lines.append(f"  - {r.feature_name}: PSI={r.psi:.3f}, KS={r.ks_statistic:.3f}")

        if self.features_warning > 0:
            lines.append("\nWARNING FEATURES:")
            for r in self.feature_results:
                if r.alert_level == 'warning':
                    lines.append(f"  - {r.feature_name}: PSI={r.psi:.3f}, KS={r.ks_statistic:.3f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class DriftMonitor:
    """
    Monitors for feature and prediction drift.

    Compares current data distribution against a reference (training) distribution
    to detect shifts that could impact model performance.
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.2,
        critical_feature_ratio: float = 0.1,
    ):
        """
        Initialize drift monitor.

        Args:
            n_bins: Number of bins for PSI calculation
            psi_threshold: PSI threshold for warning
            ks_threshold: KS statistic threshold for warning
            critical_feature_ratio: Ratio of critical features to trigger action
        """
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.critical_feature_ratio = critical_feature_ratio

        # Reference distributions (computed from training data)
        self.reference_stats: Dict[str, Dict] = {}
        self.reference_bins: Dict[str, np.ndarray] = {}

        # History
        self.drift_history: List[DriftReport] = []

    def fit_reference(self, df: pd.DataFrame) -> 'DriftMonitor':
        """
        Compute reference statistics from training data.

        Args:
            df: Training data

        Returns:
            Self for chaining
        """
        logger.info(f"Computing reference statistics from {len(df)} samples")

        for col in df.columns:
            data = df[col].dropna().values

            if len(data) < self.n_bins:
                continue

            # Compute reference statistics
            self.reference_stats[col] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'n_samples': len(data),
            }

            # Compute bin edges for PSI
            # Use percentiles for robustness
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(data, percentiles)

            # Ensure unique edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                # Not enough unique values, skip binning
                bin_edges = np.array([data.min() - 1, data.max() + 1])

            self.reference_bins[col] = bin_edges

            # Compute reference bin proportions
            ref_hist, _ = np.histogram(data, bins=bin_edges)
            ref_props = ref_hist / len(data)
            ref_props = np.clip(ref_props, 0.0001, 1)  # Avoid zero
            self.reference_stats[col]['bin_proportions'] = ref_props.tolist()

        logger.info(f"Reference computed for {len(self.reference_stats)} features")
        return self

    def analyze_drift(
        self,
        df: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        reference_predictions: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Analyze drift in current data compared to reference.

        Args:
            df: Current data to analyze
            predictions: Current model predictions (optional)
            reference_predictions: Reference predictions for comparison (optional)

        Returns:
            DriftReport with detailed analysis
        """
        if not self.reference_stats:
            raise ValueError("Must fit reference before analyzing drift")

        results = []

        for col in df.columns:
            if col not in self.reference_stats:
                continue

            current_data = df[col].dropna().values
            if len(current_data) < 10:
                continue

            result = self._analyze_feature(col, current_data)
            results.append(result)

        # Count alert levels
        features_ok = sum(1 for r in results if r.alert_level == 'ok')
        features_warning = sum(1 for r in results if r.alert_level == 'warning')
        features_critical = sum(1 for r in results if r.alert_level == 'critical')

        # Determine overall alert
        critical_ratio = features_critical / len(results) if results else 0
        warning_ratio = features_warning / len(results) if results else 0

        if critical_ratio >= self.critical_feature_ratio:
            overall_alert = 'critical'
            action_required = True
        elif critical_ratio > 0 or warning_ratio >= 0.2:
            overall_alert = 'warning'
            action_required = False
        else:
            overall_alert = 'ok'
            action_required = False

        # Analyze prediction drift if available
        prediction_psi = None
        prediction_ks = None
        if predictions is not None and reference_predictions is not None:
            prediction_psi = self._calculate_psi(reference_predictions, predictions)
            ks_stat, _ = stats.ks_2samp(reference_predictions, predictions)
            prediction_ks = ks_stat

            if prediction_psi > PSI_THRESHOLDS['high']:
                overall_alert = 'critical'
                action_required = True

        # Create report
        report = DriftReport(
            timestamp=datetime.now(),
            n_features_analyzed=len(results),
            n_samples_reference=self.reference_stats.get(
                list(self.reference_stats.keys())[0], {}
            ).get('n_samples', 0) if self.reference_stats else 0,
            n_samples_current=len(df),
            feature_results=results,
            features_ok=features_ok,
            features_warning=features_warning,
            features_critical=features_critical,
            overall_alert=overall_alert,
            action_required=action_required,
            prediction_psi=prediction_psi,
            prediction_ks=prediction_ks,
        )

        self.drift_history.append(report)

        return report

    def _analyze_feature(self, feature_name: str, current_data: np.ndarray) -> FeatureDriftResult:
        """Analyze drift for a single feature."""
        ref_stats = self.reference_stats[feature_name]
        ref_bins = self.reference_bins[feature_name]
        ref_props = np.array(ref_stats['bin_proportions'])

        # Calculate PSI
        psi = self._calculate_psi_from_bins(ref_props, current_data, ref_bins)

        # Calculate KS statistic
        # Create reference sample from stats (approximate)
        ref_mean = ref_stats['mean']
        ref_std = ref_stats['std']
        n_ref = ref_stats['n_samples']

        # Use parametric KS test
        ks_stat, ks_pvalue = stats.kstest(
            current_data,
            'norm',
            args=(ref_mean, ref_std)
        )

        # Also try 2-sample KS if we have enough reference samples
        # This is more robust but requires storing reference data

        # Calculate mean/std shifts
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)

        mean_shift = (current_mean - ref_mean) / ref_std if ref_std > 0 else 0
        std_shift = (current_std - ref_std) / ref_std if ref_std > 0 else 0

        # Determine alert level
        if psi > PSI_THRESHOLDS['high'] or ks_stat > KS_THRESHOLDS['high']:
            alert_level = 'critical'
        elif psi > PSI_THRESHOLDS['medium'] or ks_stat > KS_THRESHOLDS['medium']:
            alert_level = 'warning'
        else:
            alert_level = 'ok'

        return FeatureDriftResult(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            mean_shift=mean_shift,
            std_shift=std_shift,
            alert_level=alert_level,
            details={
                'ref_mean': ref_mean,
                'ref_std': ref_std,
                'current_mean': current_mean,
                'current_std': current_std,
            }
        )

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Population Stability Index.

        PSI = Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)
        """
        # Create bins from reference
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(reference, percentiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 3:
            return 0.0

        # Calculate proportions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        ref_props = ref_hist / len(reference)
        cur_props = cur_hist / len(current)

        # Avoid zero
        ref_props = np.clip(ref_props, 0.0001, 1)
        cur_props = np.clip(cur_props, 0.0001, 1)

        # PSI calculation
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def _calculate_psi_from_bins(
        self,
        ref_props: np.ndarray,
        current_data: np.ndarray,
        bin_edges: np.ndarray
    ) -> float:
        """Calculate PSI using pre-computed bin edges."""
        cur_hist, _ = np.histogram(current_data, bins=bin_edges)
        cur_props = cur_hist / len(current_data)
        cur_props = np.clip(cur_props, 0.0001, 1)

        # Align dimensions
        min_len = min(len(ref_props), len(cur_props))
        ref_props = ref_props[:min_len]
        cur_props = cur_props[:min_len]

        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi)

    def run_stress_test(
        self,
        df: pd.DataFrame,
        feature_name: str,
        stress_type: str = 'missing',
        stress_level: float = 0.2
    ) -> Dict[str, Any]:
        """
        Run stress test on a feature.

        Args:
            df: Data to stress test
            feature_name: Feature to modify
            stress_type: Type of stress ('missing', 'shift', 'scale', 'outliers')
            stress_level: Intensity of stress (0-1)

        Returns:
            Stress test results
        """
        df_stressed = df.copy()

        if feature_name not in df.columns:
            return {'error': f"Feature {feature_name} not found"}

        original_values = df[feature_name].values.copy()

        if stress_type == 'missing':
            # Randomly set values to NaN
            n_missing = int(len(df) * stress_level)
            missing_idx = np.random.choice(len(df), n_missing, replace=False)
            df_stressed.loc[df_stressed.index[missing_idx], feature_name] = np.nan

        elif stress_type == 'shift':
            # Shift distribution by stress_level standard deviations
            std = df[feature_name].std()
            df_stressed[feature_name] = df[feature_name] + stress_level * std * 3

        elif stress_type == 'scale':
            # Scale distribution
            mean = df[feature_name].mean()
            df_stressed[feature_name] = mean + (df[feature_name] - mean) * (1 + stress_level)

        elif stress_type == 'outliers':
            # Add extreme outliers
            n_outliers = int(len(df) * stress_level)
            outlier_idx = np.random.choice(len(df), n_outliers, replace=False)
            max_val = df[feature_name].max()
            min_val = df[feature_name].min()
            range_val = max_val - min_val
            df_stressed.loc[df_stressed.index[outlier_idx], feature_name] = max_val + range_val * 2

        else:
            return {'error': f"Unknown stress type: {stress_type}"}

        # Analyze drift after stress
        report = self.analyze_drift(df_stressed)

        # Find the stressed feature result
        feature_result = next(
            (r for r in report.feature_results if r.feature_name == feature_name),
            None
        )

        return {
            'stress_type': stress_type,
            'stress_level': stress_level,
            'feature_name': feature_name,
            'psi_before_stress': 0.0,  # Reference is same as original
            'psi_after_stress': feature_result.psi if feature_result else None,
            'ks_after_stress': feature_result.ks_statistic if feature_result else None,
            'alert_level': feature_result.alert_level if feature_result else None,
            'overall_action_required': report.action_required,
        }

    def save_reference(self, path: Path) -> None:
        """Save reference statistics to disk."""
        data = {
            'reference_stats': self.reference_stats,
            'reference_bins': {k: v.tolist() for k, v in self.reference_bins.items()},
            'config': {
                'n_bins': self.n_bins,
                'psi_threshold': self.psi_threshold,
                'ks_threshold': self.ks_threshold,
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Reference saved to {path}")

    @classmethod
    def load_reference(cls, path: Path) -> 'DriftMonitor':
        """Load reference statistics from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        monitor = cls(
            n_bins=data['config']['n_bins'],
            psi_threshold=data['config']['psi_threshold'],
            ks_threshold=data['config']['ks_threshold'],
        )
        monitor.reference_stats = data['reference_stats']
        monitor.reference_bins = {k: np.array(v) for k, v in data['reference_bins'].items()}

        logger.info(f"Reference loaded from {path}")
        return monitor
