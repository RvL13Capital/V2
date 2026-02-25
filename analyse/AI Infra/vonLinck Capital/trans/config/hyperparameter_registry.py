"""
Hyperparameter Registry for TRAnS System
=========================================

Central documentation and validation for all hyperparameters used in the system.
Each parameter includes:
- Current value
- Rationale for the choice
- Sensitivity analysis status
- Source/reference for the value

Usage:
    from config.hyperparameter_registry import HYPERPARAMETERS, get_param, validate_params

    # Get a parameter value
    bbw_pct = get_param('detection.bbw_percentile')

    # Validate all parameters
    validate_params()

NOTE: This file documents parameters but does NOT replace config/constants.py.
The actual values used in production are in constants.py. This registry
provides documentation and sensitivity analysis tracking.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class SensitivityStatus(Enum):
    """Status of sensitivity analysis for a parameter."""
    NOT_ANALYZED = "not_analyzed"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class HyperparameterSpec:
    """Specification for a single hyperparameter."""
    value: Any
    rationale: str
    sensitivity: SensitivityStatus
    sensitivity_notes: str
    source: str
    suggested_range: Optional[tuple] = None
    related_params: Optional[List[str]] = None
    last_validated: Optional[str] = None


# ============================================================================
# PATTERN DETECTION THRESHOLDS
# ============================================================================

DETECTION_THRESHOLDS: Dict[str, HyperparameterSpec] = {

    "bbw_percentile": HyperparameterSpec(
        value=0.30,
        rationale=(
            "30th percentile captures volatility contraction - stocks must be in "
            "the lower 30% of their historical BBW range to qualify as consolidating. "
            "Lower values (e.g., 20%) are too restrictive, higher (e.g., 40%) includes "
            "non-consolidating patterns."
        ),
        sensitivity=SensitivityStatus.PLANNED,
        sensitivity_notes="TODO: Run sweep 0.20-0.40 on 2015-2022 data",
        source="Empirical testing on 2015-2022 EU data",
        suggested_range=(0.20, 0.40),
        related_params=["adx_threshold"],
    ),

    "adx_threshold": HyperparameterSpec(
        value=32.0,
        rationale=(
            "ADX measures trend strength. Values <25 indicate no trend, 25-50 weak trend, "
            ">50 strong trend. We use 32 as upper bound to ensure we're capturing "
            "non-trending consolidations (avoiding runaway trends that look like consolidation)."
        ),
        sensitivity=SensitivityStatus.PLANNED,
        sensitivity_notes="TODO: Run sweep 25-40 on historical data",
        source="Wilder (1978) ADX interpretation, adjusted empirically for micro-caps",
        suggested_range=(25, 40),
        related_params=["bbw_percentile"],
    ),

    "volume_relative_threshold": HyperparameterSpec(
        value=0.35,
        rationale=(
            "Volume during qualification must be < 35% of 20-day average. "
            "Low volume during consolidation indicates lack of selling pressure "
            "and potential accumulation by informed buyers."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="Conservative threshold; may miss some valid patterns",
        source="Market microstructure theory + empirical validation",
        suggested_range=(0.25, 0.50),
        related_params=["daily_range_threshold"],
    ),

    "daily_range_threshold": HyperparameterSpec(
        value=0.65,
        rationale=(
            "Daily range must be < 65% of 20-day average range. "
            "Compressed ranges indicate equilibrium between buyers/sellers "
            "and potential for volatile breakout."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="TODO: Analyze interaction with BBW percentile",
        source="Chart pattern theory",
        suggested_range=(0.50, 0.75),
        related_params=["bbw_percentile"],
    ),

    "qualification_days": HyperparameterSpec(
        value=10,
        rationale=(
            "10 consecutive days meeting all criteria required for qualification. "
            "Shorter periods may capture noise; longer periods too restrictive. "
            "10 days = 2 trading weeks, sufficient for pattern formation."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 5, 7, 10, 15 days; 10 optimal for signal/noise balance",
        source="Empirical optimization on 2015-2022 data",
        suggested_range=(7, 15),
        last_validated="2024-01",
    ),
}


# ============================================================================
# LABELING PARAMETERS
# ============================================================================

LABELING_THRESHOLDS: Dict[str, HyperparameterSpec] = {

    "target_r_multiple": HyperparameterSpec(
        value=3.0,
        rationale=(
            "+3R target for daily patterns. This is aggressive but achievable "
            "for micro/small-caps. Lower targets (2R) would include too many "
            "noise winners; higher (4R+) would have insufficient sample size."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 2R, 3R, 4R, 5R; 3R balances hit rate vs significance",
        source="Risk management theory + empirical backtesting",
        suggested_range=(2.0, 4.0),
        related_params=["gap_limit_r"],
        last_validated="2024-01",
    ),

    "gap_limit_r": HyperparameterSpec(
        value=0.5,
        rationale=(
            "Patterns that gap up > 0.5R are marked untradeable. "
            "Large gaps destroy risk/reward for retail execution "
            "but are still valuable for model training as momentum signals."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="May need adjustment for different market regimes",
        source="Execution analysis on EOD entries",
        suggested_range=(0.3, 1.0),
        related_params=["target_r_multiple"],
    ),

    "min_outcome_window": HyperparameterSpec(
        value=10,
        rationale=(
            "Minimum 10-day outcome window for high-volatility stocks. "
            "Dynamic window = 1/volatility_proxy, clamped to [10, 60]."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="Interacts with volatility proxy calculation",
        source="V22 dynamic window implementation",
        suggested_range=(5, 15),
        related_params=["max_outcome_window", "volatility_proxy"],
    ),

    "max_outcome_window": HyperparameterSpec(
        value=60,
        rationale=(
            "Maximum 60-day outcome window for low-volatility stocks. "
            "Longer windows risk including noise from unrelated market events."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="May need to be shorter in volatile regimes",
        source="V22 dynamic window implementation",
        suggested_range=(40, 100),
        related_params=["min_outcome_window"],
    ),

    "volume_multiplier_target": HyperparameterSpec(
        value=2.0,
        rationale=(
            "Volume confirmation requires 2x 20-day average. "
            "This filters breakouts from false signals driven by "
            "thin liquidity or manipulation."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 1.5x, 2x, 3x; 2x balances signal quality vs sample size",
        source="Volume analysis theory + empirical validation",
        suggested_range=(1.5, 3.0),
        last_validated="2024-01",
    ),

    "volume_sustained_days": HyperparameterSpec(
        value=3,
        rationale=(
            "Volume confirmation requires 3 consecutive days meeting criteria. "
            "Single-day spikes can be manipulation; 3 days suggests real interest."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="TODO: Compare 2, 3, 5 day requirements",
        source="Jan 2026 volume confirmation fix",
        suggested_range=(2, 5),
        related_params=["volume_multiplier_target"],
    ),
}


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

MODEL_ARCHITECTURE: Dict[str, HyperparameterSpec] = {

    "temporal_window_size": HyperparameterSpec(
        value=20,
        rationale=(
            "20 timesteps per sequence: 10 qualification + 10 validation phase. "
            "Captures pattern evolution from formation to breakout setup."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 10, 15, 20, 30; 20 optimal for pattern capture",
        source="Pattern lifecycle analysis",
        suggested_range=(15, 30),
        last_validated="2024-01",
    ),

    "feature_dim": HyperparameterSpec(
        value=10,
        rationale=(
            "10 features per timestep. Kept minimal to avoid overfitting "
            "on small pattern datasets. Each feature carefully selected for "
            "theoretical relevance and empirical predictive power."
        ),
        sensitivity=SensitivityStatus.NOT_APPLICABLE,
        sensitivity_notes="Feature count determined by domain requirements",
        source="Feature engineering analysis",
        suggested_range=None,
    ),

    "lstm_hidden_dim": HyperparameterSpec(
        value=64,
        rationale=(
            "64-unit LSTM hidden dimension. Balances expressiveness vs overfitting "
            "risk on ~6k pattern dataset."
        ),
        sensitivity=SensitivityStatus.PLANNED,
        sensitivity_notes="TODO: Compare 32, 64, 128 in ablation study",
        source="Architecture search",
        suggested_range=(32, 128),
    ),

    "cnn_filters": HyperparameterSpec(
        value=[32, 64],
        rationale=(
            "Two-layer CNN with 32->64 filters. Captures local patterns "
            "before LSTM aggregation."
        ),
        sensitivity=SensitivityStatus.PLANNED,
        sensitivity_notes="TODO: Compare single layer vs two layer in ablation",
        source="Temporal CNN best practices",
        suggested_range=None,
    ),
}


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

TRAINING_PARAMS: Dict[str, HyperparameterSpec] = {

    "focal_gamma": HyperparameterSpec(
        value=2.0,
        rationale=(
            "Focal loss gamma=2.0 down-weights easy examples. "
            "Standard value from Lin et al. (2017)."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 0, 1, 2, 3, 5; gamma=2 is standard",
        source="Lin et al. 2017 'Focal Loss for Dense Object Detection'",
        suggested_range=(1.0, 5.0),
        last_validated="2024-01",
    ),

    "coil_weight": HyperparameterSpec(
        value=3.0,
        rationale=(
            "Coil-aware focal loss upweights high-coil patterns by 3x. "
            "Patterns with high coil intensity are more likely to be significant."
        ),
        sensitivity=SensitivityStatus.PLANNED,
        sensitivity_notes="TODO: Compare 1.5, 2.0, 3.0, 5.0 in ablation",
        source="Coil-aware focal loss implementation",
        suggested_range=(1.5, 5.0),
    ),

    "learning_rate": HyperparameterSpec(
        value=0.001,
        rationale=(
            "Standard Adam learning rate. May need decay schedule "
            "for longer training."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="Using scheduler in training; base LR less critical",
        source="Adam defaults",
        suggested_range=(0.0001, 0.01),
    ),

    "batch_size": HyperparameterSpec(
        value=32,
        rationale=(
            "Small batch size for limited dataset. Larger batches may "
            "improve gradient stability but reduce effective updates."
        ),
        sensitivity=SensitivityStatus.NOT_ANALYZED,
        sensitivity_notes="Limited by GPU memory and dataset size",
        source="Deep learning best practices",
        suggested_range=(16, 128),
    ),
}


# ============================================================================
# EXECUTION PARAMETERS
# ============================================================================

EXECUTION_PARAMS: Dict[str, HyperparameterSpec] = {

    "max_danger_prob": HyperparameterSpec(
        value=0.25,
        rationale=(
            "Reject patterns with P(Danger) > 25%. "
            "Audit found 58.9% of GOOD signals had high P(Danger). "
            "This threshold reduces false positives significantly."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="Tested 50%, 40%, 25%; 25% balances signal quality vs quantity",
        source="Jan 2026 danger filter analysis",
        suggested_range=(0.20, 0.40),
        last_validated="2026-01",
    ),

    "min_dollar_volume": HyperparameterSpec(
        value=50000,
        rationale=(
            "$50k minimum average dollar volume. Ensures sufficient "
            "liquidity for $250 risk unit positions without excessive slippage."
        ),
        sensitivity=SensitivityStatus.COMPLETED,
        sensitivity_notes="$100k was too aggressive (76% data loss), reverted to $50k",
        source="Liquidity analysis Jan 2026",
        suggested_range=(25000, 100000),
        last_validated="2026-01",
    ),

    "risk_unit_dollars": HyperparameterSpec(
        value=250.0,
        rationale=(
            "$250 fixed risk per trade. Conservative for $10k-$100k accounts. "
            "Limits single-trade impact to ~0.25-2.5% of capital."
        ),
        sensitivity=SensitivityStatus.NOT_APPLICABLE,
        sensitivity_notes="Account management decision, not model parameter",
        source="Risk management policy",
        suggested_range=None,
    ),
}


# ============================================================================
# COMBINED REGISTRY
# ============================================================================

HYPERPARAMETERS: Dict[str, Dict[str, HyperparameterSpec]] = {
    "detection": DETECTION_THRESHOLDS,
    "labeling": LABELING_THRESHOLDS,
    "model": MODEL_ARCHITECTURE,
    "training": TRAINING_PARAMS,
    "execution": EXECUTION_PARAMS,
}


def get_param(path: str) -> Any:
    """
    Get a hyperparameter value by path.

    Args:
        path: Dot-separated path (e.g., 'detection.bbw_percentile')

    Returns:
        Parameter value

    Raises:
        KeyError: If path not found
    """
    parts = path.split('.')
    if len(parts) != 2:
        raise KeyError(f"Invalid path format: {path}. Use 'category.param_name'")

    category, param = parts

    if category not in HYPERPARAMETERS:
        raise KeyError(f"Unknown category: {category}")

    if param not in HYPERPARAMETERS[category]:
        raise KeyError(f"Unknown parameter: {param} in category {category}")

    return HYPERPARAMETERS[category][param].value


def get_param_spec(path: str) -> HyperparameterSpec:
    """
    Get full hyperparameter specification by path.

    Args:
        path: Dot-separated path (e.g., 'detection.bbw_percentile')

    Returns:
        HyperparameterSpec object
    """
    parts = path.split('.')
    category, param = parts
    return HYPERPARAMETERS[category][param]


def validate_params() -> Dict[str, List[str]]:
    """
    Validate all hyperparameters and report issues.

    Returns:
        Dictionary with 'warnings' and 'needs_analysis' lists
    """
    warnings = []
    needs_analysis = []

    for category, params in HYPERPARAMETERS.items():
        for name, spec in params.items():
            path = f"{category}.{name}"

            # Check sensitivity status
            if spec.sensitivity == SensitivityStatus.NOT_ANALYZED:
                needs_analysis.append(f"{path}: Sensitivity not analyzed")
            elif spec.sensitivity == SensitivityStatus.PLANNED:
                needs_analysis.append(f"{path}: Sensitivity analysis planned")

            # Check for missing rationale
            if not spec.rationale or len(spec.rationale) < 20:
                warnings.append(f"{path}: Insufficient rationale")

            # Check for missing source
            if not spec.source:
                warnings.append(f"{path}: No source documented")

    return {
        'warnings': warnings,
        'needs_analysis': needs_analysis
    }


def generate_sensitivity_report() -> str:
    """
    Generate a markdown report of sensitivity analysis status.

    Returns:
        Markdown-formatted string
    """
    lines = [
        "# Hyperparameter Sensitivity Analysis Status",
        "",
        "## Summary",
        ""
    ]

    status_counts = {s: 0 for s in SensitivityStatus}

    for category, params in HYPERPARAMETERS.items():
        for spec in params.values():
            status_counts[spec.sensitivity] += 1

    total = sum(status_counts.values())
    lines.append(f"- **Completed**: {status_counts[SensitivityStatus.COMPLETED]} ({100*status_counts[SensitivityStatus.COMPLETED]/total:.0f}%)")
    lines.append(f"- **Planned**: {status_counts[SensitivityStatus.PLANNED]} ({100*status_counts[SensitivityStatus.PLANNED]/total:.0f}%)")
    lines.append(f"- **Not Analyzed**: {status_counts[SensitivityStatus.NOT_ANALYZED]} ({100*status_counts[SensitivityStatus.NOT_ANALYZED]/total:.0f}%)")
    lines.append(f"- **N/A**: {status_counts[SensitivityStatus.NOT_APPLICABLE]} ({100*status_counts[SensitivityStatus.NOT_APPLICABLE]/total:.0f}%)")
    lines.append("")

    # Detail by category
    for category, params in HYPERPARAMETERS.items():
        lines.append(f"## {category.title()}")
        lines.append("")
        lines.append("| Parameter | Value | Status | Notes |")
        lines.append("|-----------|-------|--------|-------|")

        for name, spec in params.items():
            status = spec.sensitivity.value.replace("_", " ").title()
            notes = spec.sensitivity_notes[:50] + "..." if len(spec.sensitivity_notes) > 50 else spec.sensitivity_notes
            lines.append(f"| {name} | {spec.value} | {status} | {notes} |")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Print validation report
    report = validate_params()

    print("=" * 60)
    print("HYPERPARAMETER VALIDATION REPORT")
    print("=" * 60)

    if report['warnings']:
        print("\nWARNINGS:")
        for w in report['warnings']:
            print(f"  - {w}")

    if report['needs_analysis']:
        print("\nNEEDS SENSITIVITY ANALYSIS:")
        for n in report['needs_analysis']:
            print(f"  - {n}")

    print("\n" + generate_sensitivity_report())
