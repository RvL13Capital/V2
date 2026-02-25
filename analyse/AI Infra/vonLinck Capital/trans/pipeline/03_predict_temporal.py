"""
Step 3: Generate Risk-Based Execution Plan (Nightly Orders)
============================================================

Loads trained model, generates predictions, and outputs position-sized
execution orders with structural risk management and liquidity clamping.

Risk Framework:
    - RISK_UNIT_DOLLARS: Amount willing to lose on structural failure ($250 default)
    - MAX_CAPITAL_PER_TRADE: Hard liquidity cap per position ($5,000 default)
    - Position Size = RISK_UNIT / Risk_Per_Share
    - Liquidity Clamp: Never exceed 10% of Avg Daily Dollar Volume

Input:
    - Trained model checkpoint
    - Test/live sequences (HDF5 with context)
    - Pattern metadata (with boundaries, dollar_volume)

Output:
    - Nightly execution plan (CSV + Parquet)
    - Order columns: ticker, trigger_price, stop_price, shares, capital_required
    - Risk metrics: risk_per_share, r_multiple_target, liquidity_pct

Runtime: ~5-10 minutes

Usage:
    python 03_predict_temporal.py --model best_model.pt --sequences live_patterns.h5 \\
        --metadata live_metadata.parquet --risk-unit 250 --max-capital 5000
"""

import sys
import os

# CRITICAL: Disable PyTorch dynamo/inductor BEFORE importing torch
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.temporal_hybrid_unified import HybridFeatureNetwork
from utils.logging_config import setup_pipeline_logging
from config import (
    STRATEGIC_VALUES,
    SIGNAL_THRESHOLDS,
    PREDICTION_TEMPERATURE,
    DEFAULT_SEQUENCE_DIR,
    DEFAULT_PREDICTION_DIR,
    USE_GRN_CONTEXT,
    USE_PROBABILITY_CALIBRATION
)
from config.context_features import NUM_CONTEXT_FEATURES
from config.constants import NUM_CLASSES, CLASS_NAMES

# Setup centralized logging
logger = setup_pipeline_logging('03_predict_temporal')

# =============================================================================
# DEFAULT RISK CONFIGURATION
# =============================================================================
DEFAULT_RISK_UNIT_DOLLARS = 250.0      # Willing to lose $250 on structural failure
DEFAULT_MAX_CAPITAL_PER_TRADE = 5000.0 # Hard cap on position size
DEFAULT_ADV_LIQUIDITY_PCT = 0.04       # Never exceed 4% of ADV (10% is untradeable in micro-caps)
DEFAULT_TRIGGER_OFFSET = 0.01          # $0.01 above upper boundary


# =============================================================================
# EOD STRUCTURAL EXECUTION (V21 - Buy Stop Limits)
# =============================================================================
# We are EOD only. We cannot see 10:00 AM volume.
# Instead, we define strict Order Instructions for the next trading day.
#
# STRATEGY: BUY STOP LIMIT
#   - Order Type: BUY STOP LIMIT
#   - Stop Price (Trigger): Upper_Boundary + $0.01
#   - Limit Price: Trigger + 0.5R (Cap slippage/gaps at 0.5R)
#
# EXECUTION RULES:
#   1. Place order EOD (night before) with GTC or DAY expiration
#   2. Order triggers when price hits Stop Price
#   3. Fills at Limit or better (caps gap risk)
#   4. If Open > Limit: DO NOT CHASE. Cancel order immediately.
#
# WHY THIS WORKS FOR RETAIL:
#   - No intraday monitoring required (we don't see 10AM volume)
#   - Slippage capped at 0.5R (we know worst-case entry cost)
#   - Gap protection: If it gaps past our limit, we don't chase
#   - Mechanical: No discretion, no FOMO, no chasing
#
# RISK MATH:
#   R = Trigger - Stop_Loss (structural risk per share)
#   Limit = Trigger + 0.5R
#   If filled at Limit, actual risk = 1.5R (still acceptable)
#   If gaps past Limit, we're flat (no entry, no risk)
# =============================================================================

# Slippage cap as multiple of R (risk per share)
SLIPPAGE_CAP_R_MULTIPLE = 0.5  # Limit price = Trigger + 0.5R


def calculate_buy_stop_limit_prices(
    upper_boundary: float,
    lower_boundary: float,
    trigger_offset: float = 0.01,
    slippage_cap_r: float = SLIPPAGE_CAP_R_MULTIPLE
) -> dict:
    """
    Calculate BUY STOP LIMIT order prices for EOD execution.

    Args:
        upper_boundary: Upper boundary of consolidation pattern
        lower_boundary: Lower boundary (technical floor / stop loss)
        trigger_offset: Offset above upper boundary for trigger ($0.01 default)
        slippage_cap_r: Maximum slippage as multiple of R (0.5 default)

    Returns:
        dict with:
            - stop_price: Price that triggers the order (upper + offset)
            - limit_price: Maximum fill price (stop + 0.5R)
            - r_per_share: Risk per share (trigger - stop_loss)
            - max_slippage: Maximum slippage in dollars
            - chase_warning_price: If Open > this, cancel order
            - order_type: 'BUY_STOP_LIMIT'
    """
    stop_price = upper_boundary + trigger_offset
    stop_loss = lower_boundary  # Technical floor

    # R = risk per share (distance from trigger to stop)
    r_per_share = stop_price - stop_loss

    # Limit price caps slippage at 0.5R
    max_slippage = r_per_share * slippage_cap_r
    limit_price = stop_price + max_slippage

    return {
        'order_type': 'BUY_STOP_LIMIT',
        'stop_price': round(stop_price, 4),
        'limit_price': round(limit_price, 4),
        'stop_loss': round(stop_loss, 4),
        'r_per_share': round(r_per_share, 4),
        'max_slippage': round(max_slippage, 4),
        'slippage_cap_r': slippage_cap_r,
        'chase_warning_price': round(limit_price, 4),  # If Open > this, CANCEL
    }


class SequenceDataset(Dataset):
    """PyTorch dataset for temporal sequences with optional context."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray = None, context: np.ndarray = None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.context = torch.FloatTensor(context) if context is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.context is not None and self.labels is not None:
            return self.sequences[idx], self.labels[idx], self.context[idx]
        elif self.context is not None:
            return self.sequences[idx], self.context[idx]
        elif self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


def calculate_expected_value(class_probs: np.ndarray) -> np.ndarray:
    """
    Calculate Expected Value from class probabilities.

    Uses V17 strategic value system (3 classes).

    Args:
        class_probs: (n_samples, 3) array of class probabilities
                     [Danger, Noise, Target]

    Returns:
        (n_samples,) array of Expected Values
    """
    num_classes = class_probs.shape[1]
    values = np.array([STRATEGIC_VALUES[i] for i in range(num_classes)])
    ev = np.sum(class_probs * values, axis=1)
    return ev


def generate_signals(
    ev: np.ndarray,
    danger_prob: np.ndarray = None,
    max_danger_prob: float = None
) -> np.ndarray:
    """
    Generate trading signals from Expected Values with optional danger filter.

    Args:
        ev: (n_samples,) array of Expected Values
        danger_prob: (n_samples,) array of danger probabilities (optional)
        max_danger_prob: Maximum danger probability threshold (optional)
                        Patterns exceeding this are downgraded to AVOID

    Returns:
        (n_samples,) array of signal strings
    """
    signals = np.full(len(ev), 'AVOID', dtype=object)

    signals[ev >= SIGNAL_THRESHOLDS['MODERATE']] = 'MODERATE'
    signals[ev >= SIGNAL_THRESHOLDS['GOOD']] = 'GOOD'
    signals[ev >= SIGNAL_THRESHOLDS['STRONG']] = 'STRONG'

    # Apply danger probability filter (Jan 2026 guardrail)
    if danger_prob is not None and max_danger_prob is not None:
        danger_mask = danger_prob > max_danger_prob
        n_filtered = (danger_mask & (signals != 'AVOID')).sum()
        if n_filtered > 0:
            signals[danger_mask] = 'AVOID'
            logger.info(f"Danger filter: {n_filtered} signals downgraded to AVOID (P(Danger) > {max_danger_prob:.0%})")

    return signals


def calculate_position_size(
    upper_boundary: float,
    lower_boundary: float,
    avg_dollar_volume: float,
    risk_unit_dollars: float,
    max_capital_per_trade: float,
    adv_liquidity_pct: float,
    trigger_offset: float = 0.01
) -> dict:
    """
    Calculate risk-based position size with liquidity clamping.

    Args:
        upper_boundary: Pattern upper boundary (resistance)
        lower_boundary: Pattern lower boundary (support/invalidation)
        avg_dollar_volume: Average daily dollar volume
        risk_unit_dollars: Max $ to lose on structural failure
        max_capital_per_trade: Hard capital cap
        adv_liquidity_pct: Max % of ADV to trade
        trigger_offset: $ above upper boundary for trigger

    Returns:
        dict with position sizing details
    """
    # Step 1: Calculate structural risk
    trigger_price = upper_boundary + trigger_offset
    stop_price = lower_boundary  # Technical floor
    risk_per_share = trigger_price - stop_price

    # Validate risk makes sense
    if risk_per_share <= 0:
        return {
            'valid': False,
            'reason': 'Invalid risk (trigger <= stop)',
            'trigger_price': trigger_price,
            'limit_price': trigger_price,
            'stop_price': stop_price,
            'risk_per_share': 0,
            'risk_width_pct': 0,
            'shares': 0,
            'capital_required': 0,
            'actual_risk_dollars': 0,
            't1_price': 0, 't1_shares': 0,
            't2_price': 0, 't2_shares': 0,
            't3_price': 0, 't3_shares': 0,
            'potential_t1': 0, 'potential_t2': 0, 'potential_t3': 0,
            'potential_total': 0, 'weighted_avg_r': 0,
            'liquidity_pct': 0,
            'clamped': False,
            'clamp_reason': None
        }

    risk_width_pct = risk_per_share / trigger_price

    # Step 2: Calculate initial position size based on risk unit
    shares_from_risk = int(risk_unit_dollars / risk_per_share)
    capital_from_risk = shares_from_risk * trigger_price

    # Step 3: Apply MAX_CAPITAL_PER_TRADE clamp
    clamped = False
    clamp_reason = None

    if capital_from_risk > max_capital_per_trade:
        shares_from_capital = int(max_capital_per_trade / trigger_price)
        if shares_from_capital < shares_from_risk:
            shares_from_risk = shares_from_capital
            capital_from_risk = shares_from_risk * trigger_price
            clamped = True
            clamp_reason = 'max_capital'

    # Step 4: Apply liquidity clamp (crucial for micro-caps)
    max_allowed_size = avg_dollar_volume * adv_liquidity_pct

    if capital_from_risk > max_allowed_size and avg_dollar_volume > 0:
        shares_from_liquidity = int(max_allowed_size / trigger_price)
        if shares_from_liquidity < shares_from_risk:
            shares_from_risk = shares_from_liquidity
            capital_from_risk = shares_from_risk * trigger_price
            clamped = True
            clamp_reason = 'liquidity_10pct_adv'

    # Final position
    final_shares = max(0, shares_from_risk)
    final_capital = final_shares * trigger_price
    liquidity_pct = (final_capital / avg_dollar_volume * 100) if avg_dollar_volume > 0 else 0

    # Calculate actual risk in dollars
    actual_risk_dollars = final_shares * risk_per_share

    # =========================================================================
    # JUSTICE EXIT PLAN - Multi-Tier Profit Taking
    # =========================================================================
    # Limit Price for BUY_STOP order (max slippage tolerance)
    limit_price = trigger_price + (0.5 * risk_per_share)

    # Target 1 (BANK): +3R - Sell 50% to lock in profit
    t1_price = trigger_price + (3.0 * risk_per_share)
    t1_shares = int(final_shares * 0.50)

    # Target 2 (RUNNER): +5R - Sell 25% for extended move
    t2_price = trigger_price + (5.0 * risk_per_share)
    t2_shares = int(final_shares * 0.25)

    # Target 3 (MOON): +10R - Hold 25% with trailing stop
    t3_price = trigger_price + (10.0 * risk_per_share)
    t3_shares = final_shares - t1_shares - t2_shares  # Remainder

    # Calculate potential gains at each tier
    potential_t1 = t1_shares * 3.0 * risk_per_share
    potential_t2 = t2_shares * 5.0 * risk_per_share
    potential_t3 = t3_shares * 10.0 * risk_per_share
    potential_total = potential_t1 + potential_t2 + potential_t3

    # Weighted average R if all targets hit
    weighted_avg_r = (0.50 * 3.0) + (0.25 * 5.0) + (0.25 * 10.0)  # = 5.25R

    return {
        'valid': final_shares > 0,
        'reason': None if final_shares > 0 else 'Zero shares after clamping',
        # Entry
        'trigger_price': round(trigger_price, 2),
        'limit_price': round(limit_price, 2),
        'stop_price': round(stop_price, 2),
        # Risk metrics
        'risk_per_share': round(risk_per_share, 4),
        'risk_width_pct': round(risk_width_pct * 100, 2),
        # Position sizing
        'shares': final_shares,
        'capital_required': round(final_capital, 2),
        'actual_risk_dollars': round(actual_risk_dollars, 2),
        # Justice Exit Plan - Multi-tier targets
        't1_price': round(t1_price, 2),
        't1_shares': t1_shares,
        't2_price': round(t2_price, 2),
        't2_shares': t2_shares,
        't3_price': round(t3_price, 2),
        't3_shares': t3_shares,
        # Potential gains
        'potential_t1': round(potential_t1, 2),
        'potential_t2': round(potential_t2, 2),
        'potential_t3': round(potential_t3, 2),
        'potential_total': round(potential_total, 2),
        'weighted_avg_r': round(weighted_avg_r, 2),
        # Liquidity
        'liquidity_pct': round(liquidity_pct, 2),
        'clamped': clamped,
        'clamp_reason': clamp_reason
    }


def generate_execution_plan(
    predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    risk_unit_dollars: float,
    max_capital_per_trade: float,
    adv_liquidity_pct: float,
    min_signal: str = 'GOOD',
    trigger_offset: float = 0.01,
    max_sector_positions: int = None
) -> pd.DataFrame:
    """
    Generate risk-based execution plan from predictions and metadata.

    Args:
        predictions_df: DataFrame with predictions (signal, ev, probs)
        metadata_df: DataFrame with pattern metadata (boundaries, dollar_volume)
        risk_unit_dollars: Max $ risk per trade
        max_capital_per_trade: Hard cap per position
        adv_liquidity_pct: Max % of ADV
        min_signal: Minimum signal strength to include
        trigger_offset: $ above upper boundary
        max_sector_positions: Max positions per sector (None = no limit)

    Returns:
        DataFrame with execution orders
    """
    # Filter to actionable signals
    signal_order = {'STRONG': 0, 'GOOD': 1, 'MODERATE': 2, 'AVOID': 3}
    min_signal_rank = signal_order.get(min_signal, 1)

    actionable_mask = predictions_df['signal'].map(
        lambda x: signal_order.get(x, 3)
    ) <= min_signal_rank

    actionable_df = predictions_df[actionable_mask].copy()
    logger.info(f"Actionable signals ({min_signal}+): {len(actionable_df)} / {len(predictions_df)}")

    if len(actionable_df) == 0:
        logger.warning("No actionable signals - returning empty execution plan")
        return pd.DataFrame()

    # Merge with metadata
    if 'pattern_id' in actionable_df.columns and 'pattern_id' in metadata_df.columns:
        merged = actionable_df.merge(metadata_df, on='pattern_id', how='left')
    else:
        # Assume same order
        if len(actionable_df) != len(metadata_df):
            logger.warning("Prediction/metadata length mismatch - using index alignment")
            merged = pd.concat([
                actionable_df.reset_index(drop=True),
                metadata_df.iloc[actionable_df.index].reset_index(drop=True)
            ], axis=1)
        else:
            merged = pd.concat([actionable_df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)

    # =========================================================================
    # UNTRADEABLE FILTER (Jan 2026 fix)
    # =========================================================================
    # Patterns marked untradeable=True (e.g., gap > 0.5R) cannot be executed by retail.
    # These patterns ARE used for model training (strong momentum signals), but
    # MUST be filtered out at prediction time to prevent generating unexecutable orders.
    # =========================================================================
    if 'untradeable' in merged.columns:
        n_untradeable = merged['untradeable'].fillna(False).sum()
        if n_untradeable > 0:
            logger.info(f"\nFiltering {n_untradeable} untradeable patterns (gap > 0.5R, etc.)")
            tradeable_mask = ~merged['untradeable'].fillna(False)
            merged = merged[tradeable_mask].copy()
            logger.info(f"Remaining tradeable patterns: {len(merged)}")

            if len(merged) == 0:
                logger.warning("All patterns are untradeable - returning empty execution plan")
                return pd.DataFrame()

    # Required columns for position sizing
    required_cols = ['upper_boundary', 'lower_boundary']
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    # Dollar volume - try multiple column names
    dollar_vol_col = None
    for col in ['avg_dollar_volume', 'dollar_volume', 'avg_daily_dollar_vol', 'adv']:
        if col in merged.columns:
            dollar_vol_col = col
            break

    if dollar_vol_col is None:
        logger.warning("No dollar volume column found - liquidity clamping disabled")
        merged['_dollar_volume'] = 1e9  # Large number = no clamping
        dollar_vol_col = '_dollar_volume'

    # Calculate position sizes
    orders = []
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Calculating positions"):
        position = calculate_position_size(
            upper_boundary=row['upper_boundary'],
            lower_boundary=row['lower_boundary'],
            avg_dollar_volume=row[dollar_vol_col],
            risk_unit_dollars=risk_unit_dollars,
            max_capital_per_trade=max_capital_per_trade,
            adv_liquidity_pct=adv_liquidity_pct,
            trigger_offset=trigger_offset
        )

        order = {
            'ticker': row.get('ticker', row.get('symbol', f'UNK_{idx}')),
            'detection_date': row.get('detection_date', row.get('date', None)),
            'signal': row['signal'],
            'expected_value': row['expected_value'],
            'danger_prob': row['danger_prob'],
            'noise_prob': row['noise_prob'],
            'target_prob': row['target_prob'],
            **position
        }

        # Add pattern metadata if available
        for col in ['pattern_id', 'base_duration', 'coil_intensity', 'price_position_at_end']:
            if col in row:
                order[col] = row[col]

        # =======================================================================
        # EOD STRUCTURAL EXECUTION V21 (Buy Stop Limits)
        # =======================================================================
        # We are EOD only. We cannot see 10:00 AM volume.
        # Instead, output strict BUY STOP LIMIT order instructions.
        #
        # ORDER TYPE: BUY STOP LIMIT
        #   Stop Price (Trigger): Upper_Boundary + $0.01
        #   Limit Price: Trigger + 0.5R (caps slippage/gaps)
        #
        # EXECUTION RULES:
        #   1. Place order EOD (night before) with GTC or DAY expiration
        #   2. Order triggers when price hits Stop Price
        #   3. If Open > Limit Price: DO NOT CHASE. Cancel order.
        # =======================================================================

        # Calculate BUY STOP LIMIT prices
        buy_stop_limit = calculate_buy_stop_limit_prices(
            upper_boundary=row['upper_boundary'],
            lower_boundary=row['lower_boundary'],
            trigger_offset=trigger_offset,
            slippage_cap_r=SLIPPAGE_CAP_R_MULTIPLE
        )

        # Add order instruction fields
        order['order_type'] = buy_stop_limit['order_type']
        order['stop_price'] = buy_stop_limit['stop_price']
        order['limit_price'] = buy_stop_limit['limit_price']
        order['r_per_share'] = buy_stop_limit['r_per_share']
        order['max_slippage'] = buy_stop_limit['max_slippage']
        order['slippage_cap_r'] = buy_stop_limit['slippage_cap_r']
        order['chase_warning_price'] = buy_stop_limit['chase_warning_price']

        # Add execution instruction as human-readable note
        order['execution_instruction'] = (
            f"BUY STOP LIMIT: Stop=${buy_stop_limit['stop_price']:.2f}, "
            f"Limit=${buy_stop_limit['limit_price']:.2f}. "
            f"If Open > ${buy_stop_limit['chase_warning_price']:.2f}, CANCEL (do not chase)."
        )

        orders.append(order)

    orders_df = pd.DataFrame(orders)

    # Filter to valid orders
    valid_orders = orders_df[orders_df['valid']].copy()
    invalid_orders = orders_df[~orders_df['valid']]

    logger.info(f"Valid orders: {len(valid_orders)} / {len(orders_df)}")
    if len(invalid_orders) > 0:
        logger.warning(f"Invalid orders: {len(invalid_orders)}")
        reasons = invalid_orders['reason'].value_counts()
        for reason, count in reasons.items():
            logger.warning(f"  {reason}: {count}")

    # Sort by expected value (best first)
    valid_orders = valid_orders.sort_values('expected_value', ascending=False)

    # =========================================================================
    # SECTOR EXPOSURE CAP (Jan 2026)
    # =========================================================================
    # Prevent overconcentration in any single sector.
    # Orders are processed in EV order (best first) to keep highest quality.
    # =========================================================================
    if max_sector_positions is not None and max_sector_positions > 0:
        try:
            from config.sector_registry import get_sector, SectorExposureChecker

            checker = SectorExposureChecker(max_per_sector=max_sector_positions)
            sector_approved = []
            sector_rejected = []

            for idx, row in valid_orders.iterrows():
                ticker = row.get('ticker', 'UNKNOWN')
                sector = get_sector(ticker)

                if checker.can_add(sector):
                    checker.add(sector)
                    sector_approved.append(idx)
                else:
                    sector_rejected.append(idx)

            # Apply sector filter
            n_rejected = len(sector_rejected)
            if n_rejected > 0:
                logger.info(f"\nSector Exposure Cap (max {max_sector_positions} per sector):")
                logger.info(f"  Approved: {len(sector_approved)}")
                logger.info(f"  Rejected: {n_rejected}")
                logger.info(f"  Exposure: {checker.get_summary()}")

                valid_orders = valid_orders.loc[sector_approved].copy()
        except ImportError:
            logger.warning("sector_registry not available - skipping sector cap")

    return valid_orders


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Risk-Based Execution Plan (Nightly Orders)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Risk Framework:
  Position Size = RISK_UNIT_DOLLARS / Risk_Per_Share
  Risk_Per_Share = Trigger_Price - Stop_Price (Technical Floor)
  Liquidity Clamp: Never exceed 10% of Average Daily Dollar Volume

Example:
  python 03_predict_temporal.py --model best_model.pt --sequences live.h5 \\
      --metadata patterns.parquet --risk-unit 250 --max-capital 5000
        """
    )

    # Model and data
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--sequences', type=str,
                        default=f'{DEFAULT_SEQUENCE_DIR}/sequences_test.h5',
                        help='Path to sequences (HDF5 with context)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels (optional, for accuracy calc)')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to pattern metadata (Parquet with boundaries)')

    # Risk configuration
    parser.add_argument('--risk-unit', type=float, default=DEFAULT_RISK_UNIT_DOLLARS,
                        help=f'Max $ to lose on structural failure (default: ${DEFAULT_RISK_UNIT_DOLLARS})')
    parser.add_argument('--max-capital', type=float, default=DEFAULT_MAX_CAPITAL_PER_TRADE,
                        help=f'Hard cap on capital per trade (default: ${DEFAULT_MAX_CAPITAL_PER_TRADE})')
    parser.add_argument('--adv-pct', type=float, default=DEFAULT_ADV_LIQUIDITY_PCT,
                        help=f'Max %% of ADV to trade (default: {DEFAULT_ADV_LIQUIDITY_PCT*100}%%)')
    parser.add_argument('--trigger-offset', type=float, default=DEFAULT_TRIGGER_OFFSET,
                        help=f'$ above upper boundary for trigger (default: ${DEFAULT_TRIGGER_OFFSET})')

    # Signal filtering
    parser.add_argument('--min-signal', type=str, default='GOOD',
                        choices=['STRONG', 'GOOD', 'MODERATE'],
                        help='Minimum signal strength for orders (default: GOOD)')
    parser.add_argument('--max-danger-prob', type=float, default=0.50,
                        help='Max danger probability for GOOD/MODERATE signals (default: 0.50). '
                             'Patterns with P(Danger) > threshold are downgraded to AVOID.')

    # Sector exposure (Jan 2026)
    parser.add_argument('--max-sector-positions', type=int, default=None,
                        help='Max positions per sector (e.g., 3). None = no limit')

    # Inference settings
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for inference (512 optimal for CPU)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--output-dir', type=str,
                        default=DEFAULT_PREDICTION_DIR,
                        help='Output directory')
    parser.add_argument('--norm-params', type=str, default=None,
                        help='Path to normalization parameters (JSON)')
    parser.add_argument('--calibrator', type=str, default=None,
                        help='Path to probability calibrator (.pkl)')
    parser.add_argument('--prediction-only', action='store_true',
                        help='Generate predictions only (skip execution plan if boundaries missing)')

    # Drift monitoring (Jan 2026)
    parser.add_argument('--check-drift', action='store_true',
                        help='Check prediction drift against baseline')
    parser.add_argument('--drift-baseline', type=str, default=None,
                        help='Path to drift baseline JSON (from scripts/monitor_drift.py)')

    return parser.parse_args()


def apply_normalization(sequences: np.ndarray, norm_params: dict) -> np.ndarray:
    """Apply normalization using saved parameters (supports both standard and robust scaling).

    Optimized: Uses in-place vectorized operations to minimize memory allocation.
    """
    # Check scaling type - support both legacy (mean/std) and robust (median/iqr)
    scaling_type = norm_params.get('scaling_type', 'standard')

    if scaling_type == 'robust' or 'median' in norm_params:
        # Robust scaling: (x - median) / iqr
        center = np.array(norm_params['median'], dtype=sequences.dtype)
        scale = np.array(norm_params['iqr'], dtype=sequences.dtype)
    else:
        # Standard scaling: (x - mean) / std
        center = np.array(norm_params['mean'], dtype=sequences.dtype)
        scale = np.array(norm_params['std'], dtype=sequences.dtype)

    # Replace zeros with 1.0 to avoid division by zero (feature unchanged)
    scale_safe = np.where(scale > 0, scale, sequences.dtype.type(1.0))

    # Create copy once, then modify in-place (memory efficient)
    normalized = sequences.copy()
    normalized -= center  # In-place subtraction
    normalized /= scale_safe  # In-place division

    return normalized


def load_model_atomic(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model with embedded scalers."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']

    # Detect actual input_features from model weights (more reliable than config)
    if 'lstm.weight_ih_l0' in state_dict:
        input_features = state_dict['lstm.weight_ih_l0'].shape[1]
    elif 'lstm.lstm.weight_ih_l0' in state_dict:
        # ContextConditionedLSTM
        input_features = state_dict['lstm.lstm.weight_ih_l0'].shape[1]
    else:
        input_features = config.get('input_features', 10)

    # Detect if model has context features by checking for context_grn keys
    has_context = any('context_grn' in k for k in state_dict.keys())
    if has_context and 'context_grn.fc1.weight' in state_dict:
        context_dim = state_dict['context_grn.fc1.weight'].shape[1]
    else:
        context_dim = 0

    # Detect CNN architecture from state_dict keys (more reliable than config)
    # V18-style: conv_3, conv_5, cnn_norm (inline CNN)
    # V20-style: cnn_encoder.conv_small, cnn_encoder.conv_large, cnn_encoder.norm
    has_v18_cnn = 'conv_3.weight' in state_dict or 'conv_5.weight' in state_dict
    has_v20_cnn = 'cnn_encoder.conv_small.weight' in state_dict

    # Detect if using conditioned LSTM
    use_conditioned_lstm = 'lstm.h_proj.weight' in state_dict

    # Determine mode from actual architecture, not just config
    ablation_mode = config.get('ablation_mode')
    if has_v18_cnn:
        # Checkpoint uses V18-style inline CNN -> must use v18_full or v18_baseline
        mode = 'v18_full'
        logger.info("Detected V18-style CNN (conv_3, conv_5) -> using mode='v18_full'")
    elif has_v20_cnn:
        # Checkpoint uses V20-style TemporalEncoderCNN
        mode = ablation_mode if ablation_mode in ('concat', 'lstm', 'cqa_only') else 'lstm'
        logger.info(f"Detected V20-style CNN (cnn_encoder) -> using mode='{mode}'")
    else:
        # Fallback to config
        mode = ablation_mode if ablation_mode else 'v18_full'
        logger.info(f"CNN architecture unclear -> using mode='{mode}' from config")

    logger.info(f"Model detected from weights: input_features={input_features}, context_features={context_dim}, mode={mode}")
    logger.info(f"GRN Context: {'ENABLED (' + str(context_dim) + ' features)' if context_dim > 0 else 'DISABLED'}")
    logger.info(f"Conditioned LSTM: {use_conditioned_lstm}")

    model = HybridFeatureNetwork(
        mode=mode,
        input_features=input_features,
        num_classes=3,
        context_features=context_dim,
        use_conditioned_lstm=use_conditioned_lstm
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Model loaded - Epoch: {checkpoint.get('epoch', '?')}, Val Acc: {checkpoint.get('val_acc', '?')}")

    norm_params = checkpoint.get('norm_params')
    if norm_params:
        logger.info("Loaded embedded norm_params from checkpoint")

    return model, norm_params


def predict(model: torch.nn.Module, dataloader: TorchDataLoader, device: torch.device, use_context: bool = False):
    """Generate predictions for all sequences.

    Optimized: Uses torch.inference_mode for slightly better performance than no_grad.
    """
    all_probs = []
    all_logits = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            context = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3 and use_context:
                    sequences, _, context = batch
                elif len(batch) == 2:
                    if use_context:
                        sequences, context = batch
                    else:
                        sequences, _ = batch
                else:
                    sequences = batch[0]
            else:
                sequences = batch

            if isinstance(sequences, np.ndarray):
                sequences = torch.FloatTensor(sequences)
            sequences = sequences.to(device)

            if context is not None:
                if isinstance(context, np.ndarray):
                    context = torch.FloatTensor(context)
                context = context.to(device)

            logits = model(sequences, context=context)
            probs = torch.softmax(logits / PREDICTION_TEMPERATURE, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_logits, axis=0)


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("RISK-BASED EXECUTION PLAN GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Risk Configuration:")
    logger.info(f"  RISK_UNIT_DOLLARS:    ${args.risk_unit:,.2f}")
    logger.info(f"  MAX_CAPITAL_PER_TRADE: ${args.max_capital:,.2f}")
    logger.info(f"  ADV_LIQUIDITY_PCT:     {args.adv_pct * 100:.1f}%")
    logger.info(f"  TRIGGER_OFFSET:        ${args.trigger_offset:.2f}")
    logger.info(f"  MIN_SIGNAL:            {args.min_signal}")
    logger.info(f"  MAX_DANGER_PROB:       {args.max_danger_prob:.0%}")
    logger.info("=" * 70)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, embedded_norm_params = load_model_atomic(args.model, device)

    # Load sequences
    logger.info(f"Loading sequences from {args.sequences}")
    seq_path = Path(args.sequences)
    context = None
    labels = None

    if seq_path.suffix in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(seq_path, 'r') as f:
            sequences = f['sequences'][:]
            labels = f['labels'][:] if 'labels' in f else None
            context = f['context'][:] if 'context' in f else None
            logger.info(f"Loaded HDF5: sequences={sequences.shape}")
            if context is not None:
                logger.info(f"  Context: {context.shape}")
    elif seq_path.suffix == '.npz':
        seq_file = np.load(args.sequences, allow_pickle=True)
        sequences = seq_file['sequences']
        labels = seq_file.get('labels')
        context = seq_file.get('context')
    else:
        sequences = np.load(args.sequences, allow_pickle=True)

    # Load metadata (REQUIRED for execution plan)
    logger.info(f"Loading metadata from {args.metadata}")
    metadata = pd.read_parquet(args.metadata)
    logger.info(f"Metadata shape: {metadata.shape}")
    logger.info(f"Metadata columns: {list(metadata.columns)}")

    # Validate metadata has required columns for execution plan
    required = ['upper_boundary', 'lower_boundary']
    missing = [c for c in required if c not in metadata.columns]
    skip_execution_plan = False
    if missing:
        if args.prediction_only:
            logger.warning(f"Missing boundary columns {missing} - skipping execution plan (--prediction-only)")
            skip_execution_plan = True
        else:
            raise ValueError(f"Metadata missing required columns: {missing}. Use --prediction-only to skip execution plan.")

    # Apply normalization
    norm_params = None
    if args.norm_params:
        with open(args.norm_params, 'r') as f:
            norm_params = json.load(f)
    elif embedded_norm_params:
        norm_params = embedded_norm_params
    else:
        model_dir = Path(args.model).parent
        norm_files = sorted(model_dir.glob("norm_params_*.json"), reverse=True)
        if norm_files:
            with open(norm_files[0], 'r') as f:
                norm_params = json.load(f)

    if norm_params:
        logger.info("Applying normalization...")
        sequences = apply_normalization(sequences, norm_params)
    else:
        logger.warning("No normalization parameters - predictions may be inaccurate!")

    # Filter out samples with NaN values (data integrity)
    nan_mask = np.isnan(sequences).any(axis=(1, 2))
    n_nan_samples = nan_mask.sum()
    if n_nan_samples > 0:
        logger.warning(f"Filtering {n_nan_samples} samples with NaN values")
        valid_mask = ~nan_mask
        sequences = sequences[valid_mask]
        if labels is not None:
            labels = labels[valid_mask]
        if context is not None:
            context = context[valid_mask]
        # Also filter metadata to keep indices aligned
        metadata = metadata[valid_mask].reset_index(drop=True)
        logger.info(f"Remaining samples after NaN filter: {len(sequences)}")

    # Validate context
    use_context = USE_GRN_CONTEXT and context is not None
    if use_context and context.shape[-1] != NUM_CONTEXT_FEATURES:
        logger.warning(f"Context mismatch: {context.shape[-1]} vs {NUM_CONTEXT_FEATURES} - disabling")
        use_context = False
        context = None

    # Create dataloader
    dataset = SequenceDataset(sequences, labels, context=context)
    dataloader = TorchDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Generate predictions
    logger.info("\nGenerating predictions...")
    raw_probs, logits = predict(model, dataloader, device, use_context=use_context)

    # Apply calibration if available
    class_probs = raw_probs
    if USE_PROBABILITY_CALIBRATION:
        calibrator_path = args.calibrator
        if calibrator_path is None:
            model_dir = Path(args.model).parent
            calibrator_files = sorted(model_dir.glob("calibrator_*.pkl"), reverse=True)
            if calibrator_files:
                calibrator_path = str(calibrator_files[0])

        if calibrator_path and Path(calibrator_path).exists():
            try:
                from utils.probability_calibration import ProbabilityCalibrator
                calibrator = ProbabilityCalibrator.load(calibrator_path)
                class_probs = calibrator.calibrate(raw_probs)
                logger.info("Applied probability calibration")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

    # Calculate EV and signals
    ev = calculate_expected_value(class_probs)
    danger_prob = class_probs[:, 0]
    signals = generate_signals(
        ev,
        danger_prob=danger_prob,
        max_danger_prob=args.max_danger_prob
    )
    predicted_classes = np.argmax(class_probs, axis=1)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'predicted_class': predicted_classes,
        'danger_prob': class_probs[:, 0],
        'noise_prob': class_probs[:, 1],
        'target_prob': class_probs[:, 2],
        'expected_value': ev,
        'signal': signals
    })

    if labels is not None:
        predictions_df['true_class'] = labels

    # Add pattern_id if available
    if 'pattern_id' in metadata.columns:
        predictions_df['pattern_id'] = metadata['pattern_id'].values[:len(predictions_df)]

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Summary")
    logger.info("=" * 60)

    logger.info("\nPredicted class distribution:")
    for cls in range(NUM_CLASSES):
        count = (predicted_classes == cls).sum()
        pct = 100 * count / len(predicted_classes)
        logger.info(f"  {CLASS_NAMES[cls]}: {count:,} ({pct:.2f}%)")

    logger.info("\nSignal distribution:")
    for signal in ['STRONG', 'GOOD', 'MODERATE', 'AVOID']:
        count = (signals == signal).sum()
        pct = 100 * count / len(signals)
        logger.info(f"  {signal}: {count:,} ({pct:.2f}%)")

    logger.info(f"\nEV stats: Mean={ev.mean():.3f}, Median={np.median(ev):.3f}, Max={ev.max():.3f}")

    # Save predictions (always, even in prediction-only mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = output_dir / f"predictions_{timestamp}.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logger.info(f"\nSaved predictions: {predictions_path}")

    # =========================================================================
    # DRIFT CHECK (Jan 2026 - Optional PSI Monitoring)
    # =========================================================================
    if args.check_drift:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION DRIFT CHECK")
        logger.info("=" * 60)

        # Find baseline path
        drift_baseline = args.drift_baseline
        if drift_baseline is None:
            # Try default locations
            for candidate in [
                output_dir / 'drift_baseline.json',
                output_dir.parent / 'drift_baseline.json',
                Path('output/drift_baseline.json')
            ]:
                if candidate.exists():
                    drift_baseline = str(candidate)
                    break

        if drift_baseline and Path(drift_baseline).exists():
            try:
                from ml.drift_monitor import DriftMonitor, PSI_THRESHOLDS
                monitor = DriftMonitor.load_reference(Path(drift_baseline))

                # Check drift on probability columns
                prob_cols = ['danger_prob', 'noise_prob', 'target_prob', 'expected_value']
                available = [c for c in prob_cols if c in predictions_df.columns]

                if available:
                    report = monitor.analyze_drift(predictions_df[available])
                    logger.info(report.summary())

                    # Save drift report
                    report_path = output_dir / f"drift_report_{timestamp}.json"
                    import json as json_mod
                    with open(report_path, 'w') as f:
                        json_mod.dump(report.to_dict(), f, indent=2, default=str)

                    if report.action_required:
                        logger.warning("DRIFT ALERT: Model retraining may be required!")
                        logger.warning(f"  PSI threshold: {PSI_THRESHOLDS['high']}")
                else:
                    logger.warning("No probability columns for drift check")
            except Exception as e:
                logger.warning(f"Drift check failed: {e}")
        else:
            logger.warning(f"Drift baseline not found. Generate with:")
            logger.warning("  python scripts/monitor_drift.py --generate-baseline ...")

    # =========================================================================
    # GENERATE EXECUTION PLAN (skip if boundaries missing)
    # =========================================================================
    if skip_execution_plan:
        logger.info("\n" + "=" * 70)
        logger.info("SKIPPING EXECUTION PLAN (--prediction-only mode)")
        logger.info("=" * 70)
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTION COMPLETE")
        logger.info("=" * 70)
        return

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING EXECUTION PLAN")
    logger.info("=" * 70)

    execution_plan = generate_execution_plan(
        predictions_df=predictions_df,
        metadata_df=metadata,
        risk_unit_dollars=args.risk_unit,
        max_capital_per_trade=args.max_capital,
        adv_liquidity_pct=args.adv_pct,
        min_signal=args.min_signal,
        trigger_offset=args.trigger_offset,
        max_sector_positions=args.max_sector_positions
    )

    if len(execution_plan) == 0:
        logger.warning("No orders generated - check signal thresholds and data")
        return

    # Execution plan summary
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION PLAN SUMMARY")
    logger.info("=" * 60)

    total_capital = execution_plan['capital_required'].sum()
    total_risk = execution_plan['actual_risk_dollars'].sum()
    total_potential = execution_plan['potential_total'].sum()
    n_clamped = execution_plan['clamped'].sum()

    logger.info(f"\nOrders: {len(execution_plan)}")
    logger.info(f"Total Capital Required: ${total_capital:,.2f}")
    logger.info(f"Total Risk at Stake:    ${total_risk:,.2f}")
    logger.info(f"Avg Risk per Trade:     ${total_risk / len(execution_plan):,.2f}")
    logger.info(f"Orders Clamped:         {n_clamped} ({100*n_clamped/len(execution_plan):.1f}%)")

    # Justice Exit Plan potential
    logger.info("\n--- JUSTICE EXIT PLAN POTENTIAL ---")
    total_t1 = execution_plan['potential_t1'].sum()
    total_t2 = execution_plan['potential_t2'].sum()
    total_t3 = execution_plan['potential_t3'].sum()
    logger.info(f"  T1 (Bank +3R,  50%): ${total_t1:,.2f}")
    logger.info(f"  T2 (Runner +5R, 25%): ${total_t2:,.2f}")
    logger.info(f"  T3 (Moon +10R, 25%): ${total_t3:,.2f}")
    logger.info(f"  TOTAL (if all hit):  ${total_potential:,.2f}")
    logger.info(f"  Risk/Reward Ratio:   1:{total_potential/total_risk:.1f}" if total_risk > 0 else "  Risk/Reward: N/A")

    if n_clamped > 0:
        clamp_reasons = execution_plan[execution_plan['clamped']]['clamp_reason'].value_counts()
        logger.info("\nClamping Breakdown:")
        for reason, count in clamp_reasons.items():
            logger.info(f"  - {reason}: {count}")

    # Signal breakdown
    logger.info("\nBy Signal Strength:")
    for signal in ['STRONG', 'GOOD', 'MODERATE']:
        mask = execution_plan['signal'] == signal
        if mask.sum() > 0:
            subset = execution_plan[mask]
            logger.info(f"  {signal}: {mask.sum()} orders, ${subset['capital_required'].sum():,.0f} capital")

    # Top orders preview with Justice targets
    logger.info("\nTop 10 Orders (by EV):")
    logger.info("-" * 100)
    logger.info(f"{'Ticker':<8} | {'Signal':<8} | {'EV':>6} | {'Shares':>6} | {'Trigger':>8} | {'Stop':>8} | {'T1(3R)':>8} | {'T2(5R)':>8} | {'T3(10R)':>8}")
    logger.info("-" * 100)

    for idx, row in execution_plan.head(10).iterrows():
        logger.info(
            f"{row['ticker']:<8} | {row['signal']:<8} | {row['expected_value']:>+5.2f} | "
            f"{row['shares']:>6.0f} | ${row['trigger_price']:>7.2f} | ${row['stop_price']:>7.2f} | "
            f"${row['t1_price']:>7.2f} | ${row['t2_price']:>7.2f} | ${row['t3_price']:>7.2f}"
        )

    # =========================================================================
    # SAVE EXECUTION PLAN OUTPUTS
    # =========================================================================

    # Save execution plan (Parquet)
    plan_parquet_path = output_dir / f"execution_plan_{timestamp}.parquet"
    execution_plan.to_parquet(plan_parquet_path, index=False)
    logger.info(f"Saved execution plan (Parquet): {plan_parquet_path}")

    # Save execution plan (CSV for easy viewing)
    plan_csv_path = output_dir / f"execution_plan_{timestamp}.csv"
    # Select key columns for CSV
    csv_cols = [
        'ticker', 'detection_date', 'signal', 'expected_value',
        'danger_prob', 'noise_prob', 'target_prob',
        'trigger_price', 'limit_price', 'stop_price',
        'risk_per_share', 'risk_width_pct', 'shares', 'capital_required',
        'actual_risk_dollars', 't1_price', 't1_shares', 't2_price', 't2_shares',
        't3_price', 't3_shares', 'potential_total', 'liquidity_pct',
        'clamped', 'clamp_reason',
        # V21 EOD Structural Execution fields (Buy Stop Limit)
        'order_type', 'stop_price', 'limit_price', 'r_per_share',
        'max_slippage', 'slippage_cap_r', 'chase_warning_price', 'execution_instruction'
    ]
    csv_cols = [c for c in csv_cols if c in execution_plan.columns]
    execution_plan[csv_cols].to_csv(plan_csv_path, index=False)
    logger.info(f"Saved execution plan (CSV): {plan_csv_path}")

    # =========================================================================
    # NIGHTLY ORDERS FILE - Broker Import Format
    # =========================================================================
    # Format: Ticker, Action, Trigger, Limit_Price, Stop_Loss, Shares, T1, T2, T3
    nightly_orders_path = output_dir / f"nightly_orders_{timestamp}.csv"

    nightly_df = execution_plan.copy()

    # Reorder columns for broker import
    # V21 EOD Structural Execution: BUY STOP LIMIT with slippage cap
    nightly_cols = [
        'ticker', 'order_type', 'stop_price', 'limit_price',
        'shares', 'chase_warning_price', 'max_slippage',
        't1_price', 't2_price', 't3_price', 'execution_instruction'
    ]
    # Filter to existing columns
    nightly_cols = [c for c in nightly_cols if c in nightly_df.columns]

    # Rename columns for clarity
    nightly_output = nightly_df[nightly_cols].copy()
    col_renames = {
        'ticker': 'Ticker',
        'order_type': 'Order_Type',
        'stop_price': 'Stop_Price',           # Price that triggers the order
        'limit_price': 'Limit_Price',         # Max fill price (caps slippage)
        'shares': 'Shares',
        'chase_warning_price': 'Do_Not_Chase_If_Open_Above',  # Cancel if Open > this
        'max_slippage': 'Max_Slippage',
        't1_price': 'T1_Price',
        't2_price': 'T2_Price',
        't3_price': 'T3_Price',
        'execution_instruction': 'Instructions'
    }
    nightly_output.columns = [col_renames.get(c, c) for c in nightly_output.columns]

    nightly_output.to_csv(nightly_orders_path, index=False)
    logger.info(f"Saved nightly orders: {nightly_orders_path}")

    # Also save a detailed Justice Exit Plan file
    justice_path = output_dir / f"justice_exit_plan_{timestamp}.csv"
    justice_cols = [
        'ticker', 'shares',
        't1_price', 't1_shares',
        't2_price', 't2_shares',
        't3_price', 't3_shares',
        'potential_t1', 'potential_t2', 'potential_t3', 'potential_total'
    ]
    justice_cols = [c for c in justice_cols if c in execution_plan.columns]

    justice_df = execution_plan[justice_cols].copy()
    justice_df.columns = [
        'Ticker', 'Total_Shares',
        'T1_Bank_Price', 'T1_Sell_Shares',
        'T2_Runner_Price', 'T2_Sell_Shares',
        'T3_Moon_Price', 'T3_Hold_Shares',
        'T1_Gain_$', 'T2_Gain_$', 'T3_Gain_$', 'Total_Potential_$'
    ]
    justice_df.insert(1, 'Exit_Strategy', 'JUSTICE_3_5_10')
    justice_df.to_csv(justice_path, index=False)
    logger.info(f"Saved Justice Exit Plan: {justice_path}")

    # Save risk summary
    risk_summary = {
        'timestamp': timestamp,
        'config': {
            'risk_unit_dollars': args.risk_unit,
            'max_capital_per_trade': args.max_capital,
            'adv_liquidity_pct': args.adv_pct,
            'trigger_offset': args.trigger_offset,
            'min_signal': args.min_signal
        },
        'summary': {
            'total_orders': len(execution_plan),
            'total_capital_required': round(total_capital, 2),
            'total_risk_at_stake': round(total_risk, 2),
            'avg_risk_per_trade': round(total_risk / len(execution_plan), 2),
            'orders_clamped': int(n_clamped),
            'orders_clamped_pct': round(100 * n_clamped / len(execution_plan), 1)
        },
        'justice_exit_plan': {
            'strategy': 'JUSTICE_3_5_10',
            't1_bank_3r': {
                'r_multiple': 3.0,
                'sell_pct': 50,
                'total_potential': round(total_t1, 2)
            },
            't2_runner_5r': {
                'r_multiple': 5.0,
                'sell_pct': 25,
                'total_potential': round(total_t2, 2)
            },
            't3_moon_10r': {
                'r_multiple': 10.0,
                'hold_pct': 25,
                'trailing_stop': True,
                'total_potential': round(total_t3, 2)
            },
            'total_potential_all_targets': round(total_potential, 2),
            'risk_reward_ratio': round(total_potential / total_risk, 2) if total_risk > 0 else 0,
            'weighted_avg_r': 5.25  # (50% * 3R) + (25% * 5R) + (25% * 10R)
        },
        'signal_breakdown': execution_plan.groupby('signal').agg({
            'capital_required': 'sum',
            'actual_risk_dollars': 'sum',
            'shares': 'count'
        }).to_dict()
    }

    summary_path = output_dir / f"risk_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(risk_summary, f, indent=2, default=str)
    logger.info(f"Saved risk summary: {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION PLAN GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
