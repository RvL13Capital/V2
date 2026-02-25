"""
Evaluate Model for Live Trading Performance

FOCUS: Precision @ Top 15% and EV > 3.0 Threshold
- Global accuracy is IRRELEVANT (includes noise/danger we'll never trade)
- Only metric that matters: win rate on highest-confidence signals
- Live trading rule: ONLY signal if Predicted EV > 3.0
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.temporal_hybrid_v18 import HybridFeatureNetwork  # V18: Context-Query Attention
from config.constants import STRATEGIC_VALUES, calculate_expected_value, USE_GRN_CONTEXT, DRIFT_CRITICAL_FEATURES
from config.context_features import NUM_CONTEXT_FEATURES
from ml.drift_monitor import DriftMonitor, PSI_THRESHOLDS

# ============================================================================
# REALISTIC TRADING COST CONSTANTS (Jan 2026)
# ============================================================================
# Spread Tax: Cost of crossing bid-ask spread on illiquid micro-caps
# - Entry: Pay ask price (0.5% above mid)
# - Exit: Receive bid price (0.5% below mid)
# - Total round-trip cost: 1.0%
SPREAD_TAX_ENTRY_PCT = 0.5   # 0.5% on entry (pay ask)
SPREAD_TAX_EXIT_PCT = 0.5    # 0.5% on exit (receive bid)
SPREAD_TAX_TOTAL_PCT = SPREAD_TAX_ENTRY_PCT + SPREAD_TAX_EXIT_PCT  # 1.0% round-trip


def calculate_gap_penalty(
    close_t: float,
    open_t_plus_1: float
) -> float:
    """
    Calculate the gap penalty between Close_T and Open_T+1.

    Gap Penalty Logic (CONSERVATIVE/PESSIMISTIC):
    - Gap Up (Open_T+1 > Close_T): We paid MORE than expected. Penalty is positive.
    - Gap Down (Open_T+1 < Close_T): Do NOT improve PnL. Assume we missed fill or hesitated.
      This is a conservative assumption - in reality, gap downs often:
      1. Cause hesitation (you second-guess the signal)
      2. Fill at worse prices due to slippage on the recovery
      3. Get missed entirely if using limit orders

    Args:
        close_t: Close price at pattern end (signal day)
        open_t_plus_1: Open price on execution day (T+1)

    Returns:
        Gap penalty as a percentage (always >= 0, conservative)
    """
    if close_t <= 0 or open_t_plus_1 <= 0:
        return 0.0

    gap_pct = (open_t_plus_1 - close_t) / close_t

    if gap_pct > 0:
        # Gap Up: We paid more. Return positive penalty.
        return gap_pct
    else:
        # Gap Down: Do NOT improve PnL (conservative assumption)
        # In reality, gap downs often mean missed fills or hesitation
        return 0.0


def calculate_realized_pnl(
    theoretical_r_multiple: float,
    close_t: float,
    open_t_plus_1: float,
    entry_price: Optional[float] = None,
    spread_tax_entry: float = SPREAD_TAX_ENTRY_PCT,
    spread_tax_exit: float = SPREAD_TAX_EXIT_PCT
) -> Dict:
    """
    Calculate realized PnL after accounting for gap penalty and spread tax.

    REALISTIC TRADING COSTS FOR MICRO-CAPS:
    ========================================

    1. GAP PENALTY (Execution Slippage):
       - Signal fires at Close_T (e.g., $10.00)
       - Execution at Open_T+1 (e.g., $10.50)
       - Gap Penalty = (10.50 - 10.00) / 10.00 = 5%
       - This 5% is LOST from your expected return

       Conservative Assumption for Gap Downs:
       - If Open_T+1 < Close_T (gap down), we do NOT credit the improvement
       - Reasons: Hesitation, missed fills, slippage on recovery

    2. SPREAD TAX (Bid-Ask Spread):
       - Entry: Pay 0.5% above mid-price (crossing ask)
       - Exit: Receive 0.5% below mid-price (hitting bid)
       - Total: 1.0% round-trip cost

       This is REALISTIC for illiquid micro-caps:
       - Blue chips: 0.01-0.05% spread
       - Small caps: 0.1-0.5% spread
       - MICRO-CAPS: 0.5-2.0% spread (we use 0.5% conservative estimate)

    Args:
        theoretical_r_multiple: The R-multiple outcome (e.g., +5 for Target, -2 for Danger)
        close_t: Close price at pattern end (signal day)
        open_t_plus_1: Open price on execution day (T+1)
        entry_price: Actual entry price if known (defaults to open_t_plus_1)
        spread_tax_entry: Entry spread cost in % (default: 0.5%)
        spread_tax_exit: Exit spread cost in % (default: 0.5%)

    Returns:
        Dictionary with:
        - theoretical_pnl: PnL based on R-multiple alone
        - gap_penalty: Gap penalty as % (always >= 0)
        - spread_tax_total: Total spread cost in %
        - realized_pnl: Adjusted PnL after all costs
        - is_winner: True if realized_pnl > 0
        - cost_ate_profit: True if theoretical winner became loser due to costs
    """
    # Use open_t_plus_1 as entry if not specified
    if entry_price is None:
        entry_price = open_t_plus_1

    # Calculate gap penalty (always >= 0)
    gap_penalty = calculate_gap_penalty(close_t, open_t_plus_1)

    # Total spread tax
    spread_tax_total = (spread_tax_entry + spread_tax_exit) / 100.0  # Convert to decimal

    # Theoretical PnL (based on R-multiple, assuming R = 1% of position for normalization)
    # In the TRANS system:
    # - Target = +5R (strategic value +5)
    # - Noise = -0.1R (strategic value -0.1)
    # - Danger = -2R (strategic value -2)
    theoretical_pnl = theoretical_r_multiple

    # Realized PnL calculation:
    # 1. Start with theoretical PnL
    # 2. Subtract gap penalty (we paid more on entry)
    # 3. Subtract spread tax (bid-ask crossing cost)
    #
    # Note: Gap penalty affects ENTRY, spread tax affects BOTH entry and exit
    # For simplicity, we model these as percentage costs subtracted from returns

    # Convert gap_penalty to same units as theoretical_pnl
    # If theoretical_pnl is in R-multiples and entry was supposed to be at Close_T,
    # then a 5% gap means we lost 5% of position value on entry
    #
    # Assuming R is approximately the average R-value (around 5% of position),
    # we express gap penalty in R-multiples:
    # gap_cost_in_r = gap_penalty / 0.05 (if R = 5%)
    #
    # For conservative estimate, assume R = 3% (typical for tight consolidations)
    R_ESTIMATE = 0.03  # 3% is typical R-multiple base
    gap_cost_in_r = gap_penalty / R_ESTIMATE if R_ESTIMATE > 0 else 0
    spread_cost_in_r = spread_tax_total / R_ESTIMATE if R_ESTIMATE > 0 else 0

    # Realized PnL
    realized_pnl = theoretical_pnl - gap_cost_in_r - spread_cost_in_r

    # Determine if winner/loser
    is_winner = realized_pnl > 0
    theoretical_winner = theoretical_pnl > 0
    cost_ate_profit = theoretical_winner and not is_winner

    return {
        'theoretical_pnl': theoretical_pnl,
        'gap_penalty_pct': gap_penalty * 100,  # As percentage
        'gap_cost_in_r': gap_cost_in_r,
        'spread_tax_pct': spread_tax_total * 100,  # As percentage
        'spread_cost_in_r': spread_cost_in_r,
        'total_cost_in_r': gap_cost_in_r + spread_cost_in_r,
        'realized_pnl': realized_pnl,
        'is_winner': is_winner,
        'theoretical_winner': theoretical_winner,
        'cost_ate_profit': cost_ate_profit
    }


def estimate_gap_adjusted_metrics(
    df_predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    avg_gap_up_pct: float = 2.0,  # Estimated average gap up percentage
    gap_up_probability: float = 0.60  # Probability of gap up (vs gap down/flat)
) -> Dict:
    """
    Estimate gap-adjusted metrics when actual price data is unavailable.

    This is a STATISTICAL ESTIMATE based on typical micro-cap behavior:
    - 60% of entries gap UP (we pay more)
    - 40% of entries gap DOWN or flat (conservative: no credit)
    - Average gap up is 2% for micro-caps

    Args:
        df_predictions: DataFrame with predictions
        metadata: Metadata DataFrame (may contain price data)
        avg_gap_up_pct: Average gap up percentage when gap occurs (default: 2%)
        gap_up_probability: Probability of gap up (default: 60%)

    Returns:
        Dictionary with estimated gap-adjusted metrics
    """
    # Check if we have actual price data in metadata
    has_price_data = (
        'close_at_end' in metadata.columns and
        'open_t_plus_1' in metadata.columns
    )

    if has_price_data:
        # Use actual price data
        return calculate_gap_adjusted_metrics_from_prices(df_predictions, metadata)
    else:
        # Estimate based on typical micro-cap behavior
        return estimate_gap_adjusted_metrics_statistical(
            df_predictions,
            avg_gap_up_pct=avg_gap_up_pct,
            gap_up_probability=gap_up_probability
        )


def estimate_gap_adjusted_metrics_statistical(
    df_predictions: pd.DataFrame,
    avg_gap_up_pct: float = 2.0,
    gap_up_probability: float = 0.60
) -> Dict:
    """
    Statistical estimate of gap-adjusted metrics without actual price data.

    MICRO-CAP TYPICAL BEHAVIOR:
    - Gap ups occur ~60% of the time on breakout signals
    - Average gap up is ~2% for micro-caps (higher for hot sectors)
    - Gap downs: We do NOT credit improvement (conservative)
    """
    # R_ESTIMATE = 3% (typical R-multiple base)
    R_ESTIMATE = 0.03

    # Expected gap cost per trade
    # = P(gap_up) * avg_gap_up_pct + P(gap_down) * 0 (conservative)
    expected_gap_pct = gap_up_probability * avg_gap_up_pct
    expected_gap_cost_in_r = expected_gap_pct / 100 / R_ESTIMATE

    # Spread cost (fixed 1% round-trip)
    spread_cost_in_r = SPREAD_TAX_TOTAL_PCT / 100 / R_ESTIMATE

    # Total expected cost per trade in R-multiples
    total_cost_in_r = expected_gap_cost_in_r + spread_cost_in_r

    # Calculate adjusted PnL for each prediction
    df_adj = df_predictions.copy()
    df_adj['theoretical_pnl'] = df_adj['actual_value']  # Strategic value = R-multiple
    df_adj['estimated_gap_cost'] = expected_gap_cost_in_r
    df_adj['spread_cost'] = spread_cost_in_r
    df_adj['realized_pnl'] = df_adj['theoretical_pnl'] - total_cost_in_r

    # Win rate calculations
    theoretical_winners = df_adj['theoretical_pnl'] > 0
    gap_adjusted_winners = df_adj['realized_pnl'] > 0

    theoretical_win_rate = theoretical_winners.sum() / len(df_adj)
    gap_adjusted_win_rate = gap_adjusted_winners.sum() / len(df_adj)

    # Cost impact
    cost_ate_profit = theoretical_winners & ~gap_adjusted_winners

    return {
        'method': 'statistical_estimate',
        'assumptions': {
            'avg_gap_up_pct': avg_gap_up_pct,
            'gap_up_probability': gap_up_probability,
            'spread_tax_total_pct': SPREAD_TAX_TOTAL_PCT,
            'r_estimate': R_ESTIMATE * 100  # As percentage
        },
        'costs': {
            'expected_gap_cost_per_trade_pct': expected_gap_pct,
            'expected_gap_cost_in_r': expected_gap_cost_in_r,
            'spread_cost_in_r': spread_cost_in_r,
            'total_cost_per_trade_in_r': total_cost_in_r
        },
        'win_rates': {
            'theoretical_win_rate': theoretical_win_rate,
            'gap_adjusted_win_rate': gap_adjusted_win_rate,
            'win_rate_reduction': theoretical_win_rate - gap_adjusted_win_rate
        },
        'impact': {
            'trades_flipped_to_loss': cost_ate_profit.sum(),
            'pct_trades_flipped': cost_ate_profit.sum() / len(df_adj) * 100,
            'avg_theoretical_pnl': df_adj['theoretical_pnl'].mean(),
            'avg_realized_pnl': df_adj['realized_pnl'].mean()
        }
    }


def calculate_gap_adjusted_metrics_from_prices(
    df_predictions: pd.DataFrame,
    metadata: pd.DataFrame
) -> Dict:
    """
    Calculate gap-adjusted metrics using actual price data from metadata.

    Requires metadata to have:
    - close_at_end: Close price at pattern end (signal day)
    - open_t_plus_1: Open price on execution day (T+1)
    """
    # Merge predictions with price data
    # Match on pattern_id or ticker+date
    if 'pattern_id' in df_predictions.columns and 'pattern_id' in metadata.columns:
        merge_cols = ['pattern_id']
    elif 'ticker' in df_predictions.columns and 'ticker' in metadata.columns:
        merge_cols = ['ticker', 'start_date'] if 'start_date' in metadata.columns else ['ticker']
    else:
        raise ValueError("Cannot match predictions to metadata - no common columns")

    # Select only needed columns from metadata
    price_cols = ['close_at_end', 'open_t_plus_1']
    available_price_cols = [c for c in price_cols if c in metadata.columns]

    if not available_price_cols:
        raise ValueError("Metadata missing price columns (close_at_end, open_t_plus_1)")

    df_merged = df_predictions.merge(
        metadata[merge_cols + available_price_cols].drop_duplicates(),
        on=merge_cols,
        how='left'
    )

    # Calculate actual gap penalty for each trade
    gap_penalties = []
    realized_pnls = []

    for _, row in df_merged.iterrows():
        close_t = row.get('close_at_end', 0)
        open_t_plus_1 = row.get('open_t_plus_1', 0)
        theoretical_pnl = row['actual_value']

        if close_t > 0 and open_t_plus_1 > 0:
            result = calculate_realized_pnl(
                theoretical_r_multiple=theoretical_pnl,
                close_t=close_t,
                open_t_plus_1=open_t_plus_1
            )
            gap_penalties.append(result['gap_penalty_pct'])
            realized_pnls.append(result['realized_pnl'])
        else:
            gap_penalties.append(0)
            realized_pnls.append(theoretical_pnl)  # Fallback to theoretical

    df_merged['gap_penalty_pct'] = gap_penalties
    df_merged['realized_pnl'] = realized_pnls

    # Win rate calculations
    theoretical_winners = df_merged['actual_value'] > 0
    gap_adjusted_winners = df_merged['realized_pnl'] > 0

    theoretical_win_rate = theoretical_winners.sum() / len(df_merged)
    gap_adjusted_win_rate = gap_adjusted_winners.sum() / len(df_merged)

    # Cost impact
    cost_ate_profit = theoretical_winners & ~gap_adjusted_winners

    # Gap statistics
    positive_gaps = df_merged[df_merged['gap_penalty_pct'] > 0]['gap_penalty_pct']

    return {
        'method': 'actual_price_data',
        'gap_statistics': {
            'avg_gap_penalty_pct': df_merged['gap_penalty_pct'].mean(),
            'median_gap_penalty_pct': df_merged['gap_penalty_pct'].median(),
            'max_gap_penalty_pct': df_merged['gap_penalty_pct'].max(),
            'pct_trades_with_gap_up': (df_merged['gap_penalty_pct'] > 0).mean() * 100,
            'avg_gap_up_when_occurs': positive_gaps.mean() if len(positive_gaps) > 0 else 0
        },
        'costs': {
            'spread_tax_total_pct': SPREAD_TAX_TOTAL_PCT
        },
        'win_rates': {
            'theoretical_win_rate': theoretical_win_rate,
            'gap_adjusted_win_rate': gap_adjusted_win_rate,
            'win_rate_reduction': theoretical_win_rate - gap_adjusted_win_rate
        },
        'impact': {
            'trades_flipped_to_loss': cost_ate_profit.sum(),
            'pct_trades_flipped': cost_ate_profit.sum() / len(df_merged) * 100,
            'avg_theoretical_pnl': df_merged['actual_value'].mean(),
            'avg_realized_pnl': df_merged['realized_pnl'].mean()
        }
    }


class TradingPerformanceEvaluator:
    """Evaluates model performance for live trading decisions"""

    def __init__(self, model_path: str, metadata_path: str, sequences_dir: str,
                 load_model: bool = True, train_cutoff: str = None, val_cutoff: str = None):
        """
        Args:
            model_path: Path to trained model checkpoint
            metadata_path: Path to metadata parquet with labels
            sequences_dir: Directory containing sequence .npy files
            load_model: Whether to load the model (False for drift-only mode)
            train_cutoff: Date string for train/val boundary (e.g., '2024-01-01')
            val_cutoff: Date string for val/test boundary (e.g., '2024-07-01')
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.sequences_dir = Path(sequences_dir)
        self.train_cutoff = train_cutoff
        self.val_cutoff = val_cutoff

        # Load model (optional for drift-only mode)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model() if load_model else None

        # Load normalization parameters (CRITICAL - must match training)
        self.norm_params = self._load_norm_params()

        # Load data
        self.metadata = pd.read_parquet(metadata_path)
        print(f"Loaded {len(self.metadata)} patterns from metadata")

        # Detect metadata format and load bulk sequences if needed
        # If no sequence_filename column, use bulk array with row index as sequence index
        self.use_bulk_sequences = 'sequence_filename' not in self.metadata.columns
        self.bulk_sequences = None

        if self.use_bulk_sequences:
            print("Detected bulk sequence format (row index = sequence index)")
            self._load_bulk_sequences()
            self._apply_robust_scaling()  # Apply global robust scaling to composite features
            self._ensure_splits()
            # Add proper sequence index based on row number
            self.metadata = self.metadata.reset_index(drop=True)
            self.metadata['_seq_idx'] = self.metadata.index

    def _load_model(self) -> HybridFeatureNetwork:
        """Load trained model checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Extract model config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Context features controlled by USE_GRN_CONTEXT flag (13-feature standard Jan 2026)
            context_dim = NUM_CONTEXT_FEATURES if USE_GRN_CONTEXT else 0

            # Check if use_conditioned_lstm was used during training (from args saved in checkpoint)
            training_args = checkpoint.get('config', {})
            use_conditioned_lstm = training_args.get('use_conditioned_lstm', False)

            config = {
                'input_features': training_args.get('input_features', 10),  # 10 after composite disabled
                'sequence_length': 20,  # Full 20 timesteps (BBW fix ensures valid values)
                'context_features': context_dim,  # 13-feature standard (8 original + 5 coil)
                'lstm_hidden': 32,
                'lstm_num_layers': 2,
                'lstm_dropout': 0.2,
                'num_classes': 3,
                'use_conditioned_lstm': use_conditioned_lstm,  # Jan 2026: Context-Conditioned LSTM
            }
            print(f"GRN Context: {'ENABLED (' + str(NUM_CONTEXT_FEATURES) + ' features)' if USE_GRN_CONTEXT else 'DISABLED (pure Temporal/Spatial)'}")
            if use_conditioned_lstm:
                print(f"Context-Conditioned LSTM: ENABLED (h0/c0 from GRN)")

        # Create model
        model = HybridFeatureNetwork(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"Loaded model from {self.model_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")

        return model

    def _load_norm_params(self) -> Optional[Dict]:
        """Load normalization parameters from JSON file (must match training)"""
        models_dir = self.model_path.parent

        # Look for norm_params matching model timestamp or most recent
        model_timestamp = self.model_path.stem.replace('best_model_', '')

        possible_files = [
            models_dir / f'norm_params_{model_timestamp}.json',
            models_dir / 'norm_params.json',
        ]

        # Also try to find most recent norm_params
        all_norm_files = sorted(models_dir.glob('norm_params_*.json'))

        for norm_file in possible_files:
            if norm_file.exists():
                with open(norm_file, 'r') as f:
                    params = json.load(f)
                print(f"Loaded normalization parameters from: {norm_file}")
                return params

        # Fall back to most recent
        if all_norm_files:
            norm_file = all_norm_files[-1]
            with open(norm_file, 'r') as f:
                params = json.load(f)
            print(f"Using most recent norm params: {norm_file}")
            return params

        print("WARNING: No normalization parameters found - predictions may be incorrect!")
        return None

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply normalization to sequence (must match training normalization)"""
        if self.norm_params is None:
            return sequence

        mean = np.array(self.norm_params['mean'])
        std = np.array(self.norm_params['std'])

        # Features to normalize (matching training script)
        # Skip features 8-11 which are already normalized (price_position, days_in_pattern, etc.)
        features_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]

        normalized = sequence.copy()
        for feat_idx in features_to_normalize:
            if feat_idx < sequence.shape[-1] and std[feat_idx] > 1e-8:
                normalized[..., feat_idx] = (sequence[..., feat_idx] - mean[feat_idx]) / std[feat_idx]

        return normalized

    def _load_bulk_sequences(self):
        """Load bulk sequences array matching metadata timestamp"""
        import h5py

        # Try to find matching sequences file based on metadata filename
        meta_name = self.metadata_path.stem  # e.g., metadata_20251219_043221
        timestamp = meta_name.replace('metadata_', '').replace('_with_outcome_day', '').replace('_refined', '').replace('_latest', '')

        # Look for matching H5 file first (preferred format)
        h5_files = [
            self.sequences_dir / f'sequences_{timestamp}.h5',
        ]

        # Also check for any H5 file in the directory
        all_h5_files = sorted(self.sequences_dir.glob('sequences_*.h5'))

        for h5_file in h5_files + all_h5_files:
            if h5_file.exists():
                print(f"Loading bulk sequences from H5: {h5_file}")
                with h5py.File(h5_file, 'r') as f:
                    self.bulk_sequences = f['sequences'][:]
                    # Also load labels if present
                    if 'labels' in f:
                        self.bulk_labels = f['labels'][:]
                print(f"Loaded {len(self.bulk_sequences)} sequences with shape {self.bulk_sequences.shape}")
                return

        # Look for matching NPY file
        possible_files = [
            self.sequences_dir / f'sequences_{timestamp}.npy',
            self.sequences_dir / f'sequences_{timestamp}_fixed.npy',
            self.sequences_dir / 'sequences_refined_latest.npy',
        ]

        # Also try to find most recent sequences file
        all_seq_files = sorted(self.sequences_dir.glob('sequences_*.npy'))

        for seq_file in possible_files:
            if seq_file.exists():
                print(f"Loading bulk sequences from: {seq_file}")
                self.bulk_sequences = np.load(seq_file)
                print(f"Loaded {len(self.bulk_sequences)} sequences with shape {self.bulk_sequences.shape}")
                return

        # Fall back to most recent sequences file
        if all_seq_files:
            seq_file = all_seq_files[-1]
            print(f"Using most recent sequences file: {seq_file}")
            self.bulk_sequences = np.load(seq_file)
            print(f"Loaded {len(self.bulk_sequences)} sequences with shape {self.bulk_sequences.shape}")
            return

        raise FileNotFoundError(f"No bulk sequences file found in {self.sequences_dir}")

    def _apply_robust_scaling(self):
        """Apply global robust scaling to composite features (vol_dryup_ratio, var_score, nes_score, lpf_score)"""
        if self.bulk_sequences is None:
            return

        # Find robust scaling parameters in model directory
        model_dir = self.model_path.parent
        robust_files = sorted(model_dir.glob("robust_scaling_params_*.json"), reverse=True)

        if not robust_files:
            print("Warning: No robust scaling parameters found - composite features may have wrong scale")
            return

        robust_path = robust_files[0]  # Most recent
        print(f"Loading robust scaling parameters: {robust_path}")

        with open(robust_path, 'r') as f:
            robust_params = json.load(f)

        # DISABLED (2026-01-18): Composite features removed from feature set
        # Apply robust scaling to composite features (indices 8-11) - if any
        composite_indices = []  # DISABLED: was [8, 9, 10, 11]
        if composite_indices:
            print("Applying robust scaling to composite features...")
            for idx in composite_indices:
                key_median = f'feat_{idx}_median'
                key_iqr = f'feat_{idx}_iqr'
                if key_median in robust_params and key_iqr in robust_params:
                    median = robust_params[key_median]
                    iqr = robust_params[key_iqr]
                    self.bulk_sequences[:, :, idx] = (self.bulk_sequences[:, :, idx] - median) / iqr
                    print(f"  Feature {idx}: median={median:.4f}, IQR={iqr:.4f} -> applied")
            print("Robust scaling applied successfully")
        else:
            print("Composite features DISABLED - skipping robust scaling")

    def _ensure_splits(self):
        """Ensure metadata has train/val/test splits"""
        if 'split' in self.metadata.columns:
            return

        # Keep original indices for sequence alignment
        self.metadata = self.metadata.reset_index(drop=True)

        # Get date column
        date_col = None
        if 'pattern_end_date' in self.metadata.columns:
            date_col = 'pattern_end_date'
        elif 'pattern_start_date' in self.metadata.columns:
            date_col = 'pattern_start_date'

        if date_col:
            self.metadata[date_col] = pd.to_datetime(self.metadata[date_col])

        # Use date-based cutoffs if provided (matches training script)
        if self.train_cutoff and self.val_cutoff and date_col:
            print(f"Creating train/val/test splits using date cutoffs...")
            print(f"  Train: < {self.train_cutoff}")
            print(f"  Val:   {self.train_cutoff} to {self.val_cutoff}")
            print(f"  Test:  >= {self.val_cutoff}")

            train_cutoff_dt = pd.Timestamp(self.train_cutoff)
            val_cutoff_dt = pd.Timestamp(self.val_cutoff)

            self.metadata['split'] = 'test'  # Default
            self.metadata.loc[self.metadata[date_col] < train_cutoff_dt, 'split'] = 'train'
            self.metadata.loc[
                (self.metadata[date_col] >= train_cutoff_dt) &
                (self.metadata[date_col] < val_cutoff_dt),
                'split'
            ] = 'val'

        else:
            # Fallback: percentage-based temporal split (70/15/15)
            print("Creating train/val/test splits (70/15/15) based on temporal order...")

            if date_col:
                sort_order = self.metadata[date_col].argsort()
            else:
                # If no dates, use original order
                sort_order = np.arange(len(self.metadata))

            n = len(self.metadata)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            # Assign splits based on temporal order
            self.metadata['split'] = 'test'  # Default
            train_indices = sort_order[:train_end]
            val_indices = sort_order[train_end:val_end]

            self.metadata.loc[train_indices, 'split'] = 'train'
            self.metadata.loc[val_indices, 'split'] = 'val'

        split_counts = self.metadata['split'].value_counts()
        print(f"  Train: {split_counts.get('train', 0)}, Val: {split_counts.get('val', 0)}, Test: {split_counts.get('test', 0)}")

        # Show date ranges for each split
        if date_col:
            for split_name in ['train', 'val', 'test']:
                split_data = self.metadata[self.metadata['split'] == split_name]
                if len(split_data) > 0:
                    min_date = split_data[date_col].min().date()
                    max_date = split_data[date_col].max().date()
                    print(f"    {split_name}: {min_date} to {max_date}")

    def _load_sequence(self, seq_identifier) -> torch.Tensor:
        """Load sequence from .npy file or bulk array and apply normalization"""
        if self.use_bulk_sequences and self.bulk_sequences is not None:
            # Load from bulk array using index
            if isinstance(seq_identifier, (int, np.integer)):
                idx = int(seq_identifier)
                if idx < 0 or idx >= len(self.bulk_sequences):
                    raise IndexError(f"Sequence index {idx} out of range (0-{len(self.bulk_sequences)-1})")
                sequence = self.bulk_sequences[idx]
            else:
                raise ValueError(f"Expected sequence_idx (int), got {type(seq_identifier)}")
        else:
            # Load from individual file
            seq_path = self.sequences_dir / seq_identifier
            if not seq_path.exists():
                raise FileNotFoundError(f"Sequence file not found: {seq_path}")
            sequence = np.load(seq_path)

        # Apply normalization (CRITICAL - must match training)
        sequence = self._normalize_sequence(sequence)

        return torch.from_numpy(sequence).float()

    def predict_batch(self, split: str = 'val') -> pd.DataFrame:
        """
        Generate predictions for all patterns in specified split

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with predictions and actual outcomes
        """
        # Filter to specified split
        split_data = self.metadata[self.metadata['split'] == split].copy()

        if len(split_data) == 0:
            raise ValueError(f"No patterns found for split '{split}'")

        print(f"\nGenerating predictions for {len(split_data)} {split} patterns...")

        predictions = []

        with torch.no_grad():
            for idx, row in split_data.iterrows():
                try:
                    # Load sequence (support both formats)
                    if self.use_bulk_sequences:
                        seq_id = row['_seq_idx']  # Use row index as sequence index
                    else:
                        seq_id = row['sequence_filename']
                    sequence = self._load_sequence(seq_id)
                    sequence = sequence.unsqueeze(0).to(self.device)  # Add batch dim

                    # Get model predictions
                    logits = self.model(sequence)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                    # Calculate EV
                    probs_dict = {i: probs[i] for i in range(len(probs))}
                    ev = calculate_expected_value(probs_dict)

                    predictions.append({
                        'pattern_id': row.get('pattern_id', idx),
                        'ticker': row.get('ticker', 'UNKNOWN'),
                        'start_date': row.get('start_date'),
                        'actual_label': row['label'],
                        'prob_danger': probs[0],
                        'prob_noise': probs[1],
                        'prob_target': probs[2],
                        'predicted_ev': ev,
                        'predicted_class': np.argmax(probs),
                        'confidence': np.max(probs)
                    })

                except Exception as e:
                    print(f"Error processing pattern {row.get('pattern_id', idx)}: {e}")
                    continue

        df_predictions = pd.DataFrame(predictions)

        # Add actual outcome value
        df_predictions['actual_value'] = df_predictions['actual_label'].map(STRATEGIC_VALUES)

        print(f"Generated {len(df_predictions)} predictions")
        return df_predictions

    def compute_precision_at_top_k(self,
                                    df: pd.DataFrame,
                                    k_percent: float = 15.0) -> Dict:
        """
        MOST IMPORTANT METRIC: Precision @ Top K%

        Of the patterns with highest predicted EV, what % actually hit target?
        This is the ONLY metric that matters for live trading.

        Args:
            df: DataFrame with predictions
            k_percent: Percentage of top predictions to evaluate (default 15%)

        Returns:
            Dictionary with precision metrics
        """
        # Sort by predicted EV (descending)
        df_sorted = df.sort_values('predicted_ev', ascending=False).copy()

        # Get top k%
        k_count = max(1, int(len(df_sorted) * k_percent / 100))
        top_k = df_sorted.head(k_count)

        # Count actual targets in top k
        target_hits = (top_k['actual_label'] == 2).sum()
        precision = target_hits / k_count if k_count > 0 else 0

        # Average EV of top k
        avg_predicted_ev = top_k['predicted_ev'].mean()
        avg_actual_value = top_k['actual_value'].mean()

        # Min EV in top k (this becomes your threshold)
        min_ev_in_top_k = top_k['predicted_ev'].min()

        # Breakdown by actual outcome
        outcome_counts = top_k['actual_label'].value_counts().to_dict()

        # Risk metrics
        danger_rate = outcome_counts.get(0, 0) / k_count
        noise_rate = outcome_counts.get(1, 0) / k_count
        target_rate = outcome_counts.get(2, 0) / k_count

        return {
            'k_percent': k_percent,
            'k_count': k_count,
            'target_hits': int(target_hits),
            'precision': precision,
            'target_rate': target_rate,
            'danger_rate': danger_rate,
            'noise_rate': noise_rate,
            'avg_predicted_ev': avg_predicted_ev,
            'avg_actual_value': avg_actual_value,
            'min_ev_in_top_k': min_ev_in_top_k,
            'calibration_error': abs(avg_predicted_ev - avg_actual_value),
            'outcome_breakdown': {
                'danger': outcome_counts.get(0, 0),
                'noise': outcome_counts.get(1, 0),
                'target': outcome_counts.get(2, 0)
            }
        }

    def analyze_ev_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find optimal EV threshold for live trading

        For each threshold, show:
        - How many signals you'd generate
        - Target hit rate (precision)
        - Avg actual value (profit)
        - Danger rate (risk)

        Args:
            df: DataFrame with predictions

        Returns:
            DataFrame with metrics per threshold
        """
        thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        results = []
        for threshold in thresholds:
            filtered = df[df['predicted_ev'] >= threshold]

            if len(filtered) == 0:
                continue

            target_rate = (filtered['actual_label'] == 2).sum() / len(filtered)
            danger_rate = (filtered['actual_label'] == 0).sum() / len(filtered)
            noise_rate = (filtered['actual_label'] == 1).sum() / len(filtered)
            avg_actual_value = filtered['actual_value'].mean()
            avg_predicted_ev = filtered['predicted_ev'].mean()

            # Expected profit per signal (simplified)
            expected_profit_per_signal = avg_actual_value

            results.append({
                'ev_threshold': threshold,
                'n_signals': len(filtered),
                'pct_of_total': len(filtered) / len(df) * 100,
                'target_rate': target_rate,
                'danger_rate': danger_rate,
                'noise_rate': noise_rate,
                'avg_predicted_ev': avg_predicted_ev,
                'avg_actual_value': avg_actual_value,
                'calibration_error': abs(avg_predicted_ev - avg_actual_value),
                'expected_profit_per_signal': expected_profit_per_signal
            })

        return pd.DataFrame(results)

    def compute_ev_calibration(self, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
        """
        EV calibration: are high-EV predictions actually more profitable?

        Groups predictions into bins by predicted EV and compares
        to actual average outcome values.

        Args:
            df: DataFrame with predictions
            n_bins: Number of bins for calibration

        Returns:
            DataFrame with calibration metrics per bin
        """
        # Create bins based on predicted EV
        df['ev_bin'] = pd.qcut(df['predicted_ev'], q=n_bins, duplicates='drop')

        calibration = df.groupby('ev_bin').agg({
            'predicted_ev': ['mean', 'std', 'count'],
            'actual_value': ['mean', 'std'],
            'actual_label': lambda x: (x == 2).sum()  # Target count
        }).round(3)

        calibration.columns = [
            'pred_ev_mean', 'pred_ev_std', 'count',
            'actual_value_mean', 'actual_value_std', 'target_count'
        ]

        # Calculate calibration error per bin
        calibration['calibration_error'] = (
            calibration['pred_ev_mean'] - calibration['actual_value_mean']
        ).abs()

        # Calculate target rate
        calibration['target_rate'] = calibration['target_count'] / calibration['count']

        return calibration.sort_index()

    def monitor_feature_drift(
        self,
        critical_features: Optional[List[str]] = None,
        save_report: bool = True
    ) -> Dict:
        """
        Monitor feature drift to detect micro-cap cycle shifts.

        CRITICAL FOR MICRO-CAPS: Vol_DryUp_Ratio drift detection
        - Micro-cap cycles shift fast (30-90 days): "AI Season" -> "Bio Season"
        - If PSI > 0.25 on Vol_DryUp_Ratio -> RETRAIN IMMEDIATELY on last 90 days

        Args:
            critical_features: List of features to monitor (defaults to drift-sensitive features)
            save_report: If True, save drift report to disk

        Returns:
            Dictionary with drift analysis and retrain recommendations
        """
        print(f"\n{'='*80}")
        print("DRIFT MONITORING: FEATURE DISTRIBUTION CHANGES")
        print(f"{'='*80}")
        print("Detecting micro-cap cycle shifts (e.g., 'AI Season' -> 'Bio Season')")

        # Use centralized critical features from config
        if critical_features is None:
            critical_features = DRIFT_CRITICAL_FEATURES
            print(f"Using DRIFT_CRITICAL_FEATURES from config: {critical_features}")

        # Load training and validation data
        print(f"\nLoading metadata for drift analysis...")

        # Check if splits exist, otherwise create them
        if 'split' in self.metadata.columns:
            train_data = self.metadata[self.metadata['split'] == 'train'].copy()
            val_data = self.metadata[self.metadata['split'] == 'val'].copy()
        else:
            print("  No 'split' column found - creating 70/30 train/val split...")
            from sklearn.model_selection import train_test_split
            train_data, val_data = train_test_split(
                self.metadata,
                test_size=0.3,
                random_state=42,
                stratify=self.metadata['label'] if 'label' in self.metadata.columns else None
            )
            train_data = train_data.copy()
            val_data = val_data.copy()

        if len(train_data) == 0 or len(val_data) == 0:
            print("ERROR: Need both train and val data for drift monitoring")
            return {'error': 'Insufficient data'}

        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")

        # Filter to critical features (if they exist in metadata)
        available_features = [f for f in critical_features if f in train_data.columns]

        # If critical features not in metadata, extract from sequences
        if not available_features and self.bulk_sequences is not None:
            print(f"\nExtracting critical features from sequences...")
            # Feature index mapping (from CLAUDE.md: 14 Temporal Features)
            FEATURE_INDICES = {
                'bbw': 5,              # bbw_20 at index 5
                'vol_dryup_ratio': 8,  # vol_dryup_ratio at index 8
            }

            # Extract features: take mean across all timesteps for each sequence
            for feat_name in critical_features:
                if feat_name in FEATURE_INDICES:
                    feat_idx = FEATURE_INDICES[feat_name]
                    # Mean across timesteps (axis=1) for the feature
                    train_indices = train_data.index.tolist()
                    val_indices = val_data.index.tolist()

                    # Extract from bulk sequences using indices
                    if hasattr(train_data, '_seq_idx'):
                        train_seq_idx = train_data['_seq_idx'].values
                        val_seq_idx = val_data['_seq_idx'].values
                    else:
                        train_seq_idx = train_indices
                        val_seq_idx = val_indices

                    # Ensure indices are within bounds
                    train_seq_idx = [i for i in train_seq_idx if i < len(self.bulk_sequences)]
                    val_seq_idx = [i for i in val_seq_idx if i < len(self.bulk_sequences)]

                    if train_seq_idx and val_seq_idx:
                        train_data[feat_name] = self.bulk_sequences[train_seq_idx, :, feat_idx].mean(axis=1)
                        val_data[feat_name] = self.bulk_sequences[val_seq_idx, :, feat_idx].mean(axis=1)
                        available_features.append(feat_name)
                        print(f"  Extracted {feat_name} from sequences (index {feat_idx})")

        if not available_features:
            print(f"\nWARNING: None of the critical features found in metadata or sequences")
            print(f"  Looking for: {critical_features}")
            print(f"  Available in metadata: {list(train_data.columns)}")
            return {'error': 'No critical features available'}

        print(f"\nMonitoring {len(available_features)} critical features:")
        for feat in available_features:
            print(f"  - {feat}")

        # Initialize drift monitor
        monitor = DriftMonitor(
            n_bins=10,
            psi_threshold=PSI_THRESHOLDS['medium'],  # 0.2 for warning
            ks_threshold=0.2,
            critical_feature_ratio=0.1
        )

        # Fit reference (training data)
        monitor.fit_reference(train_data[available_features])

        # Analyze drift (validation data)
        report = monitor.analyze_drift(val_data[available_features])

        # Print summary
        print(f"\n{report.summary()}")

        # ====================================================================
        # CRITICAL: Vol_DryUp_Ratio Drift Check
        # ====================================================================
        vol_dryup_result = None
        for result in report.feature_results:
            if result.feature_name == 'vol_dryup_ratio':
                vol_dryup_result = result
                break

        if vol_dryup_result:
            print(f"\n{'='*80}")
            print("[ALERT] VOL_DRYUP_RATIO DRIFT ANALYSIS (PRIMARY INDICATOR)")
            print(f"{'='*80}")
            print(f"PSI:            {vol_dryup_result.psi:.4f}")
            print(f"KS Statistic:   {vol_dryup_result.ks_statistic:.4f}")
            print(f"Mean Shift:     {vol_dryup_result.mean_shift:.4f} std")
            print(f"Alert Level:    {vol_dryup_result.alert_level.upper()}")

            if vol_dryup_result.psi > PSI_THRESHOLDS['high']:  # 0.25
                print(f"\n[ALERT] CRITICAL DRIFT DETECTED (PSI > {PSI_THRESHOLDS['high']})")
                print(f"   MICRO-CAP CYCLE SHIFT DETECTED!")
                print(f"   Example: 'AI Season' -> 'Bio Season' rotation")
                print(f"\n   RECOMMENDED ACTION:")
                print(f"   -> RETRAIN MODEL IMMEDIATELY on last 90 days")
                print(f"   -> Use only recent patterns (excludes stale cycle data)")
                print(f"   -> Command: python pipeline/02_train_temporal.py --recent-days 90")

            elif vol_dryup_result.psi > PSI_THRESHOLDS['medium']:  # 0.2
                print(f"\n[WARN] MODERATE DRIFT DETECTED (PSI > {PSI_THRESHOLDS['medium']})")
                print(f"   Micro-cap cycle may be shifting")
                print(f"   MONITOR CLOSELY - may need retrain soon")

            else:
                print(f"\n[OK] No significant drift (PSI < {PSI_THRESHOLDS['medium']})")
                print(f"  Vol_DryUp_Ratio distribution stable")

        else:
            print(f"\nWARNING: Vol_DryUp_Ratio not found in metadata")
            print(f"  Cannot monitor primary micro-cap cycle indicator")

        # ====================================================================
        # Retrain Recommendations
        # ====================================================================
        retrain_required = report.action_required or (
            vol_dryup_result and vol_dryup_result.psi > PSI_THRESHOLDS['high']
        )

        if retrain_required:
            print(f"\n{'='*80}")
            print("[ALERT] RETRAIN REQUIRED")
            print(f"{'='*80}")
            print(f"Reason: {'Critical Vol_DryUp_Ratio drift' if vol_dryup_result and vol_dryup_result.psi > PSI_THRESHOLDS['high'] else 'Multiple features drifted'}")
            print(f"\nRetrain Strategy:")
            print(f"  1. Filter to last 90 days of data (excludes stale cycle)")
            print(f"  2. Re-label patterns with current cycle characteristics")
            print(f"  3. Train fresh model (don't use old weights)")
            print(f"  4. Validate on most recent 20% of 90-day window")
            print(f"\nCommand:")
            print(f"  python pipeline/01_generate_sequences.py --recent-days 90 --force-relabel")
            print(f"  python pipeline/02_train_temporal.py --architecture hybrid --use-asl --epochs 100")

        else:
            print(f"\n[OK] No retrain required at this time")

        # Save drift report
        if save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = Path('output/drift_reports') / f'drift_report_{timestamp}.json'
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, cls=NumpyEncoder)

            print(f"\nDrift report saved to: {report_path}")

        # Return summary
        return {
            'overall_alert': report.overall_alert,
            'action_required': report.action_required,
            'retrain_required': retrain_required,
            'features_critical': report.features_critical,
            'features_warning': report.features_warning,
            'features_ok': report.features_ok,
            'vol_dryup_psi': vol_dryup_result.psi if vol_dryup_result else None,
            'vol_dryup_alert': vol_dryup_result.alert_level if vol_dryup_result else None,
            'feature_results': [r.to_dict() for r in report.feature_results],
            'report_path': str(report_path) if save_report else None
        }

    def generate_report(self, split: str = 'val', output_path: str = None) -> Dict:
        """
        Generate trading performance report focused on actionable metrics

        Args:
            split: Which data split to evaluate ('train', 'val', 'test')
            output_path: Optional path to save report JSON

        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*80}")
        print(f"LIVE TRADING PERFORMANCE EVALUATION - {split.upper()} SET")
        print(f"{'='*80}")
        print(f"Model: {self.model_path.name}")
        print(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFOCUS: Precision @ Top 15% and EV > 3.0 threshold")
        print(f"(Global accuracy is irrelevant - we only trade high-EV patterns)")

        # Generate predictions
        df_predictions = self.predict_batch(split)

        # Dataset overview
        print(f"\n{'='*80}")
        print("DATASET OVERVIEW")
        print(f"{'='*80}")
        print(f"Total patterns: {len(df_predictions)}")
        print(f"\nActual label distribution:")
        label_dist = df_predictions['actual_label'].value_counts().sort_index()
        for label, count in label_dist.items():
            label_name = ['Danger (K0)', 'Noise (K1)', 'Target (K2)'][label]
            pct = count / len(df_predictions) * 100
            print(f"  {label_name}: {count:4d} ({pct:5.1f}%)")

        # ========================================================================
        # PRIMARY METRIC: Precision @ Top 15%
        # ========================================================================
        print(f"\n{'='*80}")
        print("PRIMARY METRIC: PRECISION @ TOP 15%")
        print(f"{'='*80}")
        print("This is the ONLY metric that matters for live trading.")
        print("Of your highest-confidence signals, how many actually hit target?\n")

        top_k_results = {}
        for k in [5, 10, 15, 20, 25]:
            metrics = self.compute_precision_at_top_k(df_predictions, k_percent=k)
            top_k_results[f'top_{k}'] = metrics

            if k == 15:
                # Highlight Top 15%
                print(f"{'*'*80}")
                print(f"*** TOP {k}% RESULTS ***")
                print(f"{'*'*80}")
            else:
                print(f"\nTop {k}%:")

            print(f"  Patterns: {metrics['k_count']}")
            print(f"  Target hit rate: {metrics['target_rate']:.1%} <- KEY METRIC")
            print(f"  Danger rate: {metrics['danger_rate']:.1%} (risk)")
            print(f"  Noise rate: {metrics['noise_rate']:.1%}")
            print(f"  Avg predicted EV: {metrics['avg_predicted_ev']:.2f}")
            print(f"  Avg actual value: {metrics['avg_actual_value']:.2f}")
            print(f"  Calibration error: {metrics['calibration_error']:.2f}")
            print(f"  Min EV in top {k}%: {metrics['min_ev_in_top_k']:.2f}")

        # ========================================================================
        # EV THRESHOLD ANALYSIS: Find optimal cutoff
        # ========================================================================
        print(f"\n{'='*80}")
        print("EV THRESHOLD ANALYSIS: FINDING OPTIMAL CUTOFF")
        print(f"{'='*80}")
        print("For each EV threshold, show signal count and profitability.\n")

        threshold_analysis = self.analyze_ev_thresholds(df_predictions)

        # Format for display
        print(threshold_analysis.to_string(
            index=False,
            float_format=lambda x: f'{x:.2f}' if abs(x) < 100 else f'{x:.1f}',
            formatters={
                'target_rate': lambda x: f'{x:.1%}',
                'danger_rate': lambda x: f'{x:.1%}',
                'noise_rate': lambda x: f'{x:.1%}',
                'pct_of_total': lambda x: f'{x:.1f}%'
            }
        ))

        # ========================================================================
        # FOCUS: EV > 3.0 (GOOD SIGNAL THRESHOLD)
        # ========================================================================
        ev3_patterns = df_predictions[df_predictions['predicted_ev'] >= 3.0]
        if len(ev3_patterns) > 0:
            print(f"\n{'='*80}")
            print("*** FOCUS: EV > 3.0 (RECOMMENDED LIVE TRADING THRESHOLD) ***")
            print(f"{'='*80}")
            print(f"Patterns meeting threshold: {len(ev3_patterns):4d} ({len(ev3_patterns)/len(df_predictions)*100:5.1f}% of total)")
            print(f"Target hit rate:            {(ev3_patterns['actual_label'] == 2).sum() / len(ev3_patterns):5.1%} <- LIVE WIN RATE")
            print(f"Danger rate:                {(ev3_patterns['actual_label'] == 0).sum() / len(ev3_patterns):5.1%}")
            print(f"Noise rate:                 {(ev3_patterns['actual_label'] == 1).sum() / len(ev3_patterns):5.1%}")
            print(f"Avg actual value:           {ev3_patterns['actual_value'].mean():5.2f}")
            print(f"Avg predicted EV:           {ev3_patterns['predicted_ev'].mean():5.2f}")
            print(f"Calibration error:          {abs(ev3_patterns['predicted_ev'].mean() - ev3_patterns['actual_value'].mean()):5.2f}")

            # Show distribution of actual outcomes
            print(f"\nOutcome breakdown (EV > 3.0):")
            for label in [0, 1, 2]:
                label_name = ['Danger (K0)', 'Noise (K1)', 'Target (K2)'][label]
                count = (ev3_patterns['actual_label'] == label).sum()
                pct = count / len(ev3_patterns) * 100
                print(f"  {label_name}: {count:4d} ({pct:5.1f}%)")

        else:
            print(f"\n{'='*80}")
            print("WARNING: NO PATTERNS WITH EV > 3.0")
            print(f"{'='*80}")
            print("Model is not generating high-confidence signals.")
            print("Consider:")
            print("  1. Retraining with more data")
            print("  2. Adjusting loss function (e.g., increase ASL gamma)")
            print("  3. Using a lower EV threshold (e.g., 2.0)")

        # ========================================================================
        # EV CALIBRATION
        # ========================================================================
        print(f"\n{'='*80}")
        print("EV CALIBRATION: PREDICTED VS ACTUAL")
        print(f"{'='*80}")
        print("Do high-EV predictions actually deliver higher returns?\n")

        calibration = self.compute_ev_calibration(df_predictions, n_bins=10)
        print(calibration.to_string(float_format=lambda x: f'{x:.2f}'))

        # Overall calibration
        overall_pred_ev = df_predictions['predicted_ev'].mean()
        overall_actual_value = df_predictions['actual_value'].mean()
        overall_calibration_error = abs(overall_pred_ev - overall_actual_value)

        print(f"\nOverall calibration:")
        print(f"  Avg predicted EV:   {overall_pred_ev:6.2f}")
        print(f"  Avg actual value:   {overall_actual_value:6.2f}")
        print(f"  Calibration error:  {overall_calibration_error:6.2f}")

        # ========================================================================
        # GAP-ADJUSTED PNL ANALYSIS (REALISTIC TRADING COSTS)
        # ========================================================================
        print(f"\n{'='*80}")
        print("GAP-ADJUSTED PNL ANALYSIS (REALISTIC TRADING COSTS)")
        print(f"{'='*80}")
        print("Accounting for execution realities in micro-cap trading:\n")
        print("  1. GAP PENALTY: Entry at Open_T+1 vs signal at Close_T")
        print("     - Gap Up:   We paid MORE (reduce PnL)")
        print("     - Gap Down: Do NOT credit improvement (conservative)")
        print(f"\n  2. SPREAD TAX: {SPREAD_TAX_TOTAL_PCT:.1f}% round-trip")
        print(f"     - Entry:  {SPREAD_TAX_ENTRY_PCT:.1f}% (crossing ask)")
        print(f"     - Exit:   {SPREAD_TAX_EXIT_PCT:.1f}% (hitting bid)")

        # Calculate gap-adjusted metrics
        try:
            gap_adjusted = estimate_gap_adjusted_metrics(df_predictions, self.metadata)

            print(f"\n{'-'*60}")
            if gap_adjusted['method'] == 'actual_price_data':
                print("USING ACTUAL PRICE DATA FROM METADATA")
                print(f"{'-'*60}")
                gap_stats = gap_adjusted['gap_statistics']
                print(f"\nGap Statistics (from actual prices):")
                print(f"  Trades with gap up: {gap_stats['pct_trades_with_gap_up']:.1f}%")
                print(f"  Avg gap penalty:    {gap_stats['avg_gap_penalty_pct']:.2f}%")
                print(f"  Median gap penalty: {gap_stats['median_gap_penalty_pct']:.2f}%")
                print(f"  Max gap penalty:    {gap_stats['max_gap_penalty_pct']:.2f}%")
                print(f"  Avg gap up (when occurs): {gap_stats['avg_gap_up_when_occurs']:.2f}%")
            else:
                print("USING STATISTICAL ESTIMATE (no price data in metadata)")
                print(f"{'-'*60}")
                assumptions = gap_adjusted['assumptions']
                print(f"\nAssumptions:")
                print(f"  Gap up probability: {assumptions['gap_up_probability']*100:.0f}%")
                print(f"  Avg gap up:         {assumptions['avg_gap_up_pct']:.1f}%")
                print(f"  R-multiple base:    {assumptions['r_estimate']:.1f}%")

            # Cost breakdown
            costs = gap_adjusted['costs']
            print(f"\nCost Breakdown (per trade):")
            if 'expected_gap_cost_per_trade_pct' in costs:
                print(f"  Expected gap cost:  {costs['expected_gap_cost_per_trade_pct']:.2f}% ({costs['expected_gap_cost_in_r']:.2f}R)")
            print(f"  Spread tax:         {costs.get('spread_tax_total_pct', SPREAD_TAX_TOTAL_PCT):.2f}% ({costs.get('spread_cost_in_r', 0.33):.2f}R)")
            if 'total_cost_per_trade_in_r' in costs:
                print(f"  TOTAL COST:         {costs['total_cost_per_trade_in_r']:.2f}R per trade")

            # WIN RATE COMPARISON (THE KEY METRIC)
            win_rates = gap_adjusted['win_rates']
            print(f"\n{'*'*60}")
            print("WIN RATE COMPARISON (KEY METRIC)")
            print(f"{'*'*60}")
            print(f"  Theoretical Win Rate:   {win_rates['theoretical_win_rate']:.1%}")
            print(f"  Gap-Adjusted Win Rate:  {win_rates['gap_adjusted_win_rate']:.1%}")
            print(f"  Win Rate Reduction:     {win_rates['win_rate_reduction']:.1%}")

            # Impact analysis
            impact = gap_adjusted['impact']
            print(f"\nImpact Analysis:")
            print(f"  Trades flipped to loss: {impact['trades_flipped_to_loss']:.0f} ({impact['pct_trades_flipped']:.1f}%)")
            print(f"  Avg theoretical PnL:    {impact['avg_theoretical_pnl']:.2f}R")
            print(f"  Avg realized PnL:       {impact['avg_realized_pnl']:.2f}R")
            print(f"  PnL reduction:          {impact['avg_theoretical_pnl'] - impact['avg_realized_pnl']:.2f}R per trade")

        except Exception as e:
            print(f"\nWARNING: Could not calculate gap-adjusted metrics: {e}")
            gap_adjusted = None

        # ========================================================================
        # COMPILE REPORT
        # ========================================================================
        report = {
            'metadata': {
                'model_path': str(self.model_path),
                'split': split,
                'evaluation_time': datetime.now().isoformat(),
                'n_patterns': len(df_predictions)
            },
            'label_distribution': label_dist.to_dict(),
            'precision_at_top_k': top_k_results,
            'ev_threshold_analysis': threshold_analysis.to_dict('records'),
            'ev_calibration': {
                'bins': calibration.to_dict('records'),
                'overall_pred_ev': float(overall_pred_ev),
                'overall_actual_value': float(overall_actual_value),
                'calibration_error': float(overall_calibration_error)
            },
            'gap_adjusted_pnl': gap_adjusted  # Realistic trading costs analysis
        }

        # Add EV > 3.0 specific metrics
        if len(ev3_patterns) > 0:
            report['ev_3_threshold'] = {
                'n_patterns': len(ev3_patterns),
                'pct_of_total': len(ev3_patterns) / len(df_predictions) * 100,
                'target_rate': float((ev3_patterns['actual_label'] == 2).sum() / len(ev3_patterns)),
                'danger_rate': float((ev3_patterns['actual_label'] == 0).sum() / len(ev3_patterns)),
                'avg_actual_value': float(ev3_patterns['actual_value'].mean()),
                'avg_predicted_ev': float(ev3_patterns['predicted_ev'].mean())
            }

        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)

            print(f"\n{'='*80}")
            print(f"Report saved to: {output_path}")

            # Save predictions (drop interval column that can't be serialized)
            predictions_path = output_path.parent / f"{output_path.stem}_predictions.parquet"
            df_to_save = df_predictions.copy()
            if 'ev_bin' in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=['ev_bin'])
            df_to_save.to_parquet(predictions_path, index=False)
            print(f"Predictions saved to: {predictions_path}")

        # ========================================================================
        # KEY TAKEAWAYS
        # ========================================================================
        print(f"\n{'='*80}")
        print("KEY TAKEAWAYS FOR LIVE TRADING")
        print(f"{'='*80}")

        top15 = report['precision_at_top_k']['top_15']
        print(f"\n1. TOP 15% PRECISION (Primary Metric):")
        print(f"   -> Win rate: {top15['target_rate']:.1%}")
        print(f"   -> {top15['k_count']} patterns with avg EV {top15['avg_predicted_ev']:.2f}")
        print(f"   -> Actual avg value: {top15['avg_actual_value']:.2f}")
        print(f"   -> Risk: {top15['danger_rate']:.1%} danger rate")

        if 'ev_3_threshold' in report:
            ev3 = report['ev_3_threshold']
            print(f"\n2. EV > 3.0 PERFORMANCE (Recommended Threshold):")
            print(f"   -> Win rate: {ev3['target_rate']:.1%} <- THIS IS YOUR LIVE WIN RATE")
            print(f"   -> {ev3['n_patterns']} signals ({ev3['pct_of_total']:.1f}% of patterns)")
            print(f"   -> Avg profit per signal: {ev3['avg_actual_value']:.2f}")
            print(f"   -> Risk: {ev3['danger_rate']:.1%} danger rate")
        else:
            print(f"\n2. WARNING: No patterns with EV > 3.0 on {split} set")
            print(f"   -> Model may need retraining or threshold adjustment")

        calib = report['ev_calibration']
        print(f"\n3. CALIBRATION:")
        print(f"   -> Model predicts avg EV: {calib['overall_pred_ev']:.2f}")
        print(f"   -> Actual avg value: {calib['overall_actual_value']:.2f}")
        print(f"   -> Calibration error: {calib['calibration_error']:.2f}")
        if calib['calibration_error'] > 2.0:
            print(f"   -> WARNING: Large calibration error - model may be overconfident")

        # Gap-Adjusted Win Rate (THE TRUTH)
        if gap_adjusted:
            win_rates = gap_adjusted['win_rates']
            print(f"\n4. REALISTIC TRADING COSTS (Gap + Spread):")
            print(f"   *** THEORETICAL vs GAP-ADJUSTED ***")
            print(f"   -> Theoretical Win Rate:   {win_rates['theoretical_win_rate']:.1%}")
            print(f"   -> Gap-Adjusted Win Rate:  {win_rates['gap_adjusted_win_rate']:.1%} <- THE TRUTH")
            print(f"   -> Win Rate Reduction:     {win_rates['win_rate_reduction']:.1%}")
            if gap_adjusted['impact']['pct_trades_flipped'] > 5:
                print(f"   -> WARNING: {gap_adjusted['impact']['pct_trades_flipped']:.1f}% of winning trades become losers!")
        else:
            print(f"\n4. REALISTIC TRADING COSTS: Could not calculate")

        print(f"\n5. RECOMMENDED LIVE TRADING RULE:")
        print(f"   -> ONLY generate signal if Predicted EV > 3.0")
        print(f"   -> Expected signals per 100 patterns: ~{len(ev3_patterns)/len(df_predictions)*100:.0f}")
        if 'ev_3_threshold' in report:
            print(f"   -> Expected theoretical win rate: {ev3['target_rate']:.1%}")
        if gap_adjusted:
            # Estimate gap-adjusted win rate at EV > 3.0 threshold
            reduction = gap_adjusted['win_rates']['win_rate_reduction']
            if 'ev_3_threshold' in report:
                adj_win_rate = max(0, ev3['target_rate'] - reduction)
                print(f"   -> Expected GAP-ADJUSTED win rate: ~{adj_win_rate:.1%}")

        print(f"\n{'='*80}\n")

        return report


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model for live trading performance (focus on Precision @ Top 15% and EV > 3.0)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model checkpoint (default: latest best_model.pt)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        help='Path to metadata parquet (default: output/sequences/metadata_refined_latest.parquet)'
    )
    parser.add_argument(
        '--sequences-dir',
        type=str,
        default='output/sequences',
        help='Directory containing sequence .npy files'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate (default: val)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save evaluation report (JSON)'
    )
    parser.add_argument(
        '--drift-only',
        action='store_true',
        help='Run ONLY drift monitoring (skip full evaluation) - detects micro-cap cycle shifts'
    )
    parser.add_argument(
        '--train-cutoff',
        type=str,
        default=None,
        help='Train/val split date (e.g., 2024-01-01). If not set, uses 70/15/15 percentage split.'
    )
    parser.add_argument(
        '--val-cutoff',
        type=str,
        default=None,
        help='Val/test split date (e.g., 2024-07-01). If not set, uses 70/15/15 percentage split.'
    )

    args = parser.parse_args()

    # Set defaults
    if args.model is None:
        # Use most recent model
        models_dir = Path('output/models')
        model_files = sorted(models_dir.glob('best_model_*.pt'))
        if model_files:
            args.model = str(model_files[-1])
            print(f"Using most recent model: {args.model}")
        else:
            args.model = 'output/models/best_model.pt'

    if args.metadata is None:
        # Auto-detect metadata from model timestamp
        # Model: best_model_20260116_065211.pt -> look for metadata with similar date
        import re
        model_basename = os.path.basename(args.model)
        model_match = re.search(r'best_model_(\d{8})_\d{6}\.pt', model_basename)

        if model_match:
            model_date = model_match.group(1)  # e.g., '20260116'
            sequences_dir = Path('output/sequences')

            # Find metadata files matching the model date
            matching_metadata = sorted(sequences_dir.glob(f'metadata_{model_date}_*.parquet'))

            if matching_metadata:
                # Use the most recent metadata from that day
                args.metadata = str(matching_metadata[-1])
                print(f"AUTO-DETECTED metadata: {args.metadata}")
            else:
                # Fall back to most recent metadata overall
                all_metadata = sorted(sequences_dir.glob('metadata_2*.parquet'))
                if all_metadata:
                    args.metadata = str(all_metadata[-1])
                    print(f"AUTO-DETECTED metadata (most recent): {args.metadata}")
                else:
                    args.metadata = 'output/sequences/metadata_refined_latest.parquet'
        else:
            # Fall back to most recent metadata
            sequences_dir = Path('output/sequences')
            all_metadata = sorted(sequences_dir.glob('metadata_2*.parquet'))
            if all_metadata:
                args.metadata = str(all_metadata[-1])
                print(f"AUTO-DETECTED metadata (most recent): {args.metadata}")
            else:
                args.metadata = 'output/sequences/metadata_refined_latest.parquet'

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'output/evaluation/trading_performance_{timestamp}.json'

    # Verify paths exist
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable models:")
        models_dir = Path('output/models')
        if models_dir.exists():
            for model_file in sorted(models_dir.glob('best_model*.pt'))[-10:]:
                print(f"  {model_file}")
        sys.exit(1)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        print("\nAvailable metadata files:")
        sequences_dir = Path('output/sequences')
        if sequences_dir.exists():
            for meta_file in sorted(sequences_dir.glob('metadata*.parquet')):
                print(f"  {meta_file}")
        sys.exit(1)

    # Initialize evaluator
    evaluator = TradingPerformanceEvaluator(
        model_path=args.model,
        metadata_path=args.metadata,
        sequences_dir=args.sequences_dir,
        load_model=not args.drift_only,  # Skip model loading in drift-only mode
        train_cutoff=args.train_cutoff,
        val_cutoff=args.val_cutoff
    )

    # Run drift monitoring only OR full evaluation
    if args.drift_only:
        # DRIFT-ONLY MODE: Quick check for micro-cap cycle shifts
        print("\n" + "="*80)
        print("DRIFT-ONLY MODE: Monitoring for micro-cap cycle shifts")
        print("="*80)
        print("Focus: Vol_DryUp_Ratio drift detection")
        print("Use case: Daily/weekly checks for 'AI Season' -> 'Bio Season' shifts\n")

        drift_report = evaluator.monitor_feature_drift(save_report=True)

        # Print final recommendation
        print("\n" + "="*80)
        print("DRIFT MONITORING COMPLETE")
        print("="*80)

        if drift_report.get('retrain_required'):
            print("\n[ALERT] ACTION REQUIRED: Model retrain recommended")
            print("   Run full evaluation to see prediction impact:")
            print(f"   python pipeline/evaluate_trading_performance.py --split {args.split}")
        else:
            print("\n[OK] No drift detected - model stable")
            print("  Continue monitoring with weekly drift checks")

    else:
        # FULL EVALUATION MODE: Performance metrics + drift monitoring
        report = evaluator.generate_report(
            split=args.split,
            output_path=args.output
        )

        # Add drift monitoring to full evaluation
        print("\n" + "="*80)
        print("BONUS: DRIFT MONITORING CHECK")
        print("="*80)
        print("Checking for micro-cap cycle shifts...")

        try:
            drift_report = evaluator.monitor_feature_drift(save_report=True)

            # Add drift results to main report
            if args.output:
                report['drift_monitoring'] = drift_report

                # Re-save report with drift data
                output_path = Path(args.output)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, cls=NumpyEncoder)

                print(f"\nFull report (with drift analysis) saved to: {output_path}")

        except Exception as e:
            print(f"\nWARNING: Drift monitoring failed: {e}")
            print("Continuing with performance evaluation only...")


if __name__ == '__main__':
    main()
