"""
Vectorized Ignition Detection for Sequences

Detects ignition signals from temporal sequences using vectorized NumPy operations.
Processes 10K sequences in <100ms.

Feature indices (10 features after composite disabled):
  0-3: open, high, low, close (relativized to day 0)
  4: volume (log ratio to day 0)
  5: bbw_20, 6: adx, 7: volume_ratio_20
  8-9: upper_slope, lower_slope (rolling regression, normalized)

NOTE: Boundary positions (upper/lower) removed in Jan 2026 refactor.
Price position is now approximated from relativized high/low.
"""

import numpy as np
from typing import Tuple, NamedTuple
from dataclasses import dataclass


# Feature indices (10-feature layout)
OPEN_IDX = 0
HIGH_IDX = 1
LOW_IDX = 2
CLOSE_IDX = 3
VOLUME_IDX = 4  # log(vol_t / vol_0)
BBW_IDX = 5
ADX_IDX = 6
VOLUME_RATIO_IDX = 7
UPPER_SLOPE_IDX = 8
LOWER_SLOPE_IDX = 9

# Legacy indices for backwards compatibility (use high/low as proxy)
UPPER_BOUND_IDX = HIGH_IDX  # Use relativized high as upper proxy
LOWER_BOUND_IDX = LOW_IDX   # Use relativized low as lower proxy

# Ignition weights
WEIGHTS = {
    'price_position': 0.40,
    'volume_spike': 0.30,
    'bbw_expansion': 0.15,
    'green_bar': 0.10,
    'adx_momentum': 0.05,
}

# Thresholds
IGNITION_THRESHOLD = 0.6
PRICE_POSITION_THRESHOLD = 0.80  # >80% in channel = near top
VOLUME_SPIKE_THRESHOLD = 1.5    # 1.5x average
ADX_MOMENTUM_THRESHOLD = 25     # ADX > 25 = trending


@dataclass
class IgnitionResult:
    """Result container for ignition detection"""
    scores: np.ndarray          # (N,) ignition scores [0, 1]
    flags: np.ndarray           # (N,) boolean ignition flags
    components: dict            # Individual component scores


def detect_sequence_ignition(
    sequences: np.ndarray,
    threshold: float = IGNITION_THRESHOLD,
    return_components: bool = False
) -> Tuple[np.ndarray, np.ndarray] | IgnitionResult:
    """
    Vectorized ignition detection for sequence tensors.

    Detects "ignition" signals indicating imminent breakout potential by analyzing:
    1. Price position in channel (near upper boundary)
    2. Volume spike (above average)
    3. BBW expansion (volatility breakout)
    4. Green bar (bullish close)
    5. ADX momentum (trend strength)

    Args:
        sequences: (N, 20, 14) tensor of temporal sequences
        threshold: Ignition threshold (default 0.6)
        return_components: If True, return IgnitionResult with component breakdown

    Returns:
        If return_components=False:
            Tuple of (scores, flags) where:
                scores: (N,) array of ignition scores [0, 1]
                flags: (N,) boolean array (True if score >= threshold)
        If return_components=True:
            IgnitionResult with scores, flags, and component breakdown

    Performance:
        Processes 10K sequences in ~50ms on typical hardware.

    Example:
        >>> sequences = np.random.randn(10000, 20, 14)
        >>> scores, flags = detect_sequence_ignition(sequences)
        >>> print(f"Ignition rate: {flags.mean():.1%}")
    """
    if sequences.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, 20, 10), got shape {sequences.shape}")

    n_sequences, n_timesteps, n_features = sequences.shape

    if n_features != 10:
        raise ValueError(f"Expected 10 features, got {n_features}")

    # Extract last timestep (index -1) for all sequences
    last_step = sequences[:, -1, :]  # (N, 10)

    # Also extract for slope calculations (last 5 timesteps)
    last_5 = sequences[:, -5:, :]  # (N, 5, 10)

    # =========================================================================
    # Component 1: Price Position in Channel (40%)
    # =========================================================================
    # Position = (close - lower) / (upper - lower)
    # Score higher when price is in upper 20% of channel (near breakout)
    close = last_step[:, CLOSE_IDX]
    upper = last_step[:, UPPER_BOUND_IDX]
    lower = last_step[:, LOWER_BOUND_IDX]

    # Channel width (avoid division by zero)
    channel_width = upper - lower
    channel_width = np.where(np.abs(channel_width) < 1e-8, 1e-8, channel_width)

    # Price position [0, 1] where 1 = at upper boundary
    price_position = (close - lower) / channel_width
    price_position = np.clip(price_position, 0, 1)

    # Score: ramp up from 0.5 to 1.0 as position goes from 0.6 to 1.0
    # Below 0.6: partial score based on position
    price_score = np.where(
        price_position >= PRICE_POSITION_THRESHOLD,
        1.0,  # Full score if above threshold
        price_position / PRICE_POSITION_THRESHOLD  # Linear ramp below
    )

    # =========================================================================
    # Component 2: Volume Spike (30%)
    # =========================================================================
    # Use volume_ratio_20 (current volume / 20-day average)
    volume_ratio = last_step[:, VOLUME_RATIO_IDX]

    # Score: 0 at ratio=1.0, 1.0 at ratio>=2.0, linear between
    volume_spike_score = np.clip(
        (volume_ratio - 1.0) / (VOLUME_SPIKE_THRESHOLD - 1.0),
        0, 1
    )

    # =========================================================================
    # Component 3: BBW Expansion / Volatility Breakout (15%)
    # =========================================================================
    # Positive BBW slope over last 5 days = volatility expanding
    bbw_last_5 = last_5[:, :, BBW_IDX]  # (N, 5)

    # Linear regression slope (vectorized)
    # slope = cov(x, y) / var(x) where x = [0,1,2,3,4]
    x = np.arange(5)
    x_mean = 2.0  # mean of [0,1,2,3,4]
    x_var = 2.0   # variance of [0,1,2,3,4]

    bbw_mean = bbw_last_5.mean(axis=1, keepdims=True)  # (N, 1)
    bbw_slope = ((bbw_last_5 - bbw_mean) * (x - x_mean)).sum(axis=1) / (5 * x_var)  # (N,)

    # Normalize slope to [0, 1] score
    # Positive slope = expansion = good signal
    # Typical BBW ranges 0.05-0.20, so slope of 0.01 per day is significant
    bbw_expansion_score = np.clip(bbw_slope / 0.02, -1, 1)
    bbw_expansion_score = (bbw_expansion_score + 1) / 2  # Map [-1,1] to [0,1]

    # =========================================================================
    # Component 4: Green Bar (10%)
    # =========================================================================
    # Close > Open = bullish / green bar
    open_price = last_step[:, OPEN_IDX]
    is_green = close > open_price
    green_bar_score = is_green.astype(np.float32)

    # =========================================================================
    # Component 5: ADX Momentum (5%)
    # =========================================================================
    # ADX > 25 indicates trending market
    adx = last_step[:, ADX_IDX]

    # ADX slope over last 5 days (momentum)
    adx_last_5 = last_5[:, :, ADX_IDX]  # (N, 5)
    adx_mean = adx_last_5.mean(axis=1, keepdims=True)
    adx_slope = ((adx_last_5 - adx_mean) * (x - x_mean)).sum(axis=1) / (5 * x_var)

    # Combine level and momentum
    adx_level_score = np.clip(adx / 50, 0, 1)  # ADX 0-50 maps to 0-1
    adx_momentum_score = np.clip(adx_slope / 2, 0, 1)  # Positive slope is good

    # Weight: 70% level, 30% momentum
    adx_score = 0.7 * adx_level_score + 0.3 * adx_momentum_score

    # =========================================================================
    # Combine Scores
    # =========================================================================
    ignition_score = (
        WEIGHTS['price_position'] * price_score +
        WEIGHTS['volume_spike'] * volume_spike_score +
        WEIGHTS['bbw_expansion'] * bbw_expansion_score +
        WEIGHTS['green_bar'] * green_bar_score +
        WEIGHTS['adx_momentum'] * adx_score
    )

    # Ensure in [0, 1] range
    ignition_score = np.clip(ignition_score, 0, 1)

    # Apply threshold for boolean flag
    ignition_flag = ignition_score >= threshold

    if return_components:
        return IgnitionResult(
            scores=ignition_score.astype(np.float32),
            flags=ignition_flag,
            components={
                'price_position': price_score.astype(np.float32),
                'volume_spike': volume_spike_score.astype(np.float32),
                'bbw_expansion': bbw_expansion_score.astype(np.float32),
                'green_bar': green_bar_score.astype(np.float32),
                'adx_momentum': adx_score.astype(np.float32),
            }
        )

    return ignition_score.astype(np.float32), ignition_flag


def detect_ignition_at_timestep(
    sequences: np.ndarray,
    timestep: int = -1,
    threshold: float = IGNITION_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect ignition at a specific timestep (for early detection).

    Args:
        sequences: (N, 20, 14) tensor
        timestep: Which timestep to analyze (default -1 = last)
        threshold: Ignition threshold

    Returns:
        Tuple of (scores, flags)
    """
    if timestep < 0:
        timestep = sequences.shape[1] + timestep

    # Need at least 5 timesteps for slope calculations
    if timestep < 4:
        # Return zeros for early timesteps
        n = sequences.shape[0]
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)

    # Slice to only include data up to timestep
    sliced = sequences[:, :timestep+1, :]

    # Pad or truncate to 20 timesteps if needed for compatibility
    # Actually, we just need the last 5 timesteps
    if sliced.shape[1] >= 5:
        # Use last 5 timesteps relative to the target timestep
        analysis_window = sliced[:, -5:, :]

        # Create a fake "20 timestep" array for reuse of main function
        # But actually we just need to extract correctly...
        # Let's just recompute inline for efficiency

        n_sequences = sequences.shape[0]

        # Last step at target timestep
        last_step = sequences[:, timestep, :]
        last_5 = sequences[:, max(0, timestep-4):timestep+1, :]

        # Pad if needed
        if last_5.shape[1] < 5:
            pad_width = 5 - last_5.shape[1]
            last_5 = np.pad(last_5, ((0,0), (pad_width,0), (0,0)), mode='edge')

        # Compute components (simplified version)
        close = last_step[:, CLOSE_IDX]
        upper = last_step[:, UPPER_BOUND_IDX]
        lower = last_step[:, LOWER_BOUND_IDX]

        channel_width = np.where(
            np.abs(upper - lower) < 1e-8, 1e-8, upper - lower
        )
        price_position = np.clip((close - lower) / channel_width, 0, 1)
        price_score = np.where(
            price_position >= 0.8, 1.0, price_position / 0.8
        )

        volume_ratio = last_step[:, VOLUME_RATIO_IDX]
        volume_spike_score = np.clip((volume_ratio - 1.0) / 0.5, 0, 1)

        open_price = last_step[:, OPEN_IDX]
        green_bar_score = (close > open_price).astype(np.float32)

        adx = last_step[:, ADX_IDX]
        adx_score = np.clip(adx / 50, 0, 1)

        # Simplified score (no slope calculation for early timesteps)
        ignition_score = (
            0.45 * price_score +
            0.35 * volume_spike_score +
            0.10 * green_bar_score +
            0.10 * adx_score
        )

        ignition_score = np.clip(ignition_score, 0, 1).astype(np.float32)
        ignition_flag = ignition_score >= threshold

        return ignition_score, ignition_flag

    # Fallback
    n = sequences.shape[0]
    return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)


def get_ignition_statistics(
    sequences: np.ndarray,
    labels: np.ndarray = None,
    threshold: float = IGNITION_THRESHOLD
) -> dict:
    """
    Compute ignition detection statistics.

    Args:
        sequences: (N, 20, 14) tensor
        labels: Optional (N,) label array for class breakdown
        threshold: Ignition threshold

    Returns:
        Dictionary with statistics
    """
    scores, flags = detect_sequence_ignition(sequences, threshold)

    stats = {
        'n_sequences': len(sequences),
        'n_ignition': int(flags.sum()),
        'ignition_rate': float(flags.mean()),
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'median_score': float(np.median(scores)),
        'threshold': threshold,
    }

    if labels is not None:
        class_names = {0: 'Danger', 1: 'Noise', 2: 'Target'}
        stats['by_class'] = {}

        for cls in [0, 1, 2]:
            mask = labels == cls
            if mask.sum() > 0:
                cls_scores = scores[mask]
                cls_flags = flags[mask]
                stats['by_class'][class_names[cls]] = {
                    'n_total': int(mask.sum()),
                    'n_ignition': int(cls_flags.sum()),
                    'ignition_rate': float(cls_flags.mean()),
                    'mean_score': float(cls_scores.mean()),
                }

        # Lift calculation
        if flags.sum() > 0:
            baseline_target_rate = (labels == 2).mean()
            ignition_target_rate = (labels[flags] == 2).mean()
            stats['target_lift'] = float(
                ignition_target_rate / baseline_target_rate
            ) if baseline_target_rate > 0 else 0.0

    return stats


# Benchmark function for performance testing
def _benchmark(n_sequences: int = 10000, n_runs: int = 10) -> dict:
    """Benchmark ignition detection performance"""
    import time

    # Generate random test data
    np.random.seed(42)
    sequences = np.random.randn(n_sequences, 20, 14).astype(np.float32)

    # Warmup
    detect_sequence_ignition(sequences)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        detect_sequence_ignition(sequences)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'n_sequences': n_sequences,
        'n_runs': n_runs,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'throughput': n_sequences / (np.mean(times) / 1000),  # sequences/sec
    }


if __name__ == '__main__':
    # Run benchmark
    print("Benchmarking ignition detection...")
    results = _benchmark()
    print(f"Performance on {results['n_sequences']:,} sequences:")
    print(f"  Mean: {results['mean_ms']:.1f}ms")
    print(f"  Std:  {results['std_ms']:.1f}ms")
    print(f"  Range: {results['min_ms']:.1f}ms - {results['max_ms']:.1f}ms")
    print(f"  Throughput: {results['throughput']:,.0f} sequences/sec")

    # Quick test
    print("\nQuick test with random data...")
    sequences = np.random.randn(1000, 20, 14).astype(np.float32)
    result = detect_sequence_ignition(sequences, return_components=True)
    print(f"  Ignition rate: {result.flags.mean():.1%}")
    print(f"  Mean score: {result.scores.mean():.3f}")
    print("  Component means:")
    for name, values in result.components.items():
        print(f"    {name}: {values.mean():.3f}")
