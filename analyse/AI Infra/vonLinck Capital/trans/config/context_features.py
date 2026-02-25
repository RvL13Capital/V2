"""
Context Features for Branch B (Potential Energy Detector)
==========================================================

These static features capture the broader market context and "potential energy"
of a consolidation pattern, complementing the temporal "coil shape" features
from Branch A.

Branch B extracts potential energy indicators that temporal patterns alone cannot capture:
- Float turnover (accumulation activity)
- Trend position (market structure)
- Base duration (consolidation maturity)
- Relative volume (recent activity)
- Distance to high (proximity to breakout)
"""

from typing import Final, List, Callable
import numpy as np

# Context Features for Branch B (Potential Energy)
# These complement the 14 temporal features from Branch A
CONTEXT_FEATURES: Final[List[str]] = [
    # Original 7 features (market context) - relative_strength_spy REMOVED (irrelevant for uncorrelated sleepers)
    # NOTE: float_turnover REMOVED (Jan 2026) - redundant with relative_volume
    'retention_rate',         # (Close - Low) / (High - Low) - candle strength indicator
    'trend_position',         # Current_Price / 200_SMA
    'base_duration',          # Days_Since_20Pct_High / 200 (normalized)
    'relative_volume',        # Avg_Vol_20D / Avg_Vol_60D
    'distance_to_high',       # (52W_High - Price) / 52W_High
    'log_float',              # log10(shares_outstanding) - liquidity/size indicator
    'log_dollar_volume',      # log10(avg_daily_dollar_volume) - tradability indicator
    # Deep Dormancy features (Jan 2026) - Critical for sleeper detection
    'dormancy_shock',         # log10(vol_20d_avg / vol_252d_avg) - Is current activity highest in a year?
    'vol_dryup_ratio',        # vol_20d_avg / (vol_100d_avg + eps) - Degree of volume exhaustion
    # Coil features (Jan 2026) - BIAS-FREE pattern state at detection
    'price_position_at_end',  # Close position in box (0=lower, 1=upper) - LOW = better K2 rate
    'volume_shock',           # Max volume in last 3 days / 20d avg (breakout precursor)
    'bbw_slope_5d',           # BBW change over last 5 days (positive = expanding)
    'vol_trend_5d',           # Recent volume vs 20d average (>1 = increasing)
    'coil_intensity',         # Combined coil quality score (higher = better)
    # Cohort-based relative strength (Jan 2026) - Replaces SPY to avoid look-ahead bias
    'relative_strength_cohort',  # Ticker 20d return vs universe median (no time-zone leakage)
    # Structural Efficiency features (Jan 2026) - Pattern quality metrics
    'risk_width_pct',            # (Upper - Lower) / Upper - structural stop distance
    'vol_contraction_intensity', # Avg_Vol_5d / Avg_Vol_60d - volume drying up before breakout
    'obv_divergence',            # OBV slope vs Price slope - smart money accumulation
    # Liquidity Fragility (Jan 2026) - Volume decay detection
    'vol_decay_slope',           # Linear regression slope of log-volume over 20 days (negative = fragile)
    # Regime-Aware Features (Jan 2026) - Cross-regime generalization
    'vix_regime_level',          # VIX normalized (0=calm, 1=crisis) - volatility environment
    'vix_trend_20d',             # VIX 20d change (-0.5 to +0.5) - volatility direction
    'market_breadth_200',        # % stocks above 200 SMA (0-1) - market participation
    'risk_on_indicator',         # Risk-on/off composite (0=risk-off, 1=risk-on)
    'days_since_regime_change',  # Log-normalized days since bull/bear crossover
]

NUM_CONTEXT_FEATURES: Final[int] = len(CONTEXT_FEATURES)  # 24 (Jan 2026 - Regime-Aware)


# =============================================================================
# LOG-DIFF TRANSFORMATION (Jan 2026 - Dormant Stock Fix)
# =============================================================================
# Problem: Volume ratios explode to NaN/Inf on dormant stocks where denom ≈ 0
#   e.g., relative_volume = vol_20d / vol_60d → Inf when vol_60d = 0
#
# Solution: Use log-diff instead of raw ratio
#   log_diff(num, denom) = log1p(num) - log1p(denom)
#
# This is mathematically equivalent to log((1+num)/(1+denom)) but:
#   1. Handles zeros gracefully (log1p(0) = 0)
#   2. No division by zero possible
#   3. Linearizes magnitude differences (500x vs 5000x distinguishable)
#   4. Output range is naturally bounded (approximately -10 to +10 for typical volumes)
#
# Example:
#   vol_20d=0, vol_60d=100 → log1p(0) - log1p(100) = 0 - 4.6 = -4.6 (very dormant)
#   vol_20d=500, vol_60d=100 → log1p(500) - log1p(100) = 6.2 - 4.6 = 1.6 (awakening)
#   vol_20d=5000, vol_60d=100 → log1p(5000) - log1p(100) = 8.5 - 4.6 = 3.9 (volume shock)
# =============================================================================

def log_diff(numerator: float, denominator: float) -> float:
    """
    Safe volume ratio calculation using log-difference.

    Replaces raw ratio (num/denom) which explodes on dormant stocks.

    Args:
        numerator: Volume or volume average (e.g., vol_20d)
        denominator: Volume baseline (e.g., vol_60d)

    Returns:
        log1p(numerator) - log1p(denominator)
        Positive = numerator > denominator (volume increasing)
        Negative = numerator < denominator (volume decreasing)
        Zero = equal volumes

    Example:
        >>> log_diff(500, 100)  # 5x volume increase
        1.607...
        >>> log_diff(0, 100)    # Dormant (no recent volume)
        -4.615...
        >>> log_diff(100, 100)  # Stable
        0.0
    """
    return np.log1p(max(numerator, 0)) - np.log1p(max(denominator, 0))


def log_diff_batch(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    """
    Vectorized log-diff for batch processing.

    Args:
        numerators: Array of numerator values
        denominators: Array of denominator values

    Returns:
        Array of log-diff values
    """
    return np.log1p(np.maximum(numerators, 0)) - np.log1p(np.maximum(denominators, 0))


def vol_decay_slope(volumes: np.ndarray, window: int = 20) -> float:
    """
    Calculate volume decay rate via linear regression slope.

    This measures whether liquidity is declining (fragile) or increasing (healthy).
    Uses log-volume to linearize the relationship and handle zero volumes.

    Args:
        volumes: Array of volume values (at least `window` elements expected)
        window: Number of days to compute slope over (default: 20)

    Returns:
        Linear regression slope of log1p(volume)
        Positive = increasing liquidity (accumulation)
        Negative = declining liquidity (fragile, drying up)
        Zero = flat/stable volume

    Example:
        >>> volumes = np.array([100, 90, 80, 70, 60])  # Declining
        >>> vol_decay_slope(volumes, window=5)
        -0.053...  # Negative slope (liquidity declining)

        >>> volumes = np.array([50, 60, 70, 80, 100])  # Increasing
        >>> vol_decay_slope(volumes, window=5)
        0.065...  # Positive slope (liquidity increasing)

        >>> volumes = np.zeros(20)  # Dormant
        >>> vol_decay_slope(volumes, window=20)
        0.0  # Flat (dormant-safe)
    """
    if len(volumes) < window:
        return 0.0

    recent = volumes[-window:]

    # Convert to log scale (handles zeros gracefully)
    log_volumes = np.log1p(np.maximum(recent, 0))

    # Check for flat volume (all same or all zero)
    if np.std(log_volumes) < 1e-6:
        return 0.0

    # Linear regression: y = log_volumes, x = [0, 1, 2, ..., window-1]
    x = np.arange(window)
    try:
        slope = np.polyfit(x, log_volumes, 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return 0.0

    # Clip to reasonable range
    return float(np.clip(slope, -0.5, 0.5))


# Features that use LOG-DIFF transformation (volume ratios + OBV)
# These features are computed as log_diff(num, denom) instead of num/denom
# FIX (Jan 2026): Added obv_divergence - now uses log-diff for dormant stock stability
VOLUME_RATIO_FEATURES: Final[List[str]] = [
    'relative_volume',           # Index 3: log_diff(vol_20d, vol_60d)
    'dormancy_shock',            # Index 7: log_diff(vol_20d, vol_252d)
    'vol_dryup_ratio',           # Index 8: log_diff(vol_20d, vol_100d)
    'volume_shock',              # Index 10: log_diff(max_vol_3d, avg_vol_20d)
    'vol_trend_5d',              # Index 12: log_diff(avg_vol_5d, avg_vol_20d)
    'vol_contraction_intensity', # Index 16: log_diff(avg_vol_5d, avg_vol_60d)
    'obv_divergence',            # Index 17: sign(obv_delta)*log1p(|obv_delta|) - sign(price_delta)*log1p(|price_delta|)
]

# Map feature names to their indices for quick lookup
VOLUME_RATIO_INDICES: Final[List[int]] = [
    CONTEXT_FEATURES.index(f) for f in VOLUME_RATIO_FEATURES
]

# Feature Descriptions (for documentation and interpretability)
CONTEXT_FEATURE_DESCRIPTIONS: Final[dict] = {
    'retention_rate': {
        'formula': '(Close - Low) / (High - Low)',
        'interpretation': {
            '<0.25': 'Weak close (near low) - sellers dominated',
            '0.25-0.50': 'Below midpoint - mild selling pressure',
            '0.50-0.75': 'Above midpoint - mild buying pressure',
            '>0.75': 'Strong close (near high) - buyers dominated'
        },
        'optimal_range': (0.5, 1.0),
        'why_matters': 'High retention = buyers winning the day. Accumulation often shows up as high retention in tight ranges.'
    },
    'trend_position': {
        'formula': 'Current_Price / 200_SMA',
        'interpretation': {
            '<0.9': 'Downtrend (avoid)',
            '0.9-1.1': 'Neutral (depends on setup)',
            '>1.1': 'Uptrend (favorable)',
            '>1.2': 'Strong uptrend (best)'
        },
        'optimal_range': (1.1, 1.5),
        'why_matters': 'Breakouts above 200MA have higher success rate'
    },
    'base_duration': {
        'formula': 'log(1 + pattern_duration_days) / log(201)  # Log-normalized to [0,1]',
        'interpretation': {
            '<0.4': 'Short base (< 15 days)',
            '0.4-0.6': 'Good base (15-50 days)',
            '0.6-0.8': 'Excellent base (50-100 days)',
            '>0.8': 'Epic base (> 100 days)'
        },
        'optimal_range': (0.5, 0.85),
        'why_matters': 'Longer consolidations build more potential energy. Log scale prevents outliers from dominating.'
    },
    'relative_volume': {
        'formula': 'log_diff(Avg_Vol_20D, Avg_Vol_60D)  # log1p(20d) - log1p(60d)',
        'interpretation': {
            '<-0.5': 'Drying up (log ratio < 0.6x)',
            '-0.5-0.2': 'Normal activity',
            '0.2-0.7': 'Picking up (accumulation signal)',
            '>0.7': 'Surge (>2x, breakout imminent)'
        },
        'optimal_range': (0.0, 1.0),
        'why_matters': 'Volume increase in consolidation indicates accumulation. Log-diff handles dormant stocks gracefully.'
    },
    'distance_to_high': {
        'formula': '(52W_High - Current_Price) / 52W_High',
        'interpretation': {
            '<0.05': 'Near ATH (5% away) - high risk/reward',
            '0.05-0.15': 'Close (5-15% away) - favorable',
            '0.15-0.30': 'Moderate (15-30% away) - depends',
            '>0.30': 'Far from high (> 30% away) - caution'
        },
        'optimal_range': (0.05, 0.20),
        'why_matters': 'Closer to ATH = less overhead resistance'
    },
    'log_float': {
        'formula': 'log10(shares_outstanding)',
        'interpretation': {
            '<6.0': 'Nano-cap (<1M shares) - extreme illiquidity risk',
            '6.0-7.0': 'Micro-cap (1-10M shares) - target sweet spot',
            '7.0-8.0': 'Small-cap (10-100M shares) - good liquidity',
            '>8.0': 'Large float (>100M shares) - very liquid'
        },
        'optimal_range': (6.0, 8.0),
        'why_matters': 'Float size determines liquidity and move potential'
    },
    'log_dollar_volume': {
        'formula': 'log10(avg_daily_dollar_volume)',
        'interpretation': {
            '<5.0': 'Very illiquid (<$100k/day) - hard to trade',
            '5.0-6.0': 'Low liquidity ($100k-$1M/day) - manageable',
            '6.0-7.0': 'Medium liquidity ($1M-$10M/day) - good',
            '>7.0': 'High liquidity (>$10M/day) - excellent'
        },
        'optimal_range': (5.0, 7.0),
        'why_matters': 'Dollar volume determines position sizing capability'
    },
    # Deep Dormancy features (Jan 2026) - Critical for sleeper detection
    'dormancy_shock': {
        'formula': 'log_diff(vol_20d_avg, vol_252d_avg)  # log1p(20d) - log1p(252d)',
        'interpretation': {
            '<-2.0': 'Deeply dormant (current volume < 10% of yearly avg)',
            '-2.0--0.5': 'Below yearly average',
            '-0.5-0.5': 'At or above yearly average (awakening)',
            '>0.5': 'Volume surge (>1.5x yearly average) - WAKE UP signal'
        },
        'optimal_range': (-3.0, 3.0),
        'why_matters': 'Identifies if current activity is highest in a year - true sleeper detection. Log-diff is stable even when yearly avg = 0.'
    },
    'vol_dryup_ratio': {
        'formula': 'log_diff(vol_20d_avg, vol_100d_avg)  # log1p(20d) - log1p(100d)',
        'interpretation': {
            '<-1.0': 'Extreme volume dryup (supply exhaustion)',
            '-1.0--0.3': 'Volume contracting (coiling)',
            '-0.3-0.3': 'Normal volume',
            '>0.3': 'Volume expanding (accumulation or distribution)'
        },
        'optimal_range': (-2.0, 2.0),
        'why_matters': 'Volume exhaustion signals supply drying up before breakout. Log-diff eliminates epsilon hack.'
    },
    # Coil features (Jan 2026)
    'price_position_at_end': {
        'formula': '(close_at_end - lower_boundary) / (upper_boundary - lower_boundary)',
        'interpretation': {
            '<0.3': 'Coiled low (near support) - BEST for K2 (29.8% target rate)',
            '0.3-0.5': 'Mid-low position - good',
            '0.5-0.7': 'Mid-high position - neutral',
            '>0.7': 'Near resistance - likely NOISE (45.9% K1 rate)'
        },
        'optimal_range': (0.0, 0.4),
        'why_matters': 'Low position = coiled spring with energy, High position = drifting noise'
    },
    'volume_shock': {
        'formula': 'log_diff(max(volume[-3:]), avg_volume_20d)  # log1p(max_3d) - log1p(20d_avg)',
        'interpretation': {
            '<0': 'No recent volume spike (quiet coil)',
            '0-0.7': 'Mild volume increase (accumulation starting)',
            '0.7-1.25': 'Strong volume spike (2-3.5x, breakout precursor)',
            '>1.25': 'Extreme volume (>3.5x, institutional activity)'
        },
        'optimal_range': (0.0, 2.0),
        'why_matters': 'Volume spikes in last 3 days often precede breakouts - measures "wake up" signal. Log-diff handles zero-volume periods.'
    },
    'bbw_slope_5d': {
        'formula': '(bbw_at_end - bbw_5d_ago) / 5',
        'interpretation': {
            '<-0.005': 'Contracting fast (coiling tight)',
            '-0.005-0': 'Slightly contracting',
            '0-0.005': 'Slightly expanding',
            '>0.005': 'Expanding fast (breakout imminent or leak)'
        },
        'optimal_range': (-0.02, 0.02),
        'why_matters': 'BBW slope indicates whether volatility is expanding or contracting'
    },
    'vol_trend_5d': {
        'formula': 'log_diff(avg_vol_5d, avg_vol_20d)  # log1p(5d) - log1p(20d)',
        'interpretation': {
            '<-0.35': 'Volume drying up (coiling)',
            '-0.35-0': 'Below average',
            '0-0.25': 'Normal/slightly elevated',
            '>0.25': 'Volume surge (accumulation or distribution)'
        },
        'optimal_range': (-1.0, 1.0),
        'why_matters': 'Volume trend indicates institutional activity. Log-diff is stable for dormant stocks.'
    },
    'coil_intensity': {
        'formula': '(pos_score + bbw_score + vol_score) / 3, where scores favor low position, tight BBW, low volume',
        'interpretation': {
            '<0.4': 'Loose coil (low energy) - more likely NOISE',
            '0.4-0.5': 'Moderate coil',
            '0.5-0.6': 'Good coil',
            '>0.6': 'Tight coil (high energy) - better K2 rate'
        },
        'optimal_range': (0.0, 1.0),
        'why_matters': 'Combined score: high intensity = better target rate (26.2% vs 20.9%)'
    },
    # Cohort-based relative strength (Jan 2026) - Replaces SPY to avoid look-ahead bias
    'relative_strength_cohort': {
        'formula': '(Ticker_Close / Ticker_Close_20d) - Universe_Median_Return_20D',
        'interpretation': {
            '<-0.10': 'Significant underperformance vs cohort',
            '-0.10-0': 'Slight underperformance vs cohort',
            '0-0.10': 'Slight outperformance vs cohort',
            '>0.10': 'Significant outperformance vs cohort'
        },
        'optimal_range': (-0.20, 0.20),
        'why_matters': 'Compares ticker momentum to universe without time-zone leakage (EU stocks close 4.5h before US)'
    },
    # Structural Efficiency features (Jan 2026)
    'risk_width_pct': {
        'formula': '(upper_boundary - lower_boundary) / upper_boundary',
        'interpretation': {
            '<0.05': 'Very tight structure (5% risk) - excellent R:R',
            '0.05-0.15': 'Tight structure (5-15% risk) - good',
            '0.15-0.30': 'Normal structure (15-30% risk) - acceptable',
            '>0.30': 'Loose structure (>30% risk) - poor R:R, filtered at 40%'
        },
        'optimal_range': (0.02, 0.25),
        'why_matters': 'Structural stop distance determines R:R quality. Loose patterns (>40%) are untradeable.'
    },
    'vol_contraction_intensity': {
        'formula': 'log_diff(avg_vol_5d, avg_vol_60d)  # log1p(5d) - log1p(60d)',
        'interpretation': {
            '<-1.2': 'Extreme contraction (supply exhaustion) - coiling tight',
            '-1.2--0.5': 'Strong contraction (volume dying) - good setup',
            '-0.5-0': 'Mild contraction - moderate setup',
            '>0': 'Volume expanding - breakout may be starting or failed'
        },
        'optimal_range': (-2.0, 1.0),
        'why_matters': 'Volume contraction before breakout indicates supply exhaustion - spring is coiling. Log-diff handles near-zero volumes.'
    },
    'obv_divergence': {
        'formula': 'log_diff(norm_slope(OBV_20d), norm_slope(Price_20d))',
        'interpretation': {
            '<-0.2': 'Negative divergence (Price running without volume)',
            '-0.2-0.2': 'Neutral / Synced',
            '>0.2': 'Positive divergence (Smart money accumulation)'
        },
        'optimal_range': (0.0, 1.0),
        'why_matters': 'Log-diff of normalized slopes. Stabilizes dormant stocks where raw slopes explode. Detects hidden buying.'
    },
    # Liquidity Fragility (Jan 2026)
    'vol_decay_slope': {
        'formula': 'linregress(log1p(volume[-20:]), x=[0..19]).slope',
        'interpretation': {
            '<-0.1': 'Declining liquidity (fragile, drying up fast)',
            '-0.1-0': 'Slight decline (normal consolidation)',
            '0-0.1': 'Stable/slight increase (healthy interest)',
            '>0.1': 'Increasing liquidity (accumulation in progress)'
        },
        'optimal_range': (-0.3, 0.3),
        'why_matters': 'Volume decay rate via linear regression. Negative slope = liquidity drying up faster than normal. Dormant-safe (returns 0 when volume is flat).'
    },
    # =========================================================================
    # REGIME-AWARE FEATURES (Jan 2026) - Cross-Regime Generalization
    # =========================================================================
    'vix_regime_level': {
        'formula': 'clip((vix - 10) / 40, 0, 1)',
        'interpretation': {
            '<0.25': 'Low volatility regime (VIX < 15) - calm markets',
            '0.25-0.50': 'Normal volatility regime (VIX 15-20)',
            '0.50-0.75': 'Elevated volatility regime (VIX 20-30)',
            '>0.75': 'High volatility / Crisis regime (VIX > 30)'
        },
        'optimal_range': (0.0, 0.5),
        'why_matters': 'Breakouts behave differently in calm vs volatile markets. High VIX often means breakouts fail.'
    },
    'vix_trend_20d': {
        'formula': 'clip((vix_current / vix_20d_ago - 1), -0.5, 0.5)',
        'interpretation': {
            '<-0.2': 'Volatility collapsing (bullish for breakouts)',
            '-0.2-0.2': 'Stable volatility',
            '>0.2': 'Volatility expanding (bearish for breakouts)'
        },
        'optimal_range': (-0.3, 0.1),
        'why_matters': 'Rising VIX during consolidation may signal impending breakdown. Falling VIX = calming conditions.'
    },
    'market_breadth_200': {
        'formula': 'pct_stocks_above_sma200 / 100',
        'interpretation': {
            '<0.30': 'Extremely narrow market / bearish',
            '0.30-0.50': 'Weak breadth',
            '0.50-0.70': 'Healthy breadth',
            '>0.70': 'Strong breadth / potential overextension'
        },
        'optimal_range': (0.4, 0.7),
        'why_matters': 'Narrow markets have different breakout dynamics. Micro-cap breakouts more likely when breadth is improving.'
    },
    'risk_on_indicator': {
        'formula': '(spy-tlt) + (jnk-tlt) - (gld-spy), normalized to 0-1',
        'interpretation': {
            '<0.3': 'Strong risk-off (avoid breakout entries)',
            '0.3-0.7': 'Neutral',
            '>0.7': 'Strong risk-on (favorable for breakouts)'
        },
        'optimal_range': (0.4, 0.8),
        'why_matters': 'Risk appetite directly affects willingness to bid up micro-caps. Composite captures multiple risk dimensions.'
    },
    'days_since_regime_change': {
        'formula': 'log1p(days) / log1p(500)',
        'interpretation': {
            '<0.4': 'Early in regime (< 30 days) - uncertain',
            '0.4-0.6': 'Establishing (30-100 days)',
            '0.6-0.8': 'Mature regime (100-250 days)',
            '>0.8': 'Extended regime (> 250 days) - potential reversal'
        },
        'optimal_range': (0.4, 0.8),
        'why_matters': 'Breakouts in early regime phases face more uncertainty. Mature regimes provide stable backdrop.'
    },
}

# Default values for missing features (used when data unavailable)
# NOTE: Volume ratio features now use LOG-DIFF, so defaults are 0 (equal volumes)
CONTEXT_FEATURE_DEFAULTS: Final[dict] = {
    # Original 7 features (market context)
    'retention_rate': 0.5,          # Neutral (close at midpoint)
    'trend_position': 1.0,          # Neutral (at 200MA)
    'base_duration': 0.3,           # Default to 60-day base
    'relative_volume': 0.0,         # LOG-DIFF neutral (20d = 60d avg)
    'distance_to_high': 0.15,       # Moderate distance
    'log_float': 7.0,               # ~10M shares (small-cap default)
    'log_dollar_volume': 5.5,       # ~$300k/day (micro-cap default)
    # Deep Dormancy features (Jan 2026) - LOG-DIFF format
    'dormancy_shock': 0.0,          # LOG-DIFF neutral (20d = 252d avg)
    'vol_dryup_ratio': 0.0,         # LOG-DIFF neutral (20d = 100d avg)
    # Coil features (Jan 2026)
    'price_position_at_end': 0.5,   # Middle of box (neutral)
    'volume_shock': 0.0,            # LOG-DIFF neutral (max 3d = 20d avg)
    'bbw_slope_5d': 0.0,            # No change (neutral)
    'vol_trend_5d': 0.0,            # LOG-DIFF neutral (5d = 20d avg)
    'coil_intensity': 0.5,          # Moderate coil
    # Cohort-based relative strength (Jan 2026)
    'relative_strength_cohort': 0.0,  # Neutral (performing same as universe median)
    # Structural Efficiency features (Jan 2026) - LOG-DIFF for vol_contraction
    'risk_width_pct': 0.10,              # 10% structural stop (typical tight pattern)
    'vol_contraction_intensity': -0.3,   # LOG-DIFF: slight contraction (typical coil)
    'obv_divergence': 0.0,               # No divergence (neutral)
    # Liquidity Fragility (Jan 2026)
    'vol_decay_slope': 0.0,              # Flat volume trend (neutral)
    # Regime-Aware Features (Jan 2026)
    'vix_regime_level': 0.25,            # Low volatility (calm market default)
    'vix_trend_20d': 0.0,                # Stable VIX (no trend)
    'market_breadth_200': 0.5,           # Neutral breadth (50% above SMA200)
    'risk_on_indicator': 0.5,            # Neutral risk appetite
    'days_since_regime_change': 0.5,     # Mid-regime (100-150 days)
}

# Normalization ranges (for stable training)
# These prevent extreme values from dominating the network
# NOTE: Volume ratio features now use LOG-DIFF with different ranges
CONTEXT_FEATURE_RANGES: Final[dict] = {
    # Original 7 features (market context)
    'retention_rate': (0.0, 1.0),          # Already bounded [0, 1]
    'trend_position': (0.7, 1.5),          # 70% to 150% of 200MA
    'base_duration': (0.0, 1.0),           # Already normalized
    'relative_volume': (-2.0, 2.0),        # LOG-DIFF: -2 (10% ratio) to +2 (7x ratio)
    'distance_to_high': (0.0, 0.5),        # 0% to 50% from high
    'log_float': (5.0, 10.0),              # 100k to 10B shares (log10 scale)
    'log_dollar_volume': (4.0, 9.0),       # $10k to $1B/day (log10 scale)
    # Deep Dormancy features (Jan 2026) - LOG-DIFF format
    'dormancy_shock': (-5.0, 5.0),         # LOG-DIFF: extreme dormant to massive awakening
    'vol_dryup_ratio': (-3.0, 3.0),        # LOG-DIFF: extreme dryup to expansion
    # Coil features (Jan 2026) - LOG-DIFF for volume features
    'price_position_at_end': (0.0, 1.0),   # Already normalized (0=lower, 1=upper)
    'volume_shock': (-2.0, 3.0),           # LOG-DIFF: quiet to extreme spike
    'bbw_slope_5d': (-0.02, 0.02),         # Contracting to expanding
    'vol_trend_5d': (-2.0, 2.0),           # LOG-DIFF: drying up to surging
    'coil_intensity': (0.0, 1.0),          # Already normalized
    # Cohort-based relative strength (Jan 2026)
    'relative_strength_cohort': (-0.30, 0.30),  # -30% to +30% vs universe median
    # Structural Efficiency features (Jan 2026) - LOG-DIFF for vol_contraction
    'risk_width_pct': (0.02, 0.40),             # 2% to 40% (filtered at 40%)
    'vol_contraction_intensity': (-3.0, 2.0),   # LOG-DIFF: extreme contraction to expansion
    'obv_divergence': (-5.0, 5.0),              # LOG-DIFF divergence score (handles dormant stocks)
    # Liquidity Fragility (Jan 2026)
    'vol_decay_slope': (-0.5, 0.5),             # Declining to increasing liquidity
    # Regime-Aware Features (Jan 2026)
    'vix_regime_level': (0.0, 1.0),             # Already normalized [0, 1]
    'vix_trend_20d': (-0.5, 0.5),               # Already clipped to this range
    'market_breadth_200': (0.0, 1.0),           # Already normalized [0, 1]
    'risk_on_indicator': (0.0, 1.0),            # Already normalized [0, 1]
    'days_since_regime_change': (0.0, 1.0),     # Log-normalized [0, 1]
}


def normalize_context_feature(feature_name: str, value: float) -> float:
    """
    Normalize a context feature to a stable range for neural network input.

    Args:
        feature_name: Name of the feature (must be in CONTEXT_FEATURES)
        value: Raw feature value

    Returns:
        Normalized value clipped to expected range

    Example:
        >>> normalize_context_feature('float_turnover', 3.5)
        0.7  # 3.5 / 5.0 = 0.7
    """
    if feature_name not in CONTEXT_FEATURE_RANGES:
        return value

    min_val, max_val = CONTEXT_FEATURE_RANGES[feature_name]

    # Clip to range
    value = max(min_val, min(max_val, value))

    # Normalize to [0, 1]
    if max_val > min_val:
        return (value - min_val) / (max_val - min_val)
    else:
        return value


def validate_context_features(features: dict) -> dict:
    """
    Validate and fill missing context features with defaults.

    Args:
        features: Dictionary of context features

    Returns:
        Complete dictionary with all required features

    Example:
        >>> validate_context_features({'float_turnover': 2.0})
        {
            'float_turnover': 2.0,
            'trend_position': 1.0,  # filled with default
            'base_duration': 0.3,   # filled with default
            ...
        }
    """
    validated = {}

    for feature_name in CONTEXT_FEATURES:
        if feature_name in features and features[feature_name] is not None:
            validated[feature_name] = features[feature_name]
        else:
            # Use default value
            validated[feature_name] = CONTEXT_FEATURE_DEFAULTS[feature_name]

    return validated
