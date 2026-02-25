"""
Consolidation Pattern Scanner V17 - Microstructure-Aware Edition
====================================================================

Integrates sleeper_scanner_v17 microstructure logic with TRANS temporal
tracking architecture for complete pattern detection and outcome analysis.

Key Features:
- Microstructure-aware pattern detection (thin liquidity handling)
- Optional accumulation detection (configurable via ENABLE_ACCUMULATION_DETECTION)
- Walk-forward validation for outcome class calculation
- Parallel processing for large universes
- Backward compatible with TRANS pipeline API

Usage:
    scanner = ConsolidationPatternScanner()
    results = scanner.scan_universe(tickers, start_date, end_date)
"""

import os
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import time
import sys

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .aiv7_components import DataLoader, PatternPhase
from .sleeper_scanner_v17 import find_sleepers_v17, calculate_outcome_class
from .path_dependent_labeler import PathDependentLabelerV17
from .exceptions import (
    DataIntegrityError,
    TemporalConsistencyError,
    ValidationError
)
from config import (
    MIN_DATA_LENGTH,
    MIN_PATTERN_DURATION,
    MAX_TEMPORAL_GAP_DAYS,
    ENABLE_ACCUMULATION_DETECTION,
    INDICATOR_WARMUP_DAYS,
    INDICATOR_STABLE_DAYS,
    OUTCOME_WINDOW_DAYS,
    ENABLE_SHARE_DILUTION_FILTER,
    MAX_DILUTION_PCT,
    EU_SUFFIXES
)
from utils.market_cap_fetcher import MarketCapFetcher
from utils.dilution_manager import DilutionDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternScanner")

# =============================================================================
# WORKER POOL OPTIMIZATION
# =============================================================================
# Module-level scanner instance for worker processes (avoids re-creation per task)
_worker_scanner: Optional['ConsolidationPatternScanner'] = None
_worker_config: Dict = {}


def _init_worker(config: Dict):
    """
    Initialize worker process with a reusable scanner instance.

    Called once per worker process by ProcessPoolExecutor initializer.
    This avoids the overhead of creating DataLoader, MarketCapFetcher,
    and DilutionDB for every single ticker.
    """
    global _worker_scanner, _worker_config
    _worker_config = config

    # Import here to avoid circular imports
    from core.pattern_scanner import ConsolidationPatternScanner

    _worker_scanner = ConsolidationPatternScanner(
        bbw_percentile_threshold=config.get('bbw_percentile_threshold', 0.30),
        adx_threshold=config.get('adx_threshold', 32.0),
        volume_ratio_threshold=config.get('volume_ratio_threshold', 0.35),
        range_ratio_threshold=config.get('range_ratio_threshold', 0.65),
        qualifying_days=config.get('qualifying_days', 10),
        min_data_days=config.get('min_data_days', 100),
        indicator_lookback=config.get('indicator_lookback', 100),
        min_liquidity_dollar=config.get('min_liquidity_dollar', 50000),
        enable_market_cap=config.get('enable_market_cap', True),
        candidate_only=config.get('candidate_only', False),
        disable_gcs=True,  # Always disable GCS in workers for multiprocessing safety
        fast_validation=config.get('fast_validation', False),
        # Adaptive thresholds (Jan 2026)
        tightness_zscore=config.get('tightness_zscore'),
        min_float_turnover=config.get('min_float_turnover'),
        # Weekly qualification mode (Jan 2026)
        use_weekly_qualification=config.get('use_weekly_qualification', False),
        # Point-in-time market cap (Jan 2026)
        skip_market_cap_api=config.get('skip_market_cap_api', False)
    )
    logger.debug(f"Worker initialized with scanner (PID: {os.getpid()})")


def _worker_scan_ticker(args: Tuple[str, Optional[str], Optional[str]]) -> 'ScanResult':
    """
    Worker function that uses the pre-initialized scanner.

    Args:
        args: Tuple of (ticker, start_date, end_date)

    Returns:
        ScanResult for the ticker
    """
    global _worker_scanner

    ticker, start_date, end_date = args

    if _worker_scanner is None:
        # Fallback: create scanner if not initialized (shouldn't happen)
        logger.warning(f"Worker scanner not initialized, creating new instance for {ticker}")
        _init_worker(_worker_config)

    return _worker_scanner.scan_ticker(ticker, start_date, end_date)


def is_us_ticker(ticker: str) -> bool:
    """
    Check if ticker is a US stock (no EU suffix).

    US tickers have no suffix (e.g., 'AAPL', 'MSFT')
    EU tickers have suffix (e.g., 'SAP.DE', 'HSBA.L')

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if US ticker, False if EU or invalid
    """
    if not ticker:
        return False
    return not any(ticker.upper().endswith(suffix) for suffix in EU_SUFFIXES)


@dataclass
class ScanResult:
    """Structured return type for pattern scanning."""
    ticker: str
    patterns_found: int
    patterns: List[Dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    data_points_processed: int = 0
    features: Dict[str, Union[float, str]] = field(default_factory=dict)
    error: Optional[str] = None
    market_cap: Optional[float] = None  # Market cap in dollars (None if unavailable)
    market_cap_category: Optional[str] = None  # mega_cap, large_cap, mid_cap, small_cap, micro_cap, nano_cap

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class UniverseScanResult:
    """Aggregated results for universe scan."""
    total_tickers: int
    successful_tickers: int
    failed_tickers: int
    total_patterns: int
    total_time_seconds: float
    patterns_per_ticker: float
    ticker_results: List[ScanResult] = field(default_factory=list)
    all_patterns: List[Dict] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


class VectorizedPatternMixin:
    """
    Optimization Mixin: Reduces scanning loop iterations by ~95%
    using O(1) vectorization to identify 'Candidate' windows.

    Memory Peak: ~2.4 MB per 20 years of data.

    This mixin pre-filters windows using vectorized percentile calculations,
    so only windows that meet PHYSICAL tightness requirements are processed
    by the heavier V17 detection logic.
    """

    def _get_candidate_indices_vectorized(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        is_weekly: bool = False
    ) -> np.ndarray:
        """
        Fast-filter: Identifies windows that meet PHYSICAL tightness requirements.

        Uses numpy sliding_window_view for O(1) memory complexity and
        vectorized percentile calculations for massive speedup.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            window_size: Lookback window size (default 60 days/weeks)
            is_weekly: If True, uses relaxed thresholds for weekly data

        Returns:
            Array of START indices for valid candidate windows
        """
        if len(df) < window_size:
            return np.array([], dtype=np.int64)

        # 1. Prepare Data (Zero-Copy Views)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Create Sliding Views
        # Shape: (N_rows - window + 1, window_size)
        v_high = sliding_window_view(highs, window_size)
        v_low = sliding_window_view(lows, window_size)
        v_close = sliding_window_view(closes, window_size)

        # 2. Vectorized Percentile Calculation (The Speedup)
        # Calculates percentiles across axis 1 (the window) for the entire history at once

        # Wick Tightness (Loose) - uses high/low extremes
        wick_upper = np.percentile(v_high, 95, axis=1)
        wick_lower = np.percentile(v_low, 10, axis=1)

        # Body Tightness (Strict) - uses close prices only
        body_upper = np.percentile(v_close, 95, axis=1)
        body_lower = np.percentile(v_close, 10, axis=1)

        # 3. Calculate Widths
        # Add epsilon to avoid division by zero
        wick_height_pct = (wick_upper - wick_lower) / (wick_lower + 1e-9)
        body_height_pct = (body_upper - body_lower) / (body_lower + 1e-9)

        # 4. Apply Widest Possible Filter
        # Weekly mode uses relaxed thresholds because weekly candles
        # aggregate daily ranges, resulting in naturally wider bodies.
        if is_weekly:
            # Weekly mode: Wick < 90%, Body < 40%
            # Pre-filter is intentionally loose; fine-grained checks happen in find_sleepers_v17
            # BODY_TIGHTNESS_MAX_WEEKLY = 0.30, so we use 0.40 here as buffer
            max_wick = 0.90
            max_body = 0.40
        else:
            # Daily mode: Wick < 60%, Body < 15% (original nano-cap limits)
            max_wick = 0.60
            max_body = 0.15

        mask = (wick_height_pct < max_wick) & (body_height_pct < max_body)

        # Returns indices of the START of valid windows
        return np.where(mask)[0]


class ConsolidationPatternScanner(VectorizedPatternMixin):
    """
    Microstructure-Aware Consolidation Pattern Scanner.

    Integrates sleeper_scanner_v17 detection logic with TRANS temporal
    architecture for complete historical pattern analysis.

    Features:
    - Thin liquidity handling (differentiates noise vs signal volatility)
    - Optional accumulation detection (controlled by ENABLE_ACCUMULATION_DETECTION)
    - Walk-forward outcome tracking (K0-K5 classification)
    - Parallel processing support
    - Backward compatible with TRANS pipeline
    """

    def __init__(
        self,
        bbw_percentile_threshold: float = 0.30,
        adx_threshold: float = 32.0,
        volume_ratio_threshold: float = 0.35,
        range_ratio_threshold: float = 0.65,
        qualifying_days: int = 10,
        min_data_days: int = 100,
        indicator_lookback: int = 100,
        min_liquidity_dollar: float = 50000,
        enable_market_cap: bool = True,
        candidate_only: bool = False,
        disable_gcs: bool = False,
        fast_validation: bool = False,
        # Adaptive thresholds (Jan 2026)
        tightness_zscore: Optional[float] = None,
        min_float_turnover: Optional[float] = None,
        # Weekly qualification mode (Jan 2026)
        use_weekly_qualification: bool = False,
        # Point-in-time market cap (Jan 2026)
        skip_market_cap_api: bool = False
    ):
        """
        Initialize scanner with detection parameters.

        Args:
            bbw_percentile_threshold: BBW percentile (legacy param, not used in v17)
            adx_threshold: ADX threshold (legacy param, not used in v17)
            volume_ratio_threshold: Volume ratio threshold (legacy param)
            range_ratio_threshold: Range ratio threshold (legacy param)
            qualifying_days: Days to qualify pattern
            min_data_days: Minimum days of data required
            indicator_lookback: Days needed for indicator calculation
            min_liquidity_dollar: Minimum dollar volume for liquidity gate
            enable_market_cap: Fetch market cap from APIs (default True)
            candidate_only: If True, skip outcome labeling (output candidates only)
                           This ensures NO look-ahead bias - outcomes must be labeled
                           separately by 00b_label_outcomes.py after 100 days elapse.
            disable_gcs: If True, skip GCS initialization (enables multiprocessing)
            fast_validation: Skip expensive mock data detection (saves ~50-100ms/ticker)
            tightness_zscore: Max Z-Score for BBW (e.g., -1.0 = 1 std dev tighter than avg).
                             When set, overrides bbw_percentile_threshold with adaptive measurement.
            min_float_turnover: Minimum 20d float turnover (e.g., 0.10 = 10% of float traded).
                               Requires accumulation activity to detect pattern.
            use_weekly_qualification: If True, use weekly candles for 10-week qualification
                                     period instead of daily 10-day period. This finds
                                     longer-term consolidation patterns (~2.5 months).
                                     Requires ~4 years of historical data for SMA_200.
            skip_market_cap_api: If True, skip all market cap API calls and use cached
                                shares_outstanding × price for point-in-time (PIT)
                                market cap estimation. This is faster and avoids
                                look-ahead bias when analyzing historical data.
        """
        # Store skip_market_cap_api for later use
        self.skip_market_cap_api = skip_market_cap_api
        # Legacy parameters (kept for backward compatibility)
        self.bbw_percentile_threshold = bbw_percentile_threshold
        self.adx_threshold = adx_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.range_ratio_threshold = range_ratio_threshold

        # Adaptive thresholds (Jan 2026)
        self.tightness_zscore = tightness_zscore
        self.min_float_turnover = min_float_turnover

        # Weekly qualification mode (Jan 2026)
        self.use_weekly_qualification = use_weekly_qualification

        # Active parameters
        self.qualifying_days = qualifying_days
        self.min_data_days = min_data_days
        self.indicator_lookback = indicator_lookback
        self.min_liquidity_dollar = min_liquidity_dollar
        self.enable_market_cap = enable_market_cap
        self.candidate_only = candidate_only
        self.disable_gcs = disable_gcs
        self.fast_validation = fast_validation

        # Initialize data loader (disable GCS for multiprocessing support)
        self.data_loader = DataLoader(disable_gcs=disable_gcs)

        # Initialize market cap fetcher (only if enabled)
        # When skip_market_cap_api=True, use cache_only mode to avoid API calls
        if enable_market_cap:
            self.market_cap_fetcher = MarketCapFetcher(cache_only=skip_market_cap_api)
        else:
            self.market_cap_fetcher = None

        # Initialize share dilution database (US stocks only, microsecond lookups)
        # Uses local SQLite DB populated by nightly cron job (no network calls)
        self.enable_dilution_filter = ENABLE_SHARE_DILUTION_FILTER
        self.dilution_db = DilutionDB() if ENABLE_SHARE_DILUTION_FILTER else None

        # Initialize v17 labeler (v17-only system)
        self.labeler_v17 = PathDependentLabelerV17(
            indicator_warmup=INDICATOR_WARMUP_DAYS,
            indicator_stable=INDICATOR_STABLE_DAYS,
            outcome_window=OUTCOME_WINDOW_DAYS
        )
        logger.info(f"Using Path-Dependent Labeling v17")

        accumulation_status = "ENABLED" if ENABLE_ACCUMULATION_DETECTION else "DISABLED"
        if enable_market_cap:
            if skip_market_cap_api:
                market_cap_status = "CACHE-ONLY (PIT estimation, no API calls)"
            else:
                market_cap_status = "ENABLED (with API calls)"
        else:
            market_cap_status = "DISABLED"
        dilution_status = "ENABLED (US only)" if ENABLE_SHARE_DILUTION_FILTER else "DISABLED"
        candidate_status = "CANDIDATE-ONLY (no labeling)" if candidate_only else "FULL (with labeling)"
        logger.info(f"Initialized PatternScanner V17 (Microstructure-Aware)")
        logger.info(f"  - Min Liquidity: ${min_liquidity_dollar:,.0f}")
        logger.info(f"  - Accumulation Detection: {accumulation_status}")
        logger.info(f"  - Market Cap Fetching: {market_cap_status}")
        logger.info(f"  - Share Dilution Filter: {dilution_status}")
        logger.info(f"  - Mode: {candidate_status}")
        if tightness_zscore is not None:
            logger.info(f"  - ADAPTIVE: Tightness Z-Score: {tightness_zscore}")
        if min_float_turnover is not None:
            logger.info(f"  - ADAPTIVE: Min Float Turnover: {min_float_turnover:.1%}")
        if use_weekly_qualification:
            logger.info(f"  - WEEKLY MODE: Using 10-week qualification (requires ~4 years data)")
        if not candidate_only:
            logger.info(f"  - Labeling System: v17")

    def scan_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        df: Optional[pd.DataFrame] = None
    ) -> ScanResult:
        """
        Scan a single ticker for consolidation patterns using sleeper_scanner_v17 logic.

        This method uses a sliding window approach to detect consolidation patterns
        across the historical data, then tracks each pattern forward to calculate
        outcome classes (K0-K5).

        Args:
            ticker: Stock ticker symbol
            start_date: Scan start date
            end_date: Scan end date
            df: Pre-loaded DataFrame (optional, will load if not provided)

        Returns:
            ScanResult with detected patterns and metadata
        """
        start_time = time.time()

        try:
            # 1. Load and validate data
            if df is None:
                df = self._load_and_optimize(ticker, start_date, end_date)
            else:
                df = self._optimize_and_clean(df, ticker)

        except ValueError as e:
            return ScanResult(
                ticker=ticker,
                patterns_found=0,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except (DataIntegrityError, TemporalConsistencyError, ValidationError) as e:
            return ScanResult(
                ticker=ticker,
                patterns_found=0,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # 1.5 EARLY REJECTION: Share dilution filter (US stocks only)
        # Uses local SQLite DB for microsecond lookups (no network calls)
        # Reject US stocks with >20% share dilution in trailing 12 months
        if self.dilution_db is not None and is_us_ticker(ticker):
            dilution_pct = self.dilution_db.get_dilution(ticker)

            if dilution_pct is None:
                # No data in local DB - fail open (don't block scanner)
                # Run `python utils/dilution_manager.py --update tickers.txt` to populate
                logger.warning(f"{ticker}: No dilution data in local DB - skipping filter (run nightly update)")
            elif dilution_pct > 100:
                # Likely a stock split (shares more than doubled) - skip filter
                logger.debug(f"{ticker}: {dilution_pct:.1f}% share change (likely split) - skipping filter")
            elif dilution_pct > MAX_DILUTION_PCT:
                # Real dilution exceeds threshold - REJECT immediately
                logger.info(f"{ticker}: REJECTED - {dilution_pct:.1f}% dilution > {MAX_DILUTION_PCT}%")
                return ScanResult(
                    ticker=ticker,
                    patterns_found=0,
                    error=f"Dilution filter: {dilution_pct:.1f}% > {MAX_DILUTION_PCT}% threshold",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            else:
                logger.debug(f"{ticker}: {dilution_pct:.1f}% dilution OK")

        # 2. Fetch HISTORICAL market cap BEFORE pattern detection (if enabled)
        # CRITICAL: Must be done before detection so WIDTH_LIMITS can use market_cap_category
        # NEW: We use historical market cap so patterns from years ago are categorized correctly
        historical_mc_df = None
        typical_market_cap_category = None

        if self.market_cap_fetcher is not None:
            try:
                # Get historical market cap DataFrame (market_cap for each date)
                historical_mc_df = self.market_cap_fetcher.get_historical_market_cap(ticker, df)

                if historical_mc_df is not None and 'market_cap' in historical_mc_df.columns:
                    # Calculate typical (median) market cap category for WIDTH_LIMITS
                    # This ensures we use appropriate width limits during pattern detection
                    median_mc = historical_mc_df['market_cap'].median()
                    typical_market_cap_category = self.market_cap_fetcher.categorize_market_cap_value(median_mc)
                    logger.info(f"{ticker}: Historical market cap available, median category: {typical_market_cap_category}")
                else:
                    # Fallback: Use ADV (Average Dollar Volume) as proxy for market cap
                    # ADV correlates better with market cap than price alone
                    # Typical relationships (rough heuristics):
                    #   ADV < $100K → nano_cap (very illiquid)
                    #   ADV $100K - $500K → micro_cap
                    #   ADV $500K - $5M → small_cap
                    #   ADV $5M - $50M → mid_cap
                    #   ADV > $50M → large_cap
                    dollar_volume = df['close'] * df['volume']
                    median_adv = dollar_volume.rolling(40).mean().median()

                    if pd.isna(median_adv) or median_adv < 100_000:
                        typical_market_cap_category = 'nano_cap'
                    elif median_adv < 500_000:
                        typical_market_cap_category = 'micro_cap'
                    elif median_adv < 5_000_000:
                        typical_market_cap_category = 'small_cap'
                    elif median_adv < 50_000_000:
                        typical_market_cap_category = 'mid_cap'
                    else:
                        typical_market_cap_category = 'large_cap'

                    logger.info(f"{ticker}: Market cap unavailable, using ADV-based fallback (median ADV ${median_adv:,.0f} → {typical_market_cap_category})")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to fetch historical market cap: {e}")
                # Fallback: Use ADV-based heuristic (same as above)
                dollar_volume = df['close'] * df['volume']
                median_adv = dollar_volume.rolling(40).mean().median()

                if pd.isna(median_adv) or median_adv < 100_000:
                    typical_market_cap_category = 'nano_cap'
                elif median_adv < 500_000:
                    typical_market_cap_category = 'micro_cap'
                elif median_adv < 5_000_000:
                    typical_market_cap_category = 'small_cap'
                elif median_adv < 50_000_000:
                    typical_market_cap_category = 'mid_cap'
                else:
                    typical_market_cap_category = 'large_cap'

                logger.info(f"{ticker}: Using ADV-based fallback (median ADV ${median_adv:,.0f} → {typical_market_cap_category})")

        # 2.5 WEEKLY AGGREGATION (if weekly mode enabled)
        # Aggregate daily candles to weekly for longer-term consolidation detection
        weekly_df = None
        week_to_daily_map = None
        scan_df = df  # Default: use daily data
        candle_frequency = 'daily'

        if self.use_weekly_qualification:
            try:
                from utils.weekly_aggregator import (
                    resample_to_weekly,
                    calculate_weekly_indicators,
                    validate_weekly_data_requirements
                )

                # Aggregate to weekly candles
                weekly_df, week_to_daily_map = resample_to_weekly(df)

                # Validate data requirements for weekly mode
                is_valid, msg = validate_weekly_data_requirements(weekly_df)
                if not is_valid:
                    logger.warning(f"{ticker}: {msg}")
                    return ScanResult(
                        ticker=ticker,
                        patterns_found=0,
                        error=f"Weekly mode: {msg}",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Calculate weekly indicators
                weekly_df = calculate_weekly_indicators(weekly_df)

                # Use weekly data for pattern detection
                scan_df = weekly_df
                candle_frequency = 'weekly'

                logger.info(f"{ticker}: Weekly mode - aggregated {len(df)} daily to {len(weekly_df)} weekly candles")

            except Exception as e:
                logger.error(f"{ticker}: Weekly aggregation failed: {e}")
                return ScanResult(
                    ticker=ticker,
                    patterns_found=0,
                    error=f"Weekly aggregation failed: {e}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

        # 3. Detect patterns using sliding window + sleeper_scanner_v17
        # Pass historical_mc_df so each pattern uses its OWN market cap category
        # In weekly mode, scan_df is weekly candles; otherwise it's daily
        try:
            patterns = self._detect_patterns_sliding_window(
                scan_df,
                ticker,
                historical_mc_df,
                typical_market_cap_category,  # Fallback if historical MC unavailable
                candle_frequency=candle_frequency,
                week_to_daily_map=week_to_daily_map,
                daily_df=df if self.use_weekly_qualification else None
            )
        except Exception as e:
            logger.error(f"{ticker}: Pattern detection failed: {e}")
            return ScanResult(
                ticker=ticker,
                patterns_found=0,
                error=f"Pattern detection failed: {e}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # 4. Add HISTORICAL market cap to each pattern (if available)
        # NEW: Each pattern gets its market cap from the ACTUAL date it occurred
        if historical_mc_df is not None and 'market_cap' in historical_mc_df.columns:
            # Ensure historical_mc_df has date as index for fast lookup
            if 'date' in historical_mc_df.columns:
                historical_mc_indexed = historical_mc_df.set_index('date')
            else:
                historical_mc_indexed = historical_mc_df

            for pattern in patterns:
                # Get pattern date (use end_date which is when pattern completes)
                if 'end_date' in pattern:
                    pattern_date = pd.to_datetime(pattern['end_date'])
                elif 'activation_date' in pattern:
                    pattern_date = pd.to_datetime(pattern['activation_date'])
                elif 'date' in pattern:
                    pattern_date = pd.to_datetime(pattern['date'])
                else:
                    logger.warning(f"{ticker}: Pattern missing date field, cannot assign historical market cap")
                    continue

                # Look up historical market cap for this specific date
                try:
                    if pattern_date in historical_mc_indexed.index:
                        hist_mc = historical_mc_indexed.loc[pattern_date, 'market_cap']
                        pattern['market_cap'] = float(hist_mc)
                        pattern['market_cap_category'] = self.market_cap_fetcher.categorize_market_cap_value(hist_mc)
                        pattern['market_cap_source'] = 'historical'
                    else:
                        # Date not found - find nearest date
                        nearest_date = historical_mc_indexed.index[historical_mc_indexed.index.get_indexer([pattern_date], method='nearest')[0]]
                        hist_mc = historical_mc_indexed.loc[nearest_date, 'market_cap']
                        pattern['market_cap'] = float(hist_mc)
                        pattern['market_cap_category'] = self.market_cap_fetcher.categorize_market_cap_value(hist_mc)
                        pattern['market_cap_source'] = 'historical_nearest'
                        logger.debug(f"{ticker}: Used nearest date {nearest_date} for pattern at {pattern_date}")
                except Exception as e:
                    logger.warning(f"{ticker}: Failed to assign historical market cap for pattern at {pattern_date}: {e}")
                    pattern['market_cap'] = None
                    pattern['market_cap_category'] = None
                    pattern['market_cap_source'] = 'unavailable'
        else:
            # No historical market cap available - mark patterns accordingly
            for pattern in patterns:
                pattern['market_cap'] = None
                pattern['market_cap_category'] = typical_market_cap_category  # Use typical category at least
                pattern['market_cap_source'] = 'unavailable'

        # 5. Calculate scan statistics
        processing_time = (time.time() - start_time) * 1000

        # Calculate median market cap for summary
        median_market_cap = None
        if historical_mc_df is not None and 'market_cap' in historical_mc_df.columns:
            median_market_cap = historical_mc_df['market_cap'].median()

        features = {
            'data_range_days': len(df),
            'patterns_per_1000_days': (len(patterns) / len(df)) * 1000 if len(df) > 0 else 0,
            'accumulation_enabled': ENABLE_ACCUMULATION_DETECTION,
            'market_cap_enabled': self.enable_market_cap,
            'historical_market_cap_used': historical_mc_df is not None
        }

        logger.info(f"{ticker}: Found {len(patterns)} patterns in {processing_time:.1f}ms" +
                   (f" (Median Market Cap: ${median_market_cap/1e9:.2f}B, Category: {typical_market_cap_category})"
                    if median_market_cap else ""))

        return ScanResult(
            ticker=ticker,
            patterns_found=len(patterns),
            patterns=patterns,
            processing_time_ms=processing_time,
            data_points_processed=len(df),
            features=features,
            market_cap=median_market_cap,
            market_cap_category=typical_market_cap_category
        )

    def scan_universe(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        workers: int = 4,
        progress_callback: Optional[callable] = None,
        prefetch_market_caps: bool = True
    ) -> UniverseScanResult:
        """
        Scan multiple tickers for consolidation patterns.

        OPTIMIZED: Uses worker pool initialization and market cap pre-fetching
        to minimize per-ticker overhead.

        Args:
            tickers: List of ticker symbols
            start_date: Scan start date
            end_date: Scan end date
            workers: Number of parallel workers (default 4)
            progress_callback: Optional callback for progress updates
            prefetch_market_caps: Pre-fetch all market caps before scanning (default True)

        Returns:
            UniverseScanResult with aggregated results
        """
        start_time = time.time()

        logger.info(f"Scanning {len(tickers)} tickers with {workers} workers")

        # OPTIMIZATION: Pre-fetch market caps in batch before scanning
        # This warms the cache and avoids per-ticker API rate limiting
        if prefetch_market_caps and self.enable_market_cap and self.market_cap_fetcher is not None:
            logger.info("Pre-fetching market caps (this warms the cache)...")
            prefetch_start = time.time()
            self.market_cap_fetcher.prefetch_market_caps(
                tickers,
                use_async=True,
                max_workers=min(workers * 2, 8)
            )
            prefetch_time = time.time() - prefetch_start
            logger.info(f"Market cap pre-fetch completed in {prefetch_time:.1f}s")

        ticker_results = []
        all_patterns = []
        errors = {}

        if workers == 1:
            # Sequential processing
            for i, ticker in enumerate(tickers):
                result = self.scan_ticker(ticker, start_date, end_date)
                ticker_results.append(result)

                if result.success:
                    all_patterns.extend(result.patterns)
                else:
                    errors[ticker] = result.error

                if progress_callback:
                    progress_callback(i + 1, len(tickers), ticker, result)
        else:
            # OPTIMIZED: Parallel processing with worker initialization
            # Each worker process initializes scanner ONCE, not per-ticker

            # Build worker configuration
            worker_config = {
                'bbw_percentile_threshold': self.bbw_percentile_threshold,
                'adx_threshold': self.adx_threshold,
                'volume_ratio_threshold': self.volume_ratio_threshold,
                'range_ratio_threshold': self.range_ratio_threshold,
                'qualifying_days': self.qualifying_days,
                'min_data_days': self.min_data_days,
                'indicator_lookback': self.indicator_lookback,
                'min_liquidity_dollar': self.min_liquidity_dollar,
                'enable_market_cap': self.enable_market_cap,
                'candidate_only': self.candidate_only,
                'fast_validation': self.fast_validation,
                # Adaptive thresholds (Jan 2026)
                'tightness_zscore': self.tightness_zscore,
                'min_float_turnover': self.min_float_turnover,
                # Weekly qualification mode (Jan 2026)
                'use_weekly_qualification': self.use_weekly_qualification,
                # Point-in-time market cap (Jan 2026)
                'skip_market_cap_api': self.skip_market_cap_api,
            }

            # Prepare task arguments
            tasks = [(ticker, start_date, end_date) for ticker in tickers]

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(worker_config,)
            ) as executor:
                future_to_ticker = {
                    executor.submit(_worker_scan_ticker, task): task[0]
                    for task in tasks
                }

                completed = 0
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        ticker_results.append(result)

                        if result.success:
                            all_patterns.extend(result.patterns)
                        else:
                            errors[ticker] = result.error

                    except Exception as e:
                        logger.error(f"{ticker}: Worker exception: {e}")
                        errors[ticker] = str(e)
                        result = ScanResult(
                            ticker=ticker,
                            patterns_found=0,
                            error=str(e)
                        )
                        ticker_results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(tickers), ticker, result)

        # Aggregate results
        total_time = time.time() - start_time
        successful = sum(1 for r in ticker_results if r.success)

        result = UniverseScanResult(
            total_tickers=len(tickers),
            successful_tickers=successful,
            failed_tickers=len(tickers) - successful,
            total_patterns=len(all_patterns),
            total_time_seconds=total_time,
            patterns_per_ticker=len(all_patterns) / len(tickers) if tickers else 0,
            ticker_results=ticker_results,
            all_patterns=all_patterns,
            errors=errors
        )

        logger.info(f"Universe scan complete: {result.total_patterns} patterns "
                   f"from {result.successful_tickers}/{result.total_tickers} tickers "
                   f"in {result.total_time_seconds:.1f}s")

        return result

    def _scan_ticker_worker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]]
    ) -> ScanResult:
        """Worker function for parallel processing."""
        scanner = ConsolidationPatternScanner(
            bbw_percentile_threshold=self.bbw_percentile_threshold,
            adx_threshold=self.adx_threshold,
            volume_ratio_threshold=self.volume_ratio_threshold,
            range_ratio_threshold=self.range_ratio_threshold,
            qualifying_days=self.qualifying_days,
            min_data_days=self.min_data_days,
            indicator_lookback=self.indicator_lookback,
            min_liquidity_dollar=self.min_liquidity_dollar,
            enable_market_cap=self.enable_market_cap,
            candidate_only=self.candidate_only,
            disable_gcs=self.disable_gcs
        )
        return scanner.scan_ticker(ticker, start_date, end_date)

    def _load_and_optimize(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]]
    ) -> pd.DataFrame:
        """Load ticker data and optimize for scanning."""
        # Calculate extended start date for lookback
        if start_date:
            start_dt = pd.to_datetime(start_date)
            extended_start = start_dt - timedelta(days=self.indicator_lookback + 50)
        else:
            extended_start = None

        # Load data
        df = self.data_loader.load_ticker(
            ticker,
            start_date=extended_start,
            end_date=end_date,
            validate=True,
            fast_validation=self.fast_validation
        )

        if df is None:
            raise ValueError(f"No data found for {ticker}")

        return self._optimize_and_clean(df, ticker)

    def _optimize_and_clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and validate data."""
        # Standardize column names to lowercase for consistency with rest of codebase
        df.columns = [col.lower() if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'date']
                      else col for col in df.columns]

        required_cols = ['high', 'low', 'close', 'volume']

        # Schema check
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            raise ValueError(f"{ticker}: Missing columns: {missing}")

        # Length check
        if len(df) < self.min_data_days:
            raise ValueError(f"{ticker}: Insufficient data. Need {self.min_data_days}, got {len(df)}")

        df_clean = df.copy()

        # Clean prices
        mask_valid_price = (
            (df_clean['high'] > 0) &
            (df_clean['low'] > 0) &
            (df_clean['close'] > 0)
        )

        invalid_price_count = (~mask_valid_price).sum()
        if invalid_price_count > 0:
            logger.warning(f"{ticker}: Removing {invalid_price_count} rows with invalid prices")
            df_clean = df_clean[mask_valid_price]

        # Clean volume
        if (df_clean['volume'] < 0).any():
            neg_vol_count = (df_clean['volume'] < 0).sum()
            logger.warning(f"{ticker}: Clamping {neg_vol_count} negative volumes to 0")
            df_clean.loc[df_clean['volume'] < 0, 'volume'] = 0

        # Drop NaNs
        df_clean = df_clean.dropna(subset=required_cols)

        # Final length check
        if len(df_clean) < self.min_data_days:
            raise ValueError(f"{ticker}: Insufficient data after cleaning. "
                           f"Got {len(df_clean)}, needed {self.min_data_days}")

        # Add ticker column if not present
        if 'Ticker' not in df_clean.columns:
            df_clean['Ticker'] = ticker

        return df_clean

    @staticmethod
    def _select_widest_category(categories: list) -> str:
        """
        Select the widest (most permissive) category from a list.

        Width hierarchy (widest to narrowest):
        nano_cap (60%) > micro_cap (45%) > small_cap (30%) >
        mid_cap (22.5%) > large_cap (15%) > mega_cap (7.5%)

        Args:
            categories: List of market cap categories

        Returns:
            The category with the widest width limit
        """
        # Define width order (widest first)
        width_order = ['nano_cap', 'micro_cap', 'small_cap', 'mid_cap', 'large_cap', 'mega_cap']

        # Return the first category found in the order
        for cat in width_order:
            if cat in categories:
                return cat

        # Fallback
        return categories[0] if categories else 'small_cap'

    def _detect_patterns_sliding_window(
        self,
        df: pd.DataFrame,
        ticker: str,
        historical_mc_df: Optional[pd.DataFrame] = None,
        fallback_category: Optional[str] = None,
        # Weekly qualification mode parameters (Jan 2026)
        candle_frequency: str = 'daily',
        week_to_daily_map: Optional[Dict] = None,
        daily_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Detect consolidation patterns using sliding window approach.

        OPTIMIZED: Pre-computes all rolling statistics once to avoid redundant
        calculations in the sliding window loop (3-5x speedup).

        NEW: Each pattern uses its OWN market cap category based on the qualification
        period (first 10 days/weeks). If multiple categories are present during
        qualification, uses the WIDEST category to be more inclusive.

        Args:
            df: Price data DataFrame (must have DatetimeIndex).
                In weekly mode, this is weekly candles.
            ticker: Stock ticker symbol
            historical_mc_df: DataFrame with historical market cap for each date
            fallback_category: Fallback category if historical MC unavailable
            candle_frequency: 'daily' or 'weekly' (passed to find_sleepers_v17)
            week_to_daily_map: Mapping from week-end to daily date (weekly mode only)
            daily_df: Original daily DataFrame for outcome labeling (weekly mode only)

        Scans through historical data using 100-day/week windows, detects patterns
        with find_sleepers_v17(), then tracks forward to calculate outcomes.

        V17 Mode: Uses path-dependent labeling with risk multiples when configured.
        Weekly Mode: Uses weekly candles for qualification but daily prices for outcomes.
        """
        patterns = []

        # V17-only: Configure windows for indicator warmup and outcome tracking
        # Weekly mode uses smaller windows (weeks instead of days)
        if candle_frequency == 'weekly':
            # Weekly mode: ~6 months warmup, ~20 weeks forward for outcome
            min_window = 30   # 30 weeks (~7 months for indicator warmup)
            forward_window = 20  # 20 weeks (~5 months for outcome tracking)
            qualifying_periods = 10  # 10 weeks qualification
            logger.debug(f"{ticker}: Using weekly windows: min={min_window}, forward={forward_window}")
        else:
            # Daily mode: standard values
            min_window = INDICATOR_WARMUP_DAYS + INDICATOR_STABLE_DAYS  # 130 days
            forward_window = OUTCOME_WINDOW_DAYS  # 100 days
            qualifying_periods = 10  # 10 days qualification

        # OPTIMIZATION: Pre-compute all rolling statistics once for entire dataframe
        # This avoids recalculating them for each window (3-5x speedup)
        # Note: For weekly data, these are already computed by calculate_weekly_indicators()
        if 'DollarVol' not in df.columns:
            df['DollarVol'] = df['close'] * df['volume']
        if 'Vol_50MA' not in df.columns:
            df['Vol_50MA'] = df['volume'].rolling(50).mean()
        if 'Pct_Change' not in df.columns:
            df['Pct_Change'] = df['close'].pct_change()

        # Jan 2026: Pre-compute SMAs for market structure check (replaces ADX)
        # These are needed by check_thresholds() in sleeper_scanner_v17
        # For weekly data, these should already be computed by calculate_weekly_indicators()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(50).mean()
        if 'sma_200' not in df.columns:
            df['sma_200'] = df['close'].rolling(200).mean()

        # Prepare historical market cap lookup if available
        historical_mc_indexed = None
        if historical_mc_df is not None and 'market_cap' in historical_mc_df.columns:
            if 'date' in historical_mc_df.columns:
                historical_mc_indexed = historical_mc_df.set_index('date')
            else:
                historical_mc_indexed = historical_mc_df

        # VECTORIZED PRE-FILTER: Reduces loop iterations by ~95%
        # Only process windows that meet physical tightness requirements
        is_weekly = (candle_frequency == 'weekly')
        candidate_indices = self._get_candidate_indices_vectorized(df, min_window, is_weekly=is_weekly)

        # Filter candidates to ensure enough forward data for outcome tracking
        max_valid_start = len(df) - min_window - forward_window
        candidate_indices = candidate_indices[candidate_indices < max_valid_start]

        logger.debug(f"{ticker}: Vectorized pre-filter reduced {max_valid_start // 10} windows to {len(candidate_indices)} candidates")

        # Iterate ONLY over candidate windows (not full range)
        for start_idx in candidate_indices:
            end_idx = start_idx + min_window

            # Extract window (with pre-computed features)
            window_df = df.iloc[start_idx:end_idx].copy()

            # NEW: Determine market cap category for THIS SPECIFIC PATTERN
            # Check ALL days in qualification period (first 10 days) and use WIDEST category
            pattern_market_cap_category = fallback_category  # Default

            if historical_mc_indexed is not None:
                try:
                    # Get all dates in the qualification period
                    qual_end_idx = min(start_idx + qualifying_days, end_idx)
                    categories_during_qualification = set()

                    for day_idx in range(start_idx, qual_end_idx):
                        day_date = df.index[day_idx] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[day_idx]['date']

                        # Look up market cap for this day
                        if day_date in historical_mc_indexed.index:
                            day_mc = historical_mc_indexed.loc[day_date, 'market_cap']
                            day_category = self.market_cap_fetcher.categorize_market_cap_value(day_mc)
                            categories_during_qualification.add(day_category)

                    # If we found categories, select the WIDEST one
                    if categories_during_qualification:
                        pattern_market_cap_category = self._select_widest_category(list(categories_during_qualification))

                except:
                    # Fallback to ADV-based if lookup fails
                    window_dollar_volume = window_df['close'] * window_df['volume']
                    window_median_adv = window_dollar_volume.rolling(40).mean().median()
                    if pd.isna(window_median_adv) or window_median_adv < 100_000:
                        pattern_market_cap_category = 'nano_cap'
                    elif window_median_adv < 500_000:
                        pattern_market_cap_category = 'micro_cap'
                    elif window_median_adv < 5_000_000:
                        pattern_market_cap_category = 'small_cap'
                    elif window_median_adv < 50_000_000:
                        pattern_market_cap_category = 'mid_cap'
                    else:
                        pattern_market_cap_category = 'large_cap'

            # If still no category determined, use ADV-based fallback
            if pattern_market_cap_category is None:
                window_dollar_volume = window_df['close'] * window_df['volume']
                window_median_adv = window_dollar_volume.rolling(40).mean().median()
                if pd.isna(window_median_adv) or window_median_adv < 100_000:
                    pattern_market_cap_category = 'nano_cap'
                elif window_median_adv < 500_000:
                    pattern_market_cap_category = 'micro_cap'
                elif window_median_adv < 5_000_000:
                    pattern_market_cap_category = 'small_cap'
                elif window_median_adv < 50_000_000:
                    pattern_market_cap_category = 'mid_cap'
                else:
                    pattern_market_cap_category = 'large_cap'

            # Get shares outstanding for float turnover calculation (if available)
            pattern_shares_outstanding = None
            if historical_mc_indexed is not None and 'shares_outstanding' in historical_mc_indexed.columns:
                try:
                    # Use shares from the pattern end date
                    pattern_date = df.index[end_idx - 1] if isinstance(df.index, pd.DatetimeIndex) else None
                    if pattern_date is not None and pattern_date in historical_mc_indexed.index:
                        pattern_shares_outstanding = historical_mc_indexed.loc[pattern_date, 'shares_outstanding']
                except:
                    pass  # Fall back to None if lookup fails

            # Run sleeper detection (with pre-computed=True to skip redundant calculations)
            # CRITICAL: Pass THIS PATTERN'S market cap category for correct WIDTH_LIMITS
            sleeper_result = find_sleepers_v17(
                window_df,
                self.min_liquidity_dollar,
                precomputed=True,
                market_cap_category=pattern_market_cap_category,
                # Adaptive thresholds (Jan 2026)
                tightness_zscore=self.tightness_zscore,
                min_float_turnover=self.min_float_turnover,
                shares_outstanding=pattern_shares_outstanding,
                # Weekly qualification mode (Jan 2026)
                candle_frequency=candle_frequency
            )

            if sleeper_result is None:
                continue  # No pattern detected in this window

            # Pattern detected! Now track forward to calculate outcome
            pattern_start_idx = end_idx - 20  # Pattern started ~20 days ago
            pattern_end_idx = end_idx - 1
            outcome_end_idx = min(end_idx + forward_window, len(df))

            # Get pattern boundaries
            lower_boundary = sleeper_result['upper_lid'] / (1 + sleeper_result['box_width'])
            pattern_boundaries = {
                'upper': sleeper_result['upper_lid'],
                'lower': lower_boundary
            }

            # Calculate risk metrics
            entry_price = df.iloc[pattern_end_idx]['close']
            stop_loss = lower_boundary * 0.98  # 2% buffer
            R = entry_price - stop_loss

            # Build pattern dictionary (candidate format - NO outcome_class yet)
            pattern_dict = {
                'ticker': ticker,
                'pattern_id': f"{ticker}_{df.index[pattern_start_idx] if hasattr(df.index, '__getitem__') else pattern_start_idx}",
                'start_date': df.index[pattern_start_idx] if isinstance(df.index, pd.DatetimeIndex) else pattern_start_idx,
                'end_date': df.index[pattern_end_idx] if isinstance(df.index, pd.DatetimeIndex) else pattern_end_idx,
                'start_idx': pattern_start_idx,
                'end_idx': pattern_end_idx,
                'phase': 'completed',
                'upper_boundary': sleeper_result['upper_lid'],
                'lower_boundary': lower_boundary,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'risk_unit': R,
                'trigger_price': sleeper_result['trigger'],
                'box_width': sleeper_result['box_width'],  # Body height (strict <15%)
                'wick_width': sleeper_result['wick_width'],  # Wick height (loose <45% for micro-caps)
                'body_tightness_max': sleeper_result['body_tightness_max'],  # Max allowed body (15%)
                'allowed_wick_width': sleeper_result['allowed_width'],  # Max allowed wick for this market cap
                'is_thin_stock': sleeper_result['is_thin_stock'],
                'accumulation_count': sleeper_result['accumulation_count'],
                'liquidity': sleeper_result['liquidity'],
                'status': sleeper_result['status'],
                'detection_market_cap_category': pattern_market_cap_category,  # Category used during detection
                # Adaptive metrics (Jan 2026)
                'body_width_zscore': sleeper_result.get('body_width_zscore'),
                'float_turnover': sleeper_result.get('float_turnover'),
                # Relative threshold metrics (Jan 2026 - check_thresholds)
                'bbw_zscore': sleeper_result.get('bbw_zscore'),
                'market_structure_ok': sleeper_result.get('market_structure_ok'),
                # Weekly qualification mode (Jan 2026)
                'qualification_frequency': candle_frequency
            }

            # WEEKLY MODE: Add end_date_daily for outcome labeling on daily prices
            # The pattern's end_date is a week-end date; we need the actual trading day
            if candle_frequency == 'weekly' and week_to_daily_map is not None:
                pattern_end_date = pattern_dict['end_date']
                if pattern_end_date is not None:
                    if not isinstance(pattern_end_date, pd.Timestamp):
                        pattern_end_date = pd.to_datetime(pattern_end_date)
                    # Map week-end to last daily trading day
                    if pattern_end_date in week_to_daily_map:
                        pattern_dict['end_date_daily'] = week_to_daily_map[pattern_end_date]
                    else:
                        # Try to find nearest week
                        for week_end, daily_date in week_to_daily_map.items():
                            if abs((week_end - pattern_end_date).days) <= 2:
                                pattern_dict['end_date_daily'] = daily_date
                                break
                        else:
                            # Fallback to same date if mapping not found
                            pattern_dict['end_date_daily'] = pattern_dict['end_date']
                            logger.warning(
                                f"{ticker}: No daily date mapping for pattern end {pattern_end_date}"
                            )
            else:
                # Daily mode: end_date_daily is same as end_date
                pattern_dict['end_date_daily'] = pattern_dict['end_date']

            # TEMPORAL INTEGRITY: Only label if NOT in candidate-only mode
            # In candidate-only mode, outcome_class is added later by 00b_label_outcomes.py
            # after 100 days have elapsed from pattern end_date
            if self.candidate_only:
                # Candidate registry: NO outcome_class, NO labeling_version
                # These fields will be added by the outcome labeling script
                pattern_dict['outcome_class'] = None  # Explicitly null - NOT labeled yet
                pattern_dict['labeling_version'] = None
            else:
                # Legacy mode: Label immediately (LOOK-AHEAD RISK for live inference)
                # Only use this for backtesting on historical data where outcomes are known
                label = self.labeler_v17.label_pattern(
                    full_data=df,
                    pattern_end_idx=pattern_end_idx,
                    pattern_boundaries=pattern_boundaries
                )

                # Skip invalid or grey zone patterns
                if label is None or label == -1:
                    continue

                pattern_dict['outcome_class'] = label  # V17 label (0=Danger, 1=Noise, 2=Target)
                pattern_dict['labeling_version'] = 'v17'

            patterns.append(pattern_dict)

        return patterns

    def save_patterns(
        self,
        patterns: List[Dict],
        output_path: Union[str, Path],
        format: str = 'parquet'
    ) -> None:
        """
        Save detected patterns to file.

        Args:
            patterns: List of pattern dictionaries
            output_path: Output file path
            format: Output format ('parquet' or 'csv')
        """
        if not patterns:
            logger.warning("No patterns to save")
            return

        df = pd.DataFrame(patterns)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(patterns)} patterns to {output_path}")


# Convenience function for direct usage
def scan_for_patterns(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[str] = None,
    workers: int = 4
) -> UniverseScanResult:
    """
    Convenience function to scan multiple tickers for patterns.

    Args:
        tickers: List of ticker symbols
        start_date: Scan start date (YYYY-MM-DD)
        end_date: Scan end date (YYYY-MM-DD)
        output_path: Output file path (optional)
        workers: Number of parallel workers

    Returns:
        UniverseScanResult with all detected patterns
    """
    scanner = ConsolidationPatternScanner()
    result = scanner.scan_universe(tickers, start_date, end_date, workers)

    if output_path and result.all_patterns:
        scanner.save_patterns(result.all_patterns, output_path)

    return result


if __name__ == "__main__":
    # Test scanner
    logging.basicConfig(level=logging.INFO)

    # Test with a single ticker
    scanner = ConsolidationPatternScanner(enable_market_cap=True)
    result = scanner.scan_ticker('AAPL', start_date='2023-01-01')

    if result.success:
        print(f"\nFound {result.patterns_found} patterns")
        print(f"Accumulation Detection: {'ENABLED' if ENABLE_ACCUMULATION_DETECTION else 'DISABLED'}")

        # Display market cap info
        if result.market_cap:
            print(f"Market Cap: ${result.market_cap/1e9:.2f}B ({result.market_cap_category})")
        else:
            print("Market Cap: Not available")

        for pattern in result.patterns[:3]:  # Show first 3
            print(f"\n  Pattern: {pattern['pattern_id']}")
            print(f"  - Outcome Class: K{pattern['outcome_class']}")
            print(f"  - Accumulation Count: {pattern['accumulation_count']}")
            print(f"  - Liquidity: ${pattern['liquidity']:,.0f}")
            if 'market_cap' in pattern and pattern['market_cap']:
                print(f"  - Market Cap: ${pattern['market_cap']/1e9:.2f}B ({pattern.get('market_cap_category', 'N/A')})")
    else:
        print(f"Error: {result.error}")
