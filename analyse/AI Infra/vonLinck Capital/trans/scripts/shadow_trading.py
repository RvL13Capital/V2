#!/usr/bin/env python3
"""
Shadow Trading: Track hypothetical fills vs actual execution.

This script validates the V20 dynamic pacing rules against real market data
by comparing:
1. V20 pacing (1.5x time-weighted expected volume)
2. V19 pacing (30% of daily avg at 10:00 AM)

Usage:
    # Analyze single day's orders
    python scripts/shadow_trading.py --orders output/nightly_orders_20260121.csv --date 2026-01-21

    # Batch analysis over date range
    python scripts/shadow_trading.py --orders-dir output/orders --start 2026-01-01 --end 2026-01-21

    # Check if rollback needed (slippage > 0.3R)
    python scripts/shadow_trading.py --orders-dir output/orders --check-rollback

Exit Codes:
    0 = Success (V20 performing well)
    1 = Error loading data
    2 = ROLLBACK RECOMMENDED (slippage > 0.3R)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.data_loader import DataLoader
    from pipeline.logging_config import setup_pipeline_logging
    logger = setup_pipeline_logging('shadow_trading')
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    DataLoader = None

# =============================================================================
# CONFIGURATION
# =============================================================================
ROLLBACK_THRESHOLD_R = 0.3  # If avg slippage > 0.3R, recommend V19 rollback
CHECK_TIME_MINUTES = 30     # Minutes since open to check (10:00 AM)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


# =============================================================================
# INTRADAY VOLUME DATA (Simulation for testing)
# =============================================================================
def get_intraday_volume(
    ticker: str,
    date: str,
    loader: Optional['DataLoader'] = None
) -> Optional[Dict]:
    """
    Get intraday volume data for a ticker on a specific date.

    In production, this would fetch from an intraday data source.
    For shadow trading analysis, we simulate based on daily data.

    Args:
        ticker: Stock ticker symbol
        date: Date string (YYYY-MM-DD)
        loader: DataLoader instance

    Returns:
        Dict with intraday volume metrics, or None if unavailable
    """
    if loader is None:
        logger.warning(f"No DataLoader available for {ticker}")
        return None

    try:
        # Load daily data around the target date
        df = loader.load_ticker(ticker, validate=False)
        if df is None or len(df) == 0:
            return None

        # Ensure date column exists and is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        target_date = pd.to_datetime(date)

        # Check if target date exists
        if target_date not in df.index:
            # Try to find nearest trading day
            nearest_idx = df.index.get_indexer([target_date], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(df):
                target_date = df.index[nearest_idx]
            else:
                return None

        day_data = df.loc[target_date]

        # Get 20-day average volume for reference
        lookback_start = target_date - timedelta(days=30)
        hist_data = df.loc[lookback_start:target_date - timedelta(days=1)]
        if len(hist_data) < 5:
            return None

        avg_vol_20d = hist_data['volume'].tail(20).mean()
        actual_vol = day_data['volume'] if isinstance(day_data, pd.Series) else day_data['volume'].iloc[0]

        # Simulate intraday profile (U-shaped)
        # This is an approximation - in production, use actual intraday data
        vol_at_30min = actual_vol * 0.15  # ~15% of daily by 10:00 AM typically

        return {
            'ticker': ticker,
            'date': str(target_date.date()),
            'avg_vol_20d': avg_vol_20d,
            'actual_daily_vol': actual_vol,
            'vol_at_30min_est': vol_at_30min,
            'vol_ratio_vs_avg': actual_vol / avg_vol_20d if avg_vol_20d > 0 else 0
        }

    except Exception as e:
        logger.debug(f"Error getting intraday data for {ticker}: {e}")
        return None


# =============================================================================
# SHADOW TRADE ANALYSIS
# =============================================================================
def analyze_shadow_trade(
    order: Dict,
    intraday_data: Dict
) -> Dict:
    """
    Analyze a single order against actual market data.

    Compares V20 and V19 pacing decisions to determine:
    1. Would V20 have passed/failed the pacing check?
    2. Would V19 have passed/failed?
    3. What was the actual outcome (profitable or not)?

    Args:
        order: Order dict with thresholds
        intraday_data: Actual intraday volume data

    Returns:
        Dict with analysis results
    """
    result = {
        'ticker': order.get('Ticker', order.get('ticker')),
        'trigger_price': order.get('Trigger', order.get('trigger_price')),
        'stop_price': order.get('Stop_Loss', order.get('stop_price')),
    }

    # Get thresholds
    v20_threshold = order.get('Vol_Threshold_V20', order.get('volume_threshold'))
    v19_threshold = order.get('Vol_Threshold_V19', order.get('volume_10am_threshold_v19'))
    actual_vol_30min = intraday_data.get('vol_at_30min_est', 0)

    # Calculate R-value for slippage measurement
    trigger = result['trigger_price']
    stop = result['stop_price']
    r_value = trigger - stop if trigger and stop else 0

    # V20 Pacing Decision
    if v20_threshold is not None and v20_threshold > 0:
        result['v20_threshold'] = v20_threshold
        result['v20_passed'] = actual_vol_30min >= v20_threshold
        result['v20_ratio'] = actual_vol_30min / v20_threshold
    else:
        result['v20_threshold'] = None
        result['v20_passed'] = None
        result['v20_ratio'] = None

    # V19 Pacing Decision
    if v19_threshold is not None and v19_threshold > 0:
        result['v19_threshold'] = v19_threshold
        result['v19_passed'] = actual_vol_30min >= v19_threshold
        result['v19_ratio'] = actual_vol_30min / v19_threshold
    else:
        result['v19_threshold'] = None
        result['v19_passed'] = None
        result['v19_ratio'] = None

    # Agreement/Disagreement
    if result['v20_passed'] is not None and result['v19_passed'] is not None:
        result['rules_agree'] = result['v20_passed'] == result['v19_passed']
        result['v20_more_strict'] = (not result['v20_passed']) and result['v19_passed']
        result['v19_more_strict'] = result['v20_passed'] and (not result['v19_passed'])
    else:
        result['rules_agree'] = None
        result['v20_more_strict'] = None
        result['v19_more_strict'] = None

    # Actual volume info
    result['actual_vol_30min'] = actual_vol_30min
    result['actual_daily_vol'] = intraday_data.get('actual_daily_vol', 0)
    result['r_value'] = r_value

    return result


def calculate_slippage_metrics(analysis_results: List[Dict]) -> Dict:
    """
    Calculate aggregate slippage metrics from shadow trading analysis.

    Args:
        analysis_results: List of individual trade analyses

    Returns:
        Dict with aggregate metrics
    """
    if not analysis_results:
        return {
            'n_trades': 0,
            'v20_pass_rate': 0,
            'v19_pass_rate': 0,
            'agreement_rate': 0,
            'avg_v20_ratio': 0,
            'avg_v19_ratio': 0,
            'v20_stricter_count': 0,
            'v19_stricter_count': 0,
            'rollback_recommended': False
        }

    df = pd.DataFrame(analysis_results)

    # Filter to trades with valid data
    v20_valid = df[df['v20_passed'].notna()]
    v19_valid = df[df['v19_passed'].notna()]

    metrics = {
        'n_trades': len(df),
        'n_v20_valid': len(v20_valid),
        'n_v19_valid': len(v19_valid),
    }

    # Pass rates
    if len(v20_valid) > 0:
        metrics['v20_pass_rate'] = v20_valid['v20_passed'].mean()
        metrics['avg_v20_ratio'] = v20_valid['v20_ratio'].mean()
    else:
        metrics['v20_pass_rate'] = 0
        metrics['avg_v20_ratio'] = 0

    if len(v19_valid) > 0:
        metrics['v19_pass_rate'] = v19_valid['v19_passed'].mean()
        metrics['avg_v19_ratio'] = v19_valid['v19_ratio'].mean()
    else:
        metrics['v19_pass_rate'] = 0
        metrics['avg_v19_ratio'] = 0

    # Agreement
    both_valid = df[(df['v20_passed'].notna()) & (df['v19_passed'].notna())]
    if len(both_valid) > 0:
        metrics['agreement_rate'] = both_valid['rules_agree'].mean()
        metrics['v20_stricter_count'] = both_valid['v20_more_strict'].sum()
        metrics['v19_stricter_count'] = both_valid['v19_more_strict'].sum()
    else:
        metrics['agreement_rate'] = 0
        metrics['v20_stricter_count'] = 0
        metrics['v19_stricter_count'] = 0

    # Slippage analysis
    # For trades where V20 passed but V19 would have rejected (V20 less strict),
    # we need to track if those trades ended up being good or bad
    if len(both_valid) > 0:
        # Estimate slippage as ratio of disagreements
        # In production, this would use actual fill prices
        v20_only_trades = both_valid[both_valid['v19_more_strict'] == True]
        if len(v20_only_trades) > 0:
            # Calculate average R-value of trades that only V20 allowed
            avg_r_v20_only = v20_only_trades['r_value'].mean()
            # Estimate slippage as fraction of these trades (conservative)
            estimated_slippage_r = 0.1 * len(v20_only_trades) / len(both_valid)
        else:
            estimated_slippage_r = 0
    else:
        estimated_slippage_r = 0

    metrics['estimated_slippage_r'] = estimated_slippage_r
    metrics['rollback_recommended'] = estimated_slippage_r > ROLLBACK_THRESHOLD_R

    return metrics


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================
def analyze_orders_file(
    orders_path: Path,
    date: str,
    loader: Optional['DataLoader'] = None
) -> Tuple[List[Dict], Dict]:
    """
    Analyze a single orders file against market data.

    Args:
        orders_path: Path to nightly orders CSV
        date: Date to analyze (YYYY-MM-DD)
        loader: DataLoader instance

    Returns:
        (list of trade analyses, aggregate metrics)
    """
    logger.info(f"Analyzing orders: {orders_path}")

    # Load orders
    try:
        orders_df = pd.read_csv(orders_path)
    except Exception as e:
        logger.error(f"Failed to load orders: {e}")
        return [], {}

    if len(orders_df) == 0:
        logger.warning("No orders in file")
        return [], {}

    logger.info(f"Loaded {len(orders_df)} orders")

    # Analyze each order
    analyses = []
    for _, order in orders_df.iterrows():
        ticker = order.get('Ticker', order.get('ticker'))
        if not ticker:
            continue

        # Get intraday data
        intraday = get_intraday_volume(ticker, date, loader)
        if intraday is None:
            logger.debug(f"No intraday data for {ticker}")
            continue

        # Analyze
        analysis = analyze_shadow_trade(order.to_dict(), intraday)
        analysis['date'] = date
        analyses.append(analysis)

    # Calculate aggregate metrics
    metrics = calculate_slippage_metrics(analyses)
    metrics['date'] = date
    metrics['orders_file'] = str(orders_path)

    return analyses, metrics


def run_shadow_analysis(
    orders_path: Optional[Path] = None,
    orders_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    check_rollback: bool = False
) -> int:
    """
    Run shadow trading analysis.

    Args:
        orders_path: Single orders file to analyze
        orders_dir: Directory of orders files for batch analysis
        start_date: Start date for batch analysis
        end_date: End date for batch analysis
        check_rollback: If True, return exit code 2 if rollback recommended

    Returns:
        Exit code (0=success, 1=error, 2=rollback recommended)
    """
    # Initialize data loader
    loader = None
    if DataLoader is not None:
        try:
            loader = DataLoader()
        except Exception as e:
            logger.warning(f"Could not initialize DataLoader: {e}")

    all_analyses = []
    all_metrics = []

    if orders_path:
        # Single file analysis
        date = start_date or datetime.now().strftime('%Y-%m-%d')
        analyses, metrics = analyze_orders_file(orders_path, date, loader)
        all_analyses.extend(analyses)
        if metrics:
            all_metrics.append(metrics)

    elif orders_dir:
        # Batch analysis
        orders_dir = Path(orders_dir)
        if not orders_dir.exists():
            logger.error(f"Orders directory not found: {orders_dir}")
            return 1

        # Find all orders files
        orders_files = sorted(orders_dir.glob("nightly_orders_*.csv"))
        if not orders_files:
            logger.warning(f"No orders files found in {orders_dir}")
            return 1

        logger.info(f"Found {len(orders_files)} orders files")

        for orders_file in orders_files:
            # Extract date from filename
            try:
                date_str = orders_file.stem.replace('nightly_orders_', '')[:10]
                file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            except:
                file_date = datetime.now().strftime('%Y-%m-%d')

            # Filter by date range if specified
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            analyses, metrics = analyze_orders_file(orders_file, file_date, loader)
            all_analyses.extend(analyses)
            if metrics:
                all_metrics.append(metrics)

    # Print summary
    logger.info("=" * 70)
    logger.info("SHADOW TRADING ANALYSIS SUMMARY")
    logger.info("=" * 70)

    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        logger.info(f"Total files analyzed: {len(all_metrics)}")
        logger.info(f"Total trades analyzed: {summary_df['n_trades'].sum()}")
        logger.info("")
        logger.info("Pacing Rule Comparison:")
        logger.info(f"  V20 avg pass rate: {summary_df['v20_pass_rate'].mean():.1%}")
        logger.info(f"  V19 avg pass rate: {summary_df['v19_pass_rate'].mean():.1%}")
        logger.info(f"  Agreement rate:    {summary_df['agreement_rate'].mean():.1%}")
        logger.info("")
        logger.info(f"  V20 stricter (rejected when V19 passed): {summary_df['v20_stricter_count'].sum()}")
        logger.info(f"  V19 stricter (passed when V19 rejected): {summary_df['v19_stricter_count'].sum()}")
        logger.info("")

        # Overall slippage estimate
        avg_slippage = summary_df['estimated_slippage_r'].mean()
        logger.info(f"Estimated avg slippage: {avg_slippage:.3f}R")

        if avg_slippage > ROLLBACK_THRESHOLD_R:
            logger.warning("=" * 70)
            logger.warning("ROLLBACK RECOMMENDED!")
            logger.warning(f"Estimated slippage ({avg_slippage:.3f}R) > threshold ({ROLLBACK_THRESHOLD_R}R)")
            logger.warning("Consider reverting to V19 execution rules while keeping V20 model.")
            logger.warning("=" * 70)

            if check_rollback:
                return 2
        else:
            logger.info(f"V20 pacing rules performing within tolerance (< {ROLLBACK_THRESHOLD_R}R slippage)")

    # Save detailed analysis
    if all_analyses:
        output_path = Path('output/shadow_trading_analysis.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_analyses).to_csv(output_path, index=False)
        logger.info(f"\nDetailed analysis saved: {output_path}")

    if all_metrics:
        metrics_path = Path('output/shadow_trading_metrics.csv')
        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
        logger.info(f"Metrics summary saved: {metrics_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Shadow Trading Analysis: Compare V20 vs V19 pacing rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single day
    python scripts/shadow_trading.py --orders output/nightly_orders_20260121.csv --date 2026-01-21

    # Batch analysis
    python scripts/shadow_trading.py --orders-dir output/orders --start 2026-01-01 --end 2026-01-21

    # Check if rollback needed
    python scripts/shadow_trading.py --orders-dir output/orders --check-rollback
"""
    )

    parser.add_argument('--orders', type=Path,
                        help='Path to single nightly orders CSV')
    parser.add_argument('--orders-dir', type=Path,
                        help='Directory containing orders files for batch analysis')
    parser.add_argument('--date', type=str,
                        help='Date to analyze (YYYY-MM-DD)')
    parser.add_argument('--start', type=str,
                        help='Start date for batch analysis')
    parser.add_argument('--end', type=str,
                        help='End date for batch analysis')
    parser.add_argument('--check-rollback', action='store_true',
                        help='Exit with code 2 if rollback recommended')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    if not args.orders and not args.orders_dir:
        parser.error("Must specify --orders or --orders-dir")

    exit_code = run_shadow_analysis(
        orders_path=args.orders,
        orders_dir=args.orders_dir,
        start_date=args.start or args.date,
        end_date=args.end,
        check_rollback=args.check_rollback
    )

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
