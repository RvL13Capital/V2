"""
Example: Run 4-Year Full Historical Backtest
=============================================

Demonstrates comprehensive backtesting with full historical pattern detection.

This example:
1. Scans ENTIRE 4-year history (2020-2024) for ALL patterns
2. Does NOT limit to recent patterns only
3. Labels completed patterns with actual outcomes
4. Generates comprehensive performance metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from backtesting import TemporalBacktester, BacktestConfig
from backtesting.performance_metrics import PerformanceMetrics
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_4year_backtest():
    """
    Run comprehensive 4-year backtest

    Scans FULL HISTORY from 2020-01-01 to 2024-01-01
    Finds ALL consolidation patterns across entire period
    """

    # Sample tickers (in production, use larger universe)
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'NFLX', 'AMD', 'INTC'
    ]

    # Configure backtest for FULL 4-year historical scan
    config = BacktestConfig(
        start_date='2020-01-01',  # Scan from here
        end_date='2024-01-01',    # Scan to here
        tickers=sample_tickers,
        min_pattern_duration=10,
        outcome_window=100,       # 100 days to evaluate outcome
        output_dir='output/4year_backtest',
        save_all_patterns=True,   # Save ALL patterns found
        max_workers=4,
        verbose=True
    )

    logger.info("=" * 70)
    logger.info("4-YEAR FULL HISTORICAL BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Date Range: {config.start_date} → {config.end_date}")
    logger.info(f"Tickers: {len(config.tickers)}")
    logger.info(f"This will scan ENTIRE 4-year history for ALL patterns")
    logger.info("=" * 70)

    # Run backtest
    backtester = TemporalBacktester(config)
    results = backtester.run()

    # Calculate performance metrics
    metrics_calc = PerformanceMetrics()

    # Overall statistics
    stats = metrics_calc.calculate_statistics(
        patterns_df=results.all_patterns,
        completed_df=results.completed_patterns,
        active_df=results.active_patterns,
        failed_df=results.failed_patterns
    )

    logger.info("\n" + str(stats))

    # Monthly performance
    if len(results.completed_patterns) > 0:
        monthly_perf = metrics_calc.calculate_monthly_performance(results.completed_patterns)
        logger.info("\n" + "=" * 70)
        logger.info("MONTHLY PERFORMANCE ACROSS FULL HISTORY")
        logger.info("=" * 70)
        logger.info(monthly_perf.to_string())

        # Ticker performance
        ticker_perf = metrics_calc.calculate_ticker_performance(results.completed_patterns, top_n=10)
        logger.info("\n" + "=" * 70)
        logger.info("TOP 10 TICKERS (by Strategic Value)")
        logger.info("=" * 70)
        logger.info(ticker_perf.to_string())

        # Risk metrics
        risk_metrics = metrics_calc.calculate_risk_metrics(results.completed_patterns)
        logger.info("\n" + "=" * 70)
        logger.info("RISK-ADJUSTED METRICS")
        logger.info("=" * 70)
        for key, value in risk_metrics.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # Save results
    backtester.save_results(results)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SAVED TO: " + config.output_dir)
    logger.info("=" * 70)
    logger.info("Files created:")
    logger.info("  - all_patterns_full_history.parquet (ALL patterns found)")
    logger.info("  - completed_patterns_labeled.parquet (patterns with outcomes)")
    logger.info("  - backtest_metrics.json (performance metrics)")
    logger.info("  - backtest_summary.txt (summary report)")

    return results


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║   4-YEAR FULL HISTORICAL BACKTEST EXAMPLE                    ║
║                                                              ║
║   This demonstrates comprehensive pattern detection          ║
║   across ENTIRE historical dataset (not just recent)         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = run_4year_backtest()

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE!")
    print(f"Total Patterns Found: {len(results.all_patterns)}")
    print(f"Completed Patterns: {len(results.completed_patterns)}")
    print(f"Active Patterns: {len(results.active_patterns)}")
    print(f"Failed Patterns: {len(results.failed_patterns)}")
    print("=" * 70)
