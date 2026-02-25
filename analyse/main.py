"""
AIv3 System - Main Entry Point
Unified command-line interface for all analysis operations
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Import refactored modules
from core import get_config, get_data_loader, UnifiedPatternDetector

# Optional PDF generation - handle missing dependencies gracefully
try:
    from visualization.pdf_generator import UnifiedPDFGenerator
    PDF_AVAILABLE = True
except ImportError as e:
    PDF_AVAILABLE = False
    print(f"Warning: PDF generation not available. Install with: pip install reportlab kaleido")
    print(f"  Missing: {e}")
    UnifiedPDFGenerator = None


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def cmd_detect(args):
    """Pattern detection command"""
    logging.info("Starting pattern detection...")

    detector = UnifiedPatternDetector()
    data_loader = get_data_loader()

    # Get ticker list
    if args.tickers == "ALL":
        tickers = data_loader.list_available_tickers()[:args.limit]
    else:
        tickers = args.tickers.split(',')

    logging.info(f"Processing {len(tickers)} tickers...")

    # Detect patterns
    all_patterns = []
    for ticker in tickers:
        patterns = detector.detect_patterns(ticker)
        all_patterns.extend(patterns)

    logging.info(f"Found {len(all_patterns)} patterns")

    # Convert to DataFrame
    df = detector.patterns_to_dataframe(all_patterns)

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"patterns_{timestamp}.parquet"

    saved_path = data_loader.save_results(df, output_file)
    logging.info(f"Results saved to {saved_path if saved_path else output_file}")

    # Generate report if requested
    if args.report:
        if PDF_AVAILABLE:
            pdf_gen = UnifiedPDFGenerator()
            report_path = pdf_gen.generate_pattern_report(df)
            logging.info(f"Report generated: {report_path}")
        else:
            logging.warning("PDF generation not available. Install reportlab with: pip install reportlab")

    return df


def cmd_analyze(args):
    """Analysis command"""
    logging.info("Starting analysis...")

    data_loader = get_data_loader()

    # Load existing patterns
    if args.input:
        patterns_df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
    else:
        patterns_df = data_loader.load_patterns()

    logging.info(f"Loaded {len(patterns_df)} patterns")

    # Handle column name variations between old and new pattern files
    duration_col = 'pattern_duration_days' if 'pattern_duration_days' in patterns_df.columns else 'duration_days'

    # Perform analysis based on type
    if args.type == "statistical":
        # Statistical analysis
        stats = {
            'total_patterns': len(patterns_df),
            'avg_duration': patterns_df[duration_col].mean(),
            'avg_max_gain': patterns_df['outcome_max_gain'].mean(),
            'outcome_distribution': patterns_df['outcome_class'].value_counts().to_dict()
        }

        logging.info("Statistical Analysis Results:")
        for key, value in stats.items():
            logging.info(f"  {key}: {value}")

    elif args.type == "performance":
        # Performance analysis
        explosive = patterns_df[patterns_df['outcome_class'] == 'EXPLOSIVE (40%+)']
        logging.info(f"Explosive patterns: {len(explosive)} ({len(explosive) / len(patterns_df) * 100:.1f}%)")

        top_performers = patterns_df.nlargest(10, 'outcome_max_gain')
        logging.info("\nTop 10 performers:")
        for _, row in top_performers.iterrows():
            logging.info(f"  {row['ticker']}: {row['outcome_max_gain']:.1f}% gain")

    elif args.type == "quality":
        # Quality analysis based on outcome performance
        # Define quality based on successful outcomes (20%+ gains)
        successful_outcomes = ['MODERATE (10-20%)', 'STRONG (20-40%)', 'EXPLOSIVE (40%+)']
        high_quality = patterns_df[patterns_df['outcome_class'].isin(successful_outcomes)]
        logging.info(f"High quality patterns (10%+ gains): {len(high_quality)} ({len(high_quality) / len(patterns_df) * 100:.1f}%)")

        # Analyze pattern characteristics of high quality patterns
        if len(high_quality) > 0:
            logging.info("\nHigh quality pattern characteristics:")
            logging.info(f"  Avg duration: {high_quality[duration_col].mean():.1f} days")
            logging.info(f"  Avg BBW: {high_quality['avg_bbw'].mean():.3f}")
            logging.info(f"  Avg volume ratio: {high_quality['avg_volume_ratio'].mean():.3f}")
            logging.info(f"  Avg range ratio: {high_quality['avg_range_ratio'].mean():.3f}")
            logging.info(f"  Avg max gain: {high_quality['outcome_max_gain'].mean():.1f}%")

        quality_outcomes = high_quality['outcome_class'].value_counts()
        logging.info("\nHigh quality pattern outcome distribution:")
        for outcome, count in quality_outcomes.items():
            pct = count / len(high_quality) * 100
            logging.info(f"  {outcome}: {count} ({pct:.1f}%)")

    # Generate report if requested
    if args.report:
        if PDF_AVAILABLE:
            pdf_gen = UnifiedPDFGenerator()
            report_path = pdf_gen.generate_pattern_report(patterns_df, title=f"{args.type.title()} Analysis Report")
            logging.info(f"Report generated: {report_path}")
        else:
            logging.warning("PDF generation not available. Install reportlab with: pip install reportlab")


def cmd_backtest(args):
    """Backtesting command"""
    logging.info("Starting backtest...")

    # Load patterns
    data_loader = get_data_loader()

    if args.input:
        patterns_df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
    else:
        patterns_df = data_loader.load_patterns()

    logging.info(f"Loaded {len(patterns_df)} patterns for backtesting")

    # Filter by date range
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        patterns_df = patterns_df[patterns_df['pattern_start_date'] >= start_date]

    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        patterns_df = patterns_df[patterns_df['pattern_end_date'] <= end_date]

    logging.info(f"Filtered to {len(patterns_df)} patterns in date range")

    # Calculate backtest metrics
    results = {
        'total_patterns': len(patterns_df),
        'successful_breakouts': len(patterns_df[patterns_df['breakout_occurred'] == True]),
        'avg_days_to_breakout': patterns_df['days_to_breakout'].mean(),
        'win_rate': len(patterns_df[patterns_df['outcome_max_gain'] > 0]) / len(patterns_df) * 100,
        'avg_win': patterns_df[patterns_df['outcome_max_gain'] > 0]['outcome_max_gain'].mean(),
        'avg_loss': patterns_df[patterns_df['outcome_max_gain'] < 0]['outcome_max_gain'].mean(),
    }

    # Calculate expected value
    config = get_config()
    total_value = 0
    for _, pattern in patterns_df.iterrows():
        outcome_class = pattern.get('outcome_class', 'K0_STAGNANT')
        value = config.outcome.get_value(outcome_class)
        total_value += value

    results['total_strategic_value'] = total_value
    results['avg_strategic_value'] = total_value / len(patterns_df) if len(patterns_df) > 0 else 0

    logging.info("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.2f}")
        else:
            logging.info(f"  {key}: {value}")

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {args.output}")


def cmd_report(args):
    """Report generation command"""
    if not PDF_AVAILABLE:
        logging.error("PDF generation not available. Please install required dependencies:")
        logging.error("  pip install reportlab kaleido")
        sys.exit(1)

    logging.info("Generating report...")

    data_loader = get_data_loader()

    # Load data
    if args.input:
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        elif args.input.endswith('.json'):
            import json
            with open(args.input, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(args.input)
    else:
        df = data_loader.load_patterns()

    logging.info(f"Loaded {len(df)} records")

    # Generate report
    pdf_gen = UnifiedPDFGenerator(output_dir=args.output_dir)

    if args.type == "pattern":
        report_path = pdf_gen.generate_pattern_report(df, title=args.title or "Pattern Analysis Report")
    elif args.type == "backtest":
        # For backtest reports, we'd need backtest results dict
        logging.warning("Backtest report requires backtest results. Use pattern report instead.")
        report_path = pdf_gen.generate_pattern_report(df, title=args.title or "Backtest Analysis")
    else:
        report_path = pdf_gen.generate_pattern_report(df, title=args.title or "Analysis Report")

    logging.info(f"Report generated: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AIv3 System - Consolidation Pattern Detection and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--config', help='Configuration file path')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Pattern detection command
    detect_parser = subparsers.add_parser('detect', help='Detect consolidation patterns')
    detect_parser.add_argument('--tickers', default='ALL', help='Comma-separated tickers or ALL')
    detect_parser.add_argument('--limit', type=int, default=100, help='Limit number of tickers')
    detect_parser.add_argument('--output', help='Output file path')
    detect_parser.add_argument('--report', action='store_true', help='Generate PDF report')

    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze patterns')
    analyze_parser.add_argument('--input', help='Input pattern file')
    analyze_parser.add_argument('--type', choices=['statistical', 'performance', 'quality'], default='statistical',
                                 help='Analysis type')
    analyze_parser.add_argument('--report', action='store_true', help='Generate PDF report')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest patterns')
    backtest_parser.add_argument('--input', help='Input pattern file')
    backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--output', help='Output results file')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--input', required=True, help='Input data file')
    report_parser.add_argument('--type', choices=['pattern', 'backtest', 'analysis'], default='pattern',
                               help='Report type')
    report_parser.add_argument('--title', help='Report title')
    report_parser.add_argument('--output-dir', default='./reports', help='Output directory')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Load custom config if provided
    if args.config:
        config = get_config()
        # Load from file logic here

    # Execute command
    if args.command == 'detect':
        cmd_detect(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()