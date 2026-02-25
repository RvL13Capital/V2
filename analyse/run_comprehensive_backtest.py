"""
Example Script: Run Comprehensive Backtest on AIv3 Pattern Data
Demonstrates complete workflow from data loading to report generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import sys

# Import our new modules
from comprehensive_backtester import ComprehensiveBacktester
from walk_forward_validator import WalkForwardValidator
from extended_performance_metrics import PerformanceMetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_pattern_data(n_patterns: int = 2000,
                                    n_features: int = 20,
                                    start_date: str = '2020-01-01',
                                    end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Generate synthetic pattern data for testing

    In production, replace this with actual pattern data loading
    """

    logger.info(f"Generating {n_patterns} synthetic patterns for testing...")

    np.random.seed(42)

    # Generate random dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days

    dates = [start + timedelta(days=np.random.randint(0, date_range))
             for _ in range(n_patterns)]

    # Generate features (volume-based patterns)
    feature_data = {}

    # Volume features
    for days in [3, 5, 10]:
        feature_data[f'vol_strength_{days}d'] = np.random.random(n_patterns) * 2
        feature_data[f'vol_ratio_{days}d'] = np.random.random(n_patterns) * 3
        feature_data[f'accum_score_{days}d'] = np.random.random(n_patterns) * 100

    # Technical features
    feature_data['bbw'] = np.random.random(n_patterns) * 30
    feature_data['adx'] = np.random.random(n_patterns) * 50
    feature_data['rsi'] = 30 + np.random.random(n_patterns) * 40
    feature_data['obv_trend'] = np.random.randn(n_patterns)

    # Range features
    feature_data['range_contraction'] = np.random.random(n_patterns)
    feature_data['avg_range_ratio'] = np.random.random(n_patterns) * 0.5

    # Consecutive patterns
    feature_data['consec_vol_up'] = np.random.randint(0, 6, n_patterns)
    feature_data['consec_range_down'] = np.random.randint(0, 6, n_patterns)

    # Add more features to reach n_features
    for i in range(len(feature_data), n_features):
        feature_data[f'feature_{i}'] = np.random.randn(n_patterns)

    # Generate realistic outcome distribution
    # Based on AIv3 system expectations:
    # K0: 20%, K1: 25%, K2: 25%, K3: 18%, K4: 7%, K5: 5%
    outcomes = np.random.choice(
        ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
        size=n_patterns,
        p=[0.20, 0.25, 0.25, 0.18, 0.07, 0.05]
    )

    # Create DataFrame
    df = pd.DataFrame(feature_data)
    df['start_date'] = dates
    df['outcome_class'] = outcomes
    df['ticker'] = [f'TICK{i%100:03d}' for i in range(n_patterns)]

    # Add realistic max gains based on outcome class
    gain_mapping = {
        'K4': lambda: np.random.uniform(0.75, 1.50),  # >75% gains
        'K3': lambda: np.random.uniform(0.35, 0.75),  # 35-75% gains
        'K2': lambda: np.random.uniform(0.15, 0.35),  # 15-35% gains
        'K1': lambda: np.random.uniform(0.05, 0.15),  # 5-15% gains
        'K0': lambda: np.random.uniform(0.00, 0.05),  # <5% gains
        'K5': lambda: np.random.uniform(-0.20, -0.05)  # Losses
    }

    df['max_gain'] = df['outcome_class'].apply(lambda x: gain_mapping[x]())
    df['days_to_max'] = np.random.randint(5, 100, n_patterns)

    # Sort by date
    df = df.sort_values('start_date').reset_index(drop=True)

    logger.info(f"Generated data:")
    logger.info(f"  Date range: {df['start_date'].min():%Y-%m-%d} to {df['start_date'].max():%Y-%m-%d}")
    logger.info(f"  Features: {n_features}")
    logger.info(f"  Outcome distribution:\n{df['outcome_class'].value_counts().sort_index()}")

    return df


def load_real_pattern_data(data_path: str) -> pd.DataFrame:
    """
    Load real pattern data from parquet files

    Args:
        data_path: Path to directory containing parquet files or single parquet file

    Returns:
        DataFrame with pattern data
    """

    path = Path(data_path)

    if path.is_dir():
        # Load all parquet files
        parquet_files = list(path.glob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")

        logger.info(f"Loading {len(parquet_files)} parquet files from {data_path}")
        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
                logger.info(f"  Loaded {file.name}: {len(df)} patterns")
            except Exception as e:
                logger.warning(f"  Failed to load {file.name}: {e}")

        if not dfs:
            raise ValueError("No valid parquet files could be loaded")

        df = pd.concat(dfs, ignore_index=True)

    elif path.suffix == '.parquet':
        # Load single file
        logger.info(f"Loading {data_path}")
        df = pd.read_parquet(data_path)

    else:
        raise ValueError(f"Invalid data path: {data_path}")

    logger.info(f"Loaded {len(df)} patterns from {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'} tickers")

    return df


def main():
    """Main execution function"""

    logger.info("="*100)
    logger.info("COMPREHENSIVE BACKTEST - AIv3 PATTERN DETECTION SYSTEM")
    logger.info("="*100)
    logger.info("")

    # Configuration
    USE_REAL_DATA = False  # Set to True to use real data
    REAL_DATA_PATH = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\data\raw'
    OUTPUT_DIR = 'output/comprehensive_backtest'

    # Load data
    if USE_REAL_DATA and Path(REAL_DATA_PATH).exists():
        logger.info("Loading real pattern data...")
        try:
            df = load_real_pattern_data(REAL_DATA_PATH)
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            logger.info("Falling back to synthetic data...")
            df = generate_synthetic_pattern_data(n_patterns=2000)
    else:
        logger.info("Using synthetic data for demonstration...")
        df = generate_synthetic_pattern_data(n_patterns=2000)

    # Identify feature columns (exclude metadata)
    exclude_cols = ['start_date', 'outcome_class', 'ticker', 'max_gain', 'days_to_max']
    feature_columns = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"\nFeatures selected: {len(feature_columns)}")
    logger.info(f"Features: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}")
    logger.info("")

    # Initialize backtester
    logger.info("Initializing Comprehensive Backtester...")
    backtester = ComprehensiveBacktester(
        initial_train_years=2.0,      # 2 years initial training
        retrain_frequency_days=90,    # Retrain every 90 days (quarterly)
        test_window_days=100,         # 100-day test windows
        min_train_samples=500,        # Minimum 500 patterns for training
        risk_free_rate=0.02           # 2% annual risk-free rate
    )

    # Model configuration
    model_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }

    logger.info(f"Model: GradientBoostingClassifier")
    logger.info(f"Model parameters: {model_params}")
    logger.info("")

    # Run comprehensive backtest
    try:
        results = backtester.run_comprehensive_backtest(
            df=df,
            model_class=GradientBoostingClassifier,
            feature_columns=feature_columns,
            target_column='outcome_class',
            date_column='start_date',
            model_params=model_params
        )

        logger.info("\n" + "="*100)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("="*100)

        # Generate and print report
        logger.info("\nGenerating comprehensive report...")
        report = backtester.generate_publication_report()
        print("\n" + report)

        # Save results
        logger.info(f"\nSaving results to {OUTPUT_DIR}...")
        backtester.save_results(output_dir=OUTPUT_DIR)

        # Print summary
        overall = results['overall_metrics']
        logger.info("\n" + "="*100)
        logger.info("SUMMARY")
        logger.info("="*100)
        logger.info(f"✓ Win Rate: {overall['win_rate']:.1%}")
        logger.info(f"✓ Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        logger.info(f"✓ Profit Factor: {overall['profit_factor']:.2f}")
        logger.info(f"✓ Max Drawdown: {overall['max_drawdown']:.1%}")
        logger.info(f"✓ Total Return: {overall['total_return']:+.1%}")
        logger.info(f"✓ K3+K4 Hit Rate: {(overall['k3_strong_rate'] + overall['k4_exceptional_rate']):.1%}")
        logger.info("="*100)

        logger.info("\n✓ All results saved successfully!")
        logger.info(f"✓ Check {OUTPUT_DIR}/ for detailed reports")

        return results

    except Exception as e:
        logger.error(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print("\n" + "="*100)
        print("BACKTEST COMPLETE!")
        print("="*100)
        print("\nNext steps:")
        print("1. Review the comprehensive report in output/comprehensive_backtest/")
        print("2. Analyze per-window performance for consistency")
        print("3. If metrics are satisfactory, proceed to paper trading")
        print("4. Monitor performance and retrain model quarterly")
        print("="*100)
    else:
        print("\n✗ Backtest failed. Check logs for details.")
        sys.exit(1)
