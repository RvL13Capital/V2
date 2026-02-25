"""
AIv3 System - Integrated Pattern Detection with Volume ML
Combines consolidation pattern detection with volume-based ML predictions
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from analyse modules
sys.path.append(str(Path(__file__).parent.parent))

# Import core pattern detection from parent system
try:
    from core import get_config, get_data_loader, UnifiedPatternDetector
    from core.pattern_detector import Pattern
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Warning: Core pattern detection not available. Using standalone mode.")

# Import local volume ML system
from final_volume_pattern_system import VolumePatternModel, VolumeFeatureEngine, SystemConfig
from enhanced_feature_system import EnhancedFeatures
from model_manager import get_model_manager, ModelMetadata


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


class IntegratedPatternSystem:
    """Integrated system combining pattern detection with volume ML"""

    def __init__(self, use_core_detector=True, load_best_model=True):
        self.logger = logging.getLogger(__name__)
        self.config = SystemConfig()

        # Initialize volume ML components
        self.feature_engine = VolumeFeatureEngine()
        self.enhanced_features = EnhancedFeatures()
        self.model_manager = get_model_manager()

        # Load best saved model if available
        if load_best_model:
            model, metadata = self.model_manager.load_best_model()
            if model is not None:
                self.volume_model = model
                self.model_metadata = metadata
                self.logger.info(f"Loaded best model v{metadata.model_version} " +
                               f"(Accuracy: {metadata.accuracy:.3f}, " +
                               f"Win Rate High: {metadata.win_rate_high_conf:.1%})")
            else:
                self.logger.warning("No saved model found, initializing new model")
                self.volume_model = VolumePatternModel(self.config)
                self.model_metadata = None
        else:
            self.volume_model = VolumePatternModel(self.config)
            self.model_metadata = None

        # Initialize core pattern detector if available
        self.use_core_detector = use_core_detector and CORE_AVAILABLE
        if self.use_core_detector:
            self.pattern_detector = UnifiedPatternDetector()
            self.data_loader = get_data_loader()
        else:
            self.pattern_detector = None
            self.data_loader = None

    def detect_patterns(self, ticker: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Detect consolidation patterns using core detector or simplified method

        Args:
            ticker: Stock ticker symbol
            df: Optional dataframe with OHLCV data

        Returns:
            DataFrame with detected patterns
        """
        if self.use_core_detector and self.pattern_detector:
            # Use core pattern detector
            patterns = self.pattern_detector.detect_patterns(ticker)
            return self.pattern_detector.patterns_to_dataframe(patterns)
        else:
            # Use simplified pattern detection
            if df is None:
                raise ValueError("DataFrame required when core detector not available")

            return self._simple_pattern_detection(ticker, df)

    def _simple_pattern_detection(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified consolidation pattern detection"""
        patterns = []

        # Calculate indicators
        df['bbw'] = self._calculate_bbw(df)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['range'] = (df['high'] - df['low']) / df['close']

        # Find consolidation periods (BBW < 30th percentile)
        bbw_threshold = df['bbw'].quantile(0.30)
        df['consolidating'] = df['bbw'] < bbw_threshold

        # Group consecutive consolidation days
        df['group'] = (df['consolidating'] != df['consolidating'].shift()).cumsum()

        for group_id, group_df in df[df['consolidating']].groupby('group'):
            if len(group_df) >= 10:  # Minimum 10 days
                pattern = {
                    'ticker': ticker,
                    'start_date': group_df.index[0],
                    'end_date': group_df.index[-1],
                    'duration_days': len(group_df),
                    'avg_bbw': group_df['bbw'].mean(),
                    'avg_volume_ratio': group_df['volume_ratio'].mean(),
                    'upper_boundary': group_df['high'].max(),
                    'lower_boundary': group_df['low'].min(),
                    'power_boundary': group_df['high'].max() * 1.005
                }
                patterns.append(pattern)

        return pd.DataFrame(patterns)

    def _calculate_bbw(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band Width"""
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return ((upper - lower) / sma) * 100

    def predict_patterns(self, patterns_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add ML predictions to detected patterns

        Args:
            patterns_df: DataFrame with detected patterns
            price_data: Full OHLCV data

        Returns:
            DataFrame with patterns and predictions
        """
        predictions = []

        for idx, pattern in patterns_df.iterrows():
            # Get data around pattern
            start_idx = price_data.index.get_loc(pattern['start_date'])
            end_idx = price_data.index.get_loc(pattern['end_date'])

            # Extract features for the pattern period
            pattern_data = price_data.iloc[max(0, start_idx-20):end_idx+1]

            # Calculate volume features
            volume_features = self.feature_engine.extract_features(pattern_data)

            # Get enhanced features if available
            if hasattr(self, 'enhanced_features'):
                enhanced_feats = self.enhanced_features.calculate_features(pattern_data)
                features = {**volume_features, **enhanced_feats}
            else:
                features = volume_features

            # Make prediction if model is trained
            if hasattr(self.volume_model, 'model') and self.volume_model.model is not None:
                try:
                    prob = self.volume_model.predict_proba(pd.DataFrame([features]))[0, 1]
                    signal = self.volume_model.generate_signal(prob)

                    pattern['ml_probability'] = prob
                    pattern['ml_signal'] = signal
                    pattern['ml_confidence'] = self._get_confidence_level(prob)
                except Exception as e:
                    self.logger.warning(f"Prediction failed for pattern {idx}: {e}")
                    pattern['ml_probability'] = None
                    pattern['ml_signal'] = 'NO_SIGNAL'
                    pattern['ml_confidence'] = 'NONE'
            else:
                pattern['ml_probability'] = None
                pattern['ml_signal'] = 'MODEL_NOT_TRAINED'
                pattern['ml_confidence'] = 'NONE'

            predictions.append(pattern)

        return pd.DataFrame(predictions)

    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.30:
            return 'HIGH'
        elif probability >= 0.20:
            return 'MODERATE'
        elif probability >= 0.10:
            return 'LOW'
        else:
            return 'NONE'

    def train_model(self, training_data: pd.DataFrame) -> None:
        """Train the volume ML model on historical patterns"""
        self.logger.info("Training volume pattern model...")

        # Prepare training features
        X = []
        y = []

        for idx, row in training_data.iterrows():
            if 'outcome_class' in row and pd.notna(row['outcome_class']):
                # Extract features for this pattern
                features = self.feature_engine.extract_pattern_features(row)
                X.append(features)

                # Convert outcome to binary (K3/K4 = 1, others = 0)
                y.append(1 if row['outcome_class'] in ['K3', 'K4'] else 0)

        if len(X) > 100:  # Minimum training samples
            X_df = pd.DataFrame(X)
            self.volume_model.train(X_df, np.array(y))
            self.logger.info(f"Model trained on {len(X)} patterns")
        else:
            self.logger.warning(f"Insufficient training data: {len(X)} patterns")


def cmd_detect(args):
    """Pattern detection command"""
    logging.info("Starting integrated pattern detection...")

    system = IntegratedPatternSystem(use_core_detector=True)

    # Load data
    if args.data_path:
        # Load from specified path
        data_files = list(Path(args.data_path).glob('*.parquet'))
        all_data = pd.concat([pd.read_parquet(f) for f in data_files])
    else:
        # Use core data loader if available
        if system.data_loader:
            all_data = system.data_loader.load_all_data()
        else:
            logging.error("No data path specified and core data loader not available")
            return None

    # Get ticker list
    if args.tickers == "ALL":
        tickers = all_data['symbol'].unique()[:args.limit] if 'symbol' in all_data.columns else []
    else:
        tickers = args.tickers.split(',')

    logging.info(f"Processing {len(tickers)} tickers...")

    # Detect patterns
    all_patterns = []
    for ticker in tickers:
        ticker_data = all_data[all_data['symbol'] == ticker] if 'symbol' in all_data.columns else all_data

        # Detect consolidation patterns
        patterns = system.detect_patterns(ticker, ticker_data)

        if not patterns.empty:
            # Add ML predictions
            patterns_with_ml = system.predict_patterns(patterns, ticker_data)
            all_patterns.append(patterns_with_ml)

    if all_patterns:
        result_df = pd.concat(all_patterns, ignore_index=True)
        logging.info(f"Found {len(result_df)} patterns")

        # Filter to high-confidence signals if requested
        if args.high_confidence:
            result_df = result_df[result_df['ml_confidence'].isin(['HIGH', 'MODERATE'])]
            logging.info(f"Filtered to {len(result_df)} high-confidence patterns")

        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"integrated_patterns_{timestamp}.parquet"

        result_df.to_parquet(output_file)
        logging.info(f"Results saved to {output_file}")

        # Print summary
        print("\n" + "="*50)
        print("PATTERN DETECTION SUMMARY")
        print("="*50)
        print(f"Total patterns detected: {len(result_df)}")

        if 'ml_confidence' in result_df.columns:
            confidence_counts = result_df['ml_confidence'].value_counts()
            print("\nConfidence Distribution:")
            for conf, count in confidence_counts.items():
                print(f"  {conf}: {count}")

        if 'ml_signal' in result_df.columns:
            signal_counts = result_df['ml_signal'].value_counts()
            print("\nSignal Distribution:")
            for signal, count in signal_counts.items():
                print(f"  {signal}: {count}")

        return result_df
    else:
        logging.warning("No patterns found")
        return pd.DataFrame()


def cmd_train(args):
    """Train ML model on historical data"""
    logging.info("Starting model training...")

    system = IntegratedPatternSystem()

    # Load training data
    if args.training_data:
        training_df = pd.read_parquet(args.training_data)
    else:
        logging.error("Training data path required")
        return

    # Train model
    system.train_model(training_df)

    # Save model if requested
    if args.save_model:
        import joblib
        joblib.dump(system.volume_model, args.save_model)
        logging.info(f"Model saved to {args.save_model}")


def cmd_backtest(args):
    """Run backtesting on historical data"""
    logging.info("Starting backtest...")

    system = IntegratedPatternSystem()

    # Load historical data
    if args.data_path:
        data_files = list(Path(args.data_path).glob('*.parquet'))
        all_data = pd.concat([pd.read_parquet(f) for f in data_files])
    else:
        logging.error("Data path required for backtesting")
        return

    # Split data by date
    split_date = pd.to_datetime(args.split_date) if args.split_date else pd.to_datetime('2023-01-01')
    train_data = all_data[all_data.index < split_date]
    test_data = all_data[all_data.index >= split_date]

    logging.info(f"Training on data before {split_date}")
    logging.info(f"Testing on data after {split_date}")

    # Detect patterns in training data and train model
    # (Implementation would continue here...)

    print("\nBacktest results would be displayed here")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AIv3 Integrated Pattern Detection System')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect patterns')
    detect_parser.add_argument('--tickers', default='ALL', help='Comma-separated tickers or ALL')
    detect_parser.add_argument('--limit', type=int, default=100, help='Limit number of tickers')
    detect_parser.add_argument('--data-path', help='Path to data directory')
    detect_parser.add_argument('--output', help='Output file path')
    detect_parser.add_argument('--high-confidence', action='store_true', help='Only show high confidence patterns')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--training-data', required=True, help='Path to training data')
    train_parser.add_argument('--save-model', help='Path to save trained model')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--data-path', required=True, help='Path to data directory')
    backtest_parser.add_argument('--split-date', default='2023-01-01', help='Train/test split date')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Execute command
    if args.command == 'detect':
        cmd_detect(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()