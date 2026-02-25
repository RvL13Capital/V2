"""
Enhanced Pattern Export for EDA Analysis
Exports detected patterns with ALL required fields for comprehensive analysis
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

# Import your existing modules
from core.consolidation_detector import ConsolidationTracker
from core.stateful_detector import StatefulPatternDetector
from ml.feature_engineer import FeatureEngineer
from ml.stateful_labeler import StatefulPatternLabeler
from utils.data_loader import GCSDataLoader
from utils.technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPatternExporter:
    """
    Exports patterns with complete feature set for EDA analysis
    """

    def __init__(self, gcs_loader: GCSDataLoader):
        self.gcs_loader = gcs_loader
        self.detector = StatefulPatternDetector()
        self.feature_engineer = FeatureEngineer()
        self.labeler = StatefulPatternLabeler()
        self.ti = TechnicalIndicators()

    def extract_complete_pattern_data(self, ticker: str, pattern: Dict) -> Dict[str, Any]:
        """
        Extract ALL features and metrics for a single pattern

        Args:
            ticker: Stock ticker symbol
            pattern: Pattern dictionary from detector

        Returns:
            Complete pattern data with all fields for EDA
        """
        try:
            # Load price data for the ticker
            df = self.gcs_loader.load_ticker_data(ticker)
            if df is None or df.empty:
                logger.warning(f"No data available for {ticker}")
                return None

            # Get pattern dates
            start_date = pd.to_datetime(pattern['start_date'])
            end_date = pd.to_datetime(pattern.get('end_date', start_date + timedelta(days=30)))

            # Calculate pattern duration
            duration = (end_date - start_date).days

            # Get qualification period data (first 10 days)
            qual_end = start_date + timedelta(days=10)
            qual_mask = (df['date'] >= start_date) & (df['date'] <= qual_end)
            qual_data = df[qual_mask].copy()

            if qual_data.empty:
                logger.warning(f"No qualification data for {ticker} pattern starting {start_date}")
                return None

            # Calculate all technical indicators for qualification period
            qual_data = self.ti.add_all_indicators(qual_data)

            # Extract qualification metrics (the pattern "DNA")
            qualification_metrics = self._extract_qualification_metrics(qual_data)

            # Get pattern period data (full duration)
            pattern_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            pattern_data = df[pattern_mask].copy()

            if pattern_data.empty:
                return None

            # Calculate pattern characteristics
            pattern_chars = self._calculate_pattern_characteristics(pattern_data)

            # Get outcome data (100 days after pattern end)
            outcome_start = end_date
            outcome_end = end_date + timedelta(days=100)
            outcome_mask = (df['date'] > outcome_start) & (df['date'] <= outcome_end)
            outcome_data = df[outcome_mask].copy()

            # Calculate outcome metrics
            if not outcome_data.empty:
                outcome_metrics = self._calculate_outcome_metrics(
                    pattern_data.iloc[-1]['close'],
                    outcome_data
                )
            else:
                outcome_metrics = self._get_default_outcome_metrics()

            # Determine outcome class
            outcome_class = self._classify_outcome(outcome_metrics['max_gain'])

            # Build complete pattern record
            pattern_record = {
                # Primary identification
                'ticker': ticker,
                'pattern_id': f"{ticker}_{start_date.strftime('%Y%m%d')}",

                # Temporal information (CRITICAL for time analysis)
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration': duration,
                'year': start_date.year,
                'month': start_date.month,
                'quarter': f"Q{(start_date.month-1)//3 + 1}",
                'day_of_week': start_date.dayofweek,

                # Pattern boundaries and characteristics
                'boundary_width_pct': pattern_chars['boundary_width_pct'],
                'upper_boundary': pattern_chars['upper_boundary'],
                'lower_boundary': pattern_chars['lower_boundary'],
                'power_boundary': pattern_chars['power_boundary'],
                'pattern_height': pattern_chars['pattern_height'],
                'avg_price': pattern_chars['avg_price'],
                'price_stability': pattern_chars['price_stability'],

                # Outcome information (CRITICAL for performance analysis)
                'outcome_class': outcome_class,
                'max_gain': outcome_metrics['max_gain'],
                'max_loss': outcome_metrics['max_loss'],
                'days_to_max_gain': outcome_metrics['days_to_max_gain'],
                'days_to_max_loss': outcome_metrics['days_to_max_loss'],
                'final_return': outcome_metrics['final_return'],
                'volatility_after': outcome_metrics['volatility_after'],

                # Qualification metrics (ALL features for clustering/correlation)
                'qualification_metrics': qualification_metrics,

                # Pattern quality scores
                'pattern_quality': self._calculate_pattern_quality(
                    qualification_metrics,
                    pattern_chars
                ),

                # Market context
                'market_context': self._get_market_context(start_date),

                # Volume characteristics
                'volume_profile': self._calculate_volume_profile(pattern_data),

                # Additional metadata
                'detection_method': pattern.get('detection_method', 'consolidation_tracker'),
                'confidence_score': pattern.get('confidence', 0.0),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return pattern_record

        except Exception as e:
            logger.error(f"Error extracting pattern data for {ticker}: {e}")
            return None

    def _extract_qualification_metrics(self, qual_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract ALL qualification metrics (the complete feature set)
        """
        metrics = {}

        # Basic price metrics
        metrics['avg_price'] = float(qual_data['close'].mean())
        metrics['price_std'] = float(qual_data['close'].std())
        metrics['price_range'] = float(qual_data['high'].max() - qual_data['low'].min())
        metrics['avg_range'] = float((qual_data['high'] - qual_data['low']).mean())

        # Bollinger Band metrics
        if 'bbw' in qual_data.columns:
            metrics['avg_bbw'] = float(qual_data['bbw'].mean())
            metrics['min_bbw'] = float(qual_data['bbw'].min())
            metrics['max_bbw'] = float(qual_data['bbw'].max())
            metrics['bbw_slope'] = float(np.polyfit(range(len(qual_data)), qual_data['bbw'].values, 1)[0])
            metrics['bbw_stability'] = float(qual_data['bbw'].std() / qual_data['bbw'].mean() if qual_data['bbw'].mean() > 0 else 0)

        # ADX metrics
        if 'adx' in qual_data.columns:
            metrics['avg_adx'] = float(qual_data['adx'].mean())
            metrics['min_adx'] = float(qual_data['adx'].min())
            metrics['max_adx'] = float(qual_data['adx'].max())
            metrics['adx_slope'] = float(np.polyfit(range(len(qual_data)), qual_data['adx'].values, 1)[0])
            metrics['adx_trend_strength'] = float(qual_data['adx'].iloc[-1] - qual_data['adx'].iloc[0])

        # Volume metrics
        if 'volume' in qual_data.columns:
            metrics['avg_volume'] = float(qual_data['volume'].mean())
            metrics['volume_std'] = float(qual_data['volume'].std())
            metrics['volume_consistency'] = float(1 - (qual_data['volume'].std() / qual_data['volume'].mean()) if qual_data['volume'].mean() > 0 else 0)
            metrics['volume_trend'] = float(np.polyfit(range(len(qual_data)), qual_data['volume'].values, 1)[0])

            # Volume relative to average
            if 'volume_sma_20' in qual_data.columns:
                metrics['volume_ratio'] = float(qual_data['volume'].mean() / qual_data['volume_sma_20'].mean() if qual_data['volume_sma_20'].mean() > 0 else 1)
            else:
                metrics['volume_ratio'] = 1.0

        # ATR metrics
        if 'atr' in qual_data.columns:
            metrics['avg_atr'] = float(qual_data['atr'].mean())
            metrics['atr_pct'] = float(qual_data['atr'].mean() / qual_data['close'].mean() * 100)
            metrics['atr_stability'] = float(qual_data['atr'].std() / qual_data['atr'].mean() if qual_data['atr'].mean() > 0 else 0)

        # RSI metrics
        if 'rsi' in qual_data.columns:
            metrics['avg_rsi'] = float(qual_data['rsi'].mean())
            metrics['rsi_range'] = float(qual_data['rsi'].max() - qual_data['rsi'].min())
            metrics['rsi_momentum'] = float(qual_data['rsi'].iloc[-1] - qual_data['rsi'].iloc[0])

        # MACD metrics
        if 'macd' in qual_data.columns and 'macd_signal' in qual_data.columns:
            metrics['avg_macd'] = float(qual_data['macd'].mean())
            metrics['macd_histogram'] = float((qual_data['macd'] - qual_data['macd_signal']).mean())
            metrics['macd_crosses'] = int(((qual_data['macd'] > qual_data['macd_signal']) !=
                                          (qual_data['macd'].shift(1) > qual_data['macd_signal'].shift(1))).sum())

        # Stochastic metrics
        if 'stoch_k' in qual_data.columns and 'stoch_d' in qual_data.columns:
            metrics['avg_stoch_k'] = float(qual_data['stoch_k'].mean())
            metrics['avg_stoch_d'] = float(qual_data['stoch_d'].mean())
            metrics['stoch_divergence'] = float((qual_data['stoch_k'] - qual_data['stoch_d']).mean())

        # Price position within range
        metrics['price_position'] = float((qual_data['close'].iloc[-1] - qual_data['low'].min()) /
                                         (qual_data['high'].max() - qual_data['low'].min())
                                         if (qual_data['high'].max() - qual_data['low'].min()) > 0 else 0.5)

        # Volatility metrics
        returns = qual_data['close'].pct_change().dropna()
        if len(returns) > 0:
            metrics['volatility'] = float(returns.std() * np.sqrt(252))  # Annualized
            metrics['skewness'] = float(returns.skew())
            metrics['kurtosis'] = float(returns.kurtosis())

        # Pattern tightness metrics
        metrics['high_low_ratio'] = float(qual_data['high'].mean() / qual_data['low'].mean() if qual_data['low'].mean() > 0 else 1)
        metrics['close_to_high'] = float((qual_data['close'] / qual_data['high']).mean())
        metrics['close_to_low'] = float((qual_data['close'] / qual_data['low']).mean())

        # Trend metrics
        close_prices = qual_data['close'].values
        if len(close_prices) > 1:
            trend_slope, trend_intercept = np.polyfit(range(len(close_prices)), close_prices, 1)
            metrics['trend_slope'] = float(trend_slope)
            metrics['trend_strength'] = float(np.corrcoef(range(len(close_prices)), close_prices)[0, 1])

        return metrics

    def _calculate_pattern_characteristics(self, pattern_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pattern-specific characteristics
        """
        chars = {}

        # Boundaries
        chars['upper_boundary'] = float(pattern_data['high'].max())
        chars['lower_boundary'] = float(pattern_data['low'].min())
        chars['power_boundary'] = float(chars['upper_boundary'] * 1.005)  # 0.5% above upper

        # Pattern dimensions
        chars['pattern_height'] = float(chars['upper_boundary'] - chars['lower_boundary'])
        chars['avg_price'] = float(pattern_data['close'].mean())
        chars['boundary_width_pct'] = float((chars['pattern_height'] / chars['avg_price']) * 100)

        # Price stability within pattern
        chars['price_stability'] = float(1 - (pattern_data['close'].std() / pattern_data['close'].mean()))

        # Pattern symmetry
        midpoint = (chars['upper_boundary'] + chars['lower_boundary']) / 2
        chars['symmetry'] = float(1 - abs(pattern_data['close'].mean() - midpoint) / (chars['pattern_height'] / 2))

        # Touches of boundaries
        chars['upper_touches'] = int((pattern_data['high'] >= chars['upper_boundary'] * 0.98).sum())
        chars['lower_touches'] = int((pattern_data['low'] <= chars['lower_boundary'] * 1.02).sum())

        return chars

    def _calculate_outcome_metrics(self, entry_price: float, outcome_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive outcome metrics
        """
        metrics = {}

        if outcome_data.empty:
            return self._get_default_outcome_metrics()

        # Calculate returns
        returns = ((outcome_data['high'].values - entry_price) / entry_price) * 100
        losses = ((outcome_data['low'].values - entry_price) / entry_price) * 100

        # Maximum gain/loss
        metrics['max_gain'] = float(returns.max()) if len(returns) > 0 else 0
        metrics['max_loss'] = float(losses.min()) if len(losses) > 0 else 0

        # Days to max gain/loss
        if len(returns) > 0 and metrics['max_gain'] > 0:
            metrics['days_to_max_gain'] = int(np.argmax(returns) + 1)
        else:
            metrics['days_to_max_gain'] = 0

        if len(losses) > 0 and metrics['max_loss'] < 0:
            metrics['days_to_max_loss'] = int(np.argmin(losses) + 1)
        else:
            metrics['days_to_max_loss'] = 0

        # Final return (at end of outcome period)
        metrics['final_return'] = float(((outcome_data.iloc[-1]['close'] - entry_price) / entry_price) * 100)

        # Volatility after pattern
        post_returns = outcome_data['close'].pct_change().dropna()
        metrics['volatility_after'] = float(post_returns.std() * np.sqrt(252)) if len(post_returns) > 0 else 0

        # Gains at different time horizons
        for days in [5, 10, 20, 30, 50, 70, 100]:
            if len(outcome_data) >= days:
                metrics[f'gain_{days}d'] = float(((outcome_data.iloc[days-1]['high'] - entry_price) / entry_price) * 100)
                metrics[f'loss_{days}d'] = float(((outcome_data.iloc[days-1]['low'] - entry_price) / entry_price) * 100)

        return metrics

    def _get_default_outcome_metrics(self) -> Dict[str, float]:
        """
        Return default outcome metrics when no outcome data available
        """
        return {
            'max_gain': 0.0,
            'max_loss': 0.0,
            'days_to_max_gain': 0,
            'days_to_max_loss': 0,
            'final_return': 0.0,
            'volatility_after': 0.0
        }

    def _classify_outcome(self, max_gain: float) -> str:
        """
        Classify outcome based on maximum gain achieved
        """
        if max_gain >= 75:
            return 'K4'  # Exceptional
        elif max_gain >= 35:
            return 'K3'  # Strong
        elif max_gain >= 15:
            return 'K2'  # Quality
        elif max_gain >= 5:
            return 'K1'  # Minimal
        elif max_gain >= -5:
            return 'K0'  # Stagnant
        else:
            return 'K5'  # Failed

    def _calculate_pattern_quality(self, qual_metrics: Dict, pattern_chars: Dict) -> float:
        """
        Calculate overall pattern quality score (0-100)
        """
        score = 50.0  # Base score

        # Reward tight patterns
        if pattern_chars.get('boundary_width_pct', 10) < 5:
            score += 10
        elif pattern_chars.get('boundary_width_pct', 10) < 8:
            score += 5

        # Reward low ADX (non-trending)
        if qual_metrics.get('avg_adx', 30) < 25:
            score += 10
        elif qual_metrics.get('avg_adx', 30) < 32:
            score += 5

        # Reward low BBW (low volatility)
        if qual_metrics.get('avg_bbw', 30) < 20:
            score += 10
        elif qual_metrics.get('avg_bbw', 30) < 30:
            score += 5

        # Reward volume consistency
        if qual_metrics.get('volume_consistency', 0) > 0.7:
            score += 5

        # Reward price stability
        if pattern_chars.get('price_stability', 0) > 0.9:
            score += 5

        # Reward pattern symmetry
        if pattern_chars.get('symmetry', 0) > 0.8:
            score += 5

        # Cap at 100
        return min(100.0, max(0.0, score))

    def _get_market_context(self, date: pd.Timestamp) -> Dict[str, Any]:
        """
        Get market context for the pattern date
        """
        context = {}

        # Determine market regime (simplified - you could load actual market data)
        year = date.year
        month = date.month

        # Bull/bear market periods (simplified)
        if year < 2020:
            context['market_regime'] = 'bull'
        elif year == 2020 and month < 4:
            context['market_regime'] = 'crash'
        elif year == 2020:
            context['market_regime'] = 'recovery'
        elif year == 2021:
            context['market_regime'] = 'bull'
        elif year == 2022:
            context['market_regime'] = 'bear'
        elif year == 2023:
            context['market_regime'] = 'recovery'
        else:
            context['market_regime'] = 'bull'

        # Seasonality
        if month in [11, 12, 1]:
            context['season'] = 'winter_rally'
        elif month in [5, 6, 7, 8]:
            context['season'] = 'summer_doldrums'
        elif month in [9, 10]:
            context['season'] = 'fall_volatility'
        else:
            context['season'] = 'spring'

        # Day of week effects
        context['week_position'] = 'early' if date.dayofweek < 2 else 'mid' if date.dayofweek < 4 else 'late'

        return context

    def _calculate_volume_profile(self, pattern_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume profile characteristics
        """
        profile = {}

        if 'volume' not in pattern_data.columns:
            return profile

        volumes = pattern_data['volume'].values

        # Basic stats
        profile['total_volume'] = float(volumes.sum())
        profile['avg_volume'] = float(volumes.mean())
        profile['volume_variance'] = float(volumes.var())

        # Volume distribution
        profile['volume_skew'] = float(pd.Series(volumes).skew())
        profile['volume_kurtosis'] = float(pd.Series(volumes).kurtosis())

        # Volume trend
        if len(volumes) > 1:
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            profile['volume_trend'] = float(volume_trend)
            profile['volume_acceleration'] = float(np.polyfit(range(len(volumes)), volumes, 2)[0])

        # Relative volume
        profile['first_half_volume'] = float(volumes[:len(volumes)//2].mean())
        profile['second_half_volume'] = float(volumes[len(volumes)//2:].mean())
        profile['volume_shift'] = float((profile['second_half_volume'] - profile['first_half_volume']) /
                                       profile['first_half_volume'] if profile['first_half_volume'] > 0 else 0)

        return profile

    def export_patterns(self, tickers: List[str], output_file: str = None,
                       start_date: str = None, end_date: str = None) -> str:
        """
        Export patterns for specified tickers with all features

        Args:
            tickers: List of tickers to analyze
            output_file: Output JSON filename
            start_date: Start date for pattern detection
            end_date: End date for pattern detection

        Returns:
            Path to exported JSON file
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"enhanced_patterns_eda_{timestamp}.json"

        all_patterns = []

        logger.info(f"Processing {len(tickers)} tickers for enhanced pattern export...")

        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing ticker {i + 1}/{len(tickers)}: {ticker}")

            try:
                # Load data
                df = self.gcs_loader.load_ticker_data(ticker)
                if df is None or df.empty:
                    continue

                # Apply date filters if provided
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]

                # Detect patterns
                patterns = self.detector.detect_patterns(ticker, df)

                # Extract complete data for each pattern
                for pattern in patterns:
                    pattern_data = self.extract_complete_pattern_data(ticker, pattern)
                    if pattern_data:
                        all_patterns.append(pattern_data)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        # Sort patterns by date
        all_patterns.sort(key=lambda x: x['start_date'])

        # Add summary statistics
        output = {
            'metadata': {
                'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_patterns': len(all_patterns),
                'total_tickers': len(set(p['ticker'] for p in all_patterns)),
                'date_range': {
                    'start': min(p['start_date'] for p in all_patterns) if all_patterns else None,
                    'end': max(p['end_date'] for p in all_patterns) if all_patterns else None
                },
                'outcome_distribution': self._calculate_outcome_distribution(all_patterns),
                'feature_completeness': self._check_feature_completeness(all_patterns)
            },
            'patterns': all_patterns
        }

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Exported {len(all_patterns)} patterns to {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("ENHANCED PATTERN EXPORT COMPLETE")
        print("="*60)
        print(f"Total Patterns: {len(all_patterns)}")
        print(f"Output File: {output_file}")
        print(f"File Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

        if all_patterns:
            print("\nOutcome Distribution:")
            for outcome_class, count in output['metadata']['outcome_distribution'].items():
                print(f"  {outcome_class}: {count} patterns")

        print("="*60)

        return output_file

    def _calculate_outcome_distribution(self, patterns: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of outcome classes"""
        distribution = {}
        for pattern in patterns:
            outcome = pattern.get('outcome_class', 'Unknown')
            distribution[outcome] = distribution.get(outcome, 0) + 1
        return distribution

    def _check_feature_completeness(self, patterns: List[Dict]) -> Dict[str, float]:
        """Check what percentage of patterns have each feature"""
        if not patterns:
            return {}

        completeness = {}
        total = len(patterns)

        # Check main fields
        main_fields = ['ticker', 'start_date', 'end_date', 'duration', 'boundary_width_pct',
                      'outcome_class', 'max_gain', 'qualification_metrics']

        for field in main_fields:
            count = sum(1 for p in patterns if field in p and p[field] is not None)
            completeness[field] = (count / total) * 100

        # Check qualification metrics completeness
        if patterns and 'qualification_metrics' in patterns[0]:
            qual_fields = patterns[0]['qualification_metrics'].keys()
            for field in qual_fields:
                count = sum(1 for p in patterns
                          if 'qualification_metrics' in p
                          and field in p['qualification_metrics']
                          and p['qualification_metrics'][field] is not None)
                completeness[f'qual_{field}'] = (count / total) * 100

        return completeness


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Export enhanced patterns for EDA')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--num-tickers', type=int, default=100, help='Number of tickers to analyze')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output filename')

    args = parser.parse_args()

    # Initialize GCS loader
    credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
    gcs_loader = GCSDataLoader(credentials_path)

    # Get tickers
    if args.tickers:
        tickers = args.tickers
    else:
        # Get top N tickers by market cap or activity
        all_tickers = gcs_loader.get_available_tickers()
        tickers = all_tickers[:args.num_tickers]

    # Initialize exporter
    exporter = EnhancedPatternExporter(gcs_loader)

    # Export patterns
    output_file = exporter.export_patterns(
        tickers=tickers,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date
    )

    print(f"\nPatterns exported successfully to: {output_file}")
    print("You can now use this file with eda_tool.py for comprehensive analysis!")


if __name__ == "__main__":
    main()