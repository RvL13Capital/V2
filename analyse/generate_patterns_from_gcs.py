"""
Pattern Generation from GCS Raw Data
Detects consolidation patterns from historical price data in GCS
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import logging
from typing import Dict, List, Optional, Tuple
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsolidationDetector:
    """Simple consolidation pattern detector for raw price data"""
    
    def __init__(self):
        self.patterns = []
        
    def detect_patterns(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """Detect consolidation patterns in price data"""
        patterns = []
        
        if len(df) < 50:  # Need minimum data
            return patterns
        
        # Calculate indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['volume_avg'] = df['volume'].rolling(20).mean()
        df['daily_range'] = (df['high'] - df['low']) / df['low'] * 100
        df['range_avg'] = df['daily_range'].rolling(20).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        # Calculate ADX (simplified)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()
        df['adx'] = df['atr'].rolling(14).mean() * 100 / df['close']  # Simplified ADX proxy
        
        # Detect consolidation periods
        min_duration = 10
        max_duration = 60
        
        i = 50  # Start after we have enough data for indicators
        while i < len(df) - 100:  # Leave room for outcome evaluation
            
            # Check for consolidation conditions
            window = df.iloc[i:i+min_duration]
            
            if len(window) < min_duration:
                i += 1
                continue
            
            # Consolidation criteria
            avg_bbw = window['bb_width'].mean()
            avg_adx = window['adx'].mean()
            avg_volume_ratio = (window['volume'] / window['volume_avg']).mean()
            avg_range_ratio = (window['daily_range'] / window['range_avg']).mean()
            
            # Check if this qualifies as consolidation
            is_consolidating = (
                avg_bbw < 15 and  # Narrow Bollinger Bands
                avg_adx < 30 and  # Low trend strength
                avg_volume_ratio < 0.8 and  # Below average volume
                avg_range_ratio < 0.8  # Below average daily range
            )
            
            if is_consolidating:
                # Find the full consolidation period
                end_idx = i + min_duration
                
                # Extend while still consolidating
                while end_idx < min(i + max_duration, len(df) - 100):
                    next_day = df.iloc[end_idx]
                    
                    # Check if still consolidating
                    if (next_day['bb_width'] < 20 and
                        next_day['adx'] < 35 and
                        next_day['volume'] < next_day['volume_avg'] * 1.2):
                        end_idx += 1
                    else:
                        break
                
                # Get consolidation period
                consolidation = df.iloc[i:end_idx]
                
                if len(consolidation) >= min_duration:
                    # Calculate boundaries
                    upper_boundary = consolidation['high'].max()
                    lower_boundary = consolidation['low'].min()
                    power_boundary = upper_boundary * 1.005
                    
                    # Get breakout and outcome
                    post_consolidation = df.iloc[end_idx:end_idx+100]
                    
                    if len(post_consolidation) > 0:
                        breakout_price = post_consolidation.iloc[0]['close']
                        breakout_direction = 'up' if breakout_price > upper_boundary else 'down'
                        
                        # Calculate max gain in next 100 days
                        future_high = post_consolidation['high'].max()
                        max_gain = ((future_high - breakout_price) / breakout_price) * 100
                        
                        # Determine outcome class
                        if max_gain < 5:
                            outcome_class = 'K0'
                        elif max_gain < 15:
                            outcome_class = 'K1'
                        elif max_gain < 35:
                            outcome_class = 'K2'
                        elif max_gain < 75:
                            outcome_class = 'K3'
                        else:
                            outcome_class = 'K4'
                        
                        # Check for breakdown
                        if post_consolidation['low'].min() < lower_boundary * 0.95:
                            outcome_class = 'K5'
                            max_gain = -((lower_boundary - post_consolidation['low'].min()) / lower_boundary) * 100
                        
                        pattern = {
                            'ticker': ticker,
                            'start_date': consolidation.index[0].strftime('%Y-%m-%d'),
                            'end_date': consolidation.index[-1].strftime('%Y-%m-%d'),
                            'duration': len(consolidation),
                            'detection_method': 'simple_consolidation',
                            'upper_boundary': float(upper_boundary),
                            'lower_boundary': float(lower_boundary),
                            'power_boundary': float(power_boundary),
                            'boundary_width_pct': float((upper_boundary - lower_boundary) / lower_boundary * 100),
                            'qualification_metrics': {
                                'avg_bbw': float(avg_bbw),
                                'avg_adx': float(avg_adx),
                                'avg_volume_ratio': float(avg_volume_ratio),
                                'avg_range_ratio': float(avg_range_ratio)
                            },
                            'breakout_date': post_consolidation.index[0].strftime('%Y-%m-%d'),
                            'breakout_price': float(breakout_price),
                            'breakout_direction': breakout_direction,
                            'outcome_class': outcome_class,
                            'max_gain': float(max_gain),
                            'days_to_max': int(post_consolidation['high'].idxmax() - post_consolidation.index[0]).days if max_gain > 0 else 0
                        }
                        
                        patterns.append(pattern)
                        
                        # Skip ahead past this consolidation
                        i = end_idx + 10
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        return patterns


class GCSPatternGenerator:
    """Generate patterns from GCS raw data"""
    
    def __init__(self):
        self.setup_gcs()
        self.detector = ConsolidationDetector()
        self.all_patterns = []
        
    def setup_gcs(self):
        """Setup GCS connection"""
        cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
        
        if not os.path.exists(cred_path):
            # Try alternative
            cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
        
        credentials = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        self.client = storage.Client(credentials=credentials, project="ignition-ki-csv-storage")
        self.bucket = self.client.bucket("ignition-ki-csv-data-2025-user123")
        
    def load_ticker_data(self, ticker_file: str) -> Optional[pd.DataFrame]:
        """Load and parse ticker CSV data from GCS"""
        try:
            blob = self.bucket.blob(ticker_file)
            content = blob.download_as_text()
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(content))
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Ensure we have required columns
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Handle different date column names
            if 'date' not in df.columns:
                if 'datetime' in df.columns:
                    df['date'] = df['datetime']
                elif 'timestamp' in df.columns:
                    df['date'] = df['timestamp']
            
            # Check for required columns
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.warning(f"Missing columns in {ticker_file}: {missing}")
                return None
            
            # Parse dates and set index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Remove any duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN values
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {ticker_file}: {e}")
            return None
    
    def process_ticker(self, ticker_file: str) -> List[Dict]:
        """Process a single ticker file"""
        # Extract ticker name
        ticker = ticker_file.replace('_full_history.csv', '').replace('.csv', '').upper()
        
        logger.info(f"Processing {ticker}...")
        
        # Load data
        df = self.load_ticker_data(ticker_file)
        
        if df is None or len(df) < 100:
            return []
        
        # Detect patterns
        patterns = self.detector.detect_patterns(df, ticker)
        
        logger.info(f"  Found {len(patterns)} patterns in {ticker}")
        
        return patterns
    
    def generate_all_patterns(self, limit: int = None):
        """Generate patterns for all tickers in GCS"""
        
        logger.info("="*60)
        logger.info("GENERATING PATTERNS FROM GCS DATA")
        logger.info("="*60)
        
        # Get all CSV files
        csv_files = []
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('_full_history.csv') or blob.name.endswith('.csv'):
                # Skip non-ticker files
                if '/' not in blob.name or blob.name.startswith('analysis_results/'):
                    if any(char.isupper() for char in blob.name):  # Likely a ticker file
                        csv_files.append(blob.name)
        
        logger.info(f"Found {len(csv_files)} ticker files")
        
        if limit:
            csv_files = csv_files[:limit]
            logger.info(f"Processing first {limit} files")
        
        # Process files in parallel
        all_patterns = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.process_ticker, f): f for f in csv_files}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                ticker_file = futures[future]
                
                try:
                    patterns = future.result()
                    all_patterns.extend(patterns)
                    
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(csv_files)} files processed, {len(all_patterns)} patterns found")
                        
                except Exception as e:
                    logger.error(f"Error processing {ticker_file}: {e}")
        
        self.all_patterns = all_patterns
        
        logger.info("="*60)
        logger.info(f"PATTERN GENERATION COMPLETE")
        logger.info(f"Total patterns found: {len(all_patterns)}")
        logger.info("="*60)
        
        return all_patterns
    
    def save_patterns(self, output_file: str = None):
        """Save generated patterns"""
        
        if not self.all_patterns:
            logger.warning("No patterns to save")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"generated_patterns_{timestamp}.json"
        
        # Save locally
        with open(output_file, 'w') as f:
            json.dump(self.all_patterns, f, indent=2)
        
        logger.info(f"Saved {len(self.all_patterns)} patterns to {output_file}")
        
        # Also save to GCS
        try:
            blob = self.bucket.blob(f"patterns/{output_file}")
            blob.upload_from_string(json.dumps(self.all_patterns, indent=2))
            logger.info(f"Uploaded patterns to GCS: patterns/{output_file}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
        
        return output_file
    
    def get_pattern_summary(self):
        """Get summary of generated patterns"""
        
        if not self.all_patterns:
            return {}
        
        df = pd.DataFrame(self.all_patterns)
        
        summary = {
            'total_patterns': len(self.all_patterns),
            'unique_tickers': df['ticker'].nunique(),
            'outcome_distribution': df['outcome_class'].value_counts().to_dict(),
            'avg_duration': df['duration'].mean(),
            'avg_max_gain': df['max_gain'].mean(),
            'date_range': {
                'earliest': df['start_date'].min(),
                'latest': df['end_date'].max()
            }
        }
        
        return summary


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate consolidation patterns from GCS data')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 5 tickers')
    
    args = parser.parse_args()
    
    # Create generator
    generator = GCSPatternGenerator()
    
    # Set limit
    limit = args.limit
    if args.test:
        limit = 5
        logger.info("TEST MODE - Processing only 5 tickers")
    
    # Generate patterns
    patterns = generator.generate_all_patterns(limit=limit)
    
    if patterns:
        # Save patterns
        output_file = generator.save_patterns(args.output)
        
        # Show summary
        summary = generator.get_pattern_summary()
        
        print("\n" + "="*60)
        print("PATTERN GENERATION SUMMARY")
        print("="*60)
        print(f"Total Patterns: {summary['total_patterns']}")
        print(f"Unique Tickers: {summary['unique_tickers']}")
        print(f"Average Duration: {summary['avg_duration']:.1f} days")
        print(f"Average Max Gain: {summary['avg_max_gain']:.2f}%")
        print("\nOutcome Distribution:")
        for outcome, count in summary['outcome_distribution'].items():
            print(f"  {outcome}: {count}")
        print(f"\nPatterns saved to: {output_file}")
    else:
        print("No patterns generated")


if __name__ == "__main__":
    main()