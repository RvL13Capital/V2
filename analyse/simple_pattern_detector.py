"""
Simplified Pattern Detector for GCS Data
Robust pattern detection from raw price data
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
import io
import warnings
warnings.filterwarnings('ignore')

class SimplePatternDetector:
    """Simple but robust pattern detector"""
    
    def __init__(self):
        self.setup_gcs()
        self.patterns = []
        self.processed_files = 0
        self.failed_files = 0
        
    def setup_gcs(self):
        """Setup GCS connection"""
        cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
        
        if not os.path.exists(cred_path):
            cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
        
        credentials = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        self.client = storage.Client(credentials=credentials, project="ignition-ki-csv-storage")
        self.bucket = self.client.bucket("ignition-ki-csv-data-2025-user123")
        print("GCS connection established")
        
    def load_ticker_data(self, ticker_file: str) -> pd.DataFrame:
        """Load ticker data with robust error handling"""
        try:
            blob = self.bucket.blob(ticker_file)
            content = blob.download_as_text()
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(content))
            
            # Standardize columns
            df.columns = df.columns.str.lower()
            
            # Handle date column
            if 'date' in df.columns:
                # Parse dates with UTC to avoid timezone issues
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            elif 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            else:
                return None
            
            # Set index
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            
            return df
            
        except Exception as e:
            print(f"  Error loading {ticker_file}: {e}")
            return None
    
    def detect_simple_patterns(self, df: pd.DataFrame, ticker: str) -> list:
        """Detect patterns with simplified logic"""
        patterns = []
        
        if df is None or len(df) < 60:
            return patterns
        
        try:
            # Calculate simple indicators
            df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
            df['volume_avg'] = df['volume'].rolling(20, min_periods=1).mean() if 'volume' in df.columns else 1
            df['range'] = (df['high'] - df['low']) / df['low'] * 100
            df['range_avg'] = df['range'].rolling(20, min_periods=1).mean()
            
            # Simple consolidation detection
            window_size = 20
            step = 10
            
            for i in range(50, len(df) - 100, step):
                window = df.iloc[i:i+window_size]
                
                if len(window) < window_size:
                    continue
                
                # Simple consolidation criteria
                price_range = (window['high'].max() - window['low'].min()) / window['low'].min() * 100
                avg_range = window['range'].mean()
                
                # Check if consolidating (tight range)
                if price_range < 10 and avg_range < 3:
                    
                    # Get boundaries
                    upper = window['high'].max()
                    lower = window['low'].min()
                    
                    # Check outcome
                    future = df.iloc[i+window_size:i+window_size+50]
                    
                    if len(future) > 0:
                        max_future_price = future['high'].max()
                        min_future_price = future['low'].min()
                        
                        # Calculate gain and days to targets
                        if upper > 0:
                            max_gain = ((max_future_price - upper) / upper) * 100
                            
                            # Calculate days to reach different targets from actual data
                            days_to_5pct = None
                            days_to_10pct = None
                            days_to_15pct = None
                            days_to_25pct = None
                            days_to_50pct = None
                            days_to_max = None
                            
                            for day_idx, (_, day_data) in enumerate(future.iterrows(), 1):
                                day_gain = ((day_data['high'] - upper) / upper) * 100
                                
                                if days_to_5pct is None and day_gain >= 5:
                                    days_to_5pct = day_idx
                                if days_to_10pct is None and day_gain >= 10:
                                    days_to_10pct = day_idx
                                if days_to_15pct is None and day_gain >= 15:
                                    days_to_15pct = day_idx
                                if days_to_25pct is None and day_gain >= 25:
                                    days_to_25pct = day_idx
                                if days_to_50pct is None and day_gain >= 50:
                                    days_to_50pct = day_idx
                                if day_data['high'] == max_future_price:
                                    if days_to_max is None:
                                        days_to_max = day_idx
                        else:
                            max_gain = 0
                            days_to_5pct = None
                            days_to_10pct = None
                            days_to_15pct = None
                            days_to_25pct = None
                            days_to_50pct = None
                            days_to_max = None
                        
                        # Classify outcome
                        if max_gain < 5:
                            outcome = 'K0'
                        elif max_gain < 15:
                            outcome = 'K1'
                        elif max_gain < 35:
                            outcome = 'K2'
                        elif max_gain < 75:
                            outcome = 'K3'
                        else:
                            outcome = 'K4'
                        
                        # Check for breakdown
                        if min_future_price < lower * 0.95:
                            outcome = 'K5'
                            max_gain = -((lower - min_future_price) / lower) * 100
                        
                        pattern = {
                            'ticker': ticker,
                            'start_date': str(window.index[0].date()),
                            'end_date': str(window.index[-1].date()),
                            'duration': window_size,
                            'detection_method': 'simple',
                            'upper_boundary': float(upper),
                            'lower_boundary': float(lower),
                            'power_boundary': float(upper * 1.005),
                            'boundary_width_pct': float(price_range),
                            'outcome_class': outcome,
                            'max_gain': float(max_gain),
                            'days_to_max': days_to_max,
                            'days_to_5pct': days_to_5pct,
                            'days_to_10pct': days_to_10pct,
                            'days_to_15pct': days_to_15pct,
                            'days_to_25pct': days_to_25pct,
                            'days_to_50pct': days_to_50pct,
                            'qualification_metrics': {
                                'avg_range': float(avg_range),
                                'price_range': float(price_range)
                            }
                        }
                        
                        patterns.append(pattern)
            
        except Exception as e:
            print(f"  Error detecting patterns for {ticker}: {e}")
        
        return patterns
    
    def process_tickers(self, limit: int = None):
        """Process multiple tickers"""
        
        print("\n" + "="*60)
        print("DETECTING PATTERNS FROM GCS DATA")
        print("="*60)
        
        # Get CSV files
        csv_files = []
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('_full_history.csv') or (blob.name.endswith('.csv') and '_' not in blob.name):
                if any(c.isupper() for c in blob.name):  # Likely a ticker
                    csv_files.append(blob.name)
        
        print(f"Found {len(csv_files)} ticker files")
        
        if limit:
            csv_files = csv_files[:limit]
            print(f"Processing first {limit} files")
        
        # Process each file
        for idx, csv_file in enumerate(csv_files, 1):
            ticker = csv_file.replace('_full_history.csv', '').replace('.csv', '').upper()
            print(f"\n[{idx}/{len(csv_files)}] Processing {ticker}...")
            
            # Load data
            df = self.load_ticker_data(csv_file)
            
            if df is not None:
                # Detect patterns
                patterns = self.detect_simple_patterns(df, ticker)
                
                if patterns:
                    self.patterns.extend(patterns)
                    print(f"  Found {len(patterns)} patterns")
                else:
                    print(f"  No patterns found")
                
                self.processed_files += 1
            else:
                print(f"  Failed to load data")
                self.failed_files += 1
        
        print("\n" + "="*60)
        print("DETECTION COMPLETE")
        print(f"Processed: {self.processed_files} files")
        print(f"Failed: {self.failed_files} files")
        print(f"Total patterns found: {len(self.patterns)}")
        print("="*60)
    
    def save_patterns(self, output_file: str = None):
        """Save patterns to file"""
        
        if not self.patterns:
            print("No patterns to save")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"detected_patterns_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        
        print(f"\nPatterns saved to: {output_file}")
        
        # Also save to GCS
        try:
            blob = self.bucket.blob(f"patterns/{output_file}")
            blob.upload_from_string(json.dumps(self.patterns, indent=2))
            print(f"Uploaded to GCS: patterns/{output_file}")
        except Exception as e:
            print(f"Failed to upload to GCS: {e}")
        
        return output_file
    
    def get_summary(self):
        """Get pattern summary"""
        
        if not self.patterns:
            return {}
        
        df = pd.DataFrame(self.patterns)
        
        # Count outcomes
        outcome_counts = df['outcome_class'].value_counts().to_dict()
        
        print("\n" + "="*60)
        print("PATTERN SUMMARY")
        print("="*60)
        print(f"Total Patterns: {len(self.patterns)}")
        print(f"Unique Tickers: {df['ticker'].nunique()}")
        print(f"Average Duration: {df['duration'].mean():.1f} days")
        print(f"Average Max Gain: {df['max_gain'].mean():.2f}%")
        
        print("\nOutcome Distribution:")
        for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
            count = outcome_counts.get(outcome, 0)
            pct = (count / len(self.patterns)) * 100 if self.patterns else 0
            print(f"  {outcome}: {count} ({pct:.1f}%)")
        
        return {
            'total_patterns': len(self.patterns),
            'unique_tickers': df['ticker'].nunique(),
            'outcome_distribution': outcome_counts
        }


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect patterns from GCS data')
    parser.add_argument('--limit', type=int, help='Limit number of tickers')
    parser.add_argument('--test', action='store_true', help='Test mode - 5 tickers')
    parser.add_argument('--quick', action='store_true', help='Quick mode - 20 tickers')
    
    args = parser.parse_args()
    
    # Determine limit
    limit = args.limit
    if args.test:
        limit = 5
        print("TEST MODE - Processing 5 tickers")
    elif args.quick:
        limit = 20
        print("QUICK MODE - Processing 20 tickers")
    
    # Create detector
    detector = SimplePatternDetector()
    
    # Process tickers
    detector.process_tickers(limit=limit)
    
    # Save patterns
    if detector.patterns:
        output_file = detector.save_patterns()
        detector.get_summary()
        
        print(f"\nYou can now analyze these patterns with:")
        print(f"  python run_consolidation_analysis.py --patterns-file {output_file}")
    else:
        print("\nNo patterns detected. Try processing more tickers.")


if __name__ == "__main__":
    main()