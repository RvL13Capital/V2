"""
Explore and Process Tickers Directory in GCS
Specifically targets the tickers/ directory in your bucket
"""

import os
import json
import pandas as pd
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
import io
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickersDirectoryExplorer:
    """Explore and process files in the tickers/ directory"""
    
    def __init__(self):
        self.setup_gcs()
        self.tickers_data = {}
        
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
        logger.info("✓ GCS connection established")
        
    def explore_tickers_directory(self):
        """Explore the tickers/ directory structure"""
        
        print("\n" + "="*80)
        print("EXPLORING TICKERS DIRECTORY IN GCS")
        print("="*80)
        
        # List all files in tickers/ directory
        prefix = "tickers/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        files_by_type = {
            'csv': [],
            'json': [],
            'parquet': [],
            'other': []
        }
        
        subdirectories = set()
        file_count = 0
        total_size = 0
        
        print(f"\nScanning {prefix} directory...")
        
        for blob in blobs:
            file_count += 1
            total_size += blob.size if blob.size else 0
            
            # Get relative path
            relative_path = blob.name[len(prefix):]
            
            # Check for subdirectories
            if '/' in relative_path:
                subdir = relative_path.split('/')[0]
                subdirectories.add(subdir)
            
            # Categorize by file type
            if blob.name.endswith('.csv'):
                files_by_type['csv'].append(blob.name)
            elif blob.name.endswith('.json'):
                files_by_type['json'].append(blob.name)
            elif blob.name.endswith('.parquet'):
                files_by_type['parquet'].append(blob.name)
            else:
                files_by_type['other'].append(blob.name)
        
        # Display summary
        print(f"\n{'='*40}")
        print("DIRECTORY SUMMARY")
        print(f"{'='*40}")
        print(f"Total Files: {file_count}")
        print(f"Total Size: {total_size / (1024*1024):.2f} MB")
        print(f"Subdirectories: {len(subdirectories)}")
        
        print(f"\nFile Types:")
        for file_type, files in files_by_type.items():
            if files:
                print(f"  {file_type.upper()}: {len(files)} files")
        
        if subdirectories:
            print(f"\nSubdirectories found:")
            for subdir in sorted(subdirectories)[:10]:
                print(f"  - {subdir}/")
            if len(subdirectories) > 10:
                print(f"  ... and {len(subdirectories) - 10} more")
        
        # Show sample files
        for file_type, files in files_by_type.items():
            if files:
                print(f"\nSample {file_type.upper()} files:")
                for f in files[:5]:
                    blob = self.bucket.blob(f)
                    size_kb = blob.size / 1024 if blob.size else 0
                    # Extract just the filename
                    filename = f.split('/')[-1]
                    print(f"  - {filename} ({size_kb:.1f} KB)")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more {file_type} files")
        
        return files_by_type, subdirectories
    
    def analyze_ticker_file(self, file_path: str):
        """Analyze a single ticker file"""
        try:
            blob = self.bucket.blob(file_path)
            
            if file_path.endswith('.csv'):
                # Load CSV
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
                
                print(f"\nAnalyzing: {file_path}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Check for date column
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    print(f"  Date column: {date_cols[0]}")
                    # Parse dates
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], utc=True).dt.tz_localize(None)
                    print(f"  Date range: {df[date_cols[0]].min()} to {df[date_cols[0]].max()}")
                
                # Check for price columns
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                available_price_cols = [col for col in df.columns if col.lower() in price_cols]
                print(f"  Price columns: {available_price_cols}")
                
                return df
                
            elif file_path.endswith('.json'):
                # Load JSON
                content = blob.download_as_text()
                data = json.loads(content)
                
                print(f"\nAnalyzing: {file_path}")
                if isinstance(data, list):
                    print(f"  Type: List with {len(data)} items")
                    if data and isinstance(data[0], dict):
                        print(f"  Keys: {list(data[0].keys())}")
                elif isinstance(data, dict):
                    print(f"  Type: Dictionary")
                    print(f"  Keys: {list(data.keys())[:10]}")
                
                return data
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def process_ticker_files(self, file_list: list, limit: int = 5):
        """Process multiple ticker files"""
        
        print(f"\n{'='*40}")
        print("PROCESSING TICKER FILES")
        print(f"{'='*40}")
        
        processed_count = 0
        
        for file_path in file_list[:limit]:
            ticker = file_path.split('/')[-1].replace('.csv', '').replace('.json', '')
            print(f"\nProcessing: {ticker}")
            
            data = self.analyze_ticker_file(file_path)
            
            if data is not None:
                self.tickers_data[ticker] = data
                processed_count += 1
            
            if processed_count >= limit:
                break
        
        print(f"\n✓ Processed {processed_count} files")
        return self.tickers_data
    
    def detect_patterns_in_directory(self, limit: int = 10):
        """Detect patterns in ticker files from the directory"""
        
        print(f"\n{'='*80}")
        print("PATTERN DETECTION IN TICKERS DIRECTORY")
        print(f"{'='*80}")
        
        # Get CSV files from tickers directory
        prefix = "tickers/"
        csv_files = []
        
        for blob in self.bucket.list_blobs(prefix=prefix):
            if blob.name.endswith('.csv'):
                csv_files.append(blob.name)
        
        print(f"Found {len(csv_files)} CSV files in tickers/ directory")
        
        if limit:
            csv_files = csv_files[:limit]
            print(f"Processing first {limit} files")
        
        patterns = []
        
        for idx, csv_file in enumerate(csv_files, 1):
            ticker = csv_file.split('/')[-1].replace('.csv', '').upper()
            print(f"\n[{idx}/{len(csv_files)}] Processing {ticker}...")
            
            try:
                # Load data
                blob = self.bucket.blob(csv_file)
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
                
                # Standardize columns
                df.columns = df.columns.str.lower()
                
                # Find and parse date column
                date_col = None
                for col in df.columns:
                    if 'date' in col or 'time' in col:
                        date_col = col
                        break
                
                if date_col:
                    df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.tz_localize(None)
                    df = df.dropna(subset=['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Ensure we have price columns
                    required_cols = ['open', 'high', 'low', 'close']
                    if all(col in df.columns for col in required_cols):
                        
                        # Simple pattern detection
                        if len(df) >= 60:
                            # Calculate indicators
                            df['range'] = (df['high'] - df['low']) / df['low'] * 100
                            df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
                            
                            # Detect consolidations
                            for i in range(30, len(df) - 30, 10):
                                window = df.iloc[i:i+20]
                                
                                # Check if consolidating
                                price_range = (window['high'].max() - window['low'].min()) / window['low'].min() * 100
                                
                                if price_range < 10:  # Tight consolidation
                                    # Check outcome
                                    future = df.iloc[i+20:i+50]
                                    if len(future) > 0:
                                        max_gain = ((future['high'].max() - window['high'].max()) / window['high'].max()) * 100
                                        
                                        pattern = {
                                            'ticker': ticker,
                                            'start_date': str(window.index[0].date()),
                                            'end_date': str(window.index[-1].date()),
                                            'duration': 20,
                                            'price_range': price_range,
                                            'max_gain': max_gain,
                                            'source': csv_file
                                        }
                                        patterns.append(pattern)
                            
                            print(f"  Found {len([p for p in patterns if p['ticker'] == ticker])} patterns")
                    else:
                        print(f"  Missing required price columns")
                else:
                    print(f"  No date column found")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Save patterns
        if patterns:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"tickers_directory_patterns_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            print(f"\n{'='*40}")
            print("PATTERN DETECTION COMPLETE")
            print(f"{'='*40}")
            print(f"Total patterns found: {len(patterns)}")
            print(f"Patterns saved to: {output_file}")
            
            # Summary by ticker
            df_patterns = pd.DataFrame(patterns)
            ticker_summary = df_patterns.groupby('ticker').agg({
                'max_gain': ['count', 'mean', 'max']
            })
            
            print(f"\nPattern Summary by Ticker:")
            print(ticker_summary)
        else:
            print("\nNo patterns found")
        
        return patterns


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore tickers directory in GCS')
    parser.add_argument('--explore', action='store_true', help='Explore directory structure')
    parser.add_argument('--analyze', type=int, help='Analyze N files', default=0)
    parser.add_argument('--detect', type=int, help='Detect patterns in N files', default=0)
    parser.add_argument('--all', action='store_true', help='Do everything with 10 files')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = TickersDirectoryExplorer()
    
    if args.all:
        # Do everything
        files_by_type, subdirs = explorer.explore_tickers_directory()
        
        if files_by_type['csv']:
            print("\n" + "="*80)
            print("ANALYZING CSV FILES")
            print("="*80)
            explorer.process_ticker_files(files_by_type['csv'], limit=5)
        
        explorer.detect_patterns_in_directory(limit=10)
        
    else:
        if args.explore:
            files_by_type, subdirs = explorer.explore_tickers_directory()
        
        if args.analyze > 0:
            # Get files to analyze
            files = []
            for blob in explorer.bucket.list_blobs(prefix="tickers/"):
                if blob.name.endswith('.csv') or blob.name.endswith('.json'):
                    files.append(blob.name)
            
            if files:
                explorer.process_ticker_files(files, limit=args.analyze)
        
        if args.detect > 0:
            explorer.detect_patterns_in_directory(limit=args.detect)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()