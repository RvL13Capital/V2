"""
Full Production Analysis Pipeline for All Tickers
Processes all 2900+ tickers from GCS with comprehensive analysis
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
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'full_analysis_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullAnalysisPipeline:
    """Complete analysis pipeline for all GCS tickers"""
    
    def __init__(self):
        self.timestamp = timestamp
        self.output_dir = Path(f'./full_analysis_{self.timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup GCS
        self.setup_gcs()
        
        # Results storage
        self.all_patterns = []
        self.analysis_stats = {
            'start_time': datetime.now(),
            'tickers_processed': 0,
            'tickers_failed': 0,
            'patterns_found': 0,
            'errors': []
        }
        
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
        logger.info("GCS connection established")
    
    def get_all_ticker_files(self):
        """Get list of all ticker files from both root and tickers/ directory"""
        ticker_files = []
        
        # Get files from tickers/ directory
        logger.info("Scanning tickers/ directory...")
        for blob in self.bucket.list_blobs(prefix="tickers/"):
            if blob.name.endswith('.csv'):
                ticker_files.append(blob.name)
        
        # Get files from root directory (like ADNT_full_history.csv)
        logger.info("Scanning root directory for ticker files...")
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('_full_history.csv') or (
                blob.name.endswith('.csv') and 
                not '/' in blob.name and 
                any(c.isupper() for c in blob.name)
            ):
                ticker_files.append(blob.name)
        
        # Remove duplicates
        ticker_files = list(set(ticker_files))
        
        logger.info(f"Found {len(ticker_files)} total ticker files")
        return ticker_files
    
    def process_ticker(self, ticker_file: str):
        """Process a single ticker file"""
        ticker = ticker_file.split('/')[-1].replace('_full_history.csv', '').replace('.csv', '').upper()
        
        try:
            # Load data
            blob = self.bucket.blob(ticker_file)
            content = blob.download_as_text()
            df = pd.read_csv(io.StringIO(content))
            
            # Standardize columns
            df.columns = df.columns.str.lower()
            
            # Parse dates
            date_col = None
            for col in df.columns:
                if 'date' in col or 'time' in col:
                    date_col = col
                    break
            
            if not date_col:
                return ticker, []
            
            df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Check for required columns
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                return ticker, []
            
            # Convert to numeric
            for col in required + ['volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=required, inplace=True)
            
            if len(df) < 100:
                return ticker, []
            
            # Detect patterns
            patterns = self.detect_patterns(df, ticker)
            return ticker, patterns
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return ticker, []
    
    def detect_patterns(self, df: pd.DataFrame, ticker: str):
        """Detect consolidation patterns"""
        patterns = []
        
        try:
            # Calculate indicators
            df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
            df['range'] = (df['high'] - df['low']) / df['low'] * 100
            df['range_avg'] = df['range'].rolling(20, min_periods=1).mean()
            
            if 'volume' in df.columns:
                df['volume_avg'] = df['volume'].rolling(20, min_periods=1).mean()
            
            # Sliding window pattern detection
            window_size = 20
            step = 10
            
            for i in range(50, len(df) - 100, step):
                window = df.iloc[i:i+window_size]
                
                if len(window) < window_size:
                    continue
                
                # Calculate consolidation metrics
                price_range = (window['high'].max() - window['low'].min()) / window['low'].min() * 100
                avg_range = window['range'].mean()
                
                # Volume contraction (if available)
                volume_contraction = 1.0
                if 'volume' in df.columns and 'volume_avg' in df.columns:
                    avg_volume = window['volume'].mean()
                    avg_volume_baseline = window['volume_avg'].mean()
                    if avg_volume_baseline > 0:
                        volume_contraction = avg_volume / avg_volume_baseline
                
                # Detect consolidation
                is_consolidating = (
                    price_range < 15 and  # Tight price range
                    avg_range < 3 and     # Low daily volatility
                    volume_contraction < 0.8  # Volume drying up
                )
                
                if is_consolidating:
                    # Get boundaries
                    upper = window['high'].max()
                    lower = window['low'].min()
                    
                    # Evaluate outcome
                    future = df.iloc[i+window_size:i+window_size+100]
                    
                    if len(future) >= 30:
                        max_price = future['high'].max()
                        min_price = future['low'].min()
                        
                        # Calculate performance
                        max_gain = ((max_price - upper) / upper) * 100
                        max_loss = ((lower - min_price) / lower) * 100
                        
                        # Classify outcome
                        if max_loss > 10:
                            outcome = 'K5'  # Failed (breakdown)
                        elif max_gain < 5:
                            outcome = 'K0'  # Stagnant
                        elif max_gain < 15:
                            outcome = 'K1'  # Minimal
                        elif max_gain < 35:
                            outcome = 'K2'  # Quality
                        elif max_gain < 75:
                            outcome = 'K3'  # Strong
                        else:
                            outcome = 'K4'  # Exceptional
                        
                        pattern = {
                            'ticker': ticker,
                            'start_date': str(window.index[0].date()),
                            'end_date': str(window.index[-1].date()),
                            'duration': window_size,
                            'upper_boundary': float(upper),
                            'lower_boundary': float(lower),
                            'boundary_width_pct': float(price_range),
                            'avg_range': float(avg_range),
                            'volume_contraction': float(volume_contraction),
                            'outcome_class': outcome,
                            'max_gain': float(max_gain),
                            'max_loss': float(max_loss)
                        }
                        
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting patterns for {ticker}: {e}")
        
        return patterns
    
    def process_all_tickers_parallel(self, ticker_files: list, max_workers: int = 20):
        """Process all tickers in parallel batches"""
        
        logger.info(f"Processing {len(ticker_files)} tickers with {max_workers} workers...")
        
        batch_size = 100
        total_batches = (len(ticker_files) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(ticker_files))
            batch = ticker_files[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} tickers)")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_ticker, f): f for f in batch}
                
                for future in as_completed(futures):
                    ticker_file = futures[future]
                    try:
                        ticker, patterns = future.result()
                        
                        if patterns:
                            self.all_patterns.extend(patterns)
                            self.analysis_stats['patterns_found'] += len(patterns)
                            logger.info(f"  {ticker}: {len(patterns)} patterns found")
                        
                        self.analysis_stats['tickers_processed'] += 1
                        
                    except Exception as e:
                        self.analysis_stats['tickers_failed'] += 1
                        self.analysis_stats['errors'].append(str(e))
                        logger.error(f"  Failed: {ticker_file} - {e}")
            
            # Save intermediate results every 10 batches
            if (batch_num + 1) % 10 == 0:
                self.save_intermediate_results()
                gc.collect()
        
        logger.info(f"Completed processing all tickers")
    
    def save_intermediate_results(self):
        """Save intermediate results during processing"""
        intermediate_file = self.output_dir / f'patterns_intermediate_{len(self.all_patterns)}.json'
        with open(intermediate_file, 'w') as f:
            json.dump(self.all_patterns, f)
        logger.info(f"Saved intermediate results: {len(self.all_patterns)} patterns")
    
    def analyze_patterns(self):
        """Analyze all detected patterns"""
        
        logger.info("="*60)
        logger.info("ANALYZING PATTERNS")
        logger.info("="*60)
        
        if not self.all_patterns:
            logger.warning("No patterns to analyze")
            return {}
        
        df = pd.DataFrame(self.all_patterns)
        
        analysis = {
            'total_patterns': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'date_range': {
                'start': df['start_date'].min(),
                'end': df['end_date'].max()
            },
            'outcome_distribution': df['outcome_class'].value_counts().to_dict(),
            'success_metrics': {},
            'performance_metrics': {},
            'top_tickers': {},
            'optimal_characteristics': {}
        }
        
        # Success metrics
        successful = df[df['outcome_class'].isin(['K2', 'K3', 'K4'])]
        exceptional = df[df['outcome_class'] == 'K4']
        failed = df[df['outcome_class'] == 'K5']
        
        analysis['success_metrics'] = {
            'success_rate': len(successful) / len(df) * 100,
            'exceptional_rate': len(exceptional) / len(df) * 100,
            'failure_rate': len(failed) / len(df) * 100,
            'successful_patterns': len(successful),
            'exceptional_patterns': len(exceptional)
        }
        
        # Performance metrics
        analysis['performance_metrics'] = {
            'avg_max_gain': df['max_gain'].mean(),
            'median_max_gain': df['max_gain'].median(),
            'best_gain': df['max_gain'].max(),
            'avg_max_loss': df['max_loss'].mean(),
            'worst_loss': df['max_loss'].max()
        }
        
        # Top performing tickers
        ticker_stats = df.groupby('ticker').agg({
            'outcome_class': 'count',
            'max_gain': ['mean', 'max']
        }).round(2)
        
        ticker_stats.columns = ['pattern_count', 'avg_gain', 'max_gain']
        
        # Add success rate
        ticker_success = successful.groupby('ticker').size()
        ticker_stats['success_rate'] = (ticker_success / ticker_stats['pattern_count'] * 100).fillna(0).round(1)
        
        # Get top 20 by average gain
        top_by_gain = ticker_stats.nlargest(20, 'avg_gain')
        analysis['top_tickers']['by_gain'] = top_by_gain.to_dict('index')
        
        # Get top 20 by success rate (minimum 5 patterns)
        qualified = ticker_stats[ticker_stats['pattern_count'] >= 5]
        top_by_success = qualified.nlargest(20, 'success_rate')
        analysis['top_tickers']['by_success_rate'] = top_by_success.to_dict('index')
        
        # Optimal characteristics
        if len(successful) > 0:
            analysis['optimal_characteristics'] = {
                'avg_duration': successful['duration'].mean(),
                'avg_boundary_width': successful['boundary_width_pct'].mean(),
                'avg_range': successful['avg_range'].mean(),
                'avg_volume_contraction': successful['volume_contraction'].mean()
            }
        
        return analysis
    
    def generate_report(self, analysis: dict):
        """Generate comprehensive report"""
        
        report_file = self.output_dir / 'analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FULL CONSOLIDATION PATTERN ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Overview
            f.write("OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Patterns: {analysis['total_patterns']:,}\n")
            f.write(f"Unique Tickers: {analysis['unique_tickers']:,}\n")
            f.write(f"Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}\n")
            f.write(f"Tickers Processed: {self.analysis_stats['tickers_processed']:,}\n")
            f.write(f"Processing Time: {(datetime.now() - self.analysis_stats['start_time']).total_seconds() / 60:.1f} minutes\n\n")
            
            # Success Metrics
            f.write("SUCCESS METRICS\n")
            f.write("-"*40 + "\n")
            sm = analysis['success_metrics']
            f.write(f"Success Rate (K2+K3+K4): {sm['success_rate']:.2f}%\n")
            f.write(f"Exceptional Rate (K4): {sm['exceptional_rate']:.2f}%\n")
            f.write(f"Failure Rate (K5): {sm['failure_rate']:.2f}%\n")
            f.write(f"Successful Patterns: {sm['successful_patterns']:,}\n")
            f.write(f"Exceptional Patterns: {sm['exceptional_patterns']:,}\n\n")
            
            # Outcome Distribution
            f.write("OUTCOME DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
                count = analysis['outcome_distribution'].get(outcome, 0)
                pct = (count / analysis['total_patterns']) * 100 if analysis['total_patterns'] > 0 else 0
                f.write(f"{outcome}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            pm = analysis['performance_metrics']
            f.write(f"Average Max Gain: {pm['avg_max_gain']:.2f}%\n")
            f.write(f"Median Max Gain: {pm['median_max_gain']:.2f}%\n")
            f.write(f"Best Gain: {pm['best_gain']:.2f}%\n")
            f.write(f"Average Max Loss: {pm['avg_max_loss']:.2f}%\n")
            f.write(f"Worst Loss: {pm['worst_loss']:.2f}%\n\n")
            
            # Top Tickers by Gain
            f.write("TOP 20 TICKERS BY AVERAGE GAIN\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Ticker':<10} {'Count':<8} {'Avg Gain':<12} {'Max Gain':<12} {'Success%':<10}\n")
            f.write("-"*60 + "\n")
            
            for ticker, stats in list(analysis['top_tickers']['by_gain'].items())[:20]:
                f.write(f"{ticker:<10} {stats['pattern_count']:<8.0f} "
                       f"{stats['avg_gain']:<12.2f} {stats['max_gain']:<12.2f} "
                       f"{stats['success_rate']:<10.1f}\n")
            f.write("\n")
            
            # Top Tickers by Success Rate
            f.write("TOP 20 TICKERS BY SUCCESS RATE (min 5 patterns)\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Ticker':<10} {'Count':<8} {'Success%':<10} {'Avg Gain':<12}\n")
            f.write("-"*50 + "\n")
            
            for ticker, stats in list(analysis['top_tickers']['by_success_rate'].items())[:20]:
                f.write(f"{ticker:<10} {stats['pattern_count']:<8.0f} "
                       f"{stats['success_rate']:<10.1f} {stats['avg_gain']:<12.2f}\n")
            f.write("\n")
            
            # Optimal Characteristics
            if 'optimal_characteristics' in analysis and analysis['optimal_characteristics']:
                f.write("OPTIMAL PATTERN CHARACTERISTICS\n")
                f.write("-"*40 + "\n")
                oc = analysis['optimal_characteristics']
                f.write(f"Average Duration: {oc['avg_duration']:.1f} days\n")
                f.write(f"Average Boundary Width: {oc['avg_boundary_width']:.2f}%\n")
                f.write(f"Average Daily Range: {oc['avg_range']:.2f}%\n")
                f.write(f"Average Volume Contraction: {oc['avg_volume_contraction']:.2f}\n")
        
        logger.info(f"Report saved to: {report_file}")
    
    def save_final_results(self, analysis: dict):
        """Save all final results"""
        
        # Save patterns
        patterns_file = self.output_dir / f'all_patterns_{self.timestamp}.json'
        with open(patterns_file, 'w') as f:
            json.dump(self.all_patterns, f, indent=2)
        logger.info(f"Patterns saved to: {patterns_file}")
        
        # Save analysis
        analysis_file = self.output_dir / f'analysis_results_{self.timestamp}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {analysis_file}")
        
        # Save to GCS
        try:
            # Upload patterns
            blob = self.bucket.blob(f"analysis_results/patterns_{self.timestamp}.json")
            blob.upload_from_filename(str(patterns_file))
            logger.info("Patterns uploaded to GCS")
            
            # Upload analysis
            blob = self.bucket.blob(f"analysis_results/analysis_{self.timestamp}.json")
            blob.upload_from_filename(str(analysis_file))
            logger.info("Analysis uploaded to GCS")
            
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
    
    def run(self):
        """Run full analysis pipeline"""
        
        logger.info("="*80)
        logger.info("STARTING FULL ANALYSIS PIPELINE")
        logger.info("="*80)
        
        # Get all ticker files
        ticker_files = self.get_all_ticker_files()
        
        if not ticker_files:
            logger.error("No ticker files found")
            return
        
        # Process all tickers
        self.process_all_tickers_parallel(ticker_files)
        
        # Analyze patterns
        analysis = self.analyze_patterns()
        
        # Generate report
        self.generate_report(analysis)
        
        # Save results
        self.save_final_results(analysis)
        
        # Final summary
        duration = (datetime.now() - self.analysis_stats['start_time']).total_seconds() / 60
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Total Time: {duration:.1f} minutes")
        logger.info(f"Tickers Processed: {self.analysis_stats['tickers_processed']:,}")
        logger.info(f"Patterns Found: {self.analysis_stats['patterns_found']:,}")
        logger.info(f"Results saved to: {self.output_dir}")
        
        print(f"\nAnalysis complete! Results in: {self.output_dir}")
        print(f"Total patterns found: {self.analysis_stats['patterns_found']:,}")
        
        if analysis and 'success_metrics' in analysis:
            print(f"Success rate: {analysis['success_metrics']['success_rate']:.2f}%")
            print(f"Exceptional patterns: {analysis['success_metrics']['exceptional_patterns']:,}")


def main():
    """Main entry point"""
    
    print("="*80)
    print("FULL CONSOLIDATION PATTERN ANALYSIS")
    print("="*80)
    print("\nThis will analyze ALL tickers in your GCS bucket")
    print("Expected processing time: 10-30 minutes")
    print("="*80)
    
    response = input("\nProceed with full analysis? (yes/no): ")
    if response.lower() != 'yes':
        print("Analysis cancelled")
        return
    
    # Run pipeline
    pipeline = FullAnalysisPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()