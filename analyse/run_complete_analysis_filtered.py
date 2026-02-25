"""
Complete Analysis Pipeline with Price Filter
Full production analysis with $0.01 minimum price requirement
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from extended_analysis import ExtendedPatternAnalyzer
from pdf_report_generator import PDFReportGenerator

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'complete_analysis_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FilteredAnalysisPipeline:
    """Complete analysis pipeline with price filtering"""
    
    # MINIMUM PRICE REQUIREMENT
    MIN_PRICE = 0.01  # $0.01 minimum
    
    def __init__(self):
        self.timestamp = timestamp
        self.output_dir = Path(f'./filtered_analysis_{self.timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup GCS
        self.setup_gcs()
        
        # Results storage
        self.all_patterns = []
        self.extended_metrics = {}
        self.analysis_stats = {
            'start_time': datetime.now(),
            'tickers_processed': 0,
            'tickers_filtered': 0,
            'tickers_failed': 0,
            'patterns_found': 0,
            'patterns_filtered': 0,
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
        """Get list of all ticker files"""
        ticker_files = []
        
        # Get files from tickers/ directory
        logger.info("Scanning tickers/ directory...")
        for blob in self.bucket.list_blobs(prefix="tickers/"):
            if blob.name.endswith('.csv'):
                ticker_files.append(blob.name)
        
        # Get files from root directory
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
    
    def apply_price_filter(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Apply minimum price filter to dataframe"""
        
        if df is None or df.empty:
            return df
        
        # Check if we have required price columns
        price_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in price_cols):
            return df
        
        # Filter out rows where ANY price is below minimum
        original_len = len(df)
        
        # Apply filter: all prices must be >= MIN_PRICE
        df_filtered = df[
            (df['open'] >= self.MIN_PRICE) &
            (df['high'] >= self.MIN_PRICE) &
            (df['low'] >= self.MIN_PRICE) &
            (df['close'] >= self.MIN_PRICE)
        ].copy()
        
        filtered_count = original_len - len(df_filtered)
        
        if filtered_count > 0:
            logger.debug(f"{ticker}: Filtered {filtered_count} rows with price < ${self.MIN_PRICE}")
            
        # If too much data was filtered (>50%), consider the ticker invalid
        if len(df_filtered) < original_len * 0.5:
            logger.warning(f"{ticker}: >50% of data below ${self.MIN_PRICE}, skipping ticker")
            self.analysis_stats['tickers_filtered'] += 1
            return pd.DataFrame()  # Return empty dataframe
        
        return df_filtered
    
    def process_ticker(self, ticker_file: str):
        """Process a single ticker file with price filtering"""
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
            
            # APPLY PRICE FILTER
            df = self.apply_price_filter(df, ticker)
            
            if df.empty or len(df) < 100:
                return ticker, []
            
            # Check if current price meets minimum
            current_price = df['close'].iloc[-1]
            if current_price < self.MIN_PRICE:
                logger.info(f"{ticker}: Current price ${current_price:.4f} below minimum ${self.MIN_PRICE}")
                self.analysis_stats['tickers_filtered'] += 1
                return ticker, []
            
            # Detect patterns (only in valid price ranges)
            patterns = self.detect_patterns_with_filter(df, ticker)
            return ticker, patterns
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            self.analysis_stats['tickers_failed'] += 1
            return ticker, []
    
    def detect_patterns_with_filter(self, df: pd.DataFrame, ticker: str):
        """Detect consolidation patterns with price filtering"""
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
                
                # PRICE FILTER: Skip if window contains prices below minimum
                if window[['open', 'high', 'low', 'close']].min().min() < self.MIN_PRICE:
                    self.analysis_stats['patterns_filtered'] += 1
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
                    
                    # PRICE FILTER: Skip if boundaries are below minimum
                    if lower < self.MIN_PRICE or upper < self.MIN_PRICE:
                        self.analysis_stats['patterns_filtered'] += 1
                        continue
                    
                    # Evaluate outcome
                    future = df.iloc[i+window_size:i+window_size+100]
                    
                    if len(future) >= 30:
                        max_price = future['high'].max()
                        min_price = future['low'].min()
                        
                        # Calculate performance
                        max_gain = ((max_price - upper) / upper) * 100
                        max_loss = ((lower - min_price) / lower) * 100
                        
                        # Calculate actual days to reach targets from real data
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
                            if day_data['high'] == max_price and days_to_max is None:
                                days_to_max = day_idx
                        
                        # Classify outcome based on strategic value system
                        # K5: Failed pattern - price breaks below lower boundary by >5%
                        if min_price < lower * 0.95:  # 5% below lower boundary = breakdown
                            outcome = 'K5'  # Failed (breakdown)
                            strategic_value = -10
                        elif max_gain >= 75:
                            outcome = 'K4'  # Exceptional
                            strategic_value = 10
                        elif max_gain >= 35:
                            outcome = 'K3'  # Strong  
                            strategic_value = 3
                        elif max_gain >= 15:
                            outcome = 'K2'  # Quality
                            strategic_value = 1
                        elif max_gain >= 5:
                            outcome = 'K1'  # Minimal
                            strategic_value = -0.2
                        else:
                            outcome = 'K0'  # Stagnant
                            strategic_value = -2
                        
                        # Determine breakout direction
                        breakout_price = future['close'].iloc[0] if not future.empty else upper
                        breakout_direction = 'up' if breakout_price > upper else 'down'
                        
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
                            'strategic_value': float(strategic_value),
                            'max_gain': float(max_gain),
                            'max_loss': float(max_loss),
                            'days_to_max': days_to_max,
                            'days_to_5pct': days_to_5pct,
                            'days_to_10pct': days_to_10pct,
                            'days_to_15pct': days_to_15pct,
                            'days_to_25pct': days_to_25pct,
                            'days_to_50pct': days_to_50pct,
                            'breakout_direction': breakout_direction,
                            'min_price_in_pattern': float(window[['open', 'high', 'low', 'close']].min().min()),
                            'detection_method': 'filtered_consolidation'
                        }
                        
                        patterns.append(pattern)
                        self.analysis_stats['patterns_found'] += 1
        
        except Exception as e:
            logger.error(f"Error detecting patterns for {ticker}: {e}")
        
        return patterns
    
    def process_all_tickers_parallel(self, ticker_files: list, max_workers: int = 20):
        """Process all tickers in parallel with filtering"""
        
        logger.info(f"Processing {len(ticker_files)} tickers with ${self.MIN_PRICE} minimum price filter...")
        logger.info(f"Using {max_workers} parallel workers")
        
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
                            logger.info(f"  {ticker}: {len(patterns)} patterns found (above ${self.MIN_PRICE})")
                        
                        self.analysis_stats['tickers_processed'] += 1
                        
                    except Exception as e:
                        self.analysis_stats['tickers_failed'] += 1
                        self.analysis_stats['errors'].append(str(e))
                        logger.error(f"  Failed: {ticker_file} - {e}")
            
            # Save intermediate results
            if (batch_num + 1) % 10 == 0:
                self.save_intermediate_results()
                gc.collect()
        
        logger.info(f"Completed processing all tickers")
        logger.info(f"Tickers filtered due to price: {self.analysis_stats['tickers_filtered']}")
        logger.info(f"Patterns filtered due to price: {self.analysis_stats['patterns_filtered']}")
    
    def save_intermediate_results(self):
        """Save intermediate results"""
        intermediate_file = self.output_dir / f'patterns_intermediate_{len(self.all_patterns)}.json'
        with open(intermediate_file, 'w') as f:
            json.dump(self.all_patterns, f)
        logger.info(f"Saved intermediate results: {len(self.all_patterns)} patterns")
    
    def run_extended_analysis(self):
        """Run extended analysis on filtered patterns"""
        logger.info("="*80)
        logger.info("EXTENDED ANALYSIS")
        logger.info("="*80)
        
        if not self.all_patterns:
            logger.error("No patterns to analyze")
            return False
        
        # Create extended analyzer
        analyzer = ExtendedPatternAnalyzer(self.all_patterns)
        
        # Calculate all extended metrics
        logger.info("Calculating extended metrics...")
        self.extended_metrics['time_to_targets'] = analyzer.calculate_time_to_targets()
        self.extended_metrics['duration_by_outcome'] = analyzer.analyze_consolidation_duration_by_outcome()
        self.extended_metrics['false_breakouts'] = analyzer.analyze_false_breakouts()
        self.extended_metrics['by_method'] = analyzer.analyze_by_detection_method()
        self.extended_metrics['quality_metrics'] = analyzer.get_pattern_quality_metrics()
        self.extended_metrics['risk_reward'] = analyzer.calculate_risk_reward_metrics()
        self.extended_metrics['strategic_value'] = analyzer.calculate_strategic_value_metrics()
        self.extended_metrics['example_patterns'] = analyzer.get_example_patterns(n=5)
        
        # Add basic statistics
        df = pd.DataFrame(self.all_patterns)
        if not df.empty:
            self.extended_metrics['basic_stats'] = {
                'total_patterns': len(df),
                'unique_tickers': df['ticker'].nunique(),
                'date_range': {
                    'start': df['start_date'].min(),
                    'end': df['end_date'].max()
                },
                'outcome_distribution': df['outcome_class'].value_counts().to_dict(),
                'min_price_filter': self.MIN_PRICE,
                'avg_min_price_in_patterns': df['min_price_in_pattern'].mean() if 'min_price_in_pattern' in df.columns else 0
            }
        
        return True
    
    def generate_pdf_report(self):
        """Generate PDF report"""
        logger.info("="*80)
        logger.info("PDF REPORT GENERATION")
        logger.info("="*80)
        
        # Add filter information to extended metrics
        self.extended_metrics['filter_info'] = {
            'min_price': self.MIN_PRICE,
            'tickers_filtered': self.analysis_stats['tickers_filtered'],
            'patterns_filtered': self.analysis_stats['patterns_filtered']
        }
        
        # Create PDF generator
        pdf_generator = PDFReportGenerator(self.all_patterns, self.extended_metrics)
        
        # Generate report
        pdf_path = self.output_dir / f'filtered_analysis_report_{self.timestamp}.pdf'
        pdf_generator.generate_report(str(pdf_path))
        
        logger.info(f"PDF report generated: {pdf_path}")
        return True
    
    def generate_summary(self):
        """Generate summary report"""
        summary_file = self.output_dir / f'summary_{self.timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FILTERED CONSOLIDATION ANALYSIS SUMMARY\n")
            f.write(f"Minimum Price Filter: ${self.MIN_PRICE}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Tickers Processed: {self.analysis_stats['tickers_processed']:,}\n")
            f.write(f"Tickers Filtered (price): {self.analysis_stats['tickers_filtered']:,}\n")
            f.write(f"Tickers Failed: {self.analysis_stats['tickers_failed']:,}\n")
            f.write(f"Patterns Found: {self.analysis_stats['patterns_found']:,}\n")
            f.write(f"Patterns Filtered (price): {self.analysis_stats['patterns_filtered']:,}\n\n")
            
            if 'basic_stats' in self.extended_metrics:
                stats = self.extended_metrics['basic_stats']
                f.write("PATTERN STATISTICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Valid Patterns: {stats['total_patterns']:,}\n")
                f.write(f"Unique Tickers: {stats['unique_tickers']:,}\n")
                f.write(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}\n")
                f.write(f"Avg Min Price in Patterns: ${stats.get('avg_min_price_in_patterns', 0):.4f}\n\n")
            
            if 'risk_reward' in self.extended_metrics:
                rr = self.extended_metrics['risk_reward']
                f.write("RISK/REWARD METRICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Win Rate: {rr.get('win_rate', 0):.1f}%\n")
                f.write(f"Average Gain: {rr.get('avg_gain', 0):.2f}%\n")
                f.write(f"Average Loss: {rr.get('avg_loss', 0):.2f}%\n")
                f.write(f"Expectancy: {rr.get('expectancy', 0):.2f}%\n")
        
        logger.info(f"Summary saved: {summary_file}")
    
    def run_complete_pipeline(self):
        """Run complete filtered analysis pipeline"""
        
        start_time = datetime.now()
        
        logger.info("="*80)
        logger.info("STARTING FILTERED ANALYSIS PIPELINE")
        logger.info(f"Minimum Price Filter: ${self.MIN_PRICE}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*80)
        
        try:
            # Get all ticker files
            ticker_files = self.get_all_ticker_files()
            
            if not ticker_files:
                logger.error("No ticker files found")
                return False
            
            # Process all tickers with filtering
            self.process_all_tickers_parallel(ticker_files)
            
            if not self.all_patterns:
                logger.error("No patterns found after filtering")
                return False
            
            # Save patterns
            patterns_file = self.output_dir / f'filtered_patterns_{self.timestamp}.json'
            with open(patterns_file, 'w') as f:
                json.dump(self.all_patterns, f, indent=2)
            
            # Run extended analysis
            self.run_extended_analysis()
            
            # Generate PDF report
            self.generate_pdf_report()
            
            # Generate summary
            self.generate_summary()
            
            # Final statistics
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            logger.info("="*80)
            logger.info("ANALYSIS COMPLETE")
            logger.info("="*80)
            logger.info(f"Total Time: {duration:.1f} minutes")
            logger.info(f"Valid Patterns Found: {len(self.all_patterns):,}")
            logger.info(f"Tickers Filtered: {self.analysis_stats['tickers_filtered']:,}")
            logger.info(f"Patterns Filtered: {self.analysis_stats['patterns_filtered']:,}")
            
            print("\n" + "="*80)
            print("✅ FILTERED ANALYSIS COMPLETE!")
            print("="*80)
            print(f"📁 Results: {self.output_dir}/")
            print(f"📄 PDF Report: {self.output_dir}/filtered_analysis_report_{self.timestamp}.pdf")
            print(f"💰 Min Price Filter: ${self.MIN_PRICE}")
            print(f"📊 Valid Patterns: {len(self.all_patterns):,}")
            print(f"🚫 Filtered Tickers: {self.analysis_stats['tickers_filtered']:,}")
            print(f"⏱️ Duration: {duration:.1f} minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    
    print("="*80)
    print("FILTERED CONSOLIDATION ANALYSIS")
    print(f"Minimum Price Requirement: $0.01")
    print("="*80)
    print("\nThis will:")
    print("1. Filter out all stocks/patterns with prices below $0.01")
    print("2. Detect consolidation patterns only in valid price ranges")
    print("3. Generate comprehensive PDF report with filtered data")
    print("\nEstimated time: 15-45 minutes")
    print("="*80)
    
    response = input("\nProceed with filtered analysis? (yes/no): ")
    if response.lower() != 'yes':
        print("Analysis cancelled")
        return
    
    # Run pipeline
    pipeline = FilteredAnalysisPipeline()
    success = pipeline.run_complete_pipeline()
    
    if not success:
        print("\n❌ Analysis failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()