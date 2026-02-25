"""
Production Deployment Script for Full Consolidation Analysis
Processes all available data from GCS with batch processing and monitoring
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Import analysis components
from gcs_config import GCSDataLoader, setup_gcs_environment
from consolidation_analyzer import ConsolidationAnalyzer
from pattern_metrics import PatternMetrics
from breakout_validator import BreakoutValidator
from statistical_analysis import StatisticalAnalyzer
from visualization import PatternVisualizer
from report_generator import ReportGenerator

# Configure production logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'full_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionAnalysisDeployer:
    """Production-grade analysis deployment system"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.gcs_loader = None
        self.output_base = Path('./production_analysis')
        self.output_base.mkdir(exist_ok=True)
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.output_base / f'run_{self.timestamp}'
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis components
        self.consolidation_analyzer = ConsolidationAnalyzer()
        self.pattern_metrics = PatternMetrics()
        self.breakout_validator = BreakoutValidator()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = PatternVisualizer(str(self.output_dir / 'visualizations'))
        self.report_generator = ReportGenerator(str(self.output_dir / 'reports'))
        
        # Tracking
        self.stats = {
            'total_patterns': 0,
            'processed_patterns': 0,
            'failed_patterns': 0,
            'total_tickers': 0,
            'processed_tickers': 0,
            'errors': [],
            'processing_time': 0
        }
        
        # Results storage
        self.all_patterns = []
        self.analysis_results = {}
        self.ticker_results = {}
        
    def initialize_gcs(self) -> bool:
        """Initialize GCS connection with retry logic"""
        logger.info("="*80)
        logger.info("INITIALIZING GCS CONNECTION")
        logger.info("="*80)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if setup_gcs_environment():
                    self.gcs_loader = GCSDataLoader()
                    logger.info("✓ GCS connection established successfully")
                    return True
            except Exception as e:
                logger.warning(f"GCS connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        logger.error("✗ Failed to establish GCS connection after all retries")
        return False
    
    def load_all_patterns(self) -> List[Dict]:
        """Load all available patterns from GCS"""
        logger.info("\n" + "="*80)
        logger.info("LOADING ALL PATTERNS FROM GCS")
        logger.info("="*80)
        
        try:
            # List all pattern files
            pattern_files = self.gcs_loader.list_pattern_files()
            logger.info(f"Found {len(pattern_files)} pattern files in GCS")
            
            all_patterns = []
            
            # Load patterns from each file
            for file_idx, pattern_file in enumerate(pattern_files, 1):
                try:
                    logger.info(f"Loading file {file_idx}/{len(pattern_files)}: {pattern_file}")
                    patterns = self.gcs_loader.load_patterns_from_gcs(pattern_file)
                    all_patterns.extend(patterns)
                    logger.info(f"  → Loaded {len(patterns)} patterns")
                except Exception as e:
                    logger.error(f"  ✗ Error loading {pattern_file}: {e}")
                    self.stats['errors'].append(f"Failed to load {pattern_file}: {str(e)}")
            
            # If no files found, try loading without specific file
            if not pattern_files:
                logger.info("No specific pattern files found, attempting general load...")
                all_patterns = self.gcs_loader.load_patterns_from_gcs()
            
            self.stats['total_patterns'] = len(all_patterns)
            logger.info(f"\n✓ Total patterns loaded: {len(all_patterns)}")
            
            # Get unique tickers
            unique_tickers = set(p.get('ticker') for p in all_patterns if p.get('ticker'))
            self.stats['total_tickers'] = len(unique_tickers)
            logger.info(f"✓ Unique tickers: {len(unique_tickers)}")
            
            # Get date range
            if all_patterns:
                dates = [pd.to_datetime(p.get('start_date')) for p in all_patterns if p.get('start_date')]
                if dates:
                    logger.info(f"✓ Date range: {min(dates).date()} to {max(dates).date()}")
            
            return all_patterns
            
        except Exception as e:
            logger.error(f"Critical error loading patterns: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def process_patterns_batch(self, patterns: List[Dict], batch_size: int = 1000) -> Dict:
        """Process patterns in batches to manage memory"""
        logger.info(f"\nProcessing {len(patterns)} patterns in batches of {batch_size}")
        
        results = {
            'duration_stats': {},
            'outcome_stats': {},
            'qualification_metrics': {},
            'post_breakout_performance': {},
            'optimal_patterns': [],
            'method_comparison': None,
            'processed_patterns': []
        }
        
        # Process in batches
        for batch_idx in range(0, len(patterns), batch_size):
            batch = patterns[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(patterns) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} patterns)")
            
            try:
                # Convert to pattern objects
                pattern_objects = []
                for p in batch:
                    try:
                        pattern_obj = self.consolidation_analyzer._dict_to_pattern(p)
                        pattern_objects.append(pattern_obj)
                        results['processed_patterns'].append(p)
                        self.stats['processed_patterns'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to convert pattern: {e}")
                        self.stats['failed_patterns'] += 1
                
                # Update analyzer with current batch
                self.consolidation_analyzer.patterns = pattern_objects
                
                # Run analyses
                if pattern_objects:
                    # Duration analysis
                    duration_stats = self.consolidation_analyzer.analyze_consolidation_duration()
                    self._merge_stats(results['duration_stats'], duration_stats)
                    
                    # Outcome analysis
                    outcome_stats = self.consolidation_analyzer.analyze_breakout_outcomes()
                    self._merge_stats(results['outcome_stats'], outcome_stats)
                    
                    # Qualification metrics
                    qual_metrics = self.consolidation_analyzer.analyze_qualification_metrics()
                    self._merge_stats(results['qualification_metrics'], qual_metrics)
                
                # Clear memory
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                self.stats['errors'].append(f"Batch {batch_num} processing error: {str(e)}")
        
        logger.info(f"✓ Batch processing complete: {self.stats['processed_patterns']} patterns processed")
        return results
    
    def _merge_stats(self, target: Dict, source: Dict):
        """Merge statistics from batches"""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key not in target[key]:
                        target[key][sub_key] = sub_value
                    elif isinstance(sub_value, (int, float)):
                        # Average numerical values
                        target[key][sub_key] = (target[key][sub_key] + sub_value) / 2
                    elif isinstance(sub_value, dict):
                        # Merge dictionaries
                        target[key][sub_key].update(sub_value)
    
    def load_price_data_parallel(self, tickers: List[str], max_workers: int = 10) -> Dict:
        """Load price data for multiple tickers in parallel"""
        logger.info(f"\nLoading price data for {len(tickers)} tickers (parallel with {max_workers} workers)")
        
        price_data = {}
        
        def load_ticker_data(ticker):
            try:
                df = self.gcs_loader.load_price_data(ticker)
                if not df.empty:
                    return ticker, df
                return ticker, None
            except Exception as e:
                logger.warning(f"Failed to load data for {ticker}: {e}")
                return ticker, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_ticker_data, ticker): ticker for ticker in tickers}
            
            completed = 0
            for future in as_completed(futures):
                ticker, data = future.result()
                completed += 1
                
                if data is not None:
                    price_data[ticker] = data
                    self.stats['processed_tickers'] += 1
                
                if completed % 10 == 0:
                    logger.info(f"  Progress: {completed}/{len(tickers)} tickers loaded")
        
        logger.info(f"✓ Price data loaded for {len(price_data)} tickers")
        return price_data
    
    def run_comprehensive_analysis(self, patterns: List[Dict]) -> Dict:
        """Run all analysis modules"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING COMPREHENSIVE ANALYSIS")
        logger.info("="*80)
        
        analysis_results = {}
        
        try:
            # 1. Process patterns in batches
            logger.info("\n1. Processing pattern batches...")
            batch_results = self.process_patterns_batch(patterns)
            analysis_results.update(batch_results)
            
            # 2. Statistical analysis
            logger.info("\n2. Running statistical analysis...")
            patterns_by_method = {}
            for p in patterns:
                method = p.get('detection_method', 'unknown')
                if method not in patterns_by_method:
                    patterns_by_method[method] = []
                patterns_by_method[method].append(p)
            
            stat_comparison = self.statistical_analyzer.compare_methods_performance(patterns_by_method)
            analysis_results['statistical_comparison'] = stat_comparison
            
            pattern_characteristics = self.statistical_analyzer.analyze_pattern_characteristics(patterns)
            analysis_results['pattern_characteristics'] = pattern_characteristics
            
            risk_metrics = self.statistical_analyzer.calculate_risk_metrics(patterns)
            analysis_results['risk_metrics'] = risk_metrics
            
            temporal_patterns = self.statistical_analyzer.analyze_temporal_patterns(patterns)
            analysis_results['temporal_patterns'] = temporal_patterns
            
            # 3. Method comparison
            logger.info("\n3. Comparing detection methods...")
            self.consolidation_analyzer.patterns = [
                self.consolidation_analyzer._dict_to_pattern(p) for p in patterns[:5000]  # Limit for memory
            ]
            method_comparison = self.consolidation_analyzer.compare_detection_methods()
            analysis_results['method_comparison'] = method_comparison
            
            # 4. Identify optimal patterns
            logger.info("\n4. Identifying optimal patterns...")
            optimal_patterns = self.consolidation_analyzer.identify_optimal_patterns()
            analysis_results['optimal_patterns'] = optimal_patterns
            
            # 5. Breakout validation
            logger.info("\n5. Analyzing breakout patterns...")
            fakeout_analysis = self.breakout_validator.analyze_breakout_patterns(patterns[:1000])  # Sample
            analysis_results['fakeout_analysis'] = fakeout_analysis.to_dict('records') if not fakeout_analysis.empty else []
            
            # 6. Generate summary
            logger.info("\n6. Generating summary...")
            summary = self.consolidation_analyzer.generate_summary_report()
            analysis_results['summary'] = summary
            
            # Add deployment stats
            analysis_results['deployment_stats'] = self.stats
            
            logger.info("✓ Comprehensive analysis complete")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            logger.error(traceback.format_exc())
            self.stats['errors'].append(f"Analysis error: {str(e)}")
        
        return analysis_results
    
    def generate_production_reports(self, analysis_results: Dict):
        """Generate all production reports and visualizations"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING PRODUCTION REPORTS")
        logger.info("="*80)
        
        try:
            # Generate visualizations
            logger.info("\n1. Creating visualizations...")
            viz_dir = self.output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            if 'duration_stats' in analysis_results:
                self.visualizer.plot_duration_distribution(analysis_results['duration_stats'])
            
            if 'outcome_stats' in analysis_results:
                self.visualizer.plot_outcome_distribution(analysis_results['outcome_stats'])
            
            if 'method_comparison' in analysis_results and analysis_results['method_comparison'] is not None:
                self.visualizer.plot_method_comparison(analysis_results['method_comparison'])
            
            logger.info("✓ Visualizations created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            self.stats['errors'].append(f"Visualization error: {str(e)}")
        
        try:
            # Generate reports
            logger.info("\n2. Generating reports...")
            report_files = self.report_generator.generate_full_report(
                analysis_results,
                format='all'
            )
            logger.info(f"✓ Reports generated: {report_files}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            self.stats['errors'].append(f"Report generation error: {str(e)}")
    
    def save_results_to_gcs(self, analysis_results: Dict):
        """Save results back to GCS"""
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS TO GCS")
        logger.info("="*80)
        
        try:
            filename = f"analysis_results/production_analysis_{self.timestamp}.json"
            success = self.gcs_loader.save_analysis_results(analysis_results, filename)
            
            if success:
                logger.info(f"✓ Results saved to GCS: {filename}")
            else:
                logger.error("✗ Failed to save results to GCS")
                
        except Exception as e:
            logger.error(f"Error saving to GCS: {e}")
            self.stats['errors'].append(f"GCS save error: {str(e)}")
    
    def generate_deployment_summary(self):
        """Generate final deployment summary"""
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - self.start_time).total_seconds()
        
        summary = f"""
{"="*80}
PRODUCTION ANALYSIS DEPLOYMENT SUMMARY
{"="*80}

Deployment Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Completion Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {self.stats['processing_time']:.2f} seconds

PROCESSING STATISTICS:
----------------------
Total Patterns Found: {self.stats['total_patterns']:,}
Successfully Processed: {self.stats['processed_patterns']:,}
Failed Patterns: {self.stats['failed_patterns']:,}
Processing Rate: {self.stats['processed_patterns'] / self.stats['processing_time']:.1f} patterns/second

Total Unique Tickers: {self.stats['total_tickers']:,}
Price Data Loaded: {self.stats['processed_tickers']:,}

ERRORS ENCOUNTERED: {len(self.stats['errors'])}
{'-'*40}
"""
        
        if self.stats['errors']:
            for idx, error in enumerate(self.stats['errors'][:10], 1):  # Show first 10 errors
                summary += f"{idx}. {error}\n"
            if len(self.stats['errors']) > 10:
                summary += f"... and {len(self.stats['errors']) - 10} more errors\n"
        else:
            summary += "No errors encountered - Clean run!\n"
        
        summary += f"""
OUTPUT LOCATION:
----------------
Results Directory: {self.output_dir}
Log File: {log_file}

{"="*80}
DEPLOYMENT COMPLETE
{"="*80}
"""
        
        # Save summary to file
        summary_file = self.output_dir / 'deployment_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Print to console
        print(summary)
        logger.info("Deployment summary saved")
        
        return summary
    
    def deploy(self):
        """Main deployment method"""
        logger.info("="*80)
        logger.info("STARTING PRODUCTION DEPLOYMENT")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info("="*80)
        
        try:
            # 1. Initialize GCS
            if not self.initialize_gcs():
                logger.error("Cannot proceed without GCS connection")
                return False
            
            # 2. Load all patterns
            patterns = self.load_all_patterns()
            if not patterns:
                logger.error("No patterns found to analyze")
                return False
            
            # Save patterns for reference
            self.all_patterns = patterns
            
            # 3. Load price data for subset of tickers (top 100 by pattern count)
            ticker_counts = {}
            for p in patterns:
                ticker = p.get('ticker')
                if ticker:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:100]
            top_ticker_list = [t[0] for t in top_tickers]
            
            if top_ticker_list:
                logger.info(f"\nLoading price data for top {len(top_ticker_list)} tickers...")
                price_data = self.load_price_data_parallel(top_ticker_list, max_workers=10)
                self.consolidation_analyzer.price_data = price_data
            
            # 4. Run comprehensive analysis
            analysis_results = self.run_comprehensive_analysis(patterns)
            self.analysis_results = analysis_results
            
            # 5. Generate reports
            self.generate_production_reports(analysis_results)
            
            # 6. Save to GCS
            self.save_results_to_gcs(analysis_results)
            
            # 7. Generate deployment summary
            self.generate_deployment_summary()
            
            logger.info("\n✓ DEPLOYMENT SUCCESSFUL")
            return True
            
        except Exception as e:
            logger.error(f"DEPLOYMENT FAILED: {e}")
            logger.error(traceback.format_exc())
            self.stats['errors'].append(f"Critical deployment error: {str(e)}")
            self.generate_deployment_summary()
            return False


def main():
    """Main entry point for production deployment"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     CONSOLIDATION PATTERN ANALYSIS - PRODUCTION DEPLOYMENT      ║
╠══════════════════════════════════════════════════════════════════╣
║  This will process ALL available data from GCS                  ║
║  Estimated runtime: 5-30 minutes depending on data size         ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Confirm deployment
    response = input("\nProceed with full production deployment? (yes/no): ")
    if response.lower() != 'yes':
        print("Deployment cancelled.")
        return
    
    # Create deployer and run
    deployer = ProductionAnalysisDeployer()
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deployment completed successfully!")
        print(f"📁 Results available at: {deployer.output_dir}")
    else:
        print("\n❌ Deployment failed. Check logs for details.")
        print(f"📝 Log file: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()