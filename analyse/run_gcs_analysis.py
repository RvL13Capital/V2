"""
GCS-Integrated Consolidation Analysis Runner
Runs comprehensive analysis using data from Google Cloud Storage
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import GCS configuration
from gcs_config import GCSDataLoader, setup_gcs_environment, load_patterns_for_analysis

# Import analysis components
from run_consolidation_analysis import ConsolidationAnalysisRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCSAnalysisRunner(ConsolidationAnalysisRunner):
    """Extended analysis runner with GCS integration"""
    
    def __init__(self, output_dir: str = './analysis_output'):
        super().__init__(output_dir=output_dir)
        self.gcs_loader = None
        
    def initialize_gcs(self, credentials_path: str = None):
        """Initialize GCS connection"""
        
        # Setup environment
        if not setup_gcs_environment():
            logger.error("Failed to setup GCS environment")
            return False
        
        try:
            # Initialize GCS loader
            self.gcs_loader = GCSDataLoader(credentials_path)
            logger.info("GCS connection initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GCS: {e}")
            return False
    
    def load_patterns_from_gcs(self, 
                              pattern_file: str = None,
                              start_date: str = None,
                              end_date: str = None,
                              methods: List[str] = None):
        """Load patterns from GCS with filtering"""
        
        if not self.gcs_loader:
            logger.error("GCS not initialized. Call initialize_gcs() first.")
            return
        
        # Load patterns
        if pattern_file:
            patterns = self.gcs_loader.load_patterns_from_gcs(pattern_file)
        else:
            patterns = self.gcs_loader.load_historical_patterns(
                start_date=start_date,
                end_date=end_date,
                methods=methods
            )
        
        if not patterns:
            logger.warning("No patterns found in GCS")
            return
        
        # Convert to pattern objects
        self.consolidation_analyzer.patterns = [
            self.consolidation_analyzer._dict_to_pattern(p) for p in patterns
        ]
        
        # Store raw patterns
        self.all_results['raw_patterns'] = patterns
        
        # Load price data for patterns if needed
        unique_tickers = set(p.get('ticker') for p in patterns if p.get('ticker'))
        logger.info(f"Loaded {len(patterns)} patterns for {len(unique_tickers)} tickers")
        
        # Optionally load price data
        price_data = {}
        for ticker in list(unique_tickers)[:10]:  # Limit to first 10 for demo
            df = self.gcs_loader.load_price_data(ticker)
            if not df.empty:
                price_data[ticker] = df
        
        self.consolidation_analyzer.price_data = price_data
        logger.info(f"Loaded price data for {len(price_data)} tickers")
    
    def save_results_to_gcs(self):
        """Save analysis results back to GCS"""
        
        if not self.gcs_loader:
            logger.error("GCS not initialized")
            return False
        
        # Save results
        success = self.gcs_loader.save_analysis_results(self.all_results)
        
        if success:
            logger.info("Results saved to GCS successfully")
        else:
            logger.error("Failed to save results to GCS")
        
        return success
    
    def run_full_gcs_analysis(self, 
                            pattern_file: str = None,
                            start_date: str = None,
                            end_date: str = None,
                            methods: List[str] = None):
        """Run complete analysis with GCS data"""
        
        logger.info("="*60)
        logger.info("STARTING GCS-INTEGRATED CONSOLIDATION ANALYSIS")
        logger.info("="*60)
        
        # Initialize GCS
        if not self.initialize_gcs():
            logger.error("Cannot proceed without GCS connection")
            return None
        
        # Load patterns from GCS
        logger.info("Loading patterns from GCS...")
        self.load_patterns_from_gcs(
            pattern_file=pattern_file,
            start_date=start_date,
            end_date=end_date,
            methods=methods
        )
        
        if not self.consolidation_analyzer.patterns:
            logger.warning("No patterns loaded. Using sample data for demonstration...")
            self.load_sample_data()
        
        # Run analysis
        results = self.run_full_analysis()
        
        # Save results to GCS
        logger.info("Saving results to GCS...")
        self.save_results_to_gcs()
        
        return results


def main():
    """Main entry point for GCS-integrated analysis"""
    
    parser = argparse.ArgumentParser(
        description='Run consolidation analysis with GCS data'
    )
    
    parser.add_argument(
        '--credentials',
        type=str,
        help='Path to GCS credentials JSON file',
        default=r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
    )
    
    parser.add_argument(
        '--pattern-file',
        type=str,
        help='Specific pattern file in GCS to load',
        default=None
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for pattern filtering (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for pattern filtering (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        help='Detection methods to include',
        choices=['stateful', 'multi_signal', 'traditional', 'support_resistance'],
        default=None
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results',
        default='./gcs_analysis_output'
    )
    
    parser.add_argument(
        '--reports',
        nargs='+',
        choices=['json', 'csv', 'excel', 'html', 'markdown', 'all'],
        default=['excel', 'html'],
        help='Report formats to generate'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use sample data instead of GCS data'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = GCSAnalysisRunner(output_dir=args.output_dir)
    
    if args.use_sample_data:
        # Use sample data
        logger.info("Using sample data for demonstration...")
        runner.load_sample_data()
        results = runner.run_full_analysis()
    else:
        # Run GCS analysis
        results = runner.run_full_gcs_analysis(
            pattern_file=args.pattern_file,
            start_date=args.start_date,
            end_date=args.end_date,
            methods=args.methods
        )
    
    if results:
        # Generate visualizations
        if not args.no_visualizations:
            runner.generate_visualizations()
        
        # Generate reports
        runner.generate_reports(args.reports)
        
        # Export results
        runner.export_results()
        
        # Print summary
        runner.print_summary()
        
        logger.info("="*60)
        logger.info(f"Analysis complete! Results saved to {args.output_dir}")
        logger.info("="*60)
    else:
        logger.error("Analysis failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()