"""
Complete Pipeline: Generate Patterns → Analyze → Report
Full production pipeline for GCS data
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Import components
from generate_patterns_from_gcs import GCSPatternGenerator
from run_consolidation_analysis import ConsolidationAnalysisRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullPipeline:
    """Complete pattern generation and analysis pipeline"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'./pipeline_output_{self.timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        self.patterns_file = None
        self.patterns = []
        
    def step1_generate_patterns(self, limit: int = None):
        """Step 1: Generate patterns from raw GCS data"""
        
        logger.info("\n" + "="*80)
        logger.info("STEP 1: PATTERN GENERATION")
        logger.info("="*80)
        
        try:
            # Create pattern generator
            generator = GCSPatternGenerator()
            
            # Generate patterns
            patterns = generator.generate_all_patterns(limit=limit)
            
            if not patterns:
                logger.error("No patterns generated!")
                return False
            
            # Save patterns
            patterns_file = self.output_dir / f'patterns_{self.timestamp}.json'
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            self.patterns_file = patterns_file
            self.patterns = patterns
            
            # Get summary
            summary = generator.get_pattern_summary()
            
            logger.info(f"\nGeneration Complete:")
            logger.info(f"  Total Patterns: {summary['total_patterns']}")
            logger.info(f"  Unique Tickers: {summary['unique_tickers']}")
            logger.info(f"  Saved to: {patterns_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pattern generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_analyze_patterns(self):
        """Step 2: Run comprehensive analysis on generated patterns"""
        
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PATTERN ANALYSIS")
        logger.info("="*80)
        
        if not self.patterns:
            logger.error("No patterns to analyze!")
            return False
        
        try:
            # Create analysis runner
            runner = ConsolidationAnalysisRunner(
                output_dir=str(self.output_dir / 'analysis')
            )
            
            # Load patterns into analyzer
            runner.consolidation_analyzer.patterns = [
                runner.consolidation_analyzer._dict_to_pattern(p) for p in self.patterns
            ]
            runner.all_results['raw_patterns'] = self.patterns
            
            # Run full analysis
            results = runner.run_full_analysis()
            
            # Generate visualizations
            logger.info("\nGenerating visualizations...")
            runner.generate_visualizations()
            
            # Generate reports
            logger.info("\nGenerating reports...")
            runner.generate_reports(['excel', 'html', 'json'])
            
            # Print summary
            runner.print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self, ticker_limit: int = None):
        """Run complete pipeline"""
        
        start_time = datetime.now()
        
        logger.info("="*80)
        logger.info("STARTING FULL PIPELINE")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*80)
        
        # Step 1: Generate patterns
        success = self.step1_generate_patterns(limit=ticker_limit)
        
        if not success:
            logger.error("Pipeline failed at pattern generation")
            return False
        
        # Step 2: Analyze patterns
        success = self.step2_analyze_patterns()
        
        if not success:
            logger.error("Pipeline failed at analysis")
            return False
        
        # Complete
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        
        # Create summary file
        summary = {
            'timestamp': self.timestamp,
            'duration_seconds': duration,
            'patterns_generated': len(self.patterns),
            'output_directory': str(self.output_dir),
            'patterns_file': str(self.patterns_file) if self.patterns_file else None
        }
        
        summary_file = self.output_dir / 'pipeline_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return True


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete pattern generation and analysis pipeline')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 10 tickers')
    parser.add_argument('--quick', action='store_true', help='Quick mode - process only 50 tickers')
    
    args = parser.parse_args()
    
    # Determine limit
    limit = args.limit
    if args.test:
        limit = 10
        logger.info("TEST MODE - Processing only 10 tickers")
    elif args.quick:
        limit = 50
        logger.info("QUICK MODE - Processing only 50 tickers")
    
    # Confirm with user
    if not limit:
        print("\n" + "="*60)
        print("WARNING: This will process ALL tickers in GCS")
        print("This may take 30-60 minutes or more")
        print("="*60)
        response = input("\nProceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            return
    
    # Run pipeline
    pipeline = FullPipeline()
    success = pipeline.run_pipeline(ticker_limit=limit)
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Results: {pipeline.output_dir}")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()