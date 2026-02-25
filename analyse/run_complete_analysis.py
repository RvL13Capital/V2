"""
Complete Analysis Pipeline with PDF Report Generation
Full production analysis with extended metrics and comprehensive PDF report
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Import all analysis modules
from run_full_analysis import FullAnalysisPipeline
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

class CompleteAnalysisPipeline:
    """Complete analysis pipeline with PDF report"""
    
    def __init__(self):
        self.timestamp = timestamp
        self.output_dir = Path(f'./complete_analysis_{self.timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize base pipeline
        self.base_pipeline = FullAnalysisPipeline()
        self.base_pipeline.output_dir = self.output_dir
        
        self.patterns = []
        self.extended_metrics = {}
        
    def run_pattern_detection(self):
        """Run pattern detection on all tickers"""
        logger.info("="*80)
        logger.info("PHASE 1: PATTERN DETECTION")
        logger.info("="*80)
        
        # Get all ticker files
        ticker_files = self.base_pipeline.get_all_ticker_files()
        
        if not ticker_files:
            logger.error("No ticker files found")
            return False
        
        logger.info(f"Found {len(ticker_files)} ticker files")
        
        # Process all tickers
        self.base_pipeline.process_all_tickers_parallel(ticker_files)
        
        # Store patterns
        self.patterns = self.base_pipeline.all_patterns
        
        logger.info(f"Pattern detection complete: {len(self.patterns)} patterns found")
        
        # Save patterns
        patterns_file = self.output_dir / f'patterns_{self.timestamp}.json'
        with open(patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        
        return True
    
    def run_extended_analysis(self):
        """Run extended analysis on patterns"""
        logger.info("="*80)
        logger.info("PHASE 2: EXTENDED ANALYSIS")
        logger.info("="*80)
        
        if not self.patterns:
            logger.error("No patterns to analyze")
            return False
        
        # Create extended analyzer
        analyzer = ExtendedPatternAnalyzer(self.patterns)
        
        # Calculate all extended metrics
        logger.info("Calculating time to targets...")
        self.extended_metrics['time_to_targets'] = analyzer.calculate_time_to_targets()
        
        logger.info("Analyzing consolidation duration by outcome...")
        self.extended_metrics['duration_by_outcome'] = analyzer.analyze_consolidation_duration_by_outcome()
        
        logger.info("Analyzing false breakouts...")
        self.extended_metrics['false_breakouts'] = analyzer.analyze_false_breakouts()
        
        logger.info("Analyzing by detection method...")
        self.extended_metrics['by_method'] = analyzer.analyze_by_detection_method()
        
        logger.info("Calculating pattern quality metrics...")
        self.extended_metrics['quality_metrics'] = analyzer.get_pattern_quality_metrics()
        
        logger.info("Calculating risk/reward metrics...")
        self.extended_metrics['risk_reward'] = analyzer.calculate_risk_reward_metrics()
        
        logger.info("Getting example patterns...")
        self.extended_metrics['example_patterns'] = analyzer.get_example_patterns(n=5)
        
        # Add basic statistics
        df = pd.DataFrame(self.patterns)
        if not df.empty:
            self.extended_metrics['basic_stats'] = {
                'total_patterns': len(df),
                'unique_tickers': df['ticker'].nunique(),
                'date_range': {
                    'start': df['start_date'].min(),
                    'end': df['end_date'].max()
                },
                'outcome_distribution': df['outcome_class'].value_counts().to_dict()
            }
        
        logger.info("Extended analysis complete")
        
        # Save extended metrics
        metrics_file = self.output_dir / f'extended_metrics_{self.timestamp}.json'
        with open(metrics_file, 'w') as f:
            # Convert any numpy types to Python types for JSON serialization
            json.dump(self._serialize_metrics(self.extended_metrics), f, indent=2)
        
        return True
    
    def _serialize_metrics(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._serialize_metrics(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_metrics(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        logger.info("="*80)
        logger.info("PHASE 3: PDF REPORT GENERATION")
        logger.info("="*80)
        
        if not self.patterns:
            logger.error("No patterns for report generation")
            return False
        
        # Create PDF generator
        pdf_generator = PDFReportGenerator(self.patterns, self.extended_metrics)
        
        # Generate report
        pdf_path = self.output_dir / f'analysis_report_{self.timestamp}.pdf'
        pdf_generator.generate_report(str(pdf_path))
        
        logger.info(f"PDF report generated: {pdf_path}")
        
        return True
    
    def generate_summary_report(self):
        """Generate text summary report"""
        logger.info("Generating summary report...")
        
        summary_file = self.output_dir / f'summary_{self.timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPLETE CONSOLIDATION ANALYSIS SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Basic statistics
            if 'basic_stats' in self.extended_metrics:
                stats = self.extended_metrics['basic_stats']
                f.write("OVERVIEW\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Patterns: {stats['total_patterns']:,}\n")
                f.write(f"Unique Tickers: {stats['unique_tickers']:,}\n")
                f.write(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}\n\n")
            
            # Risk/Reward metrics
            if 'risk_reward' in self.extended_metrics:
                rr = self.extended_metrics['risk_reward']
                f.write("RISK/REWARD METRICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Average Gain: {rr.get('avg_gain', 0):.2f}%\n")
                f.write(f"Average Loss: {rr.get('avg_loss', 0):.2f}%\n")
                f.write(f"Win Rate: {rr.get('win_rate', 0):.1f}%\n")
                f.write(f"Risk/Reward Ratio: {rr.get('risk_reward_ratio', 0):.2f}\n")
                f.write(f"Expectancy: {rr.get('expectancy', 0):.2f}%\n\n")
            
            # False breakouts
            if 'false_breakouts' in self.extended_metrics:
                fb = self.extended_metrics['false_breakouts']
                f.write("FALSE BREAKOUT ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Upward Breakouts: {fb.get('total_upward_breakouts', 0)}\n")
                f.write(f"False Breakouts: {fb.get('false_upward_breakouts', 0)}\n")
                f.write(f"False Breakout Rate: {fb.get('false_breakout_rate', 0):.1f}%\n\n")
            
            # Quality metrics
            if 'quality_metrics' in self.extended_metrics:
                qm = self.extended_metrics['quality_metrics']
                if 'high_quality_patterns' in qm:
                    hq = qm['high_quality_patterns']
                    f.write("HIGH QUALITY PATTERNS (K3, K4)\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Count: {hq.get('count', 0)}\n")
                    f.write(f"Average Gain: {hq.get('avg_gain', 0):.2f}%\n")
                    f.write(f"Average Duration: {hq.get('avg_duration', 0):.1f} days\n\n")
        
        logger.info(f"Summary report saved: {summary_file}")
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        
        start_time = datetime.now()
        
        logger.info("="*80)
        logger.info("STARTING COMPLETE ANALYSIS PIPELINE")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*80)
        
        try:
            # Phase 1: Pattern Detection
            if not self.run_pattern_detection():
                logger.error("Pattern detection failed")
                return False
            
            # Phase 2: Extended Analysis
            if not self.run_extended_analysis():
                logger.error("Extended analysis failed")
                return False
            
            # Phase 3: PDF Report Generation
            if not self.generate_pdf_report():
                logger.error("PDF generation failed")
                return False
            
            # Generate summary
            self.generate_summary_report()
            
            # Final statistics
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            logger.info("="*80)
            logger.info("ANALYSIS COMPLETE")
            logger.info("="*80)
            logger.info(f"Total Time: {duration:.1f} minutes")
            logger.info(f"Patterns Found: {len(self.patterns):,}")
            logger.info(f"Results Directory: {self.output_dir}")
            
            print("\n" + "="*80)
            print("✅ COMPLETE ANALYSIS SUCCESSFUL!")
            print("="*80)
            print(f"📁 Results: {self.output_dir}/")
            print(f"📄 PDF Report: {self.output_dir}/analysis_report_{self.timestamp}.pdf")
            print(f"📊 Patterns: {len(self.patterns):,}")
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
    print("COMPLETE CONSOLIDATION ANALYSIS WITH PDF REPORT")
    print("="*80)
    print("\nThis will:")
    print("1. Detect patterns in ALL tickers from GCS")
    print("2. Calculate extended metrics")
    print("3. Generate comprehensive PDF report")
    print("\nEstimated time: 15-45 minutes")
    print("="*80)
    
    response = input("\nProceed with complete analysis? (yes/no): ")
    if response.lower() != 'yes':
        print("Analysis cancelled")
        return
    
    # Run pipeline
    pipeline = CompleteAnalysisPipeline()
    success = pipeline.run_complete_pipeline()
    
    if not success:
        print("\n❌ Analysis failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()