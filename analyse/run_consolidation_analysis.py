"""
Main Runner Script for Comprehensive Consolidation Pattern Analysis
Orchestrates all analysis modules and generates complete reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys

# Import analysis modules
from consolidation_analyzer import ConsolidationAnalyzer
from pattern_metrics import PatternMetrics
from breakout_validator import BreakoutValidator
from statistical_analysis import StatisticalAnalyzer
from visualization import PatternVisualizer
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsolidationAnalysisRunner:
    """Main runner for comprehensive consolidation analysis"""
    
    def __init__(self, 
                 data_path: str = None,
                 output_dir: str = './analysis_output'):
        """
        Initialize the analysis runner
        
        Args:
            data_path: Path to data files
            output_dir: Directory for output files
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all analysis components
        self.consolidation_analyzer = ConsolidationAnalyzer(data_path)
        self.pattern_metrics = PatternMetrics()
        self.breakout_validator = BreakoutValidator()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = PatternVisualizer(str(self.output_dir / 'visualizations'))
        self.report_generator = ReportGenerator(str(self.output_dir / 'reports'))
        
        # Storage for results
        self.all_results = {}
        
    def load_sample_data(self) -> None:
        """Load sample pattern data for demonstration"""
        
        logger.info("Loading sample pattern data...")
        
        # Create sample patterns for demonstration
        sample_patterns = []
        
        # Generate diverse sample patterns
        methods = ['stateful', 'multi_signal', 'traditional', 'support_resistance']
        outcomes = ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(200):
            method = np.random.choice(methods)
            outcome = np.random.choice(outcomes, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.10])
            
            # Calculate max gain based on outcome
            if outcome == 'K0':
                max_gain = np.random.uniform(0, 5)
            elif outcome == 'K1':
                max_gain = np.random.uniform(5, 15)
            elif outcome == 'K2':
                max_gain = np.random.uniform(15, 35)
            elif outcome == 'K3':
                max_gain = np.random.uniform(35, 75)
            elif outcome == 'K4':
                max_gain = np.random.uniform(75, 150)
            else:  # K5
                max_gain = np.random.uniform(-50, -5)
            
            start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 300))
            duration = np.random.randint(10, 60)
            end_date = start_date + timedelta(days=duration)
            
            pattern = {
                'ticker': f'STOCK{i % 50}',
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'detection_method': method,
                'upper_boundary': 100 + np.random.uniform(-10, 10),
                'lower_boundary': 90 + np.random.uniform(-10, 10),
                'power_boundary': 101 + np.random.uniform(-10, 10),
                'qualification_metrics': {
                    'avg_bbw': np.random.uniform(10, 30),
                    'avg_adx': np.random.uniform(15, 35),
                    'avg_volume_ratio': np.random.uniform(0.2, 0.5),
                    'avg_range_ratio': np.random.uniform(0.3, 0.7)
                },
                'breakout_date': (end_date + timedelta(days=1)).isoformat(),
                'breakout_direction': 'up' if max_gain > 0 else 'down',
                'breakout_price': 100 + np.random.uniform(-5, 5),
                'outcome_class': outcome,
                'max_gain': max_gain,
                'days_to_max': np.random.randint(5, 100),
                'duration': duration,
                'boundary_width_pct': ((100 - 90) / 90) * 100,
                'is_fakeout': np.random.choice([True, False], p=[0.2, 0.8]) if outcome == 'K5' else False,
                'sustainability_score': np.random.uniform(0, 1),
                'fakeout_days': np.random.randint(1, 10) if outcome == 'K5' else 0
            }
            
            sample_patterns.append(pattern)
        
        # Load patterns into analyzer
        self.consolidation_analyzer.patterns = [
            self.consolidation_analyzer._dict_to_pattern(p) for p in sample_patterns
        ]
        
        # Store raw patterns for other analyses
        self.all_results['raw_patterns'] = sample_patterns
        
        logger.info(f"Loaded {len(sample_patterns)} sample patterns")
        
    def run_full_analysis(self) -> Dict:
        """Run complete analysis pipeline"""
        
        logger.info("Starting comprehensive consolidation analysis...")
        
        # 1. Analyze consolidation duration
        logger.info("Analyzing consolidation durations...")
        duration_stats = self.consolidation_analyzer.analyze_consolidation_duration()
        self.all_results['duration_stats'] = duration_stats
        
        # 2. Analyze breakout outcomes
        logger.info("Analyzing breakout outcomes...")
        outcome_stats = self.consolidation_analyzer.analyze_breakout_outcomes()
        self.all_results['outcome_stats'] = outcome_stats
        
        # 3. Analyze qualification metrics
        logger.info("Analyzing qualification metrics...")
        qualification_metrics = self.consolidation_analyzer.analyze_qualification_metrics()
        self.all_results['qualification_metrics'] = qualification_metrics
        
        # 4. Analyze post-breakout performance
        logger.info("Analyzing post-breakout performance...")
        performance_stats = self.consolidation_analyzer.analyze_post_breakout_performance()
        self.all_results['post_breakout_performance'] = performance_stats
        
        # 5. Identify optimal patterns
        logger.info("Identifying optimal pattern characteristics...")
        optimal_patterns = self.consolidation_analyzer.identify_optimal_patterns()
        self.all_results['optimal_patterns'] = optimal_patterns
        
        # 6. Compare detection methods
        logger.info("Comparing detection methods...")
        method_comparison = self.consolidation_analyzer.compare_detection_methods()
        self.all_results['method_comparison'] = method_comparison
        
        # 7. Statistical analysis
        logger.info("Performing statistical analysis...")
        patterns_by_method = {}
        for pattern in self.all_results.get('raw_patterns', []):
            method = pattern['detection_method']
            if method not in patterns_by_method:
                patterns_by_method[method] = []
            patterns_by_method[method].append(pattern)
        
        stat_comparison = self.statistical_analyzer.compare_methods_performance(patterns_by_method)
        self.all_results['statistical_comparison'] = stat_comparison
        
        pattern_characteristics = self.statistical_analyzer.analyze_pattern_characteristics(
            self.all_results.get('raw_patterns', [])
        )
        self.all_results['pattern_characteristics'] = pattern_characteristics
        
        risk_metrics = self.statistical_analyzer.calculate_risk_metrics(
            self.all_results.get('raw_patterns', [])
        )
        self.all_results['risk_metrics'] = risk_metrics
        
        # 8. Generate summary
        logger.info("Generating analysis summary...")
        summary = self.consolidation_analyzer.generate_summary_report()
        self.all_results['summary'] = summary
        
        # 9. Breakout validation analysis
        logger.info("Analyzing breakout validation...")
        fakeout_analysis = self.breakout_validator.analyze_breakout_patterns(
            self.all_results.get('raw_patterns', [])
        )
        self.all_results['fakeout_analysis'] = fakeout_analysis.to_dict('records') if not fakeout_analysis.empty else []
        
        logger.info("Analysis complete!")
        return self.all_results
        
    def generate_visualizations(self) -> None:
        """Generate all visualizations"""
        
        logger.info("Generating visualizations...")
        
        # Create output directory for visualizations
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Duration distribution
        if 'duration_stats' in self.all_results:
            logger.info("Creating duration distribution plot...")
            self.visualizer.plot_duration_distribution(
                self.all_results['duration_stats']
            )
        
        # 2. Outcome distribution
        if 'outcome_stats' in self.all_results:
            logger.info("Creating outcome distribution plot...")
            self.visualizer.plot_outcome_distribution(
                self.all_results['outcome_stats']
            )
        
        # 3. Post-breakout performance
        if 'post_breakout_performance' in self.all_results:
            logger.info("Creating post-breakout performance plot...")
            self.visualizer.plot_post_breakout_performance(
                self.all_results['post_breakout_performance']
            )
        
        # 4. Method comparison
        if 'method_comparison' in self.all_results:
            logger.info("Creating method comparison plot...")
            self.visualizer.plot_method_comparison(
                self.all_results['method_comparison']
            )
        
        # 5. Fakeout analysis
        if 'fakeout_analysis' in self.all_results:
            logger.info("Creating fakeout analysis plot...")
            self.visualizer.plot_fakeout_analysis(
                self.all_results['fakeout_analysis']
            )
        
        logger.info("Visualizations complete!")
        
    def generate_reports(self, formats: List[str] = ['all']) -> Dict:
        """Generate reports in specified formats"""
        
        logger.info(f"Generating reports in formats: {formats}")
        
        generated_files = {}
        
        for format_type in formats:
            files = self.report_generator.generate_full_report(
                self.all_results,
                format=format_type
            )
            generated_files.update(files)
        
        logger.info(f"Reports generated: {generated_files}")
        return generated_files
        
    def export_results(self) -> None:
        """Export all analysis results"""
        
        logger.info("Exporting analysis results...")
        
        # Export main analyzer results
        self.consolidation_analyzer.export_results(str(self.output_dir))
        
        # Export statistical analysis results
        stat_results_file = self.output_dir / 'statistical_analysis.json'
        with open(stat_results_file, 'w') as f:
            json.dump(
                self.statistical_analyzer.analysis_results,
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"Results exported to {self.output_dir}")
        
    def print_summary(self) -> None:
        """Print analysis summary to console"""
        
        print("\n" + "="*80)
        print("CONSOLIDATION PATTERN ANALYSIS SUMMARY")
        print("="*80)
        
        summary = self.all_results.get('summary', {})
        
        print(f"\nTotal Patterns Analyzed: {summary.get('total_patterns', 0)}")
        print(f"Unique Tickers: {summary.get('unique_tickers', 0)}")
        print(f"Date Range: {summary.get('date_range', {}).get('earliest', 'N/A')} to {summary.get('date_range', {}).get('latest', 'N/A')}")
        print(f"Detection Methods: {', '.join(summary.get('detection_methods', []))}")
        
        print(f"\nOverall Success Rate: {summary.get('overall_success_rate', 0):.1%}")
        print(f"Exceptional Patterns (K4): {summary.get('exceptional_patterns', 0)}")
        print(f"Failed Patterns (K5): {summary.get('failed_patterns', 0)}")
        
        print("\nKey Findings:")
        for finding in summary.get('key_findings', []):
            print(f"  • {finding}")
        
        # Method comparison
        if 'method_comparison' in self.all_results:
            print("\n" + "-"*40)
            print("METHOD COMPARISON")
            print("-"*40)
            print(self.all_results['method_comparison'].to_string())
        
        # Risk metrics
        if 'risk_metrics' in self.all_results:
            risk = self.all_results['risk_metrics']
            print("\n" + "-"*40)
            print("RISK METRICS")
            print("-"*40)
            print(f"Failure Rate: {risk.get('failure_rate', 0):.1%}")
            
            if 'win_loss_ratios' in risk:
                wl = risk['win_loss_ratios']
                print(f"Win Rate: {wl.get('win_rate', 0):.1%}")
                print(f"Win/Loss Ratio: {wl.get('win_loss_ratio', 0):.2f}")
            
            if 'risk_reward_ratios' in risk:
                rr = risk['risk_reward_ratios']
                print(f"Average Gain: {rr.get('average_gain', 0):.2f}%")
                print(f"Average Loss: {rr.get('average_loss', 0):.2f}%")
                print(f"Risk/Reward Ratio: {rr.get('ratio', 0):.2f}")
                print(f"Kelly Criterion: {rr.get('kelly_criterion', 0):.1%}")
        
        print("\n" + "="*80)

def main():
    """Main entry point for the analysis"""
    
    parser = argparse.ArgumentParser(
        description='Run comprehensive consolidation pattern analysis'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data files',
        default=None
    )
    
    parser.add_argument(
        '--patterns-file',
        type=str,
        help='Path to patterns JSON file',
        default=None
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results',
        default='./analysis_output'
    )
    
    parser.add_argument(
        '--reports',
        nargs='+',
        choices=['json', 'csv', 'excel', 'html', 'markdown', 'all'],
        default=['all'],
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
        help='Use sample data for demonstration'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ConsolidationAnalysisRunner(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Load data
    if args.use_sample_data:
        runner.load_sample_data()
    elif args.patterns_file:
        runner.consolidation_analyzer.load_patterns(args.patterns_file)
    else:
        logger.error("No data source specified. Use --use-sample-data or --patterns-file")
        sys.exit(1)
    
    # Run analysis
    results = runner.run_full_analysis()
    
    # Generate visualizations
    if not args.no_visualizations:
        runner.generate_visualizations()
    
    # Generate reports
    runner.generate_reports(args.reports)
    
    # Export results
    runner.export_results()
    
    # Print summary
    runner.print_summary()
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()