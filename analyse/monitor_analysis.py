"""
Real-time Monitoring Dashboard for Production Analysis
Provides live updates and progress tracking during analysis execution
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AnalysisMonitor:
    """Monitor and track analysis progress"""
    
    def __init__(self, analysis_dir: str = './production_analysis'):
        self.analysis_dir = Path(analysis_dir)
        self.current_run = None
        self.start_time = None
        
    def find_latest_run(self) -> Optional[Path]:
        """Find the most recent analysis run directory"""
        if not self.analysis_dir.exists():
            return None
        
        run_dirs = [d for d in self.analysis_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            return None
        
        # Sort by modification time
        latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest
    
    def display_progress(self):
        """Display real-time progress of analysis"""
        
        print("\n" + "="*80)
        print("CONSOLIDATION ANALYSIS MONITOR")
        print("="*80)
        
        run_dir = self.find_latest_run()
        if not run_dir:
            print("No analysis runs found.")
            return
        
        print(f"Monitoring: {run_dir.name}")
        print("-"*80)
        
        # Check for log file
        log_files = list(Path('./logs').glob('full_analysis_*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            # Parse log for progress
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Extract key metrics
            patterns_loaded = 0
            patterns_processed = 0
            current_stage = "Initializing"
            errors = []
            
            for line in lines:
                if "Total patterns loaded:" in line:
                    try:
                        patterns_loaded = int(line.split(":")[-1].strip())
                    except:
                        pass
                elif "patterns processed" in line.lower():
                    try:
                        patterns_processed = int(line.split("processed")[0].split()[-1])
                    except:
                        pass
                elif "LOADING ALL PATTERNS" in line:
                    current_stage = "Loading Patterns"
                elif "RUNNING COMPREHENSIVE ANALYSIS" in line:
                    current_stage = "Running Analysis"
                elif "GENERATING PRODUCTION REPORTS" in line:
                    current_stage = "Generating Reports"
                elif "SAVING RESULTS TO GCS" in line:
                    current_stage = "Saving to GCS"
                elif "DEPLOYMENT COMPLETE" in line:
                    current_stage = "Complete"
                elif "ERROR" in line or "Error" in line:
                    errors.append(line.strip())
            
            # Display progress
            print(f"Current Stage: {current_stage}")
            print(f"Patterns Loaded: {patterns_loaded:,}")
            print(f"Patterns Processed: {patterns_processed:,}")
            
            if patterns_loaded > 0:
                progress_pct = (patterns_processed / patterns_loaded) * 100
                print(f"Progress: {progress_pct:.1f}%")
                
                # Progress bar
                bar_length = 50
                filled_length = int(bar_length * patterns_processed // patterns_loaded)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"[{bar}] {patterns_processed}/{patterns_loaded}")
            
            print(f"\nErrors Encountered: {len(errors)}")
            if errors and len(errors) <= 5:
                for error in errors[-5:]:  # Show last 5 errors
                    print(f"  - {error[:100]}...")
        
        # Check for output files
        print("\n" + "-"*80)
        print("OUTPUT FILES:")
        
        if run_dir.exists():
            # Check visualizations
            viz_dir = run_dir / 'visualizations'
            if viz_dir.exists():
                viz_files = list(viz_dir.glob('*.png'))
                print(f"  Visualizations: {len(viz_files)} files")
            
            # Check reports
            report_dir = run_dir / 'reports'
            if report_dir.exists():
                report_files = list(report_dir.iterdir())
                print(f"  Reports: {len(report_files)} files")
                for report in report_files:
                    print(f"    - {report.name} ({report.stat().st_size / 1024:.1f} KB)")
            
            # Check summary
            summary_file = run_dir / 'deployment_summary.txt'
            if summary_file.exists():
                print(f"\n  ✓ Deployment Summary Available")
    
    def get_analysis_stats(self, run_dir: Path) -> Dict:
        """Extract statistics from completed analysis"""
        stats = {}
        
        # Look for analysis results JSON
        json_files = list((run_dir / 'reports').glob('*.json')) if (run_dir / 'reports').exists() else []
        
        if json_files:
            latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_json, 'r') as f:
                    data = json.load(f)
                
                if 'summary' in data:
                    stats['total_patterns'] = data['summary'].get('total_patterns', 0)
                    stats['success_rate'] = data['summary'].get('overall_success_rate', 0)
                    stats['exceptional_patterns'] = data['summary'].get('exceptional_patterns', 0)
                    stats['failed_patterns'] = data['summary'].get('failed_patterns', 0)
                
                if 'deployment_stats' in data:
                    stats['processing_time'] = data['deployment_stats'].get('processing_time', 0)
                    stats['errors'] = len(data['deployment_stats'].get('errors', []))
                
            except Exception as e:
                logger.error(f"Error reading stats: {e}")
        
        return stats
    
    def display_summary(self):
        """Display summary of completed analysis"""
        run_dir = self.find_latest_run()
        if not run_dir:
            print("No completed analysis found.")
            return
        
        stats = self.get_analysis_stats(run_dir)
        
        if stats:
            print("\n" + "="*80)
            print("ANALYSIS SUMMARY")
            print("="*80)
            print(f"Total Patterns: {stats.get('total_patterns', 'N/A'):,}")
            print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
            print(f"Exceptional Patterns (K4): {stats.get('exceptional_patterns', 'N/A')}")
            print(f"Failed Patterns (K5): {stats.get('failed_patterns', 'N/A')}")
            print(f"Processing Time: {stats.get('processing_time', 0):.1f} seconds")
            print(f"Errors: {stats.get('errors', 0)}")
            
        # Display deployment summary if available
        summary_file = run_dir / 'deployment_summary.txt'
        if summary_file.exists():
            print("\n" + "-"*80)
            print("DEPLOYMENT SUMMARY:")
            print("-"*80)
            with open(summary_file, 'r') as f:
                print(f.read())
    
    def monitor_live(self, refresh_interval: int = 5):
        """Live monitoring with auto-refresh"""
        
        print("Starting live monitoring (Press Ctrl+C to stop)...")
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                self.display_progress()
                
                # Check if analysis is complete
                run_dir = self.find_latest_run()
                if run_dir:
                    summary_file = run_dir / 'deployment_summary.txt'
                    if summary_file.exists():
                        print("\n" + "="*80)
                        print("✓ ANALYSIS COMPLETE!")
                        print("="*80)
                        self.display_summary()
                        break
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            return


def main():
    """Main monitoring entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor consolidation analysis progress')
    parser.add_argument('--live', action='store_true', help='Enable live monitoring')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh interval in seconds')
    parser.add_argument('--summary', action='store_true', help='Show summary of latest run')
    
    args = parser.parse_args()
    
    monitor = AnalysisMonitor()
    
    if args.summary:
        monitor.display_summary()
    elif args.live:
        monitor.monitor_live(refresh_interval=args.refresh)
    else:
        monitor.display_progress()


if __name__ == "__main__":
    main()