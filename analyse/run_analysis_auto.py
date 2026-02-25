"""
Auto-run filtered analysis without user prompts
Uses only real data from GCS - no mock data generation
"""

import sys
import os
from datetime import datetime

# Import the filtered analysis pipeline
from run_complete_analysis_filtered import FilteredAnalysisPipeline

def main():
    """Run filtered analysis automatically"""
    
    print("="*80)
    print("FILTERED CONSOLIDATION ANALYSIS - AUTO RUN")
    print(f"Minimum Price Requirement: $0.01")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nThis analysis will:")
    print("1. Filter out all stocks/patterns with prices below $0.01")
    print("2. Detect consolidation patterns only in valid price ranges")
    print("3. Calculate real days to reach targets from actual price data")
    print("4. Generate comprehensive PDF report with filtered data")
    print("\nUsing only real data from GCS - no mock data")
    print("="*80)
    print("\nStarting analysis...")
    
    # Create and run pipeline
    pipeline = FilteredAnalysisPipeline()
    
    # Run the complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n" + "="*80)
        print("❌ ANALYSIS FAILED")
        print("="*80)
        print("Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()