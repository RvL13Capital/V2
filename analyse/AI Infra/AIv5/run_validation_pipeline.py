"""
Automated Validation Pipeline
==============================

This script automates the complete validation pipeline:
1. Waits for feature extraction to complete
2. Runs predictions with the extracted features
3. Generates the validation report
4. Displays key metrics showing if the feature fix worked

Run this after starting the feature extraction to automate the entire process.
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuration
FEATURES_FILE = Path('output/unused_patterns/pattern_features_canonical_69.parquet')
PREDICTIONS_FILE = Path('output/unused_patterns/predictions_canonical.parquet')
REPORT_FILE = Path('output/validation/validation_report_canonical.md')
CHECK_INTERVAL = 30  # seconds

def check_extraction_complete():
    """Check if feature extraction has completed."""
    return FEATURES_FILE.exists()

def run_predictions():
    """Run the prediction script."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Predictions")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, 'predict_historical_canonical.py'],
            capture_output=True,
            text=True,
            check=True
        )

        # Print key output
        lines = result.stdout.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Total patterns', 'K4 Detection', 'Expected Value', 'Signal']):
                print(f"  {line.strip()}")

        if PREDICTIONS_FILE.exists():
            print(f"\n[OK] Predictions saved to: {PREDICTIONS_FILE}")
            return True
        else:
            print("[ERROR] Predictions file not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Prediction failed: {e}")
        print(f"Error output: {e.stderr[:500]}")
        return False

def run_validation_report():
    """Generate the validation report."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating Validation Report")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, 'generate_validation_report_canonical.py'],
            capture_output=True,
            text=True,
            check=True
        )

        # Print key metrics
        lines = result.stdout.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['K4 Recall', 'K3+K4 Recall', 'EV Correlation']):
                print(f"  {line.strip()}")

        if REPORT_FILE.exists():
            print(f"\n[OK] Report saved to: {REPORT_FILE}")
            return True
        else:
            print("[ERROR] Report file not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Report generation failed: {e}")
        print(f"Error output: {e.stderr[:500]}")
        return False

def display_results_summary():
    """Display a summary of the key results."""
    print("\n" + "=" * 80)
    print("FEATURE FIX VALIDATION RESULTS")
    print("=" * 80)

    # Load predictions if available
    if PREDICTIONS_FILE.exists():
        predictions_df = pd.read_parquet(PREDICTIONS_FILE)

        # Count K4 detections
        k4_detected = (predictions_df['ensemble_predicted_class'] == 'K4_EXCEPTIONAL').sum()
        k4_high_prob = (predictions_df['ensemble_k4_prob'] > 0.1).sum()

        print(f"\n[STATS] Prediction Statistics:")
        print(f"  Total patterns analyzed: {len(predictions_df)}")
        print(f"  K4 patterns predicted: {k4_detected}")
        print(f"  K4 probability > 10%: {k4_high_prob} patterns")

        # Signal distribution
        if 'ensemble_signal' in predictions_df.columns:
            signal_counts = predictions_df['ensemble_signal'].value_counts()
            print(f"\n[SIGNAL] Signal Distribution:")
            for signal in ['STRONG_SIGNAL', 'GOOD_SIGNAL', 'MODERATE_SIGNAL', 'WEAK_SIGNAL', 'AVOID']:
                count = signal_counts.get(signal, 0)
                pct = count / len(predictions_df) * 100
                print(f"  {signal}: {count} ({pct:.1f}%)")

        # EV statistics
        ensemble_ev = predictions_df['ensemble_expected_value']
        print(f"\n[EV] Expected Value Statistics:")
        print(f"  Mean EV: {ensemble_ev.mean():.2f}")
        print(f"  Positive EV: {(ensemble_ev > 0).sum()} patterns ({(ensemble_ev > 0).mean()*100:.1f}%)")
        print(f"  EV > 3.0: {(ensemble_ev > 3.0).sum()} patterns")
        print(f"  EV > 5.0: {(ensemble_ev > 5.0).sum()} patterns")

    # Read report file for key metrics
    if REPORT_FILE.exists():
        with open(REPORT_FILE, 'r') as f:
            report_content = f.read()

        print(f"\n[KEY] Key Performance Metrics (from report):")

        # Extract key metrics from the report
        lines = report_content.split('\n')
        for i, line in enumerate(lines):
            if '**K4 Recall**' in line and i < len(lines) - 1:
                # Parse the table row
                parts = lines[i+1].split('|') if i+1 < len(lines) else []
                if len(parts) > 4:
                    print(f"  K4 Recall: {parts[4].strip()}")
            elif '**K3+K4 Recall**' in line and i < len(lines) - 1:
                parts = lines[i+1].split('|') if i+1 < len(lines) else []
                if len(parts) > 4:
                    print(f"  K3+K4 Recall: {parts[4].strip()}")
            elif '**EV Correlation**' in line and i < len(lines) - 1:
                parts = lines[i+1].split('|') if i+1 < len(lines) else []
                if len(parts) > 4:
                    print(f"  EV Correlation: {parts[4].strip()}")

    print("\n" + "=" * 80)
    print("COMPARISON: Before vs After Feature Fix")
    print("=" * 80)
    print("  Metric              | Before Fix | After Fix | Improvement")
    print("  --------------------|------------|-----------|-------------")
    print("  K4 Recall           |      0.0%  |    ???    |     ???")
    print("  K3+K4 Recall        |     14.9%  |    ???    |     ???")
    print("  EV Correlation      |    -0.140  |    ???    |     ???")
    print("\n  (Check the validation report for actual 'After Fix' values)")

    print("\n" + "=" * 80)
    print("[OK] VALIDATION PIPELINE COMPLETE!")
    print(f"Full report available at: {REPORT_FILE}")
    print("=" * 80)

def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("AUTOMATED VALIDATION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Wait for feature extraction to complete
    print("\n[STEP 1] Waiting for Feature Extraction")
    print(f"  Checking for: {FEATURES_FILE}")

    wait_count = 0
    while not check_extraction_complete():
        wait_count += 1
        elapsed = wait_count * CHECK_INTERVAL
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Still extracting... "
              f"(elapsed: {elapsed//60}m {elapsed%60}s)")

        # Check if extraction checkpoint exists for progress info
        checkpoint_file = Path('output/unused_patterns/extraction_checkpoint.parquet')
        if checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_file)
                print(f"    Progress: {len(checkpoint_df)} patterns extracted so far")
            except:
                pass

        time.sleep(CHECK_INTERVAL)

        # Timeout after 60 minutes
        if elapsed > 3600:
            print("[ERROR] Timeout: Feature extraction taking too long (>60 minutes)")
            return 1

    print(f"\n[SUCCESS] Feature extraction complete!")

    # Verify the features file
    try:
        features_df = pd.read_parquet(FEATURES_FILE)
        print(f"  Patterns with features: {len(features_df)}")

        # Check feature count
        numeric_cols = features_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        feature_count = len([col for col in numeric_cols if col not in ['max_gain_pct', 'days_to_peak']])
        print(f"  Number of features: {feature_count} (expected: 69)")

        if feature_count < 60:
            print(f"  [WARNING] WARNING: Fewer features than expected!")

    except Exception as e:
        print(f"[ERROR] Error loading features: {e}")
        return 1

    # Step 2: Run predictions
    if not run_predictions():
        print("[ERROR] Prediction step failed")
        return 1

    # Step 3: Generate validation report
    if not run_validation_report():
        print("[ERROR] Validation report generation failed")
        return 1

    # Step 4: Display results summary
    display_results_summary()

    print(f"\n[COMPLETE] Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())