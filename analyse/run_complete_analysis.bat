@echo off
REM Complete Analysis Pipeline - Process ALL Data + Generate Full Report
REM Runs parallel extraction on entire GCS dataset and produces comprehensive report

echo ======================================================================
echo COMPLETE ANALYSIS PIPELINE - ALL DATA + FULL REPORT
echo ======================================================================
echo.
echo This will:
echo   1. Process ALL 3,548+ tickers from GCS (parallel processing)
echo   2. Extract features for all detected patterns
echo   3. Prepare ML-ready training dataset
echo   4. Generate comprehensive analysis report
echo.
echo Estimated total time: 30-90 minutes
echo.
echo Press Ctrl+C to cancel, or
pause

REM Record start time
set START_TIME=%time%
set START_DATE=%date%

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    if exist "AI Infra\venv\Scripts\activate.bat" (
        echo Using AI Infra virtual environment...
        call "AI Infra\venv\Scripts\activate.bat"
    ) else (
        echo Warning: Virtual environment not found, using system Python
    )
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo ======================================================================
echo PHASE 1: PARALLEL PATTERN DETECTION + FEATURE EXTRACTION
echo ======================================================================
echo Processing ALL available tickers with parallel workers...
echo.

python extract_all_features_parallel.py
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo X PHASE 1 FAILED - Parallel extraction failed
    echo ======================================================================
    echo Check log file: extract_features_parallel_*.log
    pause
    exit /b 1
)

echo.
echo [OK] Phase 1 Complete - Features extracted
echo.

echo ======================================================================
echo PHASE 2: TRAINING DATA PREPARATION
echo ======================================================================
echo.

REM Find most recent parallel features file
for /f "delims=" %%i in ('dir /b /o-d "AI Infra\data\features\parallel_pattern_features_*.parquet" 2^>nul') do (
    set FEATURES_FILE=%%i
    goto :found_features
)

echo [ERROR] No features file found!
pause
exit /b 1

:found_features
echo Found features file: %FEATURES_FILE%
echo Preparing training dataset...
echo.

cd "AI Infra\hybrid_model"

REM Create production training script
echo import sys > prepare_production.py
echo from pathlib import Path >> prepare_production.py
echo sys.path.insert(0, str(Path(__file__).parent)) >> prepare_production.py
echo from test_prepare_training import prepare_training_data >> prepare_production.py
echo. >> prepare_production.py
echo result = prepare_training_data( >> prepare_production.py
echo     features_file=Path("../data/features/%FEATURES_FILE%"), >> prepare_production.py
echo     output_dir=Path("../data/raw"), >> prepare_production.py
echo     test_mode=False >> prepare_production.py
echo ) >> prepare_production.py
echo. >> prepare_production.py
echo if not result or not result['success']: >> prepare_production.py
echo     sys.exit(1) >> prepare_production.py

python prepare_production.py
set PREP_EXIT=%errorlevel%
del prepare_production.py

if %PREP_EXIT% NEQ 0 (
    echo.
    echo ======================================================================
    echo X PHASE 2 FAILED - Training data preparation failed
    echo ======================================================================
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo [OK] Phase 2 Complete - Training data ready
echo.

echo ======================================================================
echo PHASE 3: COMPREHENSIVE REPORT GENERATION
echo ======================================================================
echo.

REM Create comprehensive report generator
echo import pandas as pd > generate_report.py
echo import json >> generate_report.py
echo from pathlib import Path >> generate_report.py
echo from datetime import datetime >> generate_report.py
echo. >> generate_report.py
echo print("="*70) >> generate_report.py
echo print("GENERATING COMPREHENSIVE ANALYSIS REPORT") >> generate_report.py
echo print("="*70) >> generate_report.py
echo. >> generate_report.py
echo # Load metadata >> generate_report.py
echo metadata_file = Path("AI Infra/data/raw/production_training_metadata.json") >> generate_report.py
echo with open(metadata_file, 'r') as f: >> generate_report.py
echo     metadata = json.load(f) >> generate_report.py
echo. >> generate_report.py
echo # Load training data >> generate_report.py
echo training_file = Path("AI Infra/data/raw/production_training_data.parquet") >> generate_report.py
echo df = pd.read_parquet(training_file) >> generate_report.py
echo. >> generate_report.py
echo # Load patterns file >> generate_report.py
echo patterns_files = list(Path("output").glob("patterns_parallel_*.parquet")) >> generate_report.py
echo if patterns_files: >> generate_report.py
echo     patterns_df = pd.read_parquet(sorted(patterns_files)[-1]) >> generate_report.py
echo else: >> generate_report.py
echo     patterns_df = None >> generate_report.py
echo. >> generate_report.py
echo # Generate report >> generate_report.py
echo report_lines = [] >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("COMPLETE ANALYSIS REPORT") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append(f"Generated: {datetime.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S')}") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Dataset Overview >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("DATASET OVERVIEW") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append(f"Total Patterns: {metadata['total_patterns']:,}") >> generate_report.py
echo report_lines.append(f"Total Features: {metadata['features']['total']}") >> generate_report.py
echo if patterns_df is not None: >> generate_report.py
echo     report_lines.append(f"Unique Tickers: {patterns_df['ticker'].nunique()}") >> generate_report.py
echo     report_lines.append(f"Date Range: {patterns_df['pattern_start_date'].min().date()} to {patterns_df['pattern_end_date'].max().date()}") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Feature Breakdown >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("FEATURE BREAKDOWN") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append(f"Volume Features: {metadata['features']['volume']}") >> generate_report.py
echo report_lines.append(f"Pattern Features: {metadata['features']['pattern']}") >> generate_report.py
echo report_lines.append(f"Trend Features: {metadata['features']['trend']}") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Data Splits >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("DATA SPLITS (Time-Series)") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append(f"Train: {metadata['splits']['train']:,} patterns ({metadata['splits']['train']/metadata['total_patterns']*100:.1f}%%)") >> generate_report.py
echo report_lines.append(f"Val:   {metadata['splits']['val']:,} patterns ({metadata['splits']['val']/metadata['total_patterns']*100:.1f}%%)") >> generate_report.py
echo report_lines.append(f"Test:  {metadata['splits']['test']:,} patterns ({metadata['splits']['test']/metadata['total_patterns']*100:.1f}%%)") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Target Distribution >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("TARGET DISTRIBUTION (K3_K4 Binary)") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append(f"Positive (K3/K4): {metadata['target_distribution']['positive']:,} ({metadata['target_distribution']['positive_rate']:.1f}%%)") >> generate_report.py
echo report_lines.append(f"Negative (Others): {metadata['target_distribution']['negative']:,} ({100-metadata['target_distribution']['positive_rate']:.1f}%%)") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Outcome Class Distribution >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("OUTCOME CLASS DISTRIBUTION") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo class_dist = df['outcome_class'].value_counts().sort_index() >> generate_report.py
echo for cls, count in class_dist.items(): >> generate_report.py
echo     pct = count / len(df) * 100 >> generate_report.py
echo     report_lines.append(f"{cls:20s}: {count:>6,} ({pct:>5.1f}%%)") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Statistics by Split >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("POSITIVE CLASS BY SPLIT") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo for split_name in ['train', 'val', 'test']: >> generate_report.py
echo     split_df = df[df['split'] == split_name] >> generate_report.py
echo     if len(split_df) ^> 0: >> generate_report.py
echo         pos_count = split_df['target'].sum() >> generate_report.py
echo         pos_pct = pos_count / len(split_df) * 100 >> generate_report.py
echo         report_lines.append(f"{split_name.title():5s}: {pos_count:>4,}/{len(split_df):>6,} ({pos_pct:>5.1f}%%)") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Top Volume Features >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("TOP 10 VOLUME FEATURES (Sample)") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo volume_features = [col for col in metadata['features']['columns'] if 'vol_' in col or 'obv' in col or 'accum' in col] >> generate_report.py
echo for i, feat in enumerate(volume_features[:10]): >> generate_report.py
echo     if feat in df.columns: >> generate_report.py
echo         mean_val = df[feat].mean() >> generate_report.py
echo         std_val = df[feat].std() >> generate_report.py
echo         report_lines.append(f"{feat:30s}: mean={mean_val:>8.4f}, std={std_val:>8.4f}") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Top Tickers by Pattern Count >> generate_report.py
echo if patterns_df is not None: >> generate_report.py
echo     report_lines.append("="*70) >> generate_report.py
echo     report_lines.append("TOP 20 TICKERS BY PATTERN COUNT") >> generate_report.py
echo     report_lines.append("="*70) >> generate_report.py
echo     ticker_counts = patterns_df['ticker'].value_counts().head(20) >> generate_report.py
echo     for ticker, count in ticker_counts.items(): >> generate_report.py
echo         pct = count / len(patterns_df) * 100 >> generate_report.py
echo         report_lines.append(f"{ticker:10s}: {count:>6,} patterns ({pct:>5.2f}%%)") >> generate_report.py
echo     report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Data Quality Check >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("DATA QUALITY VALIDATION") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo feature_cols = metadata['features']['columns'] >> generate_report.py
echo nan_count = df[feature_cols].isna().sum().sum() >> generate_report.py
echo inf_count = df[feature_cols].apply(lambda x: pd.api.types.is_numeric_dtype(x) and (x == float('inf')).sum() + (x == float('-inf')).sum()).sum() >> generate_report.py
echo report_lines.append(f"NaN values: {nan_count:,} {'[OK]' if nan_count == 0 else '[WARNING]'}") >> generate_report.py
echo report_lines.append(f"Infinite values: {inf_count:,} {'[OK]' if inf_count == 0 else '[WARNING]'}") >> generate_report.py
echo report_lines.append(f"Duplicate rows: {df.duplicated().sum():,} {'[OK]' if df.duplicated().sum() == 0 else '[WARNING]'}") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo. >> generate_report.py
echo # Recommendations >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo report_lines.append("NEXT STEPS ^& RECOMMENDATIONS") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo if metadata['target_distribution']['positive'] ^>= 500: >> generate_report.py
echo     report_lines.append(f"[OK] Sufficient positive examples ({metadata['target_distribution']['positive']:,}) for ML training") >> generate_report.py
echo else: >> generate_report.py
echo     report_lines.append(f"[WARNING] Limited positive examples ({metadata['target_distribution']['positive']:,}), consider SMOTE/oversampling") >> generate_report.py
echo. >> generate_report.py
echo if metadata['total_patterns'] ^>= 5000: >> generate_report.py
echo     report_lines.append(f"[OK] Large dataset ({metadata['total_patterns']:,} patterns) - good for robust training") >> generate_report.py
echo else: >> generate_report.py
echo     report_lines.append(f"[INFO] Moderate dataset ({metadata['total_patterns']:,} patterns) - adequate for initial training") >> generate_report.py
echo. >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo report_lines.append("Recommended next steps:") >> generate_report.py
echo report_lines.append("  1. Review this report for data quality issues") >> generate_report.py
echo report_lines.append("  2. Train LightGBM model: cd AI Infra/hybrid_model ^&^& python integrated_self_training.py train") >> generate_report.py
echo report_lines.append("  3. Validate model: python integrated_self_training.py validate") >> generate_report.py
echo report_lines.append("  4. Run backtest: python automated_backtesting.py --start-date 2022-01-01") >> generate_report.py
echo report_lines.append("") >> generate_report.py
echo report_lines.append("="*70) >> generate_report.py
echo. >> generate_report.py
echo # Write report to file >> generate_report.py
echo report_file = Path(f"output/complete_analysis_report_{datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S')}.txt") >> generate_report.py
echo report_file.parent.mkdir(exist_ok=True) >> generate_report.py
echo with open(report_file, 'w') as f: >> generate_report.py
echo     f.write('\n'.join(report_lines)) >> generate_report.py
echo. >> generate_report.py
echo # Print report >> generate_report.py
echo print('\n'.join(report_lines)) >> generate_report.py
echo print(f"\n[OK] Report saved to: {report_file}") >> generate_report.py

python generate_report.py
set REPORT_EXIT=%errorlevel%
del generate_report.py

if %REPORT_EXIT% NEQ 0 (
    echo.
    echo ======================================================================
    echo X PHASE 3 FAILED - Report generation failed
    echo ======================================================================
    pause
    exit /b 1
)

REM Calculate elapsed time
set END_TIME=%time%
set END_DATE=%date%

echo.
echo ======================================================================
echo COMPLETE ANALYSIS PIPELINE FINISHED
echo ======================================================================
echo Start: %START_DATE% %START_TIME%
echo End:   %END_DATE% %END_TIME%
echo.
echo Output files created:
echo   - Patterns:  output\patterns_parallel_*.parquet
echo   - Features:  AI Infra\data\features\parallel_pattern_features_*.parquet
echo   - Training:  AI Infra\data\raw\production_training_data.parquet
echo   - Metadata:  AI Infra\data\raw\production_training_metadata.json
echo   - Report:    output\complete_analysis_report_*.txt
echo   - Log:       extract_features_parallel_*.log
echo.
echo ======================================================================
echo.
echo Review the complete_analysis_report_*.txt for detailed statistics
echo.
pause
