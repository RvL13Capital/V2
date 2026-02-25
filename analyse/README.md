# Consolidation Pattern Analysis Tool

Comprehensive analysis system for evaluating consolidation patterns detected by the AIv3 system. This tool provides deep insights into pattern behavior, performance metrics, and detection method comparisons.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Analysis

#### With GCS Data (Recommended)
```bash
# Run analysis with data from Google Cloud Storage
python run_gcs_analysis.py

# With specific date range
python run_gcs_analysis.py --start-date 2024-01-01 --end-date 2024-12-31

# Filter by detection methods
python run_gcs_analysis.py --methods stateful multi_signal
```

#### With Sample Data (Demo)
```bash
# Run with sample data for demonstration
python run_consolidation_analysis.py --use-sample-data

# Or with GCS runner
python run_gcs_analysis.py --use-sample-data
```

## 📊 Analysis Modules

### 1. **Consolidation Analyzer** (`consolidation_analyzer.py`)
- Duration statistics by detection method
- Breakout outcome analysis
- Qualification metrics analysis
- Post-breakout performance tracking
- Optimal pattern identification
- Method comparison

### 2. **Pattern Metrics** (`pattern_metrics.py`)
- Price position within consolidation
- Volume and volatility metrics
- Boundary touch analysis
- Quality scoring
- Momentum indicators
- Gap analysis

### 3. **Breakout Validator** (`breakout_validator.py`)
- Fake-out detection
- Breakout quality classification
- Sustainability scoring
- Re-entry pattern tracking
- Retest counting

### 4. **Statistical Analysis** (`statistical_analysis.py`)
- Chi-square tests for outcome distributions
- T-tests for performance comparisons
- Correlation analysis
- Risk metrics and Kelly Criterion
- Temporal pattern analysis
- Market regime effects

### 5. **Visualization** (`visualization.py`)
- Duration distribution charts
- Outcome distribution plots
- Post-breakout performance graphs
- Method comparison visualizations
- Fakeout analysis charts
- Interactive pattern charts

### 6. **Report Generator** (`report_generator.py`)
- JSON reports for programmatic access
- CSV exports for data analysis
- Excel workbooks with multiple sheets
- HTML reports for web viewing
- Markdown documentation

## 📈 Key Metrics Analyzed

### Duration Analysis
- Average, median, min, max consolidation duration
- Distribution across time buckets (10-15, 16-20, 21-30 days, etc.)
- Comparison across detection methods

### Outcome Analysis
- Distribution of outcome classes (K0-K5)
- Success rates (K2, K3, K4 patterns)
- Exceptional pattern rates (K4)
- Failure rates (K5)

### Post-Breakout Performance
- Price performance at 5, 10, 20, 30, 50, 75, 100 days
- Win rates over time
- Average gains and losses
- Maximum gains/losses

### Risk Metrics
- Failure rates by method
- Risk/reward ratios
- Kelly Criterion for position sizing
- Maximum drawdown statistics

## 🔧 Configuration

### GCS Credentials
Place your GCS credentials file at:
```
C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json
```

Or specify a custom path:
```bash
python run_gcs_analysis.py --credentials /path/to/credentials.json
```

### Environment Variables
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export PROJECT_ID="ignition-ki-csv-storage"
export GCS_BUCKET_NAME="ignition-ki-csv-data-2025-user123"
```

## 📁 Output Structure

```
analysis_output/
├── visualizations/          # Charts and graphs
│   ├── duration_distribution.png
│   ├── outcome_distribution.png
│   ├── post_breakout_performance.png
│   ├── method_comparison.png
│   └── fakeout_analysis.png
├── reports/                 # Generated reports
│   ├── analysis_report_[timestamp].json
│   ├── analysis_report_[timestamp].xlsx
│   ├── analysis_report_[timestamp].html
│   └── analysis_report_[timestamp].md
├── analysis_summary.json    # Summary statistics
├── method_comparison.csv    # Method performance comparison
└── statistical_analysis.json # Statistical test results
```

## 📊 Report Formats

### Excel Report
- **Summary Sheet**: Key metrics and findings
- **Duration Analysis**: Duration statistics by method
- **Outcome Analysis**: Outcome distributions and success rates
- **Method Comparison**: Side-by-side method performance
- **Performance**: Post-breakout performance metrics
- **Optimal Patterns**: Characteristics of best patterns

### HTML Report
- Interactive web-based report
- Styled tables and summaries
- Easy sharing and viewing

### JSON Report
- Complete analysis results
- Programmatic access
- Integration with other systems

## 🎯 Key Findings

The analysis tool helps answer:
- Which detection method performs best?
- What is the optimal consolidation duration?
- How often do fake-outs occur?
- What characteristics lead to exceptional (K4) outcomes?
- What is the risk/reward profile of patterns?
- How do patterns perform in different market conditions?

## 🚦 Command Line Options

```bash
python run_gcs_analysis.py [OPTIONS]

Options:
  --credentials PATH       Path to GCS credentials JSON
  --pattern-file FILE      Specific pattern file in GCS
  --start-date DATE        Start date (YYYY-MM-DD)
  --end-date DATE          End date (YYYY-MM-DD)
  --methods [METHODS...]   Detection methods to include
  --output-dir DIR         Output directory for results
  --reports [FORMATS...]   Report formats (json, csv, excel, html, markdown)
  --no-visualizations      Skip generating charts
  --use-sample-data        Use sample data for demo
```

## 📝 Example Usage

### Full Analysis with All Reports
```bash
python run_gcs_analysis.py --reports all
```

### Specific Time Period Analysis
```bash
python run_gcs_analysis.py \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --reports excel html
```

### Method Comparison
```bash
python run_gcs_analysis.py \
    --methods stateful multi_signal \
    --reports excel \
    --no-visualizations
```

## 🔍 Interpreting Results

### Success Rate
- **>30%**: Excellent performance
- **20-30%**: Good performance
- **10-20%**: Average performance
- **<10%**: Poor performance

### Optimal Duration
- Most successful patterns typically last 15-30 days
- Longer consolidations (>50 days) often have lower success rates

### Fake-out Rate
- **<10%**: Very reliable
- **10-20%**: Acceptable
- **>20%**: Requires additional validation

## 🛠️ Troubleshooting

### GCS Connection Issues
1. Verify credentials file exists and is valid
2. Check internet connection
3. Ensure bucket name is correct
4. Verify service account has necessary permissions

### Memory Issues with Large Datasets
1. Process patterns in batches
2. Limit date range
3. Filter by specific methods
4. Increase system memory allocation

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## 📚 Further Development

Potential enhancements:
- Real-time pattern monitoring
- Machine learning integration
- Advanced risk modeling
- Portfolio optimization
- Backtesting integration
- Alert system for high-quality patterns

## 📧 Support

For issues or questions about the analysis tool, check the logs in the output directory or review the individual module documentation.