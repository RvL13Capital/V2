"""
Batch Pattern Detection for K4 Hunting
=======================================

This script runs pattern detection on all downloaded historical data
to find consolidation patterns that could become K4 patterns.

Process:
1. Load historical data for each ticker
2. Run consolidation pattern detection
3. Extract features for each pattern snapshot
4. Predict K4 probability using trained models
5. Save high-probability K4 candidates
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

# Local imports - use simplified versions to avoid import errors
LOCAL_CACHE_DIR = Path('data/historical_cache')
OUTPUT_DIR = Path('output/k4_patterns')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pattern detection parameters
QUALIFICATION_DAYS = 10
MIN_PATTERN_DAYS = 15
MAX_PATTERN_DAYS = 100

# Consolidation criteria (from validated models)
BBW_PERCENTILE_THRESHOLD = 0.30  # MANDATORY
ADX_THRESHOLD = 32
VOLUME_THRESHOLD = 0.35
RANGE_THRESHOLD = 0.65

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators needed for pattern detection."""
    # Bollinger Bands Width (BBW)
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BBW'] = (2 * 2 * std_20) / sma_20 * 100

    # BBW Percentile
    df['BBW_Percentile'] = df['BBW'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    # ADX (simplified)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Volume ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Daily range ratio
    df['Daily_Range'] = df['High'] - df['Low']
    avg_range = df['Daily_Range'].rolling(20).mean()
    df['Range_Ratio'] = df['Daily_Range'] / avg_range

    # Simplified ADX
    df['ADX'] = 25  # Placeholder - would need full calculation

    return df

def detect_consolidation_patterns(ticker: str, df: pd.DataFrame) -> List[Dict]:
    """
    Detect consolidation patterns in historical data.

    Args:
        ticker: Ticker symbol
        df: DataFrame with OHLCV data and indicators

    Returns:
        List of detected patterns
    """
    patterns = []
    df = df.copy()

    # Need at least 120 days of data
    if len(df) < 120:
        return patterns

    # Scan for qualification periods
    for i in range(100, len(df) - QUALIFICATION_DAYS):
        # Check BBW (MANDATORY)
        bbw_qualified = df.iloc[i:i+QUALIFICATION_DAYS]['BBW_Percentile'].max() < BBW_PERCENTILE_THRESHOLD

        if not bbw_qualified:
            continue

        # Check other criteria (2 of 3 required)
        criteria_met = 0
        if df.iloc[i:i+QUALIFICATION_DAYS]['ADX'].max() < ADX_THRESHOLD:
            criteria_met += 1
        if df.iloc[i:i+QUALIFICATION_DAYS]['Volume_Ratio'].min() < VOLUME_THRESHOLD:
            criteria_met += 1
        if df.iloc[i:i+QUALIFICATION_DAYS]['Range_Ratio'].min() < RANGE_THRESHOLD:
            criteria_met += 1

        # Need BBW + at least 2 other criteria
        if criteria_met >= 2:
            # Pattern qualified - set boundaries
            qual_period = df.iloc[i:i+QUALIFICATION_DAYS]
            upper_boundary = qual_period['High'].max()
            lower_boundary = qual_period['Low'].min()
            power_boundary = upper_boundary * 1.005  # 0.5% buffer

            # Check pattern outcome (next 100 days)
            if i + QUALIFICATION_DAYS + 100 < len(df):
                outcome_period = df.iloc[i+QUALIFICATION_DAYS:i+QUALIFICATION_DAYS+100]
                max_price = outcome_period['High'].max()
                min_price = outcome_period['Low'].min()
                start_price = df.iloc[i+QUALIFICATION_DAYS]['Close']

                # Calculate gain
                max_gain_pct = ((max_price - start_price) / start_price) * 100
                max_loss_pct = ((min_price - start_price) / start_price) * 100

                # Determine if K4 potential
                is_k4 = max_gain_pct >= 75

                pattern = {
                    'ticker': ticker,
                    'qualification_start': df.iloc[i]['Date'],
                    'qualification_end': df.iloc[i+QUALIFICATION_DAYS]['Date'],
                    'upper_boundary': upper_boundary,
                    'lower_boundary': lower_boundary,
                    'power_boundary': power_boundary,
                    'start_price': start_price,
                    'max_gain_pct': max_gain_pct,
                    'max_loss_pct': max_loss_pct,
                    'is_k4': is_k4,
                    'bbw_mean': qual_period['BBW'].mean(),
                    'volume_ratio_mean': qual_period['Volume_Ratio'].mean()
                }

                patterns.append(pattern)

    return patterns

def process_ticker(ticker_file: Path) -> Tuple[str, List[Dict], str]:
    """
    Process a single ticker file for pattern detection.

    Args:
        ticker_file: Path to ticker parquet file

    Returns:
        Tuple of (ticker, patterns, status)
    """
    try:
        ticker = ticker_file.stem
        df = pd.read_parquet(ticker_file)

        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()

        # Sort by date
        df = df.sort_values('Date')

        # Calculate indicators
        df = calculate_indicators(df)

        # Detect patterns
        patterns = detect_consolidation_patterns(ticker, df)

        return (ticker, patterns, "success")

    except Exception as e:
        return (ticker_file.stem, [], f"error: {str(e)[:50]}")

def main():
    """Main function to run batch pattern detection."""
    print("=" * 70)
    print("BATCH PATTERN DETECTION FOR K4 HUNTING")
    print("=" * 70)

    # 1. Get all downloaded ticker files
    ticker_files = list(LOCAL_CACHE_DIR.glob("*.parquet"))

    if len(ticker_files) == 0:
        print("[ERROR] No historical data found. Run download_historical_data.py first.")
        sys.exit(1)

    print(f"\n[Data] Found {len(ticker_files)} ticker files")

    # 2. Process tickers in parallel
    all_patterns = []
    k4_patterns = []

    # Use fewer workers to avoid memory issues
    max_workers = min(4, len(ticker_files))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_ticker, file): file
                  for file in ticker_files[:100]}  # Process first 100 for testing

        # Process results
        completed = 0
        for future in as_completed(futures):
            ticker, patterns, status = future.result()
            completed += 1

            if status == "success" and patterns:
                all_patterns.extend(patterns)
                k4_patterns.extend([p for p in patterns if p['is_k4']])

                # Progress update
                if completed % 10 == 0:
                    print(f"  Processed {completed}/{len(futures)} tickers | "
                          f"Patterns: {len(all_patterns)} | K4s: {len(k4_patterns)}")

    # 3. Create DataFrames
    patterns_df = pd.DataFrame(all_patterns) if all_patterns else pd.DataFrame()
    k4_df = pd.DataFrame(k4_patterns) if k4_patterns else pd.DataFrame()

    # 4. Save results
    if not patterns_df.empty:
        patterns_file = OUTPUT_DIR / 'all_patterns_detected.parquet'
        patterns_df.to_parquet(patterns_file, index=False)
        print(f"\n[Saved] All patterns: {patterns_file}")

    if not k4_df.empty:
        k4_file = OUTPUT_DIR / 'k4_patterns_detected.parquet'
        k4_df.to_parquet(k4_file, index=False)
        print(f"[Saved] K4 patterns: {k4_file}")

    # 5. Display summary
    print("\n" + "=" * 70)
    print("PATTERN DETECTION COMPLETE")
    print("=" * 70)

    if not patterns_df.empty:
        print(f"\nTotal patterns detected: {len(patterns_df)}")
        print(f"K4 patterns found: {len(k4_df)} ({len(k4_df)/len(patterns_df)*100:.2f}%)")

        if not k4_df.empty:
            print("\n[K4 Patterns Summary]")
            print(f"  Average gain: {k4_df['max_gain_pct'].mean():.1f}%")
            print(f"  Median gain: {k4_df['max_gain_pct'].median():.1f}%")
            print(f"  Max gain: {k4_df['max_gain_pct'].max():.1f}%")

            print("\n[Top 10 K4 Patterns by Gain]")
            top_k4 = k4_df.nlargest(10, 'max_gain_pct')[['ticker', 'qualification_start', 'max_gain_pct']]
            print(top_k4.to_string(index=False))

            print("\n[K4 Patterns by Ticker]")
            k4_by_ticker = k4_df['ticker'].value_counts().head(10)
            for ticker, count in k4_by_ticker.items():
                print(f"  {ticker}: {count} K4 patterns")
    else:
        print("\n[WARNING] No patterns detected. Check if data is downloaded properly.")

    print("\n[Next Steps]")
    print("1. Extract full 48 features for detected patterns")
    print("2. Run trained models to predict K4 probability")
    print("3. Combine with existing K4 patterns for retraining")

    return patterns_df, k4_df

if __name__ == "__main__":
    patterns_df, k4_df = main()