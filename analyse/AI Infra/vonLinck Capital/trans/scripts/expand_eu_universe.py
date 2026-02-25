#!/usr/bin/env python3
"""
Expand EU universe by cleaning CSV files and converting to parquet.
Focuses on genuine EU equities with market cap data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_FILE = DATA_DIR / "market_cap_cache.json"

# Quality thresholds
MIN_DAYS = 100  # Minimum trading days
MAX_ZERO_VOLUME_PCT = 50  # Max % of zero volume days
MAX_GAP_PCT = 100  # Max single-day gap (%)

# US companies to exclude (cross-listings)
US_COMPANIES = {'AAPL', 'MSFT', 'NVDA', 'GOOG', 'GOOA', 'GOOC', 'AMZN', 'META', 'TSLA',
                'JPM', 'V', 'MA', 'JNJ', 'UNH', 'HD', 'PG', 'XOM', 'CVX', 'BAC',
                'WMT', 'KO', 'PEP', 'MRK', 'ABBV', 'COST', 'TMO', 'AVGO', 'NKE', 'MCD',
                'AMD', 'INTC', 'CSCO', 'ORCL', 'CRM', 'ACN', 'IBM', 'QCOM', 'TXN', 'NOW',
                'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'LLY', 'MDT', 'SYK', 'ISRG', 'REGN',
                'NFLX', 'ADBE', 'PYPL', 'DIS', 'CMCSA', 'VZ', 'T', 'PFE', 'WFC', 'C',
                'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BK'}


def load_cache():
    with open(CACHE_FILE) as f:
        return json.load(f)


def is_genuine_eu(ticker: str) -> bool:
    """Check if ticker is genuine EU equity (not US cross-listing)."""
    base = ticker.split('.')[0].upper()
    return not any(base.startswith(us) or base == us for us in US_COMPANIES)


def detect_gbp_conversion(df: pd.DataFrame) -> list:
    """Detect GBp/GBP conversion points."""
    if len(df) < 2:
        return []

    returns = df['close'].pct_change().abs()
    # Look for ~100x jumps (99% or 101%)
    conversions = []
    for i in range(1, len(returns)):
        if pd.notna(returns.iloc[i]):
            if 0.98 < returns.iloc[i] < 1.02:  # ~100% change
                conversions.append(i)
            elif returns.iloc[i] > 50:  # >5000% spike also suspicious
                # Check if it reverts
                if i + 1 < len(df):
                    next_ret = df['close'].iloc[i+1] / df['close'].iloc[i] - 1
                    if -0.99 < next_ret < -0.98:
                        conversions.append(i)
    return conversions


def fix_gbp_conversion(df: pd.DataFrame) -> tuple:
    """Fix GBp/GBP conversions."""
    df = df.copy()
    conversions = detect_gbp_conversion(df)

    if not conversions:
        return df, 0

    price_cols = ['open', 'high', 'low', 'close', 'adj_close']

    # Build segments
    segments = []
    prev_end = 0
    for conv_idx in conversions:
        if conv_idx > prev_end:
            median_price = df.loc[prev_end:conv_idx-1, 'close'].median()
            segments.append((prev_end, conv_idx - 1, median_price))
        prev_end = conv_idx

    if prev_end < len(df):
        median_price = df.loc[prev_end:, 'close'].median()
        segments.append((prev_end, len(df) - 1, median_price))

    if not segments:
        return df, 0

    # Find dominant scale
    largest_segment = max(segments, key=lambda s: s[1] - s[0])
    target_scale = largest_segment[2]

    fix_count = 0
    for start, end, median_price in segments:
        if median_price == 0:
            continue
        ratio = target_scale / median_price
        if 50 < ratio < 200:
            for col in price_cols:
                if col in df.columns:
                    df.loc[start:end, col] = df.loc[start:end, col] * 100
            fix_count += 1
        elif 0.005 < ratio < 0.02:
            for col in price_cols:
                if col in df.columns:
                    df.loc[start:end, col] = df.loc[start:end, col] / 100
            fix_count += 1

    return df, fix_count


def interpolate_spikes(df: pd.DataFrame, threshold: float = 0.5) -> tuple:
    """Interpolate single-day spikes that revert."""
    df = df.copy()
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    fix_count = 0

    for i in range(1, len(df) - 1):
        curr = df['close'].iloc[i]
        prev = df['close'].iloc[i-1]
        next_val = df['close'].iloc[i+1]

        if prev > 0 and curr > 0 and next_val > 0:
            up_move = (curr - prev) / prev
            down_move = (next_val - curr) / curr

            # Spike up then down (or vice versa)
            if up_move > threshold and down_move < -threshold * 0.8:
                # Interpolate
                for col in price_cols:
                    if col in df.columns:
                        df.loc[df.index[i], col] = (df[col].iloc[i-1] + df[col].iloc[i+1]) / 2
                fix_count += 1
            elif up_move < -threshold and down_move > threshold * 0.8:
                for col in price_cols:
                    if col in df.columns:
                        df.loc[df.index[i], col] = (df[col].iloc[i-1] + df[col].iloc[i+1]) / 2
                fix_count += 1

    return df, fix_count


def check_quality(df: pd.DataFrame) -> tuple:
    """Check data quality. Returns (is_ok, reason)."""
    if len(df) < MIN_DAYS:
        return False, f"short_history ({len(df)} days)"

    # Check zero volume
    if 'volume' in df.columns:
        zero_vol_pct = (df['volume'] == 0).sum() / len(df) * 100
        if zero_vol_pct > MAX_ZERO_VOLUME_PCT:
            return False, f"illiquid ({zero_vol_pct:.0f}% zero vol)"

    # Check max gap
    if 'close' in df.columns and len(df) > 1:
        returns = df['close'].pct_change().abs() * 100
        max_gap = returns.max()
        if max_gap > MAX_GAP_PCT:
            return False, f"extreme_gap ({max_gap:.0f}%)"

    return True, "ok"


def process_csv(csv_path: Path) -> tuple:
    """Process a single CSV file. Returns (success, df, reason)."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Check for required columns
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return False, None, "missing_columns"

        # Check for duplicate columns (corrupted CSV)
        if len(df.columns) != len(set(df.columns)):
            return False, None, "duplicate_columns"

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Add adj_close if missing
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']

        # Fix GBp/GBP for .L tickers
        ticker = csv_path.stem
        if ticker.endswith('.L'):
            df, gbp_fixes = fix_gbp_conversion(df)
        else:
            gbp_fixes = 0

        # Interpolate spikes
        df, spike_fixes = interpolate_spikes(df)

        # Check quality
        is_ok, reason = check_quality(df)
        if not is_ok:
            return False, None, reason

        return True, df, f"ok (gbp:{gbp_fixes}, spikes:{spike_fixes})"

    except Exception as e:
        return False, None, f"error: {str(e)[:30]}"


def main():
    print("=" * 60)
    print("EU UNIVERSE EXPANSION")
    print("=" * 60)

    cache = load_cache()
    eu_ext = ['.L', '.DE', '.PA', '.MI', '.AS', '.SW', '.ST', '.OL', '.VI', '.HE', '.BR', '.IR', '.MC', '.LS', '.CO']

    # Find CSV files with market cap, not in parquet
    parquet_tickers = set(f.stem for f in RAW_DIR.glob('*.parquet'))
    csv_files = list(RAW_DIR.glob('*.csv'))

    candidates = []
    for f in csv_files:
        ticker = f.stem
        if ticker in parquet_tickers:
            continue
        if not any(ticker.endswith(e) for e in eu_ext):
            continue
        if ticker not in cache:
            continue
        if cache[ticker].get('market_cap', 0) <= 0:
            continue
        if not is_genuine_eu(ticker):
            continue
        candidates.append(f)

    print(f"Candidate CSV files: {len(candidates)}")
    print()

    # Process
    success = 0
    failed = 0
    reasons = {}

    for i, csv_path in enumerate(candidates):
        ticker = csv_path.stem

        if (i + 1) % 500 == 0:
            print(f"Progress: {i+1}/{len(candidates)} ({success} ok, {failed} failed)")

        ok, df, reason = process_csv(csv_path)

        if ok:
            # Save as parquet
            parquet_path = RAW_DIR / f"{ticker}.parquet"
            df.to_parquet(parquet_path, index=False)
            success += 1
        else:
            failed += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(candidates)}")
    print(f"Success (converted to parquet): {success}")
    print(f"Failed: {failed}")
    print()
    print("FAILURE REASONS:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Final count
    new_parquet = len(list(RAW_DIR.glob('*.parquet')))
    eu_parquet = [f.stem for f in RAW_DIR.glob('*.parquet') if any(f.stem.endswith(e) for e in eu_ext)]
    usable = [t for t in eu_parquet if t in cache and cache[t].get('market_cap', 0) > 0]

    print()
    print(f"Total parquet files: {new_parquet}")
    print(f"EU parquet with market cap: {len(usable)}")


if __name__ == "__main__":
    main()
