#!/usr/bin/env python3
"""
Regime Data Download Script
============================

Downloads VIX and ETF data required for regime-aware features.

Data Sources:
    - VIX: ^VIX from Yahoo Finance
    - ETFs: SPY, TLT, JNK, GLD from Yahoo Finance

Output:
    - data/market_regime/vix_daily.parquet
    - data/market_regime/etf_daily.parquet

Usage:
    python scripts/download_regime_data.py
    python scripts/download_regime_data.py --start-date 2015-01-01
    python scripts/download_regime_data.py --update  # Incremental update

Jan 2026 - Created for regime-aware feature implementation
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_START_DATE = '2015-01-01'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'market_regime'

# Tickers to download
VIX_TICKER = '^VIX'
ETF_TICKERS = ['SPY', 'TLT', 'JNK', 'GLD']


def download_vix_data(start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Download VIX index data from Yahoo Finance.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: date, vix_close, vix_high, vix_low
    """
    logger.info(f"Downloading VIX data from {start_date}...")

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        vix = yf.Ticker(VIX_TICKER)
        df = vix.history(start=start_date, end=end_date)

        if df.empty:
            logger.error("No VIX data returned")
            return pd.DataFrame()

        # Reset index and rename columns
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Close': 'vix_close',
            'High': 'vix_high',
            'Low': 'vix_low',
            'Open': 'vix_open'
        })

        # Keep only needed columns
        df = df[['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']]

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        logger.info(f"Downloaded {len(df)} VIX records ({df['date'].min()} to {df['date'].max()})")
        return df

    except Exception as e:
        logger.error(f"Failed to download VIX: {e}")
        return pd.DataFrame()


def download_etf_data(tickers: list, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Download ETF data for risk-on/risk-off indicator.

    Args:
        tickers: List of ETF tickers (SPY, TLT, JNK, GLD)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: date, {ticker}_close, {ticker}_return_20d
    """
    logger.info(f"Downloading ETF data: {tickers}...")

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    all_data = []

    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            df = etf.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            df = df.reset_index()
            df = df.rename(columns={'Date': 'date', 'Close': f'{ticker.lower()}_close'})
            df = df[['date', f'{ticker.lower()}_close']]
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            # Calculate 20-day return
            df[f'{ticker.lower()}_return_20d'] = df[f'{ticker.lower()}_close'].pct_change(20)

            all_data.append(df)
            logger.info(f"  {ticker}: {len(df)} records")

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Merge all ETF data on date
    result = all_data[0]
    for df in all_data[1:]:
        result = result.merge(df, on='date', how='outer')

    result = result.sort_values('date').reset_index(drop=True)

    # Forward fill missing values (holidays may differ)
    result = result.ffill()

    logger.info(f"Combined ETF data: {len(result)} records")
    return result


def compute_derived_features(vix_df: pd.DataFrame, etf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived regime features from raw data.

    Args:
        vix_df: VIX daily data
        etf_df: ETF daily data

    Returns:
        DataFrame with all regime features
    """
    logger.info("Computing derived regime features...")

    # Merge VIX and ETF data
    df = vix_df.merge(etf_df, on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.ffill()  # Forward fill gaps

    # 1. VIX Regime Level (normalized 0-1)
    # Formula: clip((vix - 10) / 40, 0, 1)
    df['vix_regime_level'] = np.clip((df['vix_close'] - 10) / 40, 0.0, 1.0)

    # 2. VIX Trend 20D
    # Formula: clip((vix_current / vix_20d_ago - 1), -0.5, 0.5)
    df['vix_20d_ago'] = df['vix_close'].shift(20)
    df['vix_trend_20d'] = np.clip(
        (df['vix_close'] / df['vix_20d_ago']) - 1.0,
        -0.5, 0.5
    )

    # 3. Risk-On Indicator
    # Score = (spy - tlt) + (jnk - tlt) - (gld - spy)
    # Normalized to 0-1
    if all(col in df.columns for col in ['spy_return_20d', 'tlt_return_20d', 'jnk_return_20d', 'gld_return_20d']):
        equity_vs_bonds = df['spy_return_20d'] - df['tlt_return_20d']
        credit_strength = df['jnk_return_20d'] - df['tlt_return_20d']
        gold_safe_haven = df['gld_return_20d'] - df['spy_return_20d']

        score = equity_vs_bonds + credit_strength - gold_safe_haven
        df['risk_on_indicator'] = np.clip(score, -0.2, 0.2) / 0.4 + 0.5
    else:
        df['risk_on_indicator'] = 0.5  # Neutral default

    # Fill NaN with neutral values
    df['vix_regime_level'] = df['vix_regime_level'].fillna(0.25)  # Low vol default
    df['vix_trend_20d'] = df['vix_trend_20d'].fillna(0.0)  # Stable default
    df['risk_on_indicator'] = df['risk_on_indicator'].fillna(0.5)  # Neutral default

    logger.info(f"Computed features for {len(df)} dates")

    return df


def load_existing_data(filepath: Path) -> pd.DataFrame:
    """Load existing parquet file if it exists."""
    if filepath.exists():
        try:
            df = pd.read_parquet(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
    return pd.DataFrame()


def save_data(df: pd.DataFrame, filepath: Path):
    """Save DataFrame to parquet."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(df)} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Download regime data (VIX, ETFs)')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE,
                        help=f'Start date (default: {DEFAULT_START_DATE})')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (default: today)')
    parser.add_argument('--update', action='store_true',
                        help='Incremental update from last available date')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help=f'Output directory (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vix_path = output_dir / 'vix_daily.parquet'
    etf_path = output_dir / 'etf_daily.parquet'
    regime_path = output_dir / 'regime_features.parquet'

    # Determine start date
    start_date = args.start_date

    if args.update:
        # Check existing data for latest date
        existing_regime = load_existing_data(regime_path)
        if not existing_regime.empty:
            last_date = existing_regime['date'].max()
            start_date = (last_date - timedelta(days=5)).strftime('%Y-%m-%d')
            logger.info(f"Incremental update from {start_date}")

    # Download data
    logger.info("=" * 60)
    logger.info("REGIME DATA DOWNLOAD")
    logger.info("=" * 60)

    vix_df = download_vix_data(start_date, args.end_date)
    etf_df = download_etf_data(ETF_TICKERS, start_date, args.end_date)

    if vix_df.empty:
        logger.error("Failed to download VIX data - aborting")
        return 1

    if etf_df.empty:
        logger.warning("No ETF data - risk-on indicator will use defaults")

    # Save raw data
    save_data(vix_df, vix_path)
    if not etf_df.empty:
        save_data(etf_df, etf_path)

    # Compute derived features
    regime_df = compute_derived_features(vix_df, etf_df)

    # Merge with existing if updating
    if args.update:
        existing_regime = load_existing_data(regime_path)
        if not existing_regime.empty:
            # Remove overlapping dates from existing
            regime_df = pd.concat([
                existing_regime[existing_regime['date'] < regime_df['date'].min()],
                regime_df
            ], ignore_index=True)

    # Save regime features
    save_data(regime_df, regime_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"VIX data:     {vix_path}")
    logger.info(f"ETF data:     {etf_path}")
    logger.info(f"Regime data:  {regime_path}")
    logger.info(f"Date range:   {regime_df['date'].min()} to {regime_df['date'].max()}")
    logger.info(f"Total records: {len(regime_df)}")

    # Show sample
    logger.info("\nSample regime features (last 5 days):")
    sample_cols = ['date', 'vix_close', 'vix_regime_level', 'vix_trend_20d', 'risk_on_indicator']
    sample_cols = [c for c in sample_cols if c in regime_df.columns]
    print(regime_df[sample_cols].tail().to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
