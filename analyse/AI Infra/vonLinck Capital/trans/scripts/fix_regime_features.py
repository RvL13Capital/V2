"""
Fix Missing Regime Features: market_breadth_200 and days_since_regime_change
=============================================================================

This script calculates the two missing regime features:
1. market_breadth_200: % of stocks above their 200-day SMA
2. days_since_regime_change: Days since last 50/200 MA crossover on SPY

Usage:
    python scripts/fix_regime_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_spy_ma_crossovers(start_date: str = '2014-01-01') -> pd.DataFrame:
    """
    Calculate SPY 50/200 MA crossovers for days_since_regime_change.

    Returns DataFrame with columns: date, sma_50, sma_200, phase, crossover_date
    """
    logger.info("Downloading SPY data for MA crossover calculation...")

    spy = yf.download('SPY', start=start_date, end=datetime.now().strftime('%Y-%m-%d'), progress=False)

    if spy.empty:
        logger.error("Failed to download SPY data")
        return pd.DataFrame()

    # Flatten multi-index columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy.reset_index()
    spy.columns = [c.lower() for c in spy.columns]

    # Calculate MAs
    spy['sma_50'] = spy['close'].rolling(window=50).mean()
    spy['sma_200'] = spy['close'].rolling(window=200).mean()

    # Determine phase (1 = bull, 0 = bear)
    spy['phase'] = (spy['sma_50'] > spy['sma_200']).astype(int)

    # Track crossover dates
    spy['phase_change'] = spy['phase'].diff().abs()

    # Initialize crossover_date column
    spy['crossover_date'] = pd.NaT

    # Fill crossover dates
    last_crossover = pd.NaT
    for idx in spy.index:
        if spy.loc[idx, 'phase_change'] == 1:
            last_crossover = spy.loc[idx, 'date']
        spy.loc[idx, 'crossover_date'] = last_crossover

    # For initial period before first crossover, use a far-back date
    first_crossover_idx = spy['phase_change'].eq(1).idxmax() if (spy['phase_change'] == 1).any() else None
    if first_crossover_idx is not None:
        spy.loc[:first_crossover_idx-1, 'crossover_date'] = spy.loc[first_crossover_idx, 'date'] - pd.Timedelta(days=365)

    result = spy[['date', 'sma_50', 'sma_200', 'phase', 'crossover_date']].dropna(subset=['sma_200'])

    logger.info(f"Calculated {len(result)} days of SPY MA crossover data")
    logger.info(f"Date range: {result['date'].min()} to {result['date'].max()}")

    # Count crossovers
    n_crossovers = (spy['phase_change'] == 1).sum()
    logger.info(f"Total crossovers: {n_crossovers}")

    return result


def calculate_market_breadth(start_date: str = '2014-01-01') -> pd.DataFrame:
    """
    Calculate market breadth using Russell 3000 proxy (using available ETFs).

    Since we can't easily get individual stock data for all Russell 3000 constituents,
    we'll use sector ETFs as a proxy for breadth.

    Returns DataFrame with columns: date, pct_above_sma200
    """
    logger.info("Calculating market breadth using sector ETF proxy...")

    # Use sector ETFs as market proxy
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

    breadth_data = {}

    for etf in sector_etfs:
        try:
            data = yf.download(etf, start=start_date, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.reset_index()
                data.columns = [c.lower() for c in data.columns]
                data['sma_200'] = data['close'].rolling(window=200).mean()
                data['above_sma200'] = (data['close'] > data['sma_200']).astype(int)
                breadth_data[etf] = data[['date', 'above_sma200']].set_index('date')
        except Exception as e:
            logger.warning(f"Failed to download {etf}: {e}")

    if not breadth_data:
        logger.error("No breadth data downloaded")
        return pd.DataFrame()

    # Combine all ETFs
    breadth_df = pd.concat(breadth_data.values(), axis=1)
    breadth_df.columns = list(breadth_data.keys())

    # Calculate percentage above SMA200
    breadth_df['pct_above_sma200'] = breadth_df.mean(axis=1) * 100
    breadth_df = breadth_df.reset_index()
    breadth_df = breadth_df[['date', 'pct_above_sma200']].dropna()

    logger.info(f"Calculated breadth for {len(breadth_df)} days using {len(breadth_data)} sector ETFs")
    logger.info(f"Date range: {breadth_df['date'].min()} to {breadth_df['date'].max()}")
    logger.info(f"Breadth range: {breadth_df['pct_above_sma200'].min():.1f}% to {breadth_df['pct_above_sma200'].max():.1f}%")

    return breadth_df


def main():
    output_dir = Path(__file__).parent.parent / 'data' / 'market_regime'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIXING MISSING REGIME FEATURES")
    print("=" * 70)

    # 1. Calculate SPY MA crossovers for days_since_regime_change
    print("\n[1/3] Calculating SPY MA crossovers...")
    phase_df = calculate_spy_ma_crossovers()

    if not phase_df.empty:
        phase_path = output_dir / 'USA500IDXUSD_phases.csv'
        phase_df.to_csv(phase_path, index=False)
        print(f"Saved to: {phase_path}")

        # Show some crossover examples
        crossovers = phase_df[phase_df['date'] == phase_df['crossover_date']]
        print(f"\nRecent crossovers:")
        for _, row in crossovers.tail(5).iterrows():
            phase_name = 'BULL' if row['phase'] == 1 else 'BEAR'
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {phase_name}")

    # 2. Calculate market breadth
    print("\n[2/3] Calculating market breadth...")
    breadth_df = calculate_market_breadth()

    if not breadth_df.empty:
        breadth_path = output_dir / 'breadth_daily.parquet'
        breadth_df.to_parquet(breadth_path, index=False)
        print(f"Saved to: {breadth_path}")

        # Show yearly averages
        breadth_df['year'] = breadth_df['date'].dt.year
        print("\nYearly average breadth:")
        for year, grp in breadth_df.groupby('year'):
            print(f"  {year}: {grp['pct_above_sma200'].mean():.1f}%")

    # 3. Verify the data can be loaded
    print("\n[3/3] Verifying regime calculator...")
    from utils.regime_features import create_regime_calculator

    calculator = create_regime_calculator()

    # Test with a few dates
    test_dates = [
        pd.Timestamp('2020-03-20'),  # COVID crash
        pd.Timestamp('2022-06-15'),  # 2022 bear
        pd.Timestamp('2024-01-15'),  # 2024 bull
    ]

    print("\nTest feature extraction:")
    for date in test_dates:
        features = calculator.get_features(date)
        print(f"\n{date.strftime('%Y-%m-%d')}:")
        for k, v in features.items():
            print(f"  {k}: {v:.3f}")

    print("\n" + "=" * 70)
    print("DONE - Regime features fixed!")
    print("=" * 70)
    print("\nTo regenerate sequences with fixed features, run:")
    print("  python pipeline/01_generate_sequences.py ...")


if __name__ == '__main__':
    main()
