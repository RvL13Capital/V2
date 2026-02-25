"""
Quick K4 Pattern Hunt - Start with High-Potential Stocks
=========================================================

This script quickly downloads and analyzes a curated list of volatile
micro/small-cap stocks that are most likely to have K4 patterns.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = Path('output/quick_k4_hunt')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# High-potential stocks (known for volatility and big moves)
HIGH_POTENTIAL_STOCKS = [
    # Recent volatile movers
    'SMCI', 'NVAX', 'IONQ', 'ASTS', 'RKLB', 'LUNR', 'RDW', 'AFRM', 'UPST', 'HOOD',

    # Biotech volatility
    'SAVA', 'OCGN', 'INO', 'VXRT', 'AGEN', 'BCRX', 'GERN', 'SRPT', 'BLUE', 'RETA',

    # EV/Tech small-caps
    'NKLA', 'RIDE', 'WKHS', 'GOEV', 'CHPT', 'STEM', 'QS', 'LCID', 'RIVN', 'PSNY',

    # Mining/Commodities
    'AG', 'FSM', 'EXK', 'CDE', 'HL', 'GPL', 'SILV', 'PAAS', 'BTG', 'KGC',

    # Energy volatility
    'REI', 'INDO', 'CEI', 'ENSV', 'HUSA', 'USWS', 'BRN', 'VTNR', 'IMPP', 'GEVO',

    # Retail/Meme stocks
    'GME', 'AMC', 'BBBY', 'KOSS', 'DDS', 'BBIG', 'ATER', 'MULN', 'FFIE', 'SNDL'
]

def download_and_analyze(ticker: str) -> dict:
    """Download data and check for K4 potential."""
    try:
        # Download 5 years of data
        stock = yf.Ticker(ticker)
        df = stock.history(start='2019-01-01', auto_adjust=True)

        if len(df) < 250:
            return None

        # Calculate rolling 100-day max gain
        gains = []
        for i in range(len(df) - 100):
            window = df.iloc[i:i+100]
            start_price = df.iloc[i]['Close']
            max_price = window['High'].max()
            gain_pct = ((max_price - start_price) / start_price) * 100
            gains.append(gain_pct)

        # Check for K4 patterns (75%+ gain)
        k4_count = sum(1 for g in gains if g >= 75)
        max_gain = max(gains) if gains else 0
        avg_gain = np.mean(gains) if gains else 0

        return {
            'ticker': ticker,
            'total_days': len(df),
            'k4_count': k4_count,
            'max_gain_pct': max_gain,
            'avg_gain_pct': avg_gain,
            'k4_rate': k4_count / len(gains) * 100 if gains else 0
        }

    except Exception as e:
        print(f"  [ERROR] {ticker}: {str(e)[:50]}")
        return None

def main():
    """Main function for quick K4 hunt."""
    print("=" * 70)
    print("QUICK K4 PATTERN HUNT - HIGH POTENTIAL STOCKS")
    print("=" * 70)
    print(f"Analyzing {len(HIGH_POTENTIAL_STOCKS)} high-volatility stocks...")

    results = []
    k4_found = []

    for i, ticker in enumerate(HIGH_POTENTIAL_STOCKS, 1):
        print(f"\n[{i}/{len(HIGH_POTENTIAL_STOCKS)}] Processing {ticker}...")
        result = download_and_analyze(ticker)

        if result:
            results.append(result)
            if result['k4_count'] > 0:
                k4_found.append(result)
                print(f"  ✓ Found {result['k4_count']} K4 patterns! Max gain: {result['max_gain_pct']:.1f}%")

    # Create DataFrame
    results_df = pd.DataFrame(results)
    k4_df = pd.DataFrame(k4_found)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'all_stocks_analyzed.csv', index=False)
    if not k4_df.empty:
        k4_df.to_csv(OUTPUT_DIR / 'k4_stocks_found.csv', index=False)

    # Display summary
    print("\n" + "=" * 70)
    print("QUICK K4 HUNT RESULTS")
    print("=" * 70)
    print(f"Stocks analyzed: {len(results)}")
    print(f"Stocks with K4 patterns: {len(k4_found)} ({len(k4_found)/len(results)*100:.1f}%)")

    if k4_found:
        print("\n[TOP K4 STOCKS]")
        k4_df_sorted = k4_df.sort_values('k4_count', ascending=False)
        print(k4_df_sorted[['ticker', 'k4_count', 'max_gain_pct', 'k4_rate']].head(10).to_string(index=False))

        print(f"\n[STATISTICS]")
        print(f"Total K4 patterns found: {k4_df['k4_count'].sum()}")
        print(f"Average K4s per stock: {k4_df['k4_count'].mean():.1f}")
        print(f"Max gain observed: {k4_df['max_gain_pct'].max():.1f}%")

        # Download full data for top K4 stocks
        print(f"\n[DOWNLOADING] Full data for top K4 stocks...")
        cache_dir = Path('data/k4_stocks_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        for ticker in k4_df_sorted['ticker'].head(20):
            stock = yf.Ticker(ticker)
            df = stock.history(start='2015-01-01', auto_adjust=True)
            df.to_parquet(cache_dir / f'{ticker}.parquet')
            print(f"  Saved {ticker}: {len(df)} days")

    print(f"\n[NEXT STEPS]")
    print("1. Run full pattern detection on K4 stocks")
    print("2. Extract 48 canonical features")
    print("3. Generate predictions with trained models")
    print("4. Add to training dataset for improved K4 detection")

    return results_df, k4_df

if __name__ == "__main__":
    results_df, k4_df = main()
    print("\nQuick K4 hunt complete! Check output/quick_k4_hunt/ for results.")