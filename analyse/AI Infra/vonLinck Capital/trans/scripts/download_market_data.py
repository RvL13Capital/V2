#!/usr/bin/env python3
"""
Download market data using configured API sources.

This script downloads historical price data from TwelveData and AlphaVantage
for a set of tickers and saves them in the format expected by the TRANS system.
"""

import os
import sys
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class MarketDataDownloader:
    """Download market data from multiple sources with rate limiting."""

    def __init__(self):
        """Initialize with API keys and rate limits."""
        self.twelvedata_key = os.getenv('TWELVEDATA_API_KEY')
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')

        # Rate limits (requests per minute)
        self.twelvedata_rpm = int(os.getenv('TWELVEDATA_RPM', 8))
        self.alphavantage_rpm = int(os.getenv('ALPHAVANTAGE_RPM', 5))

        # Track last request times for rate limiting
        self.last_twelvedata_request = 0
        self.last_alphavantage_request = 0

        # Calculate minimum delay between requests (seconds)
        self.twelvedata_delay = 60 / self.twelvedata_rpm
        self.alphavantage_delay = 60 / self.alphavantage_rpm

    def _rate_limit_wait(self, api: str):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()

        if api == 'twelvedata':
            time_since_last = current_time - self.last_twelvedata_request
            if time_since_last < self.twelvedata_delay:
                wait_time = self.twelvedata_delay - time_since_last
                print(f"  Rate limiting: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_twelvedata_request = time.time()

        elif api == 'alphavantage':
            time_since_last = current_time - self.last_alphavantage_request
            if time_since_last < self.alphavantage_delay:
                wait_time = self.alphavantage_delay - time_since_last
                print(f"  Rate limiting: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_alphavantage_request = time.time()

    def download_twelvedata(self, ticker: str, outputsize: int = 5000) -> Optional[pd.DataFrame]:
        """
        Download data from TwelveData API.

        Args:
            ticker: Stock symbol
            outputsize: Number of data points (max 5000)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.twelvedata_key:
            print("  TwelveData API key not configured")
            return None

        self._rate_limit_wait('twelvedata')

        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': ticker,
                'interval': '1day',
                'outputsize': outputsize,
                'apikey': self.twelvedata_key,
                'format': 'JSON'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'status' in data and data['status'] == 'error':
                print(f"  API error: {data.get('message', 'Unknown error')}")
                return None

            if 'values' not in data:
                print(f"  No data returned")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['date'] = pd.to_datetime(df['datetime'])
            df = df.drop('datetime', axis=1)

            # Convert string columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Sort by date (oldest first)
            df = df.sort_values('date').reset_index(drop=True)

            # Add ticker column
            df['ticker'] = ticker

            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

        except Exception as e:
            print(f"  TwelveData error: {e}")
            return None

    def download_alphavantage(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download data from AlphaVantage API.

        Args:
            ticker: Stock symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.alphavantage_key:
            print("  AlphaVantage API key not configured")
            return None

        self._rate_limit_wait('alphavantage')

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': self.alphavantage_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                print(f"  API error: {data['Error Message']}")
                return None

            if 'Note' in data:
                print(f"  API limit reached: {data['Note']}")
                return None

            if 'Time Series (Daily)' not in data:
                print(f"  No data returned")
                return None

            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

            # Convert string columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Sort by date (oldest first)
            df = df.sort_values('date').reset_index(drop=True)

            # Add ticker column
            df['ticker'] = ticker

            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

        except Exception as e:
            print(f"  AlphaVantage error: {e}")
            return None

    def download_ticker(self, ticker: str, prefer_source: str = 'twelvedata') -> bool:
        """
        Download data for a ticker from available sources.

        Args:
            ticker: Stock symbol
            prefer_source: Preferred API source

        Returns:
            True if successful, False otherwise
        """
        print(f"\nDownloading {ticker}...")

        df = None

        # Try preferred source first
        if prefer_source == 'twelvedata':
            print(f"  Trying TwelveData...")
            df = self.download_twelvedata(ticker)

            if df is None:
                print(f"  Trying AlphaVantage as fallback...")
                df = self.download_alphavantage(ticker)

        else:  # alphavantage
            print(f"  Trying AlphaVantage...")
            df = self.download_alphavantage(ticker)

            if df is None:
                print(f"  Trying TwelveData as fallback...")
                df = self.download_twelvedata(ticker)

        if df is None or df.empty:
            print(f"  [FAILED] Failed to download data")
            return False

        # Save to parquet
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{ticker}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"  [SUCCESS] Saved {len(df)} days of data to {output_path}")
        return True


def main():
    """Main function to download market data."""

    # Initialize downloader
    downloader = MarketDataDownloader()

    # Check if API keys are configured
    if not downloader.twelvedata_key and not downloader.alphavantage_key:
        print("ERROR: No API keys configured!")
        print("Please set TWELVEDATA_API_KEY or ALPHAVANTAGE_API_KEY in .env file")
        sys.exit(1)

    # Sample tickers for TRANS testing
    # Focus on micro/small-cap stocks which are ideal for consolidation patterns
    tickers = [
        # Large cap for reference
        "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",

        # Small/mid cap tech (good for patterns)
        "AMD", "PLTR", "SOFI", "ROKU", "SNAP",

        # High volatility stocks
        "GME", "AMC", "BB", "RIOT", "MARA",

        # ETFs for market context
        "SPY", "QQQ", "IWM"
    ]

    print("=" * 60)
    print("TRANS Market Data Downloader")
    print("=" * 60)
    print(f"Configured APIs:")
    if downloader.twelvedata_key:
        print(f"  [OK] TwelveData (Rate limit: {downloader.twelvedata_rpm} req/min)")
    if downloader.alphavantage_key:
        print(f"  [OK] AlphaVantage (Rate limit: {downloader.alphavantage_rpm} req/min)")
    print(f"Tickers to download: {len(tickers)}")
    print("-" * 60)

    # Download each ticker
    successful = []
    failed = []

    for ticker in tickers:
        if downloader.download_ticker(ticker):
            successful.append(ticker)
        else:
            failed.append(ticker)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(tickers)}")
    if successful:
        print(f"  Downloaded: {', '.join(successful)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # List files in data/raw
    data_dir = Path("data/raw")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
        print(f"\nTotal files in data/raw: {len(files)}")

        if files:
            print("Sample files:")
            for f in sorted(files)[:5]:
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")

    print("\n[OK] Data ready for TRANS pipeline processing!")


if __name__ == "__main__":
    main()