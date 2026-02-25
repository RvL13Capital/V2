"""
US Small Cap EOD Data Downloader
================================

Downloads historical EOD data for US small cap stocks.
Supports multiple data sources with fallback.

Usage:
    python scripts/download_us_smallcaps.py --source yfinance --tickers data/us_tickers.txt
    python scripts/download_us_smallcaps.py --source polygon --api-key YOUR_KEY
    python scripts/download_us_smallcaps.py --source tiingo --api-key YOUR_KEY

Requirements:
    pip install yfinance pandas tqdm
    pip install polygon-api-client  # for Polygon
    pip install tiingo              # for Tiingo
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
TICKERS_FILE = PROJECT_ROOT / 'data' / 'us_tickers.txt'


def load_ticker_list(filepath: Path) -> List[str]:
    """Load tickers from file."""
    with open(filepath, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers


def download_yfinance(
    tickers: List[str],
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    output_dir: Path = DATA_DIR,
    skip_existing: bool = True,
    batch_size: int = 50,
    delay: float = 1.0
) -> dict:
    """
    Download data using yfinance (free).

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (default: today)
        output_dir: Directory to save parquet files
        skip_existing: Skip tickers that already have data
        batch_size: Number of tickers per batch
        delay: Delay between batches (seconds)

    Returns:
        Dict with success/failure counts
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {'success': 0, 'failed': 0, 'skipped': 0}

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {'success': 0, 'failed': 0, 'skipped': 0}
    failed_tickers = []

    # Filter existing
    if skip_existing:
        existing = {f.stem.upper() for f in output_dir.glob("*.parquet")}
        tickers_to_download = [t for t in tickers if t not in existing]
        results['skipped'] = len(tickers) - len(tickers_to_download)
        logger.info(f"Skipping {results['skipped']} existing tickers")
    else:
        tickers_to_download = tickers

    logger.info(f"Downloading {len(tickers_to_download)} tickers...")

    # Process in batches
    for i in tqdm(range(0, len(tickers_to_download), batch_size), desc="Batches"):
        batch = tickers_to_download[i:i + batch_size]

        try:
            # Download batch
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False,
                threads=True
            )

            # Save individual tickers
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            failed_tickers.append(ticker)
                            results['failed'] += 1
                            continue
                        df = data[ticker].copy()

                    # Clean up
                    df = df.dropna(how='all')
                    if len(df) < 100:  # Minimum 100 days
                        failed_tickers.append(ticker)
                        results['failed'] += 1
                        continue

                    # Standardize columns
                    df.columns = [c.lower() for c in df.columns]
                    df = df.reset_index()
                    df = df.rename(columns={'Date': 'date', 'index': 'date'})

                    # Save
                    output_path = output_dir / f"{ticker}.parquet"
                    df.to_parquet(output_path, index=False)
                    results['success'] += 1

                except Exception as e:
                    logger.debug(f"{ticker}: {e}")
                    failed_tickers.append(ticker)
                    results['failed'] += 1

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Batch error: {e}")
            results['failed'] += len(batch)

    # Save failed tickers
    if failed_tickers:
        failed_path = output_dir.parent / 'failed_downloads.txt'
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_tickers))
        logger.info(f"Failed tickers saved to {failed_path}")

    return results


def download_polygon(
    tickers: List[str],
    api_key: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    output_dir: Path = DATA_DIR,
    skip_existing: bool = True,
    delay: float = 0.1
) -> dict:
    """
    Download data using Polygon.io (paid, fast).
    """
    try:
        from polygon import RESTClient
    except ImportError:
        logger.error("polygon not installed. Run: pip install polygon-api-client")
        return {'success': 0, 'failed': 0, 'skipped': 0}

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = RESTClient(api_key=api_key)

    results = {'success': 0, 'failed': 0, 'skipped': 0}

    # Filter existing
    if skip_existing:
        existing = {f.stem.upper() for f in output_dir.glob("*.parquet")}
        tickers_to_download = [t for t in tickers if t not in existing]
        results['skipped'] = len(tickers) - len(tickers_to_download)
    else:
        tickers_to_download = tickers

    for ticker in tqdm(tickers_to_download, desc="Downloading"):
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000
            )

            if not aggs:
                results['failed'] += 1
                continue

            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs])

            if len(df) < 100:
                results['failed'] += 1
                continue

            # Save
            output_path = output_dir / f"{ticker}.parquet"
            df.to_parquet(output_path, index=False)
            results['success'] += 1

            time.sleep(delay)

        except Exception as e:
            logger.debug(f"{ticker}: {e}")
            results['failed'] += 1

    return results


def get_russell2000_tickers() -> List[str]:
    """
    Get Russell 2000 (small cap) ticker list.
    """
    try:
        import yfinance as yf
        # IWM is the Russell 2000 ETF
        iwm = yf.Ticker("IWM")
        # This doesn't give holdings directly, but we can use other sources
    except:
        pass

    # Fallback: Common small cap screening
    # You can also get lists from:
    # - https://www.ishares.com/us/products/239710/ishares-russell-2000-etf (holdings)
    # - https://www.ftserussell.com/products/indices/russell-us

    logger.info("For Russell 2000 constituents, download from:")
    logger.info("  1. iShares IWM holdings: https://www.ishares.com/us/products/239710/")
    logger.info("  2. FTSE Russell: https://www.ftserussell.com/")

    return []


def get_nasdaq_smallcap_tickers(min_price: float = 1.0, max_price: float = 50.0) -> List[str]:
    """
    Get NASDAQ small cap tickers via screening.
    """
    try:
        import yfinance as yf

        # Get NASDAQ listed stocks
        # This is a simplified approach - for full list use NASDAQ API
        logger.info("Fetching NASDAQ small caps (this may take a while)...")

        # You can get the full NASDAQ list from:
        # https://www.nasdaq.com/market-activity/stocks/screener
        # Download as CSV and filter by market cap

    except Exception as e:
        logger.error(f"Error: {e}")

    return []


def main():
    parser = argparse.ArgumentParser(description='Download US Small Cap EOD Data')
    parser.add_argument('--source', choices=['yfinance', 'polygon', 'tiingo'],
                        default='yfinance', help='Data source')
    parser.add_argument('--tickers', type=str, default=str(TICKERS_FILE),
                        help='Path to ticker list file')
    parser.add_argument('--api-key', type=str, help='API key for paid sources')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default=str(DATA_DIR),
                        help='Output directory for parquet files')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip tickers with existing data')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for yfinance')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tickers to download')

    args = parser.parse_args()

    # Load tickers
    ticker_path = Path(args.tickers)
    if ticker_path.exists():
        tickers = load_ticker_list(ticker_path)
        logger.info(f"Loaded {len(tickers)} tickers from {ticker_path}")
    else:
        logger.error(f"Ticker file not found: {ticker_path}")
        return

    # Apply limit
    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limited to {len(tickers)} tickers")

    output_dir = Path(args.output_dir)

    # Download
    if args.source == 'yfinance':
        results = download_yfinance(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size
        )
    elif args.source == 'polygon':
        if not args.api_key:
            logger.error("Polygon requires --api-key")
            return
        results = download_polygon(
            tickers=tickers,
            api_key=args.api_key,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            skip_existing=args.skip_existing
        )
    else:
        logger.error(f"Source {args.source} not implemented yet")
        return

    # Summary
    logger.info("=" * 50)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Failed:  {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
