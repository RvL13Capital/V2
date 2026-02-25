#!/usr/bin/env python3
"""
Expand EU Ticker Dataset
========================

Downloads historical data (back to 2010) for European tickers with proper
validation for the TRANS consolidation pattern detection system.

Features:
- Downloads data from 2010-01-01 to present
- Includes adj_close column (CRITICAL for historical market cap)
- Validates data quality requirements
- Fetches market cap for filtering
- Supports incremental downloads (skip existing)
- Rate limiting to avoid API bans
- Progress tracking and resumable

Required Data Columns:
- date, open, high, low, close, volume (OHLCV)
- adj_close (split-adjusted, CRITICAL)

Market Cap Categories (for reference):
- Nano: < $50M
- Micro: $50M - $300M (Target)
- Small: $300M - $2B (Sweet Spot)
- Mid: $2B - $10B
- Large/Mega: > $10B (Usually excluded by physics filter)

Usage:
    # Download missing EU tickers
    python scripts/expand_eu_dataset.py --mode missing --limit 500

    # Re-download to extend date range to 2010
    python scripts/expand_eu_dataset.py --mode extend --limit 1000

    # Download specific markets
    python scripts/expand_eu_dataset.py --mode missing --markets .PA,.MI,.LS

    # Full expansion (all missing + extend existing)
    python scripts/expand_eu_dataset.py --mode full --workers 2

Author: TRANS System
Date: January 2026
"""

import os
import sys
import argparse
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.error("yfinance not installed. Run: pip install yfinance")

# EU Market Configuration
EU_MARKETS = {
    '.DE': 'Germany (Frankfurt/Xetra)',
    '.L': 'UK (London)',
    '.PA': 'France (Euronext Paris)',
    '.MI': 'Italy (Borsa Italiana)',
    '.MC': 'Spain (Madrid)',
    '.AS': 'Netherlands (Euronext Amsterdam)',
    '.LS': 'Portugal (Euronext Lisbon)',
    '.BR': 'Belgium (Euronext Brussels)',
    '.SW': 'Switzerland (SIX)',
    '.ST': 'Sweden (Stockholm)',
    '.OL': 'Norway (Oslo)',
    '.CO': 'Denmark (Copenhagen)',
    '.HE': 'Finland (Helsinki)',
    '.IR': 'Ireland (Dublin)',
    '.VI': 'Austria (Vienna)',
}

# Market cap thresholds (in USD)
MICRO_CAP_MAX = 300_000_000     # $300M
SMALL_CAP_MAX = 2_000_000_000   # $2B
MID_CAP_MAX = 10_000_000_000    # $10B

# Data requirements
MIN_ROWS = 100          # Minimum days of data needed
TARGET_START = '2010-01-01'
REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']


def load_eu_tickers(ticker_file: Path) -> List[str]:
    """Load EU ticker list from file."""
    if not ticker_file.exists():
        logger.error(f"Ticker file not found: {ticker_file}")
        return []

    with open(ticker_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    return tickers


def get_existing_tickers(data_dir: Path) -> Dict[str, Dict]:
    """
    Get existing tickers and their metadata.

    Returns dict with ticker -> {path, has_adj_close, min_date, max_date, rows}
    """
    existing = {}

    for f in data_dir.glob('*'):
        if f.suffix not in ['.parquet', '.csv']:
            continue

        ticker = f.stem

        # Only include EU tickers
        if not any(s in ticker for s in EU_MARKETS.keys()):
            continue

        try:
            if f.suffix == '.parquet':
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)

            df['date'] = pd.to_datetime(df['date'])

            existing[ticker] = {
                'path': f,
                'has_adj_close': 'adj_close' in df.columns,
                'min_date': df['date'].min(),
                'max_date': df['date'].max(),
                'rows': len(df)
            }
        except Exception as e:
            logger.debug(f"Error reading {f}: {e}")

    return existing


def download_ticker(
    ticker: str,
    start_date: str = TARGET_START,
    end_date: Optional[str] = None
) -> Optional[Tuple[pd.DataFrame, Dict]]:
    """
    Download ticker data with adj_close using yfinance.

    Returns:
        Tuple of (DataFrame, metadata_dict) or None if failed
    """
    if not YFINANCE_AVAILABLE:
        return None

    try:
        stock = yf.Ticker(ticker)

        # Download historical data
        df = stock.history(
            start=start_date,
            end=end_date,
            auto_adjust=False  # Keep both Close and Adj Close
        )

        if df.empty or len(df) < MIN_ROWS:
            return None

        # Reset index and standardize columns
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Ensure required columns
        column_map = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adj_close': 'adj_close',
            'volume': 'volume'
        }

        # Check for required columns
        missing_cols = [c for c in column_map.values() if c not in df.columns]
        if missing_cols:
            logger.debug(f"{ticker}: Missing columns {missing_cols}")
            return None

        # Select and order columns
        df = df[list(column_map.values())]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Get market cap info
        try:
            info = stock.info
            market_cap = info.get('marketCap', 0) or 0
            currency = info.get('currency', 'Unknown')
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            name = info.get('shortName', ticker)
        except:
            market_cap = 0
            currency = 'Unknown'
            sector = 'Unknown'
            industry = 'Unknown'
            name = ticker

        metadata = {
            'ticker': ticker,
            'market_cap': market_cap,
            'currency': currency,
            'sector': sector,
            'industry': industry,
            'name': name,
            'rows': len(df),
            'min_date': str(df['date'].min().date()),
            'max_date': str(df['date'].max().date()),
        }

        return df, metadata

    except Exception as e:
        logger.debug(f"{ticker}: Download failed - {e}")
        return None


def classify_market_cap(market_cap: float) -> str:
    """Classify market cap into category."""
    if market_cap <= 0:
        return 'unknown'
    elif market_cap < 50_000_000:
        return 'nano'
    elif market_cap < MICRO_CAP_MAX:
        return 'micro'
    elif market_cap < SMALL_CAP_MAX:
        return 'small'
    elif market_cap < MID_CAP_MAX:
        return 'mid'
    else:
        return 'large_mega'


def save_ticker_data(
    ticker: str,
    df: pd.DataFrame,
    output_dir: Path,
    format: str = 'csv'
) -> Path:
    """Save ticker data to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
        output_path = output_dir / f"{ticker}.parquet"
        df.to_parquet(output_path, index=False)
    else:
        output_path = output_dir / f"{ticker}.csv"
        df.to_csv(output_path, index=False)

    return output_path


def download_batch(
    tickers: List[str],
    output_dir: Path,
    start_date: str = TARGET_START,
    delay: float = 0.5,
    format: str = 'csv'
) -> Dict:
    """
    Download a batch of tickers sequentially with rate limiting.

    Returns summary statistics.
    """
    stats = {
        'total': len(tickers),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'by_cap': {'nano': 0, 'micro': 0, 'small': 0, 'mid': 0, 'large_mega': 0, 'unknown': 0},
        'downloaded': [],
        'failed_tickers': []
    }

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Downloading {ticker}...")

        result = download_ticker(ticker, start_date)

        if result is None:
            stats['failed'] += 1
            stats['failed_tickers'].append(ticker)
            logger.warning(f"  {ticker}: FAILED (no data or < {MIN_ROWS} rows)")
        else:
            df, metadata = result

            # Save data
            output_path = save_ticker_data(ticker, df, output_dir, format)

            # Update stats
            cap_category = classify_market_cap(metadata['market_cap'])
            stats['success'] += 1
            stats['by_cap'][cap_category] += 1
            stats['downloaded'].append(metadata)

            cap_str = f"${metadata['market_cap']/1e6:.0f}M" if metadata['market_cap'] > 0 else "Unknown"
            logger.info(f"  {ticker}: {metadata['rows']} rows, {cap_str} ({cap_category}), "
                       f"{metadata['min_date']} to {metadata['max_date']}")

        # Rate limiting
        time.sleep(delay)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Expand EU ticker dataset')
    parser.add_argument('--mode', type=str, default='missing',
                       choices=['missing', 'extend', 'full', 'redownload'],
                       help='Download mode: missing (new tickers), extend (older data), '
                            'full (both), redownload (all)')
    parser.add_argument('--markets', type=str, default=None,
                       help='Comma-separated market suffixes (e.g., .PA,.MI,.LS)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to download')
    parser.add_argument('--start-date', type=str, default=TARGET_START,
                       help='Start date for data (default: 2010-01-01)')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for data files')
    parser.add_argument('--ticker-file', type=str, default='data/eu_tickers.txt',
                       help='File containing EU ticker list')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'parquet'],
                       help='Output format')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between downloads (seconds)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be downloaded without downloading')

    args = parser.parse_args()

    # Check yfinance
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available. Install with: pip install yfinance")
        sys.exit(1)

    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    ticker_file = base_dir / args.ticker_file

    # Load available tickers
    all_tickers = load_eu_tickers(ticker_file)
    logger.info(f"Loaded {len(all_tickers):,} tickers from {ticker_file}")

    # Filter by market if specified
    if args.markets:
        markets = [m.strip() for m in args.markets.split(',')]
        all_tickers = [t for t in all_tickers if any(t.endswith(m) for m in markets)]
        logger.info(f"Filtered to {len(all_tickers):,} tickers for markets: {markets}")

    # Get existing ticker info
    logger.info("Scanning existing data...")
    existing = get_existing_tickers(output_dir)
    logger.info(f"Found {len(existing):,} existing EU ticker files")

    # Determine which tickers to download based on mode
    target_start_date = pd.Timestamp(args.start_date).tz_localize(None)

    if args.mode == 'missing':
        # Only tickers not already downloaded
        to_download = [t for t in all_tickers if t not in existing]
        logger.info(f"Mode: MISSING - {len(to_download):,} new tickers to download")

    elif args.mode == 'extend':
        # Tickers that exist but don't go back to target start date
        to_download = [
            t for t, info in existing.items()
            if pd.Timestamp(info['min_date']).tz_localize(None) > target_start_date
        ]
        logger.info(f"Mode: EXTEND - {len(to_download):,} tickers need date extension")

    elif args.mode == 'full':
        # Both missing and extension needed
        missing = set(t for t in all_tickers if t not in existing)
        extend = set(
            t for t, info in existing.items()
            if pd.Timestamp(info['min_date']).tz_localize(None) > target_start_date
        )
        to_download = list(missing | extend)
        logger.info(f"Mode: FULL - {len(missing):,} missing + {len(extend):,} to extend = "
                   f"{len(to_download):,} total")

    elif args.mode == 'redownload':
        # Redownload everything
        to_download = all_tickers
        logger.info(f"Mode: REDOWNLOAD - {len(to_download):,} tickers (all)")

    # Apply limit
    if args.limit:
        to_download = to_download[:args.limit]
        logger.info(f"Limited to {len(to_download)} tickers")

    if not to_download:
        logger.info("Nothing to download!")
        return

    # Show breakdown by market
    logger.info("\nTickers to download by market:")
    for suffix, name in EU_MARKETS.items():
        count = len([t for t in to_download if t.endswith(suffix)])
        if count > 0:
            logger.info(f"  {suffix} ({name}): {count}")

    # Dry run - just show what would happen
    if args.dry_run:
        logger.info("\n[DRY RUN] Would download:")
        for t in to_download[:20]:
            logger.info(f"  {t}")
        if len(to_download) > 20:
            logger.info(f"  ... and {len(to_download) - 20} more")
        return

    # Start downloading
    logger.info("\n" + "="*60)
    logger.info("Starting EU Ticker Download")
    logger.info("="*60)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Tickers: {len(to_download)}")
    logger.info("="*60 + "\n")

    start_time = time.time()

    stats = download_batch(
        to_download,
        output_dir,
        start_date=args.start_date,
        delay=args.delay,
        format=args.format
    )

    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Total processed: {stats['total']}")
    logger.info(f"Successful: {stats['success']} ({100*stats['success']/stats['total']:.1f}%)")
    logger.info(f"Failed: {stats['failed']}")

    logger.info("\nBy market cap category:")
    for cap, count in stats['by_cap'].items():
        if count > 0:
            logger.info(f"  {cap}: {count}")

    if stats['failed_tickers']:
        logger.info(f"\nFailed tickers ({len(stats['failed_tickers'])}):")
        for t in stats['failed_tickers'][:20]:
            logger.info(f"  {t}")
        if len(stats['failed_tickers']) > 20:
            logger.info(f"  ... and {len(stats['failed_tickers']) - 20} more")

    # Save download log
    log_path = output_dir / 'download_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': args.mode,
            'start_date': args.start_date,
            'stats': {
                'total': stats['total'],
                'success': stats['success'],
                'failed': stats['failed'],
                'by_cap': stats['by_cap']
            },
            'downloaded': stats['downloaded'][:100],  # First 100 for reference
            'failed_tickers': stats['failed_tickers']
        }, f, indent=2)

    logger.info(f"\nDownload log saved to: {log_path}")
    logger.info("\n[OK] EU dataset expansion complete!")


if __name__ == '__main__':
    main()
