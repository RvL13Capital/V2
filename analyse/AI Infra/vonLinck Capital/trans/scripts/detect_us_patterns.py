"""
Detect Patterns in US Universe (Optimized)
==========================================

Runs pattern detection on all US tickers in the local data cache.
Uses the V17 sleeper scanner for microstructure-aware pattern detection.

WARNING: US model has NO PREDICTIVE POWER (49.6% danger baseline).
This script is for research/analysis only. DO NOT use for live trading.

Optimizations:
- **Streaming I/O**: Merges chunks via PyArrow without loading full dataset into RAM.
- **Single-Pass Data Access**: Extracts prices in the worker to prevent double-loading data.
- **Early Filtering**: Discards non-target market caps immediately to save memory.
- **Incremental Stats**: Tracks metrics on the fly to avoid loading final file for reporting.
- Share dilution filter: Rejects US stocks with >20% trailing 12-month dilution

Usage:
    python scripts/detect_us_patterns.py --output output/us_patterns.parquet
    python scripts/detect_us_patterns.py --workers 4
"""

import os
import sys
import argparse
import logging
import shutil
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env', override=True)

# Non-US market suffixes (to exclude)
NON_US_SUFFIXES = [
    '.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS', '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI',
    '.TO', '.AX', '.NZ', '.HK', '.T', '.SS', '.SZ', '.KS', '.TW', '.SI', '.JK', '.BK', '.MX', '.SA', '.F'
]

# Target market cap categories (system optimized for these)
TARGET_MARKET_CAPS = ['small_cap', 'micro_cap']

# Process-level scanner instance (initialized once per worker)
_worker_scanner = None


def _init_worker():
    """Initialize scanner once per worker process."""
    global _worker_scanner
    from core.pattern_scanner import ConsolidationPatternScanner
    _worker_scanner = ConsolidationPatternScanner(
        enable_market_cap=False  # Market cap added post-hoc from cache
    )


def is_us_ticker(ticker: str) -> bool:
    """Check if ticker is a US stock (no international suffix)."""
    return not any(ticker.endswith(suffix) for suffix in NON_US_SUFFIXES)


def get_us_tickers_from_cache(data_dir: Path) -> list:
    """Get all US tickers from local cache."""
    tickers = []
    # Check Parquet
    for f in data_dir.glob('*.parquet'):
        if is_us_ticker(f.stem):
            tickers.append(f.stem)
    # Check CSV
    for f in data_dir.glob('*.csv'):
        if f.stem not in tickers and is_us_ticker(f.stem):
            tickers.append(f.stem)
    return sorted(tickers)


def load_ticker_data(ticker: str, data_dir: Path, start_date: str = None) -> pd.DataFrame:
    """Load ticker data from local cache."""
    parquet_path = data_dir / f"{ticker}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = data_dir / f"{ticker}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            return None

    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    if 'adjclose' in df.columns:
        df = df.rename(columns={'adjclose': 'adj_close'})
    elif 'adjusted_close' in df.columns:
        df = df.rename(columns={'adjusted_close': 'adj_close'})

    if 'adj_close' not in df.columns:
        if 'close' in df.columns:
            df['adj_close'] = df['close']

    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'date'})

    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]

    return df.sort_values('date').set_index('date')


def detect_patterns_for_ticker(ticker: str, start_date: str, data_dir: Path) -> list:
    """
    Worker: Detects patterns AND extracts price data.
    Optimization: Extracts price here to avoid reloading the DF in the main process.
    """
    global _worker_scanner

    try:
        df = load_ticker_data(ticker, data_dir, start_date)

        if df is None or len(df) < 100 or 'adj_close' not in df.columns:
            return []

        # Scanner performs detection (includes share dilution filter for US stocks)
        result = _worker_scanner.scan_ticker(ticker, start_date=start_date, df=df)

        labeled_patterns = []
        if result and result.patterns_found > 0:
            for p in result.patterns:
                # Filter valid classes
                if 'outcome_class' in p and p['outcome_class'] in [0, 1, 2]:

                    # OPTIMIZATION: Extract price now while DF is in memory
                    # This saves the main process from reloading the file
                    try:
                        p_end = pd.to_datetime(p['end_date'])
                        if p_end in df.index:
                            price = df.loc[p_end]['adj_close']
                        else:
                            # Nearest match if exact date missing
                            idx = df.index.get_indexer([p_end], method='nearest')[0]
                            price = df.iloc[idx]['adj_close']

                        # Store as temp metadata
                        p['_meta_price'] = float(price) if pd.notnull(price) else None
                    except Exception:
                        p['_meta_price'] = None

                    labeled_patterns.append(p)

        return labeled_patterns

    except Exception as e:
        logger.debug(f"{ticker}: Error - {e}")
        return []


def enrich_patterns_with_market_cap(patterns: list, market_cap_cache: dict) -> list:
    """
    Main Process: Enrich using pre-extracted price.
    Optimization: No disk I/O here (uses _meta_price from worker).
    """
    from utils.market_cap_fetcher import MarketCapFetcher

    for pattern in patterns:
        ticker = pattern['ticker']
        mc_data = market_cap_cache.get(ticker)
        price = pattern.get('_meta_price')

        # Clean up temp metadata to keep output schema clean
        if '_meta_price' in pattern:
            del pattern['_meta_price']

        if mc_data and mc_data.shares_outstanding and price:
            try:
                market_cap = float(price * mc_data.shares_outstanding)
                pattern['market_cap'] = market_cap
                pattern['market_cap_category'] = MarketCapFetcher.categorize_market_cap_value(market_cap)
                pattern['market_cap_source'] = 'historical'
            except Exception:
                _set_unknown_cap(pattern)
        else:
            _set_unknown_cap(pattern)

    return patterns


def _set_unknown_cap(pattern):
    pattern['market_cap'] = None
    pattern['market_cap_category'] = 'unknown'
    pattern['market_cap_source'] = 'unavailable'


def flush_chunk(patterns: list, chunk_idx: int, chunks_dir: Path, quiet: bool = False) -> None:
    """Write patterns to chunk file using PyArrow."""
    if not patterns:
        return

    df = pd.DataFrame(patterns)
    # Enforce schema consistency for PyArrow
    if 'outcome_class' in df.columns:
        df['outcome_class'] = df['outcome_class'].astype('int32')

    chunk_path = chunks_dir / f"chunk_{chunk_idx:04d}.parquet"

    # Use pyarrow engine explicitly
    df.to_parquet(chunk_path, index=False, engine='pyarrow')

    if not quiet:
        logger.info(f"Flushed {len(patterns)} patterns to {chunk_path.name}")

    # Free memory
    del df
    gc.collect()


def combine_chunks_stream(chunks_dir: Path, output_path: Path, quiet: bool = False):
    """
    Stream-combine chunks using PyArrow.
    Never loads full dataset into RAM.
    """
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return

    try:
        # Read schema from first file
        first_table = pq.read_table(chunk_files[0])
        schema = first_table.schema

        with pq.ParquetWriter(output_path, schema) as writer:
            # Write first chunk
            writer.write_table(first_table)

            # Stream remaining chunks
            for f in tqdm(chunk_files[1:], desc="Merging chunks", disable=quiet):
                try:
                    table = pq.read_table(f)
                    # Cast if schema evolved slightly (e.g. nulls to floats)
                    if table.schema != schema:
                        table = table.cast(schema)
                    writer.write_table(table)
                    del table
                except Exception as e:
                    logger.error(f"Error merging {f}: {e}")

        if not quiet:
            logger.info(f"Stream-combined {len(chunk_files)} chunks into {output_path}")

        # Cleanup
        for f in chunk_files:
            f.unlink()
        try:
            chunks_dir.rmdir()
        except:
            pass

    except Exception as e:
        logger.error(f"Failed to stream-combine chunks: {e}")


def main():
    parser = argparse.ArgumentParser(description='Detect patterns in US universe')
    parser.add_argument('--output', type=str, default='output/us_patterns.parquet')
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--start-date', type=str, default='2020-01-01')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--skip-market-cap', action='store_true')
    parser.add_argument('--include-all-caps', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not args.quiet:
        logger.warning("NOTE: US model has NO PREDICTIVE POWER (49.6% danger baseline)")
        logger.warning("This data is for research/analysis only. DO NOT use for live trading.")

    # Init Cache
    tickers = get_us_tickers_from_cache(data_dir)
    if args.limit:
        tickers = tickers[:args.limit]

    market_cap_cache = {}
    if not args.skip_market_cap:
        try:
            from utils.market_cap_fetcher import MarketCapFetcher
            market_cap_cache = MarketCapFetcher().cache
            if not args.quiet:
                logger.info(f"Loaded market cap cache: {len(market_cap_cache)} entries")
        except Exception:
            pass

    # Clean Chunks Dir
    chunks_dir = output_path.parent / "chunks_temp_us"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Stats Accumulator (Track without holding DF)
    stats = {'total': 0, 'classes': {}}

    # Processing Loop
    chunk_patterns = []
    chunk_idx = 0
    tickers_in_chunk = 0

    if not args.quiet:
        logger.info(f"Scanning {len(tickers)} tickers with {args.workers} workers...")

    # Helper to process batch results
    def process_batch(patterns):
        nonlocal chunk_patterns, chunk_idx, tickers_in_chunk

        if patterns:
            # Enrichment (Low Cost)
            if market_cap_cache and not args.skip_market_cap:
                patterns = enrich_patterns_with_market_cap(patterns, market_cap_cache)

            # EARLY FILTERING (Critical for Memory)
            if not args.include_all_caps and not args.skip_market_cap:
                patterns = [
                    p for p in patterns
                    if p.get('market_cap_category') in TARGET_MARKET_CAPS
                    or p.get('market_cap_category') == 'unknown'
                ]

            # Accumulate Stats
            for p in patterns:
                stats['total'] += 1
                stats['classes'][p.get('outcome_class')] = stats['classes'].get(p.get('outcome_class'), 0) + 1

            chunk_patterns.extend(patterns)

        tickers_in_chunk += 1

        # Flush
        if tickers_in_chunk >= args.chunk_size:
            flush_chunk(chunk_patterns, chunk_idx, chunks_dir, args.quiet)
            chunk_patterns = []
            chunk_idx += 1
            tickers_in_chunk = 0
            gc.collect()

    # Parallel Execution
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
            futures = {
                executor.submit(detect_patterns_for_ticker, t, args.start_date, data_dir): t
                for t in tickers
            }
            for future in tqdm(as_completed(futures), total=len(tickers), disable=args.quiet):
                try:
                    process_batch(future.result())
                except Exception:
                    pass
    else:
        # Sequential
        _init_worker()
        for t in tqdm(tickers, disable=args.quiet):
            try:
                process_batch(detect_patterns_for_ticker(t, args.start_date, data_dir))
            except Exception:
                pass

    # Flush remainder
    if chunk_patterns:
        flush_chunk(chunk_patterns, chunk_idx, chunks_dir, args.quiet)

    # Merge via Streaming
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_chunks_stream(chunks_dir, output_path, args.quiet)

    # Report
    if stats['total'] > 0 and not args.quiet:
        logger.info(f"\nDetected {stats['total']} patterns")
        logger.info("Distribution:")
        for k, v in sorted(stats['classes'].items()):
            logger.info(f"  Class {k}: {v} ({v/stats['total']*100:.1f}%)")
        logger.warning("\nREMINDER: US model has NO PREDICTIVE POWER")
        logger.warning("DO NOT use us_model.pt for live trading")

    if not args.quiet:
        logger.info("Done!")


if __name__ == '__main__':
    main()
