"""
Zero-RAM Dilution Database
===========================

Decouples the slow SEC API from the fast Scanner.

The scanner queries a local SQLite file (microseconds);
a separate nightly script handles the slow internet updates.

Usage:
    # In scanner (fast path):
    from utils.dilution_manager import DilutionDB
    db = DilutionDB()
    dilution = db.get_dilution('AAPL')  # Microseconds

    # In nightly cron job (slow path):
    from utils.dilution_manager import run_nightly_update
    run_nightly_update(tickers=['AAPL', 'MSFT', ...])
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("DilutionManager")


class DilutionDB:
    """
    Zero-RAM, Low-Latency Dilution Lookup.

    Decouples the slow SEC API from the fast Scanner.
    All lookups are O(1) via SQLite index - typically < 1ms.
    """

    def __init__(self, db_path: str = "data/dilution.db"):
        """
        Initialize the dilution database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize lightweight schema with indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dilution (
                    ticker TEXT PRIMARY KEY,
                    dilution_pct REAL,
                    is_diluted BOOLEAN,
                    updated_at INTEGER
                )
            """)
            # Index for fast staleness checks
            conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON dilution(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_diluted ON dilution(is_diluted)")

    def get_dilution(self, ticker: str, max_age_days: int = 7) -> Optional[float]:
        """
        Microsecond lookup.

        Args:
            ticker: Stock ticker symbol
            max_age_days: Maximum age of data before considered stale

        Returns:
            Dilution percentage or None if data is missing/stale
        """
        min_ts = int(time.time()) - (max_age_days * 86400)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT dilution_pct FROM dilution WHERE ticker = ? AND updated_at > ?",
                    (ticker.upper(), min_ts)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logger.warning(f"SQLite error for {ticker}: {e}")
            return None

    def is_diluted(self, ticker: str, max_age_days: int = 7) -> Optional[bool]:
        """
        Fast check if ticker exceeds dilution threshold.

        Args:
            ticker: Stock ticker symbol
            max_age_days: Maximum age of data before considered stale

        Returns:
            True if diluted, False if not, None if data missing/stale
        """
        min_ts = int(time.time()) - (max_age_days * 86400)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT is_diluted FROM dilution WHERE ticker = ? AND updated_at > ?",
                    (ticker.upper(), min_ts)
                )
                row = cur.fetchone()
                return bool(row[0]) if row else None
        except sqlite3.Error:
            return None

    def update_ticker(self, ticker: str, dilution_pct: float, threshold: float = 20.0):
        """
        Write new dilution data to disk.

        Args:
            ticker: Stock ticker symbol
            dilution_pct: Dilution percentage
            threshold: Threshold for marking as diluted (default 20%)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dilution (ticker, dilution_pct, is_diluted, updated_at) VALUES (?, ?, ?, ?)",
                (ticker.upper(), dilution_pct, dilution_pct > threshold, int(time.time()))
            )

    def bulk_update(self, data: List[Tuple[str, float]], threshold: float = 20.0):
        """
        Bulk write dilution data (faster than individual updates).

        Args:
            data: List of (ticker, dilution_pct) tuples
            threshold: Threshold for marking as diluted
        """
        now = int(time.time())
        rows = [
            (ticker.upper(), pct, pct > threshold, now)
            for ticker, pct in data
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO dilution (ticker, dilution_pct, is_diluted, updated_at) VALUES (?, ?, ?, ?)",
                rows
            )

    def get_stale_tickers(self, tickers: List[str], max_age_days: int = 7) -> List[str]:
        """
        Get list of tickers that need updating.

        Args:
            tickers: List of tickers to check
            max_age_days: Maximum age before considered stale

        Returns:
            List of tickers that are missing or stale
        """
        min_ts = int(time.time()) - (max_age_days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            # Get all fresh tickers
            placeholders = ','.join('?' * len(tickers))
            cur = conn.execute(
                f"SELECT ticker FROM dilution WHERE ticker IN ({placeholders}) AND updated_at > ?",
                [t.upper() for t in tickers] + [min_ts]
            )
            fresh = {row[0] for row in cur.fetchall()}

        # Return tickers not in fresh set
        return [t for t in tickers if t.upper() not in fresh]

    def get_diluted_tickers(self) -> List[str]:
        """Get all tickers marked as diluted."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT ticker FROM dilution WHERE is_diluted = 1")
            return [row[0] for row in cur.fetchall()]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM dilution").fetchone()[0]
            diluted = conn.execute("SELECT COUNT(*) FROM dilution WHERE is_diluted = 1").fetchone()[0]

            # Freshness stats (last 7 days)
            week_ago = int(time.time()) - (7 * 86400)
            fresh = conn.execute(
                "SELECT COUNT(*) FROM dilution WHERE updated_at > ?",
                (week_ago,)
            ).fetchone()[0]

            return {
                'total_tickers': total,
                'diluted_count': diluted,
                'fresh_count': fresh,
                'stale_count': total - fresh,
                'dilution_rate': (diluted / total * 100) if total > 0 else 0
            }


# --- The Nightly Updater Script ---

def run_nightly_update(
    tickers: List[str],
    db_path: str = "data/dilution.db",
    threshold: float = 20.0,
    rate_limit_sec: float = 0.5,
    max_age_days: int = 3
):
    """
    Run this via cron/scheduler at night.

    Isolates the memory-heavy JSON parsing and network waiting
    from the fast scanning path.

    Args:
        tickers: List of US tickers to update
        db_path: Path to SQLite database
        threshold: Dilution threshold percentage
        rate_limit_sec: Delay between API calls to avoid rate limiting
        max_age_days: Skip tickers updated within this many days

    Example cron (run at 2 AM daily):
        0 2 * * * cd /path/to/trans && python -c "from utils.dilution_manager import run_nightly_update; run_nightly_update(open('data/us_tickers.txt').read().splitlines())"
    """
    from utils.share_dilution_fetcher import ShareDilutionFetcher

    db = DilutionDB(db_path)
    fetcher = ShareDilutionFetcher()

    # Filter to only stale/missing tickers
    stale_tickers = db.get_stale_tickers(tickers, max_age_days=max_age_days)

    print(f"Dilution Update: {len(stale_tickers)} stale of {len(tickers)} total")

    updated = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(stale_tickers):
        try:
            # Fetch from SEC (blocking call)
            pct = fetcher.get_dilution_pct(ticker)

            if pct is not None:
                db.update_ticker(ticker, pct, threshold)
                updated += 1
                if pct > threshold:
                    print(f"[{i+1}/{len(stale_tickers)}] {ticker}: {pct:.1f}% DILUTED")
                else:
                    logger.debug(f"{ticker}: {pct:.1f}% OK")
            else:
                skipped += 1
                logger.debug(f"{ticker}: No SEC data available")

            # Rate limit to avoid SEC throttling
            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        except Exception as e:
            failed += 1
            logger.warning(f"{ticker}: Update failed - {e}")

    # Final stats
    stats = db.get_stats()
    print(f"\nUpdate Complete:")
    print(f"  - Updated: {updated}")
    print(f"  - Skipped (no data): {skipped}")
    print(f"  - Failed: {failed}")
    print(f"  - Total in DB: {stats['total_tickers']}")
    print(f"  - Diluted (>{threshold}%): {stats['diluted_count']}")
    print(f"  - Fresh (<7 days): {stats['fresh_count']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dilution Database Manager")
    parser.add_argument("--update", type=str, help="Path to ticker list file")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    parser.add_argument("--check", type=str, help="Check dilution for a ticker")
    parser.add_argument("--list-diluted", action="store_true", help="List all diluted tickers")

    args = parser.parse_args()

    db = DilutionDB()

    if args.stats:
        stats = db.get_stats()
        print("Dilution Database Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.check:
        pct = db.get_dilution(args.check)
        if pct is not None:
            status = "DILUTED" if pct > 20 else "OK"
            print(f"{args.check}: {pct:.1f}% ({status})")
        else:
            print(f"{args.check}: No data (run --update first)")

    elif args.list_diluted:
        diluted = db.get_diluted_tickers()
        print(f"Diluted tickers ({len(diluted)}):")
        for t in sorted(diluted):
            pct = db.get_dilution(t)
            print(f"  {t}: {pct:.1f}%")

    elif args.update:
        with open(args.update) as f:
            tickers = [line.strip() for line in f if line.strip()]
        run_nightly_update(tickers)

    else:
        parser.print_help()
