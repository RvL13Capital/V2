"""
Modern Foreman Service - Async-capable data acquisition orchestration.

Modernized version of ForemanService with:
- Async/await support for parallel downloads
- Backward compatible sync methods
- Uses UnifiedAsyncDownloader for best performance
- Maintains all original functionality
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ModernForemanService:
    """
    Modern Foreman Service with async support.

    Provides both synchronous and asynchronous interfaces for data acquisition:
    - Async methods for high-performance parallel downloads
    - Sync methods for backward compatibility
    - Automatic source selection (yfinance, Alpha Vantage, etc.)
    - Statistics tracking and reporting

    Example (Async):
        foreman = ModernForemanService(config)
        results = await foreman.download_tickers_async(
            [('AAPL', ''), ('MSFT', ''), ('GOOGL', '')],
            use_async=True
        )

    Example (Sync):
        foreman = ModernForemanService(config)
        results = foreman.download_tickers_sync(
            [('AAPL', ''), ('MSFT', '')]
        )
    """

    def __init__(self, config, api_key_manager=None):
        """
        Initialize Modern Foreman Service.

        Args:
            config: Foreman configuration
            api_key_manager: API key manager (optional, for Alpha Vantage)
        """
        self.config = config
        self.api_key_manager = api_key_manager

        # Setup logging
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("Modern Foreman Service Initializing")
        logger.info("=" * 60)

        # Initialize async downloader (lazy - only when needed)
        self._async_downloader = None

        # Initialize legacy components (for backward compatibility)
        self._init_legacy_components()

        # Statistics
        self.start_time = None
        self.end_time = None
        self.tickers_processed = 0
        self.tickers_successful = 0
        self.tickers_failed = 0
        self.tickers_skipped = 0

        logger.info("=" * 60)
        logger.info("Modern Foreman Service Initialized")
        logger.info("=" * 60)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.log_to_file:
            import os
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)

            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

            logging.getLogger().addHandler(file_handler)

    def _init_legacy_components(self):
        """Initialize legacy components for backward compatibility."""
        try:
            from data_acquisition.sources_orig.exchange_mapper import ExchangeMapper
            from data_acquisition.sources_orig.ticker_screener import TickerScreener
            from data_acquisition.sources_orig.data_downloader import DataDownloader
            from data_acquisition.sources_orig.storage_manager import StorageManager

            logger.info("Initializing legacy components...")

            self.exchange_mapper = ExchangeMapper(self.config)
            self.ticker_screener = TickerScreener(self.config)

            if self.api_key_manager:
                self.data_downloader = DataDownloader(self.config, self.api_key_manager)
            else:
                self.data_downloader = None

            self.storage_manager = StorageManager(self.config)

            logger.info("Legacy components initialized successfully")

        except ImportError as e:
            logger.warning(f"Could not initialize legacy components: {e}")
            self.exchange_mapper = None
            self.ticker_screener = None
            self.data_downloader = None
            self.storage_manager = None

    def _get_async_downloader(self):
        """Get or create async downloader (lazy initialization)."""
        if self._async_downloader is None:
            from data_acquisition.sources import UnifiedAsyncDownloader

            max_workers = getattr(self.config, 'parallel_downloads', 10)

            self._async_downloader = UnifiedAsyncDownloader(
                config=self.config,
                api_key_manager=self.api_key_manager,
                prefer_source='yfinance',
                max_workers=max_workers
            )

            logger.info(f"Async downloader initialized (max_workers={max_workers})")

        return self._async_downloader

    async def download_tickers_async(
        self,
        tickers: List[tuple],
        source: str = 'yfinance',
        skip_existing: bool = True
    ) -> List[Dict]:
        """
        Download multiple tickers asynchronously (high performance).

        Args:
            tickers: List of (ticker, exchange_suffix) tuples
            source: Data source ('yfinance', 'alphavantage', etc.)
            skip_existing: Skip tickers that already exist in storage

        Returns:
            List of result dictionaries with download status
        """
        logger.info(f"Starting async download of {len(tickers)} tickers from {source}")
        self.start_time = datetime.now()

        # Filter out existing tickers if requested
        if skip_existing and self.storage_manager:
            filtered_tickers = []
            for ticker, exchange_suffix in tickers:
                if not self.storage_manager.file_exists_in_gcs(ticker, exchange_suffix, 'csv'):
                    filtered_tickers.append((ticker, exchange_suffix))
                else:
                    logger.debug(f"Skipping {ticker}{exchange_suffix} (already exists)")
                    self.tickers_skipped += 1

            tickers = filtered_tickers
            logger.info(f"After filtering: {len(tickers)} tickers to download")

        if len(tickers) == 0:
            logger.info("No tickers to download")
            return []

        # Get async downloader
        downloader = self._get_async_downloader()

        # Download in parallel
        results = await downloader.download_multiple(tickers, source=source)

        # Process and store results
        processed_results = []

        for ticker_full, df, metadata in results:
            self.tickers_processed += 1

            if df is not None and len(df) > 0:
                # Store data if storage manager available
                if self.storage_manager:
                    # Extract ticker and suffix
                    # For simplicity, assume ticker_full format is "TICKER.SUFFIX" or just "TICKER"
                    parts = ticker_full.rsplit('.', 1) if '.' in ticker_full else (ticker_full, '')
                    ticker = parts[0]
                    exchange_suffix = f".{parts[1]}" if len(parts) > 1 and parts[1] else ''

                    success = self.storage_manager.save_ticker_data(
                        df=df,
                        ticker=ticker,
                        exchange_suffix=exchange_suffix,
                        metadata=metadata
                    )

                    if success:
                        self.tickers_successful += 1
                        logger.info(f"+ {ticker_full}: SUCCESS ({len(df)} data points)")
                    else:
                        self.tickers_failed += 1
                        logger.warning(f"✗ {ticker_full}: STORAGE FAILED")
                else:
                    # No storage, just count as success
                    self.tickers_successful += 1
                    logger.info(f"+ {ticker_full}: SUCCESS ({len(df)} data points)")

                processed_results.append({
                    'ticker': ticker_full,
                    'success': True,
                    'data_points': len(df),
                    'metadata': metadata
                })
            else:
                self.tickers_failed += 1
                logger.warning(f"✗ {ticker_full}: DOWNLOAD FAILED")

                processed_results.append({
                    'ticker': ticker_full,
                    'success': False,
                    'data_points': 0,
                    'metadata': {}
                })

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Async download complete in {duration:.2f}s")
        logger.info(f"Successful: {self.tickers_successful}")
        logger.info(f"Failed: {self.tickers_failed}")
        logger.info(f"Skipped: {self.tickers_skipped}")
        logger.info("=" * 60)

        return processed_results

    def download_tickers_sync(
        self,
        tickers: List[tuple],
        skip_existing: bool = True
    ) -> List[Dict]:
        """
        Download tickers synchronously (backward compatible).

        Uses async downloader with sync wrapper for better performance
        than pure threading.

        Args:
            tickers: List of (ticker, exchange_suffix) tuples
            skip_existing: Skip existing tickers

        Returns:
            List of result dictionaries
        """
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.download_tickers_async(tickers, skip_existing=skip_existing)
        )

    async def run_full_pipeline_async(
        self,
        exchange_codes: Optional[List[str]] = None,
        include_secondary_range: bool = True,
        skip_existing: bool = True,
        source: str = 'yfinance'
    ) -> Dict:
        """
        Run complete async data acquisition pipeline.

        Args:
            exchange_codes: Exchange codes to process (None = all)
            include_secondary_range: Include secondary market cap range
            skip_existing: Skip existing files
            source: Data source to use

        Returns:
            Statistics dictionary
        """
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("STARTING ASYNC PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Exchanges: {exchange_codes or 'ALL'}")
        logger.info(f"Include secondary range: {include_secondary_range}")
        logger.info(f"Skip existing: {skip_existing}")
        logger.info(f"Source: {source}")
        logger.info("=" * 60)

        try:
            # Step 1: Screen stocks (uses sync ticker screener)
            if not self.ticker_screener:
                raise RuntimeError("Ticker screener not available")

            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: Screening stocks by market cap")
            logger.info("=" * 60)

            screened_stocks = self.ticker_screener.get_all_screened_tickers(
                exchange_codes=exchange_codes,
                include_secondary=include_secondary_range
            )

            logger.info(f"Screening complete: {len(screened_stocks)} stocks identified")

            # Convert to ticker tuples
            tickers = []
            for stock in screened_stocks:
                exchange_def = self.config.exchanges.get_exchange_by_code(stock.exchange)
                if exchange_def:
                    exchange_suffix = exchange_def.yahoo_suffix
                    tickers.append((stock.ticker, exchange_suffix))

            # Step 2: Download async
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: Downloading ticker data (ASYNC)")
            logger.info("=" * 60)

            results = await self.download_tickers_async(
                tickers,
                source=source,
                skip_existing=skip_existing
            )

            # Pipeline complete
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            logger.info("ASYNC PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")

            self._print_final_statistics()

            return self.get_statistics()

        except Exception as e:
            logger.error(f"Async pipeline failed: {e}", exc_info=True)
            raise

    def run_full_pipeline_sync(
        self,
        exchange_codes: Optional[List[str]] = None,
        include_secondary_range: bool = True,
        skip_existing: bool = True
    ) -> Dict:
        """
        Run complete pipeline synchronously (backward compatible).

        Args:
            exchange_codes: Exchange codes
            include_secondary_range: Include secondary range
            skip_existing: Skip existing files

        Returns:
            Statistics dictionary
        """
        # Run async pipeline in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.run_full_pipeline_async(
                exchange_codes=exchange_codes,
                include_secondary_range=include_secondary_range,
                skip_existing=skip_existing
            )
        )

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        stats = {
            'pipeline': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'tickers_processed': self.tickers_processed,
                'tickers_successful': self.tickers_successful,
                'tickers_failed': self.tickers_failed,
                'tickers_skipped': self.tickers_skipped,
                'success_rate': (
                    self.tickers_successful / max(1, self.tickers_processed)
                    if self.tickers_processed > 0 else 0
                )
            }
        }

        # Add async downloader stats if available
        if self._async_downloader:
            stats['async_downloader'] = self._async_downloader.get_statistics()

        return stats

    def _print_final_statistics(self):
        """Print comprehensive final statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)

        stats = self.get_statistics()
        pipeline = stats['pipeline']

        logger.info(f"Tickers processed: {pipeline['tickers_processed']}")
        logger.info(f"  Successful: {pipeline['tickers_successful']}")
        logger.info(f"  Failed: {pipeline['tickers_failed']}")
        logger.info(f"  Skipped: {pipeline['tickers_skipped']}")
        logger.info(f"Success rate: {pipeline['success_rate']*100:.1f}%")
        logger.info(f"Duration: {pipeline['duration_seconds']/60:.1f} minutes")

        logger.info("=" * 60)
