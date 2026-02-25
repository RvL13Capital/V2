"""
Foreman Service - Main Orchestration Class

Coordinates all components to identify international stocks by market cap,
download historical data, and store in GCS.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.foreman_config import ForemanConfig, get_foreman_config
from foreman.api_key_manager import APIKeyManager
from foreman.exchange_mapper import ExchangeMapper
from foreman.ticker_screener import TickerScreener, ScreenedStock
from foreman.data_downloader import DataDownloader
from foreman.storage_manager import StorageManager


logger = logging.getLogger(__name__)


class ForemanService:
    """
    Main Foreman Service for International Stock Data Acquisition

    Pipeline:
    1. Screen stocks by market cap using FMP
    2. Map ticker symbols across exchanges using EODHD
    3. Download historical data using Alpha Vantage (4-key rotation)
    4. Store data in CSV + Parquet formats
    5. Upload to GCS

    Features:
    - Multi-exchange support (US, Canada, Europe)
    - Market cap filtering ($200M - $4B range)
    - Intelligent API key rotation
    - Robust error handling
    - Parallel processing
    """

    def __init__(self, config: Optional[ForemanConfig] = None):
        """
        Initialize Foreman Service

        Args:
            config: Foreman configuration (uses default if not provided)
        """
        self.config = config or get_foreman_config()

        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # Setup logging
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("Foreman Service Initializing")
        logger.info("=" * 60)

        # Initialize components
        logger.info("Initializing API Key Manager...")
        active_keys = self.config.get_active_alpha_vantage_keys()
        self.api_key_manager = APIKeyManager(
            api_keys=active_keys,
            calls_per_minute=self.config.api.alpha_vantage_calls_per_minute,
            calls_per_day=self.config.api.alpha_vantage_calls_per_day
        )

        logger.info("Initializing Exchange Mapper...")
        self.exchange_mapper = ExchangeMapper(self.config)

        logger.info("Initializing Ticker Screener...")
        self.ticker_screener = TickerScreener(self.config)

        logger.info("Initializing Data Downloader...")
        self.data_downloader = DataDownloader(self.config, self.api_key_manager)

        logger.info("Initializing Storage Manager...")
        self.storage_manager = StorageManager(self.config)

        # Statistics
        self.start_time = None
        self.end_time = None
        self.tickers_processed = 0
        self.tickers_successful = 0
        self.tickers_failed = 0
        self.tickers_skipped = 0

        logger.info("=" * 60)
        logger.info("Foreman Service Initialized Successfully")
        logger.info("=" * 60)

    def _setup_logging(self):
        """Setup logging configuration"""
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File logging if enabled
        if self.config.log_to_file:
            import os
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)

            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

            logging.getLogger().addHandler(file_handler)

    def run_full_pipeline(
        self,
        exchange_codes: Optional[List[str]] = None,
        include_secondary_range: bool = True,
        skip_existing: bool = True
    ) -> Dict:
        """
        Run complete data acquisition pipeline

        Args:
            exchange_codes: List of exchange codes to process (None = all)
            include_secondary_range: Include secondary market cap range ($2B-$4B)
            skip_existing: Skip tickers that already exist in GCS

        Returns:
            Statistics dictionary
        """
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Exchanges: {exchange_codes or 'ALL'}")
        logger.info(f"Include secondary range: {include_secondary_range}")
        logger.info(f"Skip existing: {skip_existing}")
        logger.info("=" * 60)

        try:
            # Step 1: Screen stocks by market cap
            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: Screening stocks by market cap")
            logger.info("=" * 60)

            screened_stocks = self.ticker_screener.get_all_screened_tickers(
                exchange_codes=exchange_codes,
                include_secondary=include_secondary_range
            )

            logger.info(f"Screening complete: {len(screened_stocks)} stocks identified")
            self.ticker_screener.print_statistics(screened_stocks)

            # Step 2: Process each ticker
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: Downloading and storing ticker data")
            logger.info("=" * 60)

            self._process_tickers(screened_stocks, skip_existing)

            # Pipeline complete
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")

            # Print all statistics
            self._print_final_statistics()

            return self.get_statistics()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _process_tickers(
        self,
        screened_stocks: List[ScreenedStock],
        skip_existing: bool
    ):
        """
        Process list of screened tickers

        Args:
            screened_stocks: List of stocks to process
            skip_existing: Skip if file already exists in GCS
        """
        total = len(screened_stocks)

        logger.info(f"Processing {total} tickers...")
        logger.info(f"Parallel downloads: {self.config.parallel_downloads}")
        logger.info(f"Batch size: {self.config.batch_size}")

        # Process in batches to manage memory
        batch_size = self.config.batch_size
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch = screened_stocks[start_idx:end_idx]

            logger.info(
                f"\nProcessing batch {batch_idx+1}/{num_batches} "
                f"(tickers {start_idx+1}-{end_idx})"
            )

            self._process_batch(batch, skip_existing)

        logger.info(f"\nCompleted processing {total} tickers")

    def _process_batch(
        self,
        batch: List[ScreenedStock],
        skip_existing: bool
    ):
        """
        Process a batch of tickers in parallel

        Args:
            batch: List of stocks to process
            skip_existing: Skip existing files
        """
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.config.parallel_downloads) as executor:
            # Submit all tasks
            futures = {}

            for stock in batch:
                # Get exchange suffix
                exchange_def = self.config.exchanges.get_exchange_by_code(stock.exchange)
                if not exchange_def:
                    logger.warning(f"Unknown exchange: {stock.exchange}, skipping {stock.ticker}")
                    self.tickers_skipped += 1
                    continue

                exchange_suffix = exchange_def.yahoo_suffix  # Use Yahoo suffix for consistency

                # Check if already exists
                if skip_existing:
                    if self.storage_manager.file_exists_in_gcs(stock.ticker, exchange_suffix, 'csv'):
                        logger.debug(f"Skipping {stock.ticker}{exchange_suffix} (already exists)")
                        self.tickers_skipped += 1
                        continue

                # Submit download task
                future = executor.submit(
                    self._process_single_ticker,
                    stock.ticker,
                    exchange_suffix,
                    stock.exchange
                )
                futures[future] = (stock.ticker, exchange_suffix)

            # Process completed futures
            for future in as_completed(futures):
                ticker, exchange_suffix = futures[future]
                self.tickers_processed += 1

                try:
                    success = future.result()

                    if success:
                        self.tickers_successful += 1
                        logger.info(
                            f"✓ [{self.tickers_processed}/{len(batch)}] "
                            f"{ticker}{exchange_suffix} - SUCCESS"
                        )
                    else:
                        self.tickers_failed += 1
                        logger.warning(
                            f"✗ [{self.tickers_processed}/{len(batch)}] "
                            f"{ticker}{exchange_suffix} - FAILED"
                        )

                except Exception as e:
                    self.tickers_failed += 1
                    logger.error(
                        f"✗ [{self.tickers_processed}/{len(batch)}] "
                        f"{ticker}{exchange_suffix} - ERROR: {e}"
                    )

    def _process_single_ticker(
        self,
        ticker: str,
        exchange_suffix: str,
        exchange_code: str
    ) -> bool:
        """
        Download and store data for a single ticker

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix
            exchange_code: Exchange code

        Returns:
            True if successful
        """
        try:
            # Download data
            df, metadata = self.data_downloader.download_ticker_data(
                ticker=ticker,
                exchange_suffix=exchange_suffix
            )

            if df is None or len(df) == 0:
                return False

            # Store data
            success = self.storage_manager.save_ticker_data(
                df=df,
                ticker=ticker,
                exchange_suffix=exchange_suffix,
                metadata=metadata
            )

            return success

        except Exception as e:
            logger.error(f"Error processing {ticker}{exchange_suffix}: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
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
            },
            'api_keys': self.api_key_manager.get_statistics(),
            'downloads': self.data_downloader.get_statistics(),
            'storage': self.storage_manager.get_statistics()
        }

    def _print_final_statistics(self):
        """Print comprehensive final statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)

        stats = self.get_statistics()

        # Pipeline stats
        pipeline = stats['pipeline']
        logger.info(f"Tickers processed: {pipeline['tickers_processed']}")
        logger.info(f"  Successful: {pipeline['tickers_successful']}")
        logger.info(f"  Failed: {pipeline['tickers_failed']}")
        logger.info(f"  Skipped: {pipeline['tickers_skipped']}")
        logger.info(f"Success rate: {pipeline['success_rate']*100:.1f}%")
        logger.info(f"Duration: {pipeline['duration_seconds']/60:.1f} minutes")

        # Component stats
        logger.info("")
        self.api_key_manager.print_statistics()
        logger.info("")
        self.data_downloader.print_statistics()
        logger.info("")
        self.storage_manager.print_statistics()

        logger.info("=" * 60)
