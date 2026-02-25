"""
API Key Manager - Intelligent 4-Key Rotation System

Manages multiple Alpha Vantage API keys with rate limit tracking
and automatic switching to maximize throughput.
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock


logger = logging.getLogger(__name__)


@dataclass
class KeyUsageStats:
    """Track usage statistics for a single API key"""

    key_id: int
    api_key: str

    # Call counters
    calls_this_minute: int = 0
    calls_today: int = 0
    total_calls: int = 0

    # Timing
    last_call_time: Optional[datetime] = None
    minute_window_start: Optional[datetime] = None
    day_start: Optional[datetime] = None

    # Limits (from config)
    max_calls_per_minute: int = 5
    max_calls_per_day: int = 500

    # Status
    is_available: bool = True
    cooldown_until: Optional[datetime] = None

    def reset_minute_counter(self):
        """Reset per-minute counter"""
        self.calls_this_minute = 0
        self.minute_window_start = datetime.now()

    def reset_day_counter(self):
        """Reset daily counter"""
        self.calls_today = 0
        self.day_start = datetime.now()

    def check_and_update_limits(self) -> bool:
        """
        Check if key can be used and update counters

        Returns:
            bool: True if key is available, False if rate limited
        """
        now = datetime.now()

        # Initialize timers on first use
        if self.minute_window_start is None:
            self.reset_minute_counter()

        if self.day_start is None:
            self.reset_day_counter()

        # Check if in cooldown period
        if self.cooldown_until and now < self.cooldown_until:
            return False

        # Check daily limit
        if self.calls_today >= self.max_calls_per_day:
            # Check if we've moved to a new day
            if now.date() > self.day_start.date():
                self.reset_day_counter()
            else:
                logger.warning(f"Key {self.key_id}: Daily limit reached ({self.calls_today}/{self.max_calls_per_day})")
                self.is_available = False
                # Set cooldown until midnight
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                self.cooldown_until = tomorrow
                return False

        # Check per-minute limit
        if (now - self.minute_window_start).total_seconds() >= 60:
            # New minute started
            self.reset_minute_counter()

        if self.calls_this_minute >= self.max_calls_per_minute:
            # Calculate wait time until next minute
            seconds_until_reset = 60 - (now - self.minute_window_start).total_seconds()
            logger.debug(f"Key {self.key_id}: Minute limit reached, waiting {seconds_until_reset:.1f}s")
            self.cooldown_until = now + timedelta(seconds=seconds_until_reset)
            return False

        # Key is available
        return True

    def record_call(self):
        """Record an API call"""
        now = datetime.now()

        self.calls_this_minute += 1
        self.calls_today += 1
        self.total_calls += 1
        self.last_call_time = now
        self.is_available = True  # Reset availability flag
        self.cooldown_until = None

        logger.debug(
            f"Key {self.key_id}: Call recorded "
            f"(minute: {self.calls_this_minute}/{self.max_calls_per_minute}, "
            f"day: {self.calls_today}/{self.max_calls_per_day})"
        )

    def get_status_dict(self) -> Dict:
        """Get status as dictionary for logging"""
        return {
            'key_id': self.key_id,
            'calls_this_minute': self.calls_this_minute,
            'calls_today': self.calls_today,
            'total_calls': self.total_calls,
            'is_available': self.is_available,
            'last_call': self.last_call_time.isoformat() if self.last_call_time else None,
        }


class APIKeyManager:
    """
    Manages multiple Alpha Vantage API keys with intelligent rotation

    Features:
    - Automatic key rotation to maximize throughput
    - Per-minute and per-day rate limit tracking
    - Cooldown management
    - Thread-safe operations
    """

    def __init__(
        self,
        api_keys: List[str],
        calls_per_minute: int = 5,
        calls_per_day: int = 500
    ):
        """
        Initialize API Key Manager

        Args:
            api_keys: List of Alpha Vantage API keys (up to 4)
            calls_per_minute: Rate limit per key per minute
            calls_per_day: Rate limit per key per day
        """
        if not api_keys or len(api_keys) == 0:
            raise ValueError("At least one API key is required")

        # Filter out empty keys
        self.api_keys = [key for key in api_keys if key]

        if not self.api_keys:
            raise ValueError("No valid API keys provided")

        logger.info(f"Initialized APIKeyManager with {len(self.api_keys)} keys")

        # Create usage trackers for each key
        self.key_stats: List[KeyUsageStats] = []
        for i, key in enumerate(self.api_keys):
            stats = KeyUsageStats(
                key_id=i,
                api_key=key,
                max_calls_per_minute=calls_per_minute,
                max_calls_per_day=calls_per_day
            )
            self.key_stats.append(stats)

        # Current key index (round-robin starting point)
        self.current_key_index = 0

        # Thread safety
        self._lock = Lock()

        # Statistics
        self.total_calls_made = 0
        self.total_waits = 0
        self.total_wait_time_seconds = 0.0

    def get_available_key(self, wait_if_needed: bool = True, max_wait_seconds: float = 300) -> Optional[str]:
        """
        Get an available API key

        Strategy:
        1. Try to find immediately available key (round-robin)
        2. If none available and wait_if_needed=True, wait for next available
        3. If max_wait_seconds exceeded, return None

        Args:
            wait_if_needed: If True, wait for a key to become available
            max_wait_seconds: Maximum time to wait for a key

        Returns:
            API key string or None if no key available
        """
        with self._lock:
            start_time = time.time()

            while True:
                # Try to find an available key
                available_key = self._find_available_key()

                if available_key:
                    # Record the call
                    key_id = self._get_key_id(available_key)
                    self.key_stats[key_id].record_call()
                    self.total_calls_made += 1
                    return available_key

                # No key available
                if not wait_if_needed:
                    logger.warning("No API keys available and wait_if_needed=False")
                    return None

                # Check if we've exceeded max wait time
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    logger.error(f"No API key available after waiting {elapsed:.1f}s")
                    return None

                # Calculate wait time until next key is available
                wait_time = self._get_min_wait_time()

                if wait_time is None or wait_time > max_wait_seconds - elapsed:
                    logger.error(f"All keys rate limited, wait time exceeds max_wait_seconds")
                    return None

                logger.info(f"All keys busy, waiting {wait_time:.1f}s for next available key...")
                self.total_waits += 1
                self.total_wait_time_seconds += wait_time

                # Release lock while sleeping
                self._lock.release()
                time.sleep(wait_time)
                self._lock.acquire()

    def _find_available_key(self) -> Optional[str]:
        """
        Find an available key using round-robin strategy

        Returns:
            API key or None if all keys are rate limited
        """
        # Try all keys starting from current position
        for i in range(len(self.key_stats)):
            index = (self.current_key_index + i) % len(self.key_stats)
            stats = self.key_stats[index]

            if stats.check_and_update_limits():
                # Update current key index for next call
                self.current_key_index = (index + 1) % len(self.key_stats)
                logger.debug(f"Selected key {stats.key_id}")
                return stats.api_key

        return None

    def _get_min_wait_time(self) -> Optional[float]:
        """
        Get minimum wait time until next key is available

        Returns:
            Wait time in seconds, or None if all keys exhausted for the day
        """
        now = datetime.now()
        min_wait = None

        for stats in self.key_stats:
            # Skip keys that are exhausted for the day
            if stats.calls_today >= stats.max_calls_per_day:
                if now.date() >= stats.day_start.date():
                    continue  # This key is done for today

            # Check cooldown
            if stats.cooldown_until:
                wait = (stats.cooldown_until - now).total_seconds()
                if wait > 0:
                    if min_wait is None or wait < min_wait:
                        min_wait = wait

        return min_wait

    def _get_key_id(self, api_key: str) -> int:
        """Get key ID for a given API key"""
        for stats in self.key_stats:
            if stats.api_key == api_key:
                return stats.key_id
        return -1

    def get_statistics(self) -> Dict:
        """
        Get usage statistics for all keys

        Returns:
            Dictionary with overall and per-key statistics
        """
        with self._lock:
            return {
                'total_calls': self.total_calls_made,
                'total_waits': self.total_waits,
                'total_wait_time_seconds': self.total_wait_time_seconds,
                'num_keys': len(self.key_stats),
                'keys': [stats.get_status_dict() for stats in self.key_stats]
            }

    def print_statistics(self):
        """Print usage statistics to logger"""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("API Key Manager Statistics")
        logger.info("=" * 60)
        logger.info(f"Total API calls made: {stats['total_calls']}")
        logger.info(f"Total waits: {stats['total_waits']}")
        logger.info(f"Total wait time: {stats['total_wait_time_seconds']:.1f}s")
        logger.info(f"Number of keys: {stats['num_keys']}")
        logger.info("")

        for key_stats in stats['keys']:
            logger.info(f"Key {key_stats['key_id']}:")
            logger.info(f"  Calls today: {key_stats['calls_today']}")
            logger.info(f"  Calls this minute: {key_stats['calls_this_minute']}")
            logger.info(f"  Total calls: {key_stats['total_calls']}")
            logger.info(f"  Available: {key_stats['is_available']}")
            logger.info(f"  Last call: {key_stats['last_call']}")
            logger.info("")

        logger.info("=" * 60)

    def reset_all_keys(self):
        """Reset all key counters (for testing)"""
        with self._lock:
            for stats in self.key_stats:
                stats.reset_minute_counter()
                stats.reset_day_counter()
                stats.is_available = True
                stats.cooldown_until = None

        logger.info("All API keys reset")
