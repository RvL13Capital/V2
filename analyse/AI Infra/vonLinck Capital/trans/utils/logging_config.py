"""
Structured Logging Configuration for TRANS Production System
=============================================================

Production-ready logging with JSON format, request tracing, and database integration.
Supports both legacy simple logging and new structured logging for backward compatibility.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import uuid
import traceback
from contextlib import contextmanager
import time


# =====================================================================
# Production Structured Logging Components
# =====================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs logs in JSON format with all relevant metadata for production monitoring.
    """

    def __init__(self, include_stacktrace: bool = True):
        """
        Initialize JSON formatter.

        Args:
            include_stacktrace: Include full stacktrace for errors
        """
        super().__init__()
        self.include_stacktrace = include_stacktrace
        self.hostname = os.environ.get('HOSTNAME', 'localhost')

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'hostname': self.hostname
        }

        # Add custom fields if present
        for field in ['request_id', 'component', 'ticker', 'pattern_id',
                     'execution_time_ms', 'memory_usage_mb', 'model_version_id']:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        # Add exception info if present
        if record.exc_info and self.include_stacktrace:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }
        elif record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1])
            }

        return json.dumps(log_data)


class RequestContextFilter(logging.Filter):
    """Add request context to log records for tracking."""

    def __init__(self):
        """Initialize context filter."""
        super().__init__()
        self.request_id = None

    def set_request_id(self, request_id: str):
        """Set current request ID."""
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request ID to log record."""
        if self.request_id:
            record.request_id = self.request_id
        return True


class DatabaseLogHandler(logging.Handler):
    """Handler that writes logs to database."""

    def __init__(self, db_manager=None):
        """
        Initialize database log handler.

        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        self.db_manager = db_manager

    def emit(self, record: logging.LogRecord):
        """Write log record to database."""
        if not self.db_manager:
            return

        try:
            # Import here to avoid circular dependency
            from database.models import SystemLog

            with self.db_manager.get_session() as session:
                log_entry = SystemLog(
                    timestamp=datetime.utcfromtimestamp(record.created),
                    level=record.levelname,
                    component=getattr(record, 'component', 'unknown'),
                    message=record.getMessage(),
                    ticker=getattr(record, 'ticker', None),
                    pattern_id=getattr(record, 'pattern_id', None),
                    model_version_id=getattr(record, 'model_version_id', None),
                    error_type=record.exc_info[0].__name__ if record.exc_info else None,
                    error_trace=''.join(traceback.format_exception(*record.exc_info)) if record.exc_info else None,
                    execution_time_ms=getattr(record, 'execution_time_ms', None),
                    memory_usage_mb=getattr(record, 'memory_usage_mb', None),
                    request_id=getattr(record, 'request_id', None),
                    user_id=getattr(record, 'user_id', None)
                )
                session.add(log_entry)
                session.commit()
        except Exception:
            # Don't let logging errors break the application
            pass


class ProductionLoggerManager:
    """
    Centralized logger management for production TRANS system.
    Provides structured JSON logging with request tracing.
    """

    def __init__(self,
                 log_level: str = "INFO",
                 log_dir: Optional[Path] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_database: bool = False,
                 db_manager=None,
                 use_json_format: bool = True):
        """
        Initialize production logger manager.

        Args:
            log_level: Logging level
            log_dir: Directory for log files
            enable_console: Enable console output
            enable_file: Enable file output
            enable_database: Enable database logging
            db_manager: Database manager for database logging
            use_json_format: Use JSON format (False for legacy format)
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir or Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_database = enable_database
        self.db_manager = db_manager
        self.use_json_format = use_json_format
        self.context_filter = RequestContextFilter()

        # Create log directory if needed
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        self._configure_root_logger()

    def _configure_root_logger(self):
        """Configure the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Clear existing handlers
        root_logger.handlers = []

        # Choose formatter based on config
        if self.use_json_format:
            console_formatter = JSONFormatter(include_stacktrace=False)
            file_formatter = JSONFormatter(include_stacktrace=True)
        else:
            # Legacy format for backward compatibility
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_formatter = file_formatter = formatter

        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(self.context_filter)
            root_logger.addHandler(console_handler)

        # Add file handlers
        if self.enable_file:
            # Main log file
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "trans.log",
                maxBytes=100 * 1024 * 1024,  # 100 MB
                backupCount=10
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(self.context_filter)
            root_logger.addHandler(file_handler)

            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "trans_errors.log",
                maxBytes=50 * 1024 * 1024,  # 50 MB
                backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            error_handler.addFilter(self.context_filter)
            root_logger.addHandler(error_handler)

        # Add database handler
        if self.enable_database and self.db_manager:
            db_handler = DatabaseLogHandler(self.db_manager)
            db_handler.setLevel(logging.WARNING)  # Only warnings and above
            db_handler.addFilter(self.context_filter)
            root_logger.addHandler(db_handler)

    def get_logger(self, name: str, component: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance with optional component tracking.

        Args:
            name: Logger name (usually __name__)
            component: Component name for tracking

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Add component as a filter if specified
        if component:
            class ComponentFilter(logging.Filter):
                def filter(self, record):
                    record.component = component
                    return True
            logger.addFilter(ComponentFilter())

        return logger

    @contextmanager
    def request_context(self, request_id: Optional[str] = None):
        """
        Context manager for request tracking.

        Args:
            request_id: Request ID (generates UUID if not provided)

        Example:
            with logger_manager.request_context() as req_id:
                logger.info("Processing request", ticker="AAPL")
        """
        request_id = request_id or str(uuid.uuid4())
        old_request_id = self.context_filter.request_id

        try:
            self.context_filter.set_request_id(request_id)
            yield request_id
        finally:
            self.context_filter.set_request_id(old_request_id)


# Global production logger manager
_production_logger_manager: Optional[ProductionLoggerManager] = None


def initialize_production_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_database: bool = False,
    db_manager=None,
    use_json_format: bool = True
) -> ProductionLoggerManager:
    """
    Initialize production logging configuration.

    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_console: Enable console output
        enable_file: Enable file output
        enable_database: Enable database logging
        db_manager: Database manager for database logging
        use_json_format: Use JSON format for structured logging

    Returns:
        Production logger manager instance
    """
    global _production_logger_manager
    _production_logger_manager = ProductionLoggerManager(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_database=enable_database,
        db_manager=db_manager,
        use_json_format=use_json_format
    )
    return _production_logger_manager


def get_production_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Get a production logger instance.

    Args:
        name: Logger name (usually __name__)
        component: Component name for tracking

    Returns:
        Configured logger instance
    """
    global _production_logger_manager
    if not _production_logger_manager:
        _production_logger_manager = initialize_production_logging()
    return _production_logger_manager.get_logger(name, component)


def get_production_logger_manager() -> ProductionLoggerManager:
    """
    Get the global production logger manager instance.

    Returns:
        Production logger manager instance
    """
    global _production_logger_manager
    if not _production_logger_manager:
        _production_logger_manager = initialize_production_logging()
    return _production_logger_manager


# =====================================================================
# Legacy Functions (Maintained for Backward Compatibility)
# =====================================================================

def setup_pipeline_logging(
    step_name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    include_timestamp: bool = True,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """Setup standardized logging for pipeline steps

    Creates a logger with consistent formatting and handlers for both
    file and console output. Ensures log directory exists and handles
    all edge cases.

    Args:
        step_name: Name of the pipeline step (e.g., '01_generate_sequences')
        log_dir: Directory for log files (default: 'logs')
        level: Logging level (default: INFO)
        include_timestamp: Include timestamp in log filename
        console_output: Enable console output
        file_output: Enable file output

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_pipeline_logging('01_generate_sequences')
        >>> logger.info("Starting sequence generation")
    """
    # Use default log directory if not specified
    if log_dir is None:
        log_dir = Path('logs')

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger instance
    logger = logging.getLogger(step_name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with consistent format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add file handler if requested
    if file_output:
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{step_name}_{timestamp}.log"
        else:
            log_filename = f"{step_name}.log"

        log_filepath = log_dir / log_filename

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Log initial setup message
    logger.info(f"Logger initialized for {step_name}")
    if file_output:
        logger.info(f"Log file: {log_filepath}")

    return logger


def setup_module_logger(
    module_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger for a specific module

    Simpler logging setup for individual modules that don't need
    file output.

    Args:
        module_name: Name of the module
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Only add handler if none exist
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class LoggerManager:
    """Manager for multiple loggers in a pipeline

    Useful for complex pipelines that need multiple specialized loggers.
    """

    def __init__(self, base_log_dir: Path = Path('logs')):
        """Initialize logger manager

        Args:
            base_log_dir: Base directory for all log files
        """
        self.base_log_dir = base_log_dir
        self.loggers = {}

    def get_logger(
        self,
        name: str,
        subdir: Optional[str] = None,
        **kwargs
    ) -> logging.Logger:
        """Get or create a logger

        Args:
            name: Logger name
            subdir: Subdirectory under base_log_dir
            **kwargs: Additional arguments for setup_pipeline_logging

        Returns:
            Logger instance
        """
        if name not in self.loggers:
            log_dir = self.base_log_dir
            if subdir:
                log_dir = log_dir / subdir

            self.loggers[name] = setup_pipeline_logging(
                name,
                log_dir=log_dir,
                **kwargs
            )

        return self.loggers[name]

    def close_all(self):
        """Close all file handlers for all loggers"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()


def configure_root_logger(level: int = logging.WARNING):
    """Configure the root logger to reduce noise from libraries

    Sets up the root logger to only show warnings and above,
    preventing verbose output from third-party libraries.

    Args:
        level: Logging level for root logger
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant

    Args:
        level_str: String representation of log level

    Returns:
        Logging level constant

    Raises:
        ValueError: If level_str is not recognized
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level_str = level_str.upper()
    if level_str not in levels:
        raise ValueError(f"Unknown log level: {level_str}")

    return levels[level_str]


def cleanup_old_logs(
    log_dir: Path,
    days_to_keep: int = 30,
    pattern: str = "*.log"
) -> int:
    """Clean up old log files

    Removes log files older than specified number of days.

    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep logs
        pattern: Glob pattern for log files

    Returns:
        Number of files removed
    """
    if not log_dir.exists():
        return 0

    import time
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)

    removed_count = 0
    for log_file in log_dir.glob(pattern):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {log_file}: {e}")

    return removed_count


class ProgressLogger:
    """Helper class for logging progress of long-running operations"""

    def __init__(self, logger: logging.Logger, total_items: int, log_interval: int = 10):
        """Initialize progress logger

        Args:
            logger: Logger instance to use
            total_items: Total number of items to process
            log_interval: Log progress every N items
        """
        self.logger = logger
        self.total_items = total_items
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = datetime.now()

    def update(self, items: int = 1, message: Optional[str] = None):
        """Update progress

        Args:
            items: Number of items processed
            message: Optional message to include
        """
        self.processed += items

        if self.processed % self.log_interval == 0 or self.processed == self.total_items:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.processed / elapsed if elapsed > 0 else 0
            pct = (self.processed / self.total_items) * 100

            log_msg = f"Progress: {self.processed}/{self.total_items} ({pct:.1f}%) - Rate: {rate:.1f} items/sec"
            if message:
                log_msg += f" - {message}"

            self.logger.info(log_msg)

    def finish(self, message: Optional[str] = None):
        """Log completion message

        Args:
            message: Optional completion message
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log_msg = f"Completed {self.processed} items in {elapsed:.1f} seconds"
        if message:
            log_msg += f" - {message}"

        self.logger.info(log_msg)