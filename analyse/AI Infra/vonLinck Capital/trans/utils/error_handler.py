"""
Error Handling and Recovery Framework
======================================

Production-ready error handling with retry logic, circuit breakers, and graceful degradation.
"""

import functools
import time
import random
import logging
from typing import Any, Callable, Optional, Type, Union, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import traceback
import threading
from collections import deque

from utils.logging_config import get_production_logger

logger = get_production_logger(__name__, "error_handler")


class ErrorSeverity(Enum):
    """Error severity levels for categorization and response."""
    LOW = "low"  # Can continue with warning
    MEDIUM = "medium"  # Should retry or handle
    HIGH = "high"  # Requires intervention
    CRITICAL = "critical"  # System should halt


class RetryStrategy(Enum):
    """Retry strategies for error recovery."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


class TransError(Exception):
    """Base exception class for TRANS system errors."""

    def __init__(self,
                 message: str,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recoverable: bool = True,
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize TRANS error.

        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether error is recoverable
            context: Additional context information
        """
        super().__init__(message)
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = datetime.utcnow()


class DataError(TransError):
    """Errors related to data loading, validation, or processing."""
    pass


class ModelError(TransError):
    """Errors related to model loading, training, or prediction."""
    pass


class PatternError(TransError):
    """Errors related to pattern detection or labeling."""
    pass


class DatabaseError(TransError):
    """Errors related to database operations."""
    pass


class ConfigurationError(TransError):
    """Errors related to configuration or environment setup."""
    pass


class RetryExhaustedError(TransError):
    """Raised when all retry attempts have been exhausted."""
    pass


class CircuitBreakerError(TransError):
    """Raised when circuit breaker is open."""
    pass


def exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exceptions to retry on
        on_retry: Callback function called on each retry

    Example:
        @exponential_backoff(max_retries=5, initial_delay=2)
        def fetch_data():
            return api_call()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1

                    if attempt >= max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}",
                            exc_info=True,
                            execution_time_ms=(attempt * delay * 1000)
                        )
                        raise RetryExhaustedError(
                            f"Failed after {max_retries} attempts: {str(e)}",
                            severity=ErrorSeverity.HIGH,
                            recoverable=False,
                            context={"function": func.__name__, "attempts": max_retries}
                        )

                    # Calculate next delay
                    delay = min(delay * exponential_base, max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt}/{max_retries} for {func.__name__} after {delay:.2f}s",
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(delay)

            return None

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name for identification
            failure_threshold: Number of failures to open circuit
            success_threshold: Number of successes to close circuit
            timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to track
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == self.State.OPEN:
                if self._should_attempt_reset():
                    self.state = self.State.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        severity=ErrorSeverity.HIGH,
                        recoverable=True,
                        context={"breaker": self.name, "state": self.state.value}
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time and
                datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout))

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == self.State.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = self.State.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
            elif self.state == self.State.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.state == self.State.HALF_OPEN:
                self.state = self.State.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after failure in HALF_OPEN")
            elif self.failure_count >= self.failure_threshold:
                self.state = self.State.OPEN
                logger.error(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = self.State.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class ErrorHandler:
    """
    Centralized error handling and recovery management.
    """

    def __init__(self,
                 max_error_history: int = 100,
                 alert_threshold: int = 10,
                 alert_window_minutes: int = 5):
        """
        Initialize error handler.

        Args:
            max_error_history: Maximum number of errors to keep in history
            alert_threshold: Number of errors to trigger alert
            alert_window_minutes: Time window for alert threshold
        """
        self.max_error_history = max_error_history
        self.alert_threshold = alert_threshold
        self.alert_window = timedelta(minutes=alert_window_minutes)

        self.error_history = deque(maxlen=max_error_history)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        self._lock = threading.Lock()

    def register_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker):
        """Register a circuit breaker for monitoring."""
        self.circuit_breakers[name] = circuit_breaker

    def register_recovery_strategy(self,
                                  error_type: Type[Exception],
                                  strategy: Callable[[Exception], Any]):
        """
        Register a recovery strategy for specific error type.

        Args:
            error_type: Exception type to handle
            strategy: Recovery function
        """
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error: The exception to handle
            context: Additional context information

        Returns:
            Recovery result if applicable
        """
        with self._lock:
            # Record error
            error_record = {
                "timestamp": datetime.utcnow(),
                "type": type(error).__name__,
                "message": str(error),
                "context": context or {},
                "traceback": traceback.format_exc()
            }
            self.error_history.append(error_record)

            # Check for alert condition
            self._check_alert_condition()

            # Find and apply recovery strategy
            for error_type, strategy in self.recovery_strategies.items():
                if isinstance(error, error_type):
                    logger.info(f"Applying recovery strategy for {error_type.__name__}")
                    return strategy(error)

            # No recovery strategy found
            if isinstance(error, TransError):
                if not error.recoverable:
                    logger.critical(f"Non-recoverable error: {error}")
                    raise
                elif error.severity == ErrorSeverity.CRITICAL:
                    logger.critical(f"Critical error requiring intervention: {error}")
                    raise

            # Default behavior for unknown errors
            logger.error(f"No recovery strategy for {type(error).__name__}: {error}")
            raise

    def _check_alert_condition(self):
        """Check if error rate triggers alert."""
        cutoff_time = datetime.utcnow() - self.alert_window
        recent_errors = [e for e in self.error_history
                        if e["timestamp"] > cutoff_time]

        if len(recent_errors) >= self.alert_threshold:
            logger.critical(
                f"Alert: {len(recent_errors)} errors in last {self.alert_window.total_seconds()/60:.1f} minutes",
                error_count=len(recent_errors),
                error_types=list({e["type"] for e in recent_errors})
            )
            # Here you could trigger additional alerts (email, Slack, etc.)

    @contextmanager
    def error_context(self, operation_name: str, **context):
        """
        Context manager for error handling with automatic recovery.

        Args:
            operation_name: Name of the operation
            **context: Additional context information

        Example:
            with error_handler.error_context("data_loading", ticker="AAPL"):
                load_data()
        """
        start_time = time.time()
        context["operation"] = operation_name

        try:
            yield
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"Operation '{operation_name}' completed successfully",
                        execution_time_ms=execution_time,
                        **context)
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            context["execution_time_ms"] = execution_time

            logger.error(f"Operation '{operation_name}' failed",
                        exc_info=True,
                        **context)

            # Try to handle the error
            result = self.handle_error(e, context)
            if result is not None:
                logger.info(f"Operation '{operation_name}' recovered",
                           recovery_result=result,
                           **context)
            else:
                raise


class GracefulDegradation:
    """
    Implement graceful degradation strategies for system resilience.
    """

    def __init__(self):
        """Initialize graceful degradation handler."""
        self.fallback_strategies = {}
        self.degradation_levels = {}

    def register_fallback(self,
                         feature: str,
                         primary: Callable,
                         fallback: Callable,
                         condition: Optional[Callable[[Exception], bool]] = None):
        """
        Register a fallback strategy for a feature.

        Args:
            feature: Feature name
            primary: Primary implementation
            fallback: Fallback implementation
            condition: Optional condition to trigger fallback
        """
        self.fallback_strategies[feature] = {
            "primary": primary,
            "fallback": fallback,
            "condition": condition or (lambda e: True)
        }

    def execute(self, feature: str, *args, **kwargs) -> Any:
        """
        Execute feature with fallback if needed.

        Args:
            feature: Feature to execute
            *args: Feature arguments
            **kwargs: Feature keyword arguments

        Returns:
            Feature result
        """
        if feature not in self.fallback_strategies:
            raise ValueError(f"No fallback strategy for feature: {feature}")

        strategy = self.fallback_strategies[feature]

        try:
            return strategy["primary"](*args, **kwargs)
        except Exception as e:
            if strategy["condition"](e):
                logger.warning(f"Falling back for feature '{feature}': {e}")
                return strategy["fallback"](*args, **kwargs)
            raise


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler
    if not _error_handler:
        _error_handler = ErrorHandler()
    return _error_handler


def safe_execute(func: Callable,
                 *args,
                 default_value: Any = None,
                 log_errors: bool = True,
                 **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        default_value: Default value if execution fails
        log_errors: Whether to log errors
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default_value


def validate_data_integrity(data: Any,
                           validations: Dict[str, Callable[[Any], bool]]) -> List[str]:
    """
    Validate data integrity with multiple checks.

    Args:
        data: Data to validate
        validations: Dictionary of validation name to validation function

    Returns:
        List of failed validation names
    """
    failures = []

    for name, validation in validations.items():
        try:
            if not validation(data):
                failures.append(name)
        except Exception as e:
            logger.error(f"Validation '{name}' raised error: {e}")
            failures.append(name)

    return failures


if __name__ == "__main__":
    # Test error handling framework
    import pandas as pd

    # Initialize components
    error_handler = get_error_handler()

    # Register circuit breaker for API calls
    api_breaker = CircuitBreaker(
        name="data_api",
        failure_threshold=3,
        success_threshold=2,
        timeout=30
    )
    error_handler.register_circuit_breaker("data_api", api_breaker)

    # Test exponential backoff
    @exponential_backoff(max_retries=3, initial_delay=1)
    def flaky_operation():
        """Simulate flaky operation."""
        import random
        if random.random() < 0.7:
            raise ConnectionError("Connection failed")
        return "Success"

    # Test circuit breaker
    def api_call():
        """Simulate API call."""
        import random
        if random.random() < 0.5:
            raise ConnectionError("API unavailable")
        return {"status": "ok"}

    # Test graceful degradation
    degradation = GracefulDegradation()

    def primary_data_source():
        raise ConnectionError("Primary source unavailable")

    def fallback_data_source():
        return pd.DataFrame({"value": [1, 2, 3]})

    degradation.register_fallback(
        "data_loading",
        primary_data_source,
        fallback_data_source
    )

    # Run tests
    print("Testing error handling framework...")

    # Test flaky operation with retry
    try:
        result = flaky_operation()
        print(f"Flaky operation succeeded: {result}")
    except RetryExhaustedError as e:
        print(f"Flaky operation failed: {e}")

    # Test circuit breaker
    for i in range(10):
        try:
            result = api_breaker.call(api_call)
            print(f"API call {i+1} succeeded")
        except (ConnectionError, CircuitBreakerError) as e:
            print(f"API call {i+1} failed: {e}")
        time.sleep(0.5)

    # Test graceful degradation
    data = degradation.execute("data_loading")
    print(f"Data loaded with degradation: {data.shape}")