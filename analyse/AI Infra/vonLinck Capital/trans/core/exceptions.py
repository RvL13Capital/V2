"""
Exception Hierarchy for TRANS Temporal Architecture

Provides custom exceptions for consistent error handling throughout the system.
Ensures proper error categorization and debugging information.
"""

from typing import Optional, Any, Dict


class TemporalPipelineError(Exception):
    """Base exception for all temporal pipeline errors

    This is the base class for all custom exceptions in the TRANS system.
    It provides enhanced error context and debugging information.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the exception with enhanced context

        Args:
            message: The error message
            details: Additional context as a dictionary
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self):
        """String representation of the error"""
        parts = [self.message]

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {detail_str}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")

        return " | ".join(parts)


class DataIntegrityError(TemporalPipelineError):
    """Raised when data integrity violations are detected

    Examples:
    - Temporal gaps or duplicates in time series
    - Mock or synthetic data detected
    - OHLCV relationship violations (e.g., high < low)
    """
    pass


class TemporalConsistencyError(TemporalPipelineError):
    """Raised when temporal ordering or consistency is violated

    Examples:
    - Data not in chronological order
    - Time going backwards
    - Future data appearing before past data
    """
    pass


class ValidationError(TemporalPipelineError):
    """Raised when data validation fails

    Examples:
    - Missing required columns
    - Invalid data types
    - Out-of-range values
    - Insufficient data length
    """
    pass


class FeatureCalculationError(TemporalPipelineError):
    """Raised when feature calculation fails

    Examples:
    - Indicator calculation errors
    - Insufficient data for calculation
    - Mathematical errors (division by zero, etc.)
    """
    pass


class ModelError(TemporalPipelineError):
    """Raised for model-related errors

    Examples:
    - Model loading failures
    - Shape mismatches
    - Checkpoint corruption
    - Training failures
    """
    pass


class ConfigError(TemporalPipelineError):
    """Raised for configuration errors

    Examples:
    - Invalid configuration values
    - Missing required configuration
    - Conflicting configuration settings
    """
    pass


class DataLoadError(TemporalPipelineError):
    """Raised when data loading fails

    Examples:
    - File not found
    - Corrupt data files
    - Network errors (for GCS)
    - Permission errors
    """
    pass


class SequenceGenerationError(TemporalPipelineError):
    """Raised when sequence generation fails

    Examples:
    - Insufficient pattern duration
    - Invalid window size
    - Feature extraction failures
    """
    pass


class PredictionError(TemporalPipelineError):
    """Raised when prediction generation fails

    Examples:
    - Model not loaded
    - Invalid input shape
    - Batch processing errors
    """
    pass


class EvaluationError(TemporalPipelineError):
    """Raised when evaluation fails

    Examples:
    - Missing predictions
    - Label mismatches
    - Metric calculation errors
    """
    pass


# Utility functions for error handling

def handle_data_error(
    error: Exception,
    ticker: Optional[str] = None,
    operation: Optional[str] = None
) -> DataIntegrityError:
    """Convert a generic data error to a DataIntegrityError with context

    Args:
        error: The original exception
        ticker: The ticker being processed
        operation: The operation that failed

    Returns:
        DataIntegrityError with enhanced context
    """
    details = {}
    if ticker:
        details['ticker'] = ticker
    if operation:
        details['operation'] = operation

    message = f"Data error during {operation or 'processing'}"
    if ticker:
        message += f" for {ticker}"

    return DataIntegrityError(
        message=message,
        details=details,
        cause=error
    )


def handle_model_error(
    error: Exception,
    model_path: Optional[str] = None,
    operation: Optional[str] = None
) -> ModelError:
    """Convert a generic model error to a ModelError with context

    Args:
        error: The original exception
        model_path: Path to the model file
        operation: The operation that failed

    Returns:
        ModelError with enhanced context
    """
    details = {}
    if model_path:
        details['model_path'] = model_path
    if operation:
        details['operation'] = operation

    message = f"Model error during {operation or 'processing'}"
    if model_path:
        message += f" with {model_path}"

    return ModelError(
        message=message,
        details=details,
        cause=error
    )


def validate_not_none(value: Any, name: str) -> Any:
    """Validate that a value is not None

    Args:
        value: The value to check
        name: The name of the value (for error messages)

    Returns:
        The value if not None

    Raises:
        ValidationError: If the value is None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive

    Args:
        value: The value to check
        name: The name of the value (for error messages)

    Returns:
        The value if positive

    Raises:
        ValidationError: If the value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive",
            details={'value': value, 'name': name}
        )
    return value


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str
) -> float:
    """Validate that a value is within a range

    Args:
        value: The value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: The name of the value (for error messages)

    Returns:
        The value if within range

    Raises:
        ValidationError: If the value is out of range
    """
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}",
            details={
                'value': value,
                'min': min_val,
                'max': max_val,
                'name': name
            }
        )
    return value