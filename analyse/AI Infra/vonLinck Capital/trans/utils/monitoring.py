"""
System Monitoring and Metrics Collection
========================================

Production monitoring with metrics aggregation, alerting, and performance tracking.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path

from database.connection import get_db_session
from database.models import MetricSnapshot, SystemLog
from utils.logging_config import get_production_logger
# Labeling version is always v17

logger = get_production_logger(__name__, "monitoring")


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Metric:
    """Base class for metrics."""

    def __init__(self, name: str, description: str, metric_type: MetricType):
        """
        Initialize metric.

        Args:
            name: Metric name
            description: Metric description
            metric_type: Type of metric
        """
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self._lock = threading.Lock()


class Counter(Metric):
    """Counter metric that only increases."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description, MetricType.COUNTER)
        self.value = 0

    def increment(self, amount: float = 1):
        """Increment counter."""
        with self._lock:
            self.value += amount

    def get(self) -> float:
        """Get current value."""
        return self.value


class Gauge(Metric):
    """Gauge metric that can go up or down."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description, MetricType.GAUGE)
        self.value = 0

    def set(self, value: float):
        """Set gauge value."""
        with self._lock:
            self.value = value

    def increment(self, amount: float = 1):
        """Increment gauge."""
        with self._lock:
            self.value += amount

    def decrement(self, amount: float = 1):
        """Decrement gauge."""
        with self._lock:
            self.value -= amount

    def get(self) -> float:
        """Get current value."""
        return self.value


class Histogram(Metric):
    """Histogram metric for distributions."""

    def __init__(self, name: str, description: str, buckets: Optional[List[float]] = None):
        super().__init__(name, description, MetricType.HISTOGRAM)
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.observations = []
        self.sum_value = 0
        self.count = 0

    def observe(self, value: float):
        """Record an observation."""
        with self._lock:
            self.observations.append(value)
            self.sum_value += value
            self.count += 1

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value."""
        if not self.observations:
            return 0
        sorted_obs = sorted(self.observations)
        index = int(len(sorted_obs) * percentile)
        return sorted_obs[min(index, len(sorted_obs) - 1)]

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.observations:
            return {"count": 0, "sum": 0, "mean": 0}

        return {
            "count": self.count,
            "sum": self.sum_value,
            "mean": self.sum_value / self.count if self.count > 0 else 0,
            "p50": self.get_percentile(0.5),
            "p95": self.get_percentile(0.95),
            "p99": self.get_percentile(0.99)
        }


class MetricsCollector:
    """
    Centralized metrics collection and aggregation.
    """

    def __init__(self,
                 collection_interval: int = 60,
                 retention_hours: int = 24):
        """
        Initialize metrics collector.

        Args:
            collection_interval: Seconds between collections
            retention_hours: Hours to retain metrics in memory
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours

        # Metric registries
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

        # Time series data
        self.time_series = defaultdict(lambda: deque(maxlen=retention_hours * 3600 // collection_interval))

        # Alert thresholds
        self.alert_rules = []

        # Background thread for collection
        self._collection_thread = None
        self._stop_event = threading.Event()

        # Initialize standard metrics
        self._initialize_standard_metrics()

    def _initialize_standard_metrics(self):
        """Initialize standard system metrics."""
        # Request metrics
        self.register_counter("requests_total", "Total API requests")
        self.register_counter("requests_failed", "Failed API requests")
        self.register_histogram("request_duration_seconds", "Request duration in seconds")

        # Pattern metrics
        self.register_counter("patterns_detected", "Total patterns detected")
        self.register_counter("patterns_labeled", "Total patterns labeled")
        self.register_counter("predictions_made", "Total predictions made")

        # System metrics
        self.register_gauge("cpu_usage_percent", "CPU usage percentage")
        self.register_gauge("memory_usage_mb", "Memory usage in MB")
        self.register_gauge("disk_usage_gb", "Disk usage in GB")
        self.register_gauge("active_connections", "Active database connections")

        # Model metrics
        self.register_gauge("model_accuracy", "Current model accuracy")
        self.register_gauge("model_ev_correlation", "Model EV correlation")
        self.register_histogram("model_inference_seconds", "Model inference time")

        # Error metrics
        self.register_counter("errors_total", "Total errors")
        self.register_gauge("error_rate_per_minute", "Errors per minute")

    def register_counter(self, name: str, description: str) -> Counter:
        """Register a counter metric."""
        counter = Counter(name, description)
        self.counters[name] = counter
        return counter

    def register_gauge(self, name: str, description: str) -> Gauge:
        """Register a gauge metric."""
        gauge = Gauge(name, description)
        self.gauges[name] = gauge
        return gauge

    def register_histogram(self, name: str, description: str, buckets: Optional[List[float]] = None) -> Histogram:
        """Register a histogram metric."""
        histogram = Histogram(name, description, buckets)
        self.histograms[name] = histogram
        return histogram

    def increment_counter(self, name: str, amount: float = 1):
        """Increment a counter."""
        if name in self.counters:
            self.counters[name].increment(amount)

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        if name in self.gauges:
            self.gauges[name].set(value)

    def observe_histogram(self, name: str, value: float):
        """Record a histogram observation."""
        if name in self.histograms:
            self.histograms[name].observe(value)

    def add_alert_rule(self,
                      metric_name: str,
                      threshold: float,
                      comparator: str,
                      level: AlertLevel,
                      message: str,
                      callback: Optional[Callable] = None):
        """
        Add an alert rule.

        Args:
            metric_name: Metric to monitor
            threshold: Threshold value
            comparator: Comparison operator (>, <, >=, <=, ==)
            level: Alert severity
            message: Alert message
            callback: Optional callback function
        """
        self.alert_rules.append({
            "metric": metric_name,
            "threshold": threshold,
            "comparator": comparator,
            "level": level,
            "message": message,
            "callback": callback
        })

    def start_collection(self):
        """Start background metrics collection."""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_event.clear()
            self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._collection_thread.start()
            logger.info("Started metrics collection")

    def stop_collection(self):
        """Stop background metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_event.set()
            self._collection_thread.join(timeout=5)
            logger.info("Stopped metrics collection")

    def _collect_loop(self):
        """Background collection loop."""
        while not self._stop_event.is_set():
            try:
                self._collect_metrics()
                self._check_alerts()
                self._store_snapshot()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

            self._stop_event.wait(self.collection_interval)

    def _collect_metrics(self):
        """Collect current metric values."""
        timestamp = datetime.utcnow()

        # Collect system metrics
        self.set_gauge("cpu_usage_percent", psutil.cpu_percent())
        self.set_gauge("memory_usage_mb", psutil.virtual_memory().used / 1024 / 1024)

        disk = psutil.disk_usage('/')
        self.set_gauge("disk_usage_gb", disk.used / 1024 / 1024 / 1024)

        # Store time series data
        for name, counter in self.counters.items():
            self.time_series[name].append((timestamp, counter.get()))

        for name, gauge in self.gauges.items():
            self.time_series[name].append((timestamp, gauge.get()))

        for name, histogram in self.histograms.items():
            summary = histogram.get_summary()
            self.time_series[f"{name}_mean"].append((timestamp, summary.get("mean", 0)))
            self.time_series[f"{name}_p95"].append((timestamp, summary.get("p95", 0)))

    def _check_alerts(self):
        """Check alert rules and trigger if needed."""
        for rule in self.alert_rules:
            metric_name = rule["metric"]
            threshold = rule["threshold"]
            comparator = rule["comparator"]

            # Get current value
            current_value = None
            if metric_name in self.counters:
                current_value = self.counters[metric_name].get()
            elif metric_name in self.gauges:
                current_value = self.gauges[metric_name].get()

            if current_value is None:
                continue

            # Check threshold
            triggered = False
            if comparator == ">" and current_value > threshold:
                triggered = True
            elif comparator == "<" and current_value < threshold:
                triggered = True
            elif comparator == ">=" and current_value >= threshold:
                triggered = True
            elif comparator == "<=" and current_value <= threshold:
                triggered = True
            elif comparator == "==" and current_value == threshold:
                triggered = True

            if triggered:
                self._trigger_alert(rule, current_value)

    def _trigger_alert(self, rule: Dict, value: float):
        """Trigger an alert."""
        message = f"{rule['message']} (current: {value}, threshold: {rule['threshold']})"

        if rule["level"] == AlertLevel.CRITICAL:
            logger.critical(message)
        elif rule["level"] == AlertLevel.ERROR:
            logger.error(message)
        elif rule["level"] == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        # Execute callback if provided
        if rule["callback"]:
            try:
                rule["callback"](rule, value)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _store_snapshot(self):
        """Store metrics snapshot to database."""
        try:
            with get_db_session() as session:
                # Calculate aggregated metrics
                patterns_detected = self.counters.get("patterns_detected", Counter("", "")).get()
                patterns_labeled = self.counters.get("patterns_labeled", Counter("", "")).get()
                patterns_predicted = self.counters.get("predictions_made", Counter("", "")).get()

                snapshot = MetricSnapshot(
                    period="MINUTE",  # Store minute-level snapshots
                    patterns_detected=int(patterns_detected),
                    patterns_labeled=int(patterns_labeled),
                    patterns_predicted=int(patterns_predicted),
                    cpu_usage_percent=self.gauges.get("cpu_usage_percent", Gauge("", "")).get(),
                    memory_usage_mb=self.gauges.get("memory_usage_mb", Gauge("", "")).get(),
                    disk_usage_gb=self.gauges.get("disk_usage_gb", Gauge("", "")).get(),
                    error_rate=self.gauges.get("error_rate_per_minute", Gauge("", "")).get()
                )

                session.add(snapshot)
                session.commit()

        except Exception as e:
            logger.error(f"Failed to store metric snapshot: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        metrics = {}

        # Collect counters
        for name, counter in self.counters.items():
            metrics[name] = counter.get()

        # Collect gauges
        for name, gauge in self.gauges.items():
            metrics[name] = gauge.get()

        # Collect histogram summaries
        for name, histogram in self.histograms.items():
            metrics[f"{name}_summary"] = histogram.get_summary()

        return metrics

    def record_response_time(self, endpoint: str, duration_ms: float):
        """
        Record API response time.

        Args:
            endpoint: API endpoint path
            duration_ms: Response time in milliseconds
        """
        # Convert to seconds for histogram
        duration_seconds = duration_ms / 1000.0
        self.observe_histogram("request_duration_seconds", duration_seconds)

        # Track endpoint-specific metrics
        endpoint_key = endpoint.replace("/", "_").strip("_")
        if endpoint_key:
            self.observe_histogram(f"response_time_{endpoint_key}", duration_ms)

    def get_average_response_time(self, window_minutes: int = 60) -> float:
        """
        Get average response time over the specified window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Average response time in milliseconds
        """
        # Get recent response times from histogram
        if "request_duration_seconds" in self.histograms:
            histogram = self.histograms["request_duration_seconds"]
            if histogram.count > 0:
                # Return mean in milliseconds
                return (histogram.sum_value / histogram.count) * 1000.0
        return 0.0

    def start(self):
        """Alias for start_collection for compatibility."""
        self.start_collection()

    def stop(self):
        """Alias for stop_collection for compatibility."""
        self.stop_collection()

    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format (json, prometheus)

        Returns:
            Formatted metrics string
        """
        metrics = self.get_current_metrics()

        if format == "json":
            return json.dumps(metrics, indent=2, default=str)
        elif format == "prometheus":
            lines = []
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{name} {value}")
                elif isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            lines.append(f"{name}_{sub_name} {sub_value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown export format: {format}")


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()

    def track_request(self, endpoint: str, duration: float, success: bool):
        """
        Track API request metrics.

        Args:
            endpoint: API endpoint
            duration: Request duration in seconds
            success: Whether request succeeded
        """
        self.metrics.increment_counter("requests_total")
        if not success:
            self.metrics.increment_counter("requests_failed")

        self.metrics.observe_histogram("request_duration_seconds", duration)

        # Update endpoint-specific metrics
        self.metrics.increment_counter(f"requests_{endpoint}_total")
        self.metrics.observe_histogram(f"request_{endpoint}_duration", duration)

    def track_pattern_detection(self, ticker: str, duration: float, patterns_found: int):
        """
        Track pattern detection metrics.

        Args:
            ticker: Ticker symbol
            duration: Detection duration in seconds
            patterns_found: Number of patterns found
        """
        self.metrics.increment_counter("patterns_detected", patterns_found)
        self.metrics.observe_histogram("pattern_detection_seconds", duration)

        logger.debug(f"Pattern detection for {ticker}: {patterns_found} patterns in {duration:.2f}s")

    def track_prediction(self, pattern_id: str, duration: float, ev: float, signal: str):
        """
        Track prediction metrics.

        Args:
            pattern_id: Pattern ID
            duration: Inference duration in seconds
            ev: Expected value
            signal: Signal strength
        """
        self.metrics.increment_counter("predictions_made")
        self.metrics.observe_histogram("model_inference_seconds", duration)

        # Track signal distribution
        self.metrics.increment_counter(f"signal_{signal.lower()}_count")

    def track_error(self, error_type: str, component: str):
        """
        Track error occurrence.

        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.metrics.increment_counter("errors_total")
        self.metrics.increment_counter(f"errors_{component}_{error_type}")

        # Update error rate
        # This is simplified - in production, calculate actual rate
        current_rate = self.metrics.gauges.get("error_rate_per_minute", Gauge("", "")).get()
        self.metrics.set_gauge("error_rate_per_minute", current_rate + 1)


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_performance_monitor: Optional[PerformanceMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if not _metrics_collector:
        _metrics_collector = MetricsCollector()
        _metrics_collector.start_collection()
    return _metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    if not _performance_monitor:
        _performance_monitor = PerformanceMonitor(get_metrics_collector())
    return _performance_monitor


if __name__ == "__main__":
    # Test monitoring system
    import random

    # Initialize monitoring
    collector = get_metrics_collector()
    monitor = get_performance_monitor()

    # Add alert rules
    collector.add_alert_rule(
        "cpu_usage_percent",
        80,
        ">",
        AlertLevel.WARNING,
        "High CPU usage detected"
    )

    collector.add_alert_rule(
        "error_rate_per_minute",
        10,
        ">",
        AlertLevel.ERROR,
        "High error rate detected"
    )

    # Simulate activity
    print("Simulating system activity...")

    for i in range(10):
        # Track requests
        duration = random.uniform(0.1, 2.0)
        success = random.random() > 0.1
        monitor.track_request("scan/pattern", duration, success)

        # Track pattern detection
        patterns = random.randint(0, 5)
        monitor.track_pattern_detection("TEST", duration, patterns)

        # Track predictions
        if patterns > 0:
            ev = random.uniform(-2, 10)
            signal = "STRONG" if ev > 5 else "GOOD" if ev > 3 else "MODERATE"
            monitor.track_prediction(f"TEST_{i}", 0.05, ev, signal)

        # Occasionally track errors
        if random.random() < 0.2:
            monitor.track_error("ConnectionError", "data_loader")

        time.sleep(1)

    # Export metrics
    print("\nCurrent Metrics:")
    print(collector.export_metrics("json"))

    # Stop collection
    collector.stop_collection()