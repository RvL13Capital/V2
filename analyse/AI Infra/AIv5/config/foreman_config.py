"""
Minimal ForemanConfig for backward compatibility with sources_orig modules.

This is a compatibility layer to support legacy code in data_acquisition/sources_orig.
New code should use shared.config instead.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ExchangeDefinition:
    """Exchange configuration."""
    name: str
    code: str
    country: str
    suffix: str = ""
    currency: str = "USD"


@dataclass
class ForemanConfig:
    """
    Minimal Foreman configuration for backward compatibility.

    This provides basic configuration structure that sources_orig modules expect.
    For full configuration, use shared.config.Settings instead.
    """

    # API Keys
    fmp_api_key: Optional[str] = None
    eodhd_api_key: Optional[str] = None
    alpha_vantage_keys: List[str] = field(default_factory=lambda: [])
    twelvedata_api_key: Optional[str] = None

    # Data storage
    output_dir: str = "data/raw"
    use_gcs: bool = False
    gcs_bucket: Optional[str] = None
    project_id: Optional[str] = None

    # Market cap filtering
    min_market_cap_millions: float = 200.0
    max_market_cap_millions: float = 4000.0

    # Exchanges
    exchanges: List[ExchangeDefinition] = field(default_factory=lambda: [
        ExchangeDefinition(name="NASDAQ", code="NASDAQ", country="US", suffix="", currency="USD"),
        ExchangeDefinition(name="NYSE", code="NYSE", country="US", suffix="", currency="USD"),
        ExchangeDefinition(name="TSX", code="TSX", country="CA", suffix=".TO", currency="CAD"),
        ExchangeDefinition(name="TSXV", code="TSXV", country="CA", suffix=".V", currency="CAD"),
    ])

    # Processing settings
    max_workers: int = 5
    batch_size: int = 10
    delay_between_requests: float = 0.5

    # Filtering
    excluded_sectors: List[str] = field(default_factory=list)
    min_price: float = 0.5
    max_price: float = 100.0


def get_foreman_config() -> ForemanConfig:
    """
    Get default Foreman configuration.

    Returns:
        ForemanConfig instance with default values
    """
    return ForemanConfig()
