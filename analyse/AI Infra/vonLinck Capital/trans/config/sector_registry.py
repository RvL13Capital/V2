"""
Sector Registry for Position Risk Management
=============================================

Tracks ticker → sector mapping for portfolio exposure limits.
Prevents overconcentration in any single sector.

Usage:
    from config.sector_registry import get_sector, MAX_SECTOR_POSITIONS

    sector = get_sector('AAPL')  # 'Technology'
    if sector_counts[sector] >= MAX_SECTOR_POSITIONS:
        reject_order(reason=f'Sector cap exceeded: {sector}')

Jan 2026 - Created for sector exposure management
"""

from typing import Final, Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# SECTOR CLASSIFICATION
# =============================================================================

# Maximum positions per sector (prevents concentration risk)
MAX_SECTOR_POSITIONS: Final[int] = 3

# Default sector for unknown tickers
DEFAULT_SECTOR: Final[str] = 'Unknown'

# =============================================================================
# SECTOR CATEGORIES
# =============================================================================
# Standard GICS-aligned sectors for micro/small-cap trading
# Focus on sectors common in breakout patterns

SECTORS: Final[list] = [
    'Technology',      # Software, semiconductors, IT services
    'BioTech',         # Biotechnology, pharmaceuticals
    'Healthcare',      # Medical devices, healthcare services
    'Energy',          # Oil, gas, renewables
    'Mining',          # Gold, silver, base metals, rare earth
    'Cannabis',        # Marijuana, CBD
    'EV/CleanTech',    # Electric vehicles, batteries, solar
    'Financials',      # Banks, fintech, insurance
    'Consumer',        # Retail, restaurants, entertainment
    'Industrial',      # Manufacturing, aerospace, defense
    'Materials',       # Chemicals, construction materials
    'SPAC',            # Special purpose acquisition companies
    'Unknown',         # Default for unmapped tickers
]

# =============================================================================
# TICKER → SECTOR MAPPING
# =============================================================================
# Start with common patterns, extend via external data or API
# Format: 'TICKER': 'Sector'

# Common tech tickers
TECH_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'AVGO',
    'CSCO', 'ORCL', 'IBM', 'CRM', 'ADBE', 'PYPL', 'SQ', 'SHOP', 'SNOW',
    'PLTR', 'NET', 'DDOG', 'ZS', 'CRWD', 'OKTA', 'MDB', 'TWLO', 'DOCN',
]

# Common biotech tickers
BIOTECH_TICKERS = [
    'MRNA', 'BNTX', 'NVAX', 'PFE', 'JNJ', 'MRK', 'LLY', 'ABBV', 'BMY',
    'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'ISRG', 'EXAS', 'SGEN', 'ALNY',
]

# Common EV/CleanTech tickers
EV_TICKERS = [
    'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'GOEV', 'WKHS',
    'NKLA', 'HYLN', 'RIDE', 'QS', 'CHPT', 'BLNK', 'EVGO', 'PLUG', 'FCEL',
    'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR',
]

# Common mining tickers
MINING_TICKERS = [
    'NEM', 'GOLD', 'AEM', 'KGC', 'AU', 'GFI', 'HMY', 'AGI', 'EGO',
    'SLV', 'PAAS', 'AG', 'HL', 'CDE', 'FSM', 'SILV', 'MAG',
    'FCX', 'SCCO', 'VALE', 'RIO', 'BHP', 'TECK',
    'MP', 'LAC', 'LTHM', 'ALB', 'SQM',  # Rare earth / lithium
]

# Common cannabis tickers
CANNABIS_TICKERS = [
    'TLRY', 'CGC', 'ACB', 'CRON', 'OGI', 'HEXO', 'VFF', 'SNDL', 'GRWG',
    'CURLF', 'GTBIF', 'TCNNF', 'CRLBF', 'TRSSF',
]

# Build the mapping
_SECTOR_MAP: Dict[str, str] = {}

for ticker in TECH_TICKERS:
    _SECTOR_MAP[ticker] = 'Technology'

for ticker in BIOTECH_TICKERS:
    _SECTOR_MAP[ticker] = 'BioTech'

for ticker in EV_TICKERS:
    _SECTOR_MAP[ticker] = 'EV/CleanTech'

for ticker in MINING_TICKERS:
    _SECTOR_MAP[ticker] = 'Mining'

for ticker in CANNABIS_TICKERS:
    _SECTOR_MAP[ticker] = 'Cannabis'


# =============================================================================
# PUBLIC API
# =============================================================================

def get_sector(ticker: str) -> str:
    """
    Get sector for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Sector name (e.g., 'Technology', 'BioTech', 'Unknown')

    Example:
        >>> get_sector('TSLA')
        'EV/CleanTech'
        >>> get_sector('XYZ123')
        'Unknown'
    """
    return _SECTOR_MAP.get(ticker.upper(), DEFAULT_SECTOR)


def register_ticker(ticker: str, sector: str) -> None:
    """
    Register a ticker's sector (runtime only, not persisted).

    Args:
        ticker: Stock ticker symbol
        sector: Sector name (should be in SECTORS list)

    Example:
        >>> register_ticker('NEWCO', 'Technology')
    """
    if sector not in SECTORS:
        logger.warning(f"Sector '{sector}' not in standard list: {SECTORS}")
    _SECTOR_MAP[ticker.upper()] = sector


def get_all_mappings() -> Dict[str, str]:
    """Get copy of all ticker→sector mappings."""
    return _SECTOR_MAP.copy()


def load_mappings_from_file(filepath: str) -> int:
    """
    Load additional mappings from JSON file.

    File format: {"TICKER": "Sector", ...}

    Args:
        filepath: Path to JSON mapping file

    Returns:
        Number of mappings loaded

    Example:
        >>> load_mappings_from_file('data/sector_mappings.json')
        150
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Sector mapping file not found: {filepath}")
        return 0

    try:
        with open(path, 'r') as f:
            mappings = json.load(f)

        count = 0
        for ticker, sector in mappings.items():
            _SECTOR_MAP[ticker.upper()] = sector
            count += 1

        logger.info(f"Loaded {count} sector mappings from {filepath}")
        return count

    except Exception as e:
        logger.error(f"Failed to load sector mappings: {e}")
        return 0


def save_mappings_to_file(filepath: str) -> None:
    """
    Save current mappings to JSON file.

    Args:
        filepath: Output path for JSON file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(_SECTOR_MAP, f, indent=2, sort_keys=True)

    logger.info(f"Saved {len(_SECTOR_MAP)} sector mappings to {filepath}")


def infer_sector_from_suffix(ticker: str) -> Optional[str]:
    """
    Attempt to infer sector from ticker patterns.

    This is a heuristic for unknown tickers based on common naming patterns.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Inferred sector or None

    Example:
        >>> infer_sector_from_suffix('ABCBIO')
        'BioTech'
    """
    upper = ticker.upper()

    # Biotech patterns
    if any(pattern in upper for pattern in ['BIO', 'GEN', 'PHARM', 'MED', 'THERA']):
        return 'BioTech'

    # Mining patterns
    if any(pattern in upper for pattern in ['GOLD', 'SILVER', 'MINE', 'METAL']):
        return 'Mining'

    # Cannabis patterns
    if any(pattern in upper for pattern in ['CANN', 'WEED', 'POT', 'CBD']):
        return 'Cannabis'

    # EV patterns
    if any(pattern in upper for pattern in ['EV', 'ELEC', 'BATT', 'SOLAR']):
        return 'EV/CleanTech'

    return None


# =============================================================================
# SECTOR EXPOSURE CHECKER
# =============================================================================

class SectorExposureChecker:
    """
    Track and limit sector exposure during order generation.

    Usage:
        checker = SectorExposureChecker(max_per_sector=3)

        for order in orders:
            sector = get_sector(order['ticker'])
            if checker.can_add(sector):
                checker.add(sector)
                approved_orders.append(order)
            else:
                order['rejected'] = True
                order['rejection_reason'] = f'Sector cap exceeded: {sector}'
    """

    def __init__(self, max_per_sector: int = MAX_SECTOR_POSITIONS):
        """
        Args:
            max_per_sector: Maximum positions allowed per sector
        """
        self.max_per_sector = max_per_sector
        self.sector_counts: Dict[str, int] = {}

    def can_add(self, sector: str) -> bool:
        """Check if another position can be added to this sector."""
        current = self.sector_counts.get(sector, 0)
        return current < self.max_per_sector

    def add(self, sector: str) -> None:
        """Record a position in this sector."""
        self.sector_counts[sector] = self.sector_counts.get(sector, 0) + 1

    def get_count(self, sector: str) -> int:
        """Get current count for a sector."""
        return self.sector_counts.get(sector, 0)

    def get_summary(self) -> Dict[str, int]:
        """Get summary of all sector exposures."""
        return dict(sorted(self.sector_counts.items(), key=lambda x: -x[1]))

    def reset(self) -> None:
        """Reset all counts."""
        self.sector_counts = {}


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Sector Registry")
    print("=" * 40)
    print(f"Sectors: {SECTORS}")
    print(f"Max positions per sector: {MAX_SECTOR_POSITIONS}")
    print(f"Total mappings: {len(_SECTOR_MAP)}")
    print()

    # Test some lookups
    test_tickers = ['AAPL', 'TSLA', 'MRNA', 'NEM', 'TLRY', 'XYZ123']
    print("Sample lookups:")
    for ticker in test_tickers:
        print(f"  {ticker}: {get_sector(ticker)}")

    print()

    # Test exposure checker
    checker = SectorExposureChecker(max_per_sector=2)
    test_orders = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'RIVN', 'LCID']
    print("Exposure check (max=2):")
    for ticker in test_orders:
        sector = get_sector(ticker)
        can_add = checker.can_add(sector)
        print(f"  {ticker} ({sector}): {'OK' if can_add else 'BLOCKED'}")
        if can_add:
            checker.add(sector)

    print()
    print(f"Final exposure: {checker.get_summary()}")
