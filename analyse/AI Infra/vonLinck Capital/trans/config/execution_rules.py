"""
Execution Rules Configuration (V19/V20)

Provides centralized configuration for execution rules, supporting
easy rollback between V19 (static pacing) and V20 (dynamic pacing).

Usage:
    from config.execution_rules import get_pacing_config, should_use_v19_fallback

    config = get_pacing_config()
    if config['pacing_version'] == 'v19':
        # Use static 30% rule
    else:
        # Use dynamic 1.5x time-weighted rule
"""

from typing import Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# VERSION DEFINITIONS
# =============================================================================

EXECUTION_RULES_V19 = {
    'version': 'v19',
    'description': 'Static intraday pacing (30% at 10:00 AM)',
    'pacing_type': 'static',
    'pacing_threshold_pct': 0.30,      # 30% of daily avg
    'check_time_minutes': 30,           # 10:00 AM (30 min since open)
    'adv_liquidity_pct': 0.04,          # 4% of ADV
    'gap_limit_r': 0.5,                 # Max gap for tradeable
}

EXECUTION_RULES_V20 = {
    'version': 'v20',
    'description': 'Dynamic time-weighted pacing (1.5x expected)',
    'pacing_type': 'dynamic',
    'pacing_multiplier': 1.5,           # Vol_Current > 1.5 * Vol_Expected
    'check_time_minutes': 30,           # 10:00 AM (30 min since open)
    'adv_liquidity_pct': 0.04,          # 4% of ADV
    'gap_limit_r': 0.5,                 # Max gap for tradeable
    # U-shaped intraday profile
    'intraday_profile': {
        0: 0.00, 15: 0.08, 30: 0.15, 45: 0.20, 60: 0.25,
        90: 0.32, 120: 0.38, 150: 0.43, 180: 0.48, 210: 0.52,
        240: 0.57, 270: 0.62, 300: 0.68, 330: 0.76, 360: 0.85, 390: 1.00
    }
}

# Default version
DEFAULT_VERSION = 'v20'

# Rollback threshold (if shadow trading slippage exceeds this, auto-rollback)
ROLLBACK_SLIPPAGE_THRESHOLD_R = 0.3


# =============================================================================
# STATE FILE (persists version selection)
# =============================================================================
STATE_FILE = Path(__file__).parent.parent / 'output' / '.execution_rules_state.json'


def get_saved_state() -> Dict:
    """Load saved state from file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load execution rules state: {e}")
    return {}


def save_state(state: Dict):
    """Save state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save execution rules state: {e}")


# =============================================================================
# PUBLIC API
# =============================================================================

def get_pacing_config(version: Optional[str] = None) -> Dict:
    """
    Get pacing configuration for specified version.

    Args:
        version: 'v19', 'v20', or None (uses saved/default)

    Returns:
        Dict with pacing configuration
    """
    if version is None:
        # Check saved state
        state = get_saved_state()
        version = state.get('active_version', DEFAULT_VERSION)

    if version == 'v19':
        return EXECUTION_RULES_V19.copy()
    else:
        return EXECUTION_RULES_V20.copy()


def set_pacing_version(version: str, reason: str = 'manual'):
    """
    Set the active pacing version.

    Args:
        version: 'v19' or 'v20'
        reason: Reason for change (e.g., 'manual', 'rollback', 'shadow_trading')
    """
    if version not in ('v19', 'v20'):
        raise ValueError(f"Invalid version: {version}. Must be 'v19' or 'v20'")

    state = get_saved_state()
    old_version = state.get('active_version', DEFAULT_VERSION)

    state['active_version'] = version
    state['last_change_reason'] = reason
    state['last_change_timestamp'] = str(Path('').resolve())  # Current time would be better

    # Keep history
    if 'history' not in state:
        state['history'] = []
    state['history'].append({
        'from': old_version,
        'to': version,
        'reason': reason
    })

    save_state(state)
    logger.info(f"Execution rules version changed: {old_version} -> {version} (reason: {reason})")


def trigger_rollback(slippage_r: float, reason: str = 'shadow_trading'):
    """
    Trigger automatic rollback to V19 if slippage exceeds threshold.

    Args:
        slippage_r: Observed slippage in R units
        reason: Reason for rollback check

    Returns:
        True if rollback was triggered, False otherwise
    """
    if slippage_r > ROLLBACK_SLIPPAGE_THRESHOLD_R:
        logger.warning(f"ROLLBACK TRIGGERED: slippage {slippage_r:.3f}R > threshold {ROLLBACK_SLIPPAGE_THRESHOLD_R}R")
        set_pacing_version('v19', reason=f'auto_rollback_{reason}')
        return True
    return False


def get_current_version() -> str:
    """Get the currently active pacing version."""
    state = get_saved_state()
    return state.get('active_version', DEFAULT_VERSION)


def should_use_v19_fallback() -> bool:
    """Check if V19 fallback rules should be used."""
    return get_current_version() == 'v19'


def get_version_comparison() -> Dict:
    """
    Get comparison between V19 and V20 rules for documentation.

    Returns:
        Dict with side-by-side comparison
    """
    return {
        'v19': {
            'name': 'Static Pacing',
            'rule': 'Vol @ 10AM > 30% of avg daily',
            'pros': ['Simple', 'Proven', 'Conservative'],
            'cons': ['Ignores time-of-day profile', 'May miss valid trades']
        },
        'v20': {
            'name': 'Dynamic Pacing',
            'rule': 'Vol_Current > 1.5 * Vol_Expected_At_Time',
            'pros': ['Time-aware', 'More accurate', 'Captures U-shape'],
            'cons': ['More complex', 'Needs validation via shadow trading']
        },
        'current': get_current_version(),
        'rollback_threshold': ROLLBACK_SLIPPAGE_THRESHOLD_R
    }


# =============================================================================
# CLI for manual version control
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Execution Rules Version Control')
    parser.add_argument('--get', action='store_true', help='Get current version')
    parser.add_argument('--set', type=str, choices=['v19', 'v20'], help='Set version')
    parser.add_argument('--compare', action='store_true', help='Show version comparison')
    parser.add_argument('--rollback', action='store_true', help='Trigger manual rollback to V19')

    args = parser.parse_args()

    if args.get:
        print(f"Current execution rules version: {get_current_version()}")
        config = get_pacing_config()
        print(f"  Type: {config['pacing_type']}")
        print(f"  Description: {config['description']}")

    elif args.set:
        set_pacing_version(args.set, reason='manual_cli')
        print(f"Version set to: {args.set}")

    elif args.compare:
        comp = get_version_comparison()
        print("\nExecution Rules Comparison:")
        print("-" * 50)
        for v in ['v19', 'v20']:
            info = comp[v]
            marker = " (ACTIVE)" if comp['current'] == v else ""
            print(f"\n{v.upper()}: {info['name']}{marker}")
            print(f"  Rule: {info['rule']}")
            print(f"  Pros: {', '.join(info['pros'])}")
            print(f"  Cons: {', '.join(info['cons'])}")
        print(f"\nRollback threshold: {comp['rollback_threshold']}R")

    elif args.rollback:
        if get_current_version() == 'v19':
            print("Already on V19 - no rollback needed")
        else:
            set_pacing_version('v19', reason='manual_rollback')
            print("Rolled back to V19 execution rules")

    else:
        parser.print_help()
