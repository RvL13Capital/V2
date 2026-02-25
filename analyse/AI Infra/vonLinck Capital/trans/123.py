#!/usr/bin/env python3
"""
Step 3: User-Centric Execution & Behavioral Guardrails
======================================================
SOLVES: P1 (Behavioral Fit), P3 (Actionability), P4 (Feedback)

This module acts as the "Application Layer" for the TRAnS pipeline.
It transforms raw ML signals into a "Justice Exit Plan" by applying
psychological and structural guardrails.

KEY FEATURES:
    1. User Profiling: Calibrates risk thresholds ($25k vs $100k) to prevent "Lobster Traps".
    2. Behavioral Guardrails: Detects & flags FOMO (chasing), Anchoring, and Liquidity Risk.
    3. Justice Exit Plan: Auto-calculates position size and R-multiples (3R/5R).
    4. Active Nudges: Downgrades confidence or blocks trades based on behavioral risk.

USAGE:
    # Generate a plan for a specific pattern ID (from dashboard/metadata)
    python 03_user_execution.py --ticker XYZ --pattern-id <ID> --confidence 0.85
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean format for dashboard-like output
)
logger = logging.getLogger('execution')

# =============================================================================
# P1: USER PROFILE & BEHAVIORAL DIAGNOSTICS
# =============================================================================

@dataclass
class UserProfile:
    """
    Captures user constraints to tailor signal delivery.
    Addresses P1(b) - Competency Gaps (Microstructure) & Risk Capacity.
    """
    account_size: float = 25000.0       # Default $25k (Retail Constraint)
    risk_per_trade_pct: float = 0.01    # 1% standard risk ($250)
    
    # Behavioral Biases (0.0 - 1.0) - Calibrated via Onboarding
    fomo_susceptibility: float = 0.7    # High tendency to chase extensions
    loss_aversion: float = 0.6          # Reluctance to take stops
    
    @property
    def risk_unit_dollars(self) -> float:
        return self.account_size * self.risk_per_trade_pct

    @property
    def min_dollar_volume(self) -> float:
        """
        P2: LOBSTER TRAP PREVENTION logic.
        Ensures position size is < 1% of Daily Dollar Volume to guarantee exitability.
        """
        # Estimated Position = Risk_Unit * 10 (assuming ~10% stop width avg)
        est_position_size = self.risk_unit_dollars * 10
        
        # We need the position to be < 1% of daily volume to avoid slippage
        required_liquidity = est_position_size * 100
        
        # Hard floor for safety ($50k min) - Filters "Micro-Cap Traps"
        return max(50000.0, required_liquidity)

# =============================================================================
# P3: BEHAVIORAL ENGINE (Active Nudges)
# =============================================================================

class BehavioralGuardrails:
    """
    Applies active nudges based on 'Physics' features.
    Translates data (e.g., 'log_dollar_volume') into Psychology (e.g., 'Lobster Trap').
    """
    
    def __init__(self, user: UserProfile):
        self.user = user
        
    def diagnose_trade(self, pattern_data: Dict, model_prob: float) -> Dict:
        """
        Run diagnostic checks on a candidate trade.
        """
        alerts = []
        is_blocked = False
        adjusted_prob = model_prob

        # --- 1. LOBSTER TRAP GUARD (P2: Market Asymmetry) ---
        # "Easy to get in, impossible to get out."
        log_dvol = pattern_data.get('log_dollar_volume', 0)
        dollar_vol = 10**log_dvol if log_dvol > 0 else 0
        
        if dollar_vol < self.user.min_dollar_volume:
            # BLOCK trade: Safety violation
            is_blocked = True
            alerts.append({
                'type': '⛔ DANGER',
                'msg': f"LOBSTER TRAP: Daily liquidity ${dollar_vol:,.0f} is too low. "
                       f"You risk getting stuck. Min required: ${self.user.min_dollar_volume:,.0f}."
            })
        elif dollar_vol < self.user.min_dollar_volume * 1.5:
            # WARN trade: Execution risk
            alerts.append({
                'type': '⚠️ WARNING',
                'msg': f"LOW LIQUIDITY: Slippage likely. Use LIMIT orders only."
            })

        # --- 2. FOMO BRAKE (P1: Behavioral Fit) ---
        # Buying parabolic extensions or climax volume
        vol_shock = pattern_data.get('volume_shock', 0)
        price_pos = pattern_data.get('price_position_at_end', 0.5)
        trend_pos = pattern_data.get('trend_position', 1.0)
        
        # Logic: High volume at range highs + extended price = Climax (Retail Bait)
        if vol_shock > 2.0 and price_pos > 0.8:
            if self.user.fomo_susceptibility > 0.5:
                alerts.append({
                    'type': '🧠 PSYCH',
                    'msg': "FOMO CHECK: Volume is climaxing at highs. Professional money sells here."
                })
                adjusted_prob *= 0.85 # Downgrade confidence
        
        if trend_pos > 1.3:
             alerts.append({
                'type': '🧠 PSYCH',
                'msg': f"CHASING: Stock is {trend_pos:.2f}x extended from SMA200. Wait for pullback."
            })

        # --- 3. STRUCTURAL DISCIPLINE (P3: Actionability) ---
        # Rejecting loose structures that wreck R:R math
        risk_width = pattern_data.get('risk_width_pct', 0.1)
        
        if risk_width > 0.35:
            is_blocked = True
            alerts.append({
                'type': '⛔ BLOCK',
                'msg': f"LOOSE STRUCTURE: {risk_width*100:.1f}% risk width. Stop distance destroys expectancy."
            })

        # --- 4. ZOMBIE CHECK (Dormancy Detection) ---
        dormancy = pattern_data.get('dormancy_shock', 0)
        if dormancy < -2.0 and model_prob < 0.7:
             alerts.append({
                'type': 'ℹ️ INFO',
                'msg': "DEEP SLEEPER: Stock is extremely dormant. Patience required (Dead Money risk)."
            })

        return {
            'is_blocked': is_blocked,
            'original_prob': model_prob,
            'adjusted_prob': adjusted_prob,
            'alerts': alerts,
            'risk_width': risk_width,
            'entry_price': pattern_data.get('upper_boundary'),
            'stop_price': pattern_data.get('lower_boundary')
        }

def print_justice_plan(diagnosis: Dict, user: UserProfile, ticker: str):
    """
    Renders the 'Justice Exit Plan' card - the scaffold for decision making.
    """
    print("\n" + "="*60)
    print(f"🛡️  JUSTICE EXECUTION PLAN: {ticker}  |  Risk Unit: ${user.risk_unit_dollars:.0f}")
    print("="*60)
    
    # 1. ALERTS SECTION
    if diagnosis['alerts']:
        print("\n[GUARDRAIL ALERTS]")
        for alert in diagnosis['alerts']:
            print(f"  {alert['type']}: {alert['msg']}")
            
    if diagnosis['is_blocked']:
        print("\n❌ TRADE STATUS: BLOCKED (Safety Constraints Violated)")
        print("   Action: Do not execute. Preserve capital for cleaner setups.")
        return

    # 2. SIZING CALCULATION
    entry = diagnosis['entry_price']
    stop = diagnosis['stop_price']
    
    if entry and stop and entry > stop:
        risk_per_share = entry - stop
        shares = int(user.risk_unit_dollars / risk_per_share)
        position_size = shares * entry
        
        print("\n[POSITION SIZING]")
        print(f"  Entry Trigger:   ${entry:.2f}")
        print(f"  Hard Stop:       ${stop:.2f} (-{diagnosis['risk_width']*100:.1f}%)")
        print(f"  Max Shares:      {shares}")
        print(f"  Total Exposure:  ${position_size:,.2f}")
        
        # 3. TARGETS (Scaffolding for "Hold Winners")
        r3 = entry + (3 * risk_per_share)
        r5 = entry + (5 * risk_per_share)
        print("\n[EXECUTION TARGETS]")
        print(f"  🎯 Target 1 (3R): ${r3:.2f} (Sell 1/2 - Secure Win)")
        print(f"  🚀 Target 2 (5R): ${r5:.2f} (Sell Remainder - Runner)")
        
        # 4. CONFIDENCE
        print(f"\n[AI CONFIDENCE]")
        print(f"  Raw Signal:      {diagnosis['original_prob']*100:.1f}%")
        if diagnosis['adjusted_prob'] < diagnosis['original_prob']:
            print(f"  Adjusted:        {diagnosis['adjusted_prob']*100:.1f}% (Downgraded due to Behavioral Risk)")
            
    else:
        print("\n⚠️  Error: Invalid boundary data for sizing.")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test Harness for "Retail Trap" Scenario
    mock_pattern = {
        'log_dollar_volume': 4.8,       # ~$63k (Marginal for $25k acc)
        'volume_shock': 2.5,            # Huge spike (Retail bait)
        'price_position_at_end': 0.9,   # At highs (FOMO)
        'risk_width_pct': 0.12,         # 12% stop (Okay)
        'trend_position': 1.4,          # Extended
        'upper_boundary': 5.50,
        'lower_boundary': 4.84
    }
    
    user = UserProfile(account_size=25000, fomo_susceptibility=0.8)
    guard = BehavioralGuardrails(user)
    diag = guard.diagnose_trade(mock_pattern, model_prob=0.88)
    print_justice_plan(diag, user, "TEST_TICKER")
