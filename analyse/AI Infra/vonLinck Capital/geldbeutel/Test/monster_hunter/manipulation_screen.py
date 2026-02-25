"""
manipulation_screen.py

Stage 1: The Gatekeeper.
Scans SEC filing text to detect toxic lenders and death spiral financing.
MUST run before any technical analysis - rejects toxic structures immediately.

Philosophy: No amount of good technicals can save you from floorless convertibles.
"""

import re
from typing import Tuple, List, Dict


class ManipulationScreen:
    """
    Screens for toxic financing structures in micro-cap stocks.

    Known Death Spiral Patterns:
    1. Toxic Lenders: Funds that specialize in predatory convertibles
    2. Floorless Convertibles: Variable conversion with no price floor
    3. Shell Companies: No real operations, just a ticker for manipulation
    """

    def __init__(self):
        # THE BLACKLIST: Known Death Spiral Financiers
        # These funds specialize in toxic convertible notes that dilute shareholders
        self.TOXIC_LENDERS = [
            r"streeterville capital",
            r"jmj financial",
            r"st\.?\s*george investments",
            r"auctus fund",
            r"adar bays",
            r"commonwealth investment",
            r"crown bridge",
            r"eagle equities",
            r"ema financial",
            r"firstfire",
            r"geneva roth",
            r"grey\s*wood",
            r"gs capital",
            r"gsummit",
            r"ironridge",
            r"jefferson street",
            r"kingsbrook",
            r"labrys",
            r"lincoln park capital",
            r"magna group",
            r"mammoth corporation",
            r"maxim group",
            r"mercer promotion",
            r"morningview",
            r"power up lending",
            r"redstart",
            r"sbi investments",
            r"southridge",
            r"tangiers capital",
            r"tonaquint",
            r"triton funds",
            r"typenex",
            r"vis vires",
            r"vystar",
            r"yorkville advisors"
        ]

        # TOXIC PATTERNS: The Mechanisms of Death Spiral Financing
        self.TOXIC_PATTERNS = {
            # Floorless conversion: Price tied to VWAP/lowest price with discount
            "floorless_conversion": re.compile(
                r"(conversion\s+price|convertible\s+at).{0,50}"
                r"(discount|percentage).{0,50}"
                r"(lowest|average|closing|trading\s+price|vwap)",
                re.IGNORECASE | re.DOTALL
            ),
            # Variable rate: Conversion rate changes based on market price
            "variable_rate": re.compile(
                r"(variable|floating).{0,30}(conversion\s+rate|conversion\s+price)",
                re.IGNORECASE
            ),
            # Explicit no-floor language
            "no_floor": re.compile(
                r"(no\s+floor|without\s+a\s+floor).{0,100}(conversion|price)",
                re.IGNORECASE
            ),
            # Death spiral explicit mention
            "death_spiral": re.compile(
                r"death\s+spiral|toxic\s+convert|dilutive\s+financing",
                re.IGNORECASE
            )
        }

        # SHELL INDICATORS: Signs of a company with no real business
        self.SHELL_KEYWORDS = [
            r"nominal operations",
            r"shell company",
            r"blank check",
            r"seeking a business combination",
            r"no revenue",
            r"development stage",
            r"no operating history"
        ]

    def scan_filing(self, filing_text: str, filing_type: str = "8-K") -> Tuple[bool, List[str]]:
        """
        Scans SEC filing text for toxic financing indicators.

        Args:
            filing_text: Raw text from SEC filing (8-K, 10-Q, 10-K, S-1, etc.)
            filing_type: Type of filing for context-aware scanning

        Returns:
            Tuple of (passed: bool, red_flags: List[str])
            - passed=True means no CRITICAL flags found, stock can proceed
            - passed=False means CRITICAL toxic financing detected, REJECT
            - red_flags contains all warnings/critical issues found
        """
        red_flags = []
        text_lower = filing_text.lower()

        # 1. CHECK TOXIC LENDERS (CRITICAL)
        for lender_pattern in self.TOXIC_LENDERS:
            if re.search(lender_pattern, text_lower):
                # Extract the matched lender name for the flag
                match = re.search(lender_pattern, text_lower)
                lender_name = match.group(0) if match else lender_pattern
                red_flags.append(f"CRITICAL: Toxic Lender Detected ({lender_name})")

        # 2. CHECK TOXIC TERMS (CRITICAL - only in relevant filing types)
        relevant_filings = ["8-K", "10-Q", "10-K", "S-1", "S-3", "424B"]
        if any(ft in filing_type.upper() for ft in relevant_filings):
            for pattern_name, pattern in self.TOXIC_PATTERNS.items():
                if pattern.search(text_lower):
                    red_flags.append(f"CRITICAL: Toxic Financing Pattern ({pattern_name})")

        # 3. CHECK SHELL STATUS (WARNING - not critical but concerning)
        for keyword in self.SHELL_KEYWORDS:
            if re.search(keyword, text_lower):
                red_flags.append(f"WARNING: Shell Company Indicator ({keyword})")
                break  # One shell warning is enough

        # 4. DETERMINE PASS/FAIL
        # Critical flags = automatic fail
        critical_flags = [f for f in red_flags if "CRITICAL" in f]
        passed = len(critical_flags) == 0

        return passed, red_flags

    def quick_check(self, filing_text: str) -> bool:
        """
        Fast pass/fail check without detailed flag reporting.
        Use this for bulk screening.
        """
        passed, _ = self.scan_filing(filing_text)
        return passed

    def get_toxic_lender_list(self) -> List[str]:
        """Returns the list of toxic lender patterns for reference."""
        return self.TOXIC_LENDERS.copy()

    def add_toxic_lender(self, lender_pattern: str) -> None:
        """Add a new toxic lender pattern to the blacklist."""
        if lender_pattern.lower() not in [l.lower() for l in self.TOXIC_LENDERS]:
            self.TOXIC_LENDERS.append(lender_pattern.lower())


# Convenience function for quick screening
def is_toxic(filing_text: str, filing_type: str = "8-K") -> bool:
    """
    Quick check if a filing contains toxic financing.
    Returns True if toxic (FAIL), False if clean (PASS).
    """
    screen = ManipulationScreen()
    passed, _ = screen.scan_filing(filing_text, filing_type)
    return not passed  # Invert: passed=True means not toxic
