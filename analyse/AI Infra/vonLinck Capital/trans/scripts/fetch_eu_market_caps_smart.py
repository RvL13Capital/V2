#!/usr/bin/env python3
"""
Smart market cap fetcher - prioritizes major EU stocks and uses adaptive delays.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "market_cap_cache.json"

# Longer delays to avoid rate limiting
DELAY_BETWEEN_TICKERS = 2.0  # seconds
DELAY_ON_FAILURE = 5.0  # seconds
MAX_FAILURES_BEFORE_BACKOFF = 5
BACKOFF_DELAY = 60  # seconds

# Major EU tickers by exchange (known liquid stocks)
PRIORITY_TICKERS = [
    # Germany (DAX components)
    'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE',
    'MBG.DE', 'VOW3.DE', 'ADS.DE', 'MUV2.DE', 'DBK.DE', 'IFX.DE', 'HEN3.DE',
    'RWE.DE', 'DPW.DE', 'CON.DE', 'FRE.DE', 'MTX.DE', 'HEI.DE',
    # UK (FTSE components)
    'HSBA.L', 'BP.L', 'SHEL.L', 'GSK.L', 'AZN.L', 'ULVR.L', 'RIO.L', 'GLEN.L',
    'VOD.L', 'LLOY.L', 'BARC.L', 'NWG.L', 'REL.L', 'CPG.L', 'AAL.L', 'DGE.L',
    'PRU.L', 'LSEG.L', 'NG.L', 'SSE.L', 'EXPN.L', 'CRH.L', 'IMB.L', 'BA.L',
    # France (CAC components)
    'OR.PA', 'MC.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'AI.PA',
    'DG.PA', 'CS.PA', 'BN.PA', 'EL.PA', 'KER.PA', 'RI.PA', 'SGO.PA', 'CAP.PA',
    'STMPA.PA', 'ORA.PA', 'VIV.PA', 'EN.PA',
    # Netherlands (AEX)
    'ASML.AS', 'SHELL.AS', 'UNA.AS', 'PRX.AS', 'INGA.AS', 'ADYEN.AS', 'ABN.AS',
    'ASM.AS', 'WKL.AS', 'RAND.AS', 'AD.AS', 'AKZA.AS', 'HEIA.AS', 'PHIA.AS',
    # Switzerland
    'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'CSGN.SW', 'ABBN.SW', 'ZURN.SW',
    'SREN.SW', 'GIVN.SW', 'LONN.SW', 'GEBN.SW', 'SIKA.SW', 'SCMN.SW', 'ALC.SW',
    # Spain (IBEX)
    'SAN.MC', 'IBE.MC', 'ITX.MC', 'BBVA.MC', 'TEF.MC', 'REP.MC', 'FER.MC',
    'ELE.MC', 'ACS.MC', 'AMS.MC', 'CABK.MC', 'GRF.MC', 'COL.MC', 'ENG.MC',
    # Italy (MIB)
    'ENI.MI', 'ENEL.MI', 'ISP.MI', 'UCG.MI', 'STM.MI', 'G.MI', 'RACE.MI',
    'PRY.MI', 'LDO.MI', 'SFER.MI', 'TEN.MI', 'BAMI.MI', 'MB.MI', 'AMP.MI',
    # Nordic
    'NOVO-B.CO', 'MAERSK-B.CO', 'CARL-B.CO', 'ORSTED.CO', 'VWS.CO',  # Denmark
    'VOLV-B.ST', 'ERIC-B.ST', 'ATCO-B.ST', 'SEB-A.ST', 'SKA-B.ST',  # Sweden
    'EQNR.OL', 'DNB.OL', 'TEL.OL', 'MOWI.OL', 'ORK.OL',  # Norway
    'NESTE.HE', 'FORTUM.HE', 'NOKIA.HE', 'STERV.HE', 'UPM.HE',  # Finland
    # Austria
    'OMV.VI', 'VOE.VI', 'EBS.VI', 'POST.VI', 'VER.VI',
    # Belgium
    'ABI.BR', 'UCB.BR', 'SOLB.BR', 'KBC.BR', 'ARGX.BR',
    # Portugal
    'EDP.LS', 'GALP.LS', 'JMT.LS', 'SON.LS', 'BCP.LS',
    # Ireland
    'CRH.IR', 'AIB.IR', 'BIRG.IR',
]


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def fetch_market_cap(ticker: str) -> tuple:
    """Fetch market cap for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        mktcap = info.get('marketCap')
        name = info.get('shortName') or info.get('longName')
        currency = info.get('currency')

        if mktcap and mktcap > 0:
            return mktcap, name, currency
        return None, name, currency
    except Exception as e:
        return None, None, str(e)[:50]


def main():
    print("=" * 60)
    print("SMART EU MARKET CAP FETCHER")
    print("=" * 60)

    cache = load_cache()
    print(f"Current cache size: {len(cache)}")

    # Filter to tickers not in cache
    to_fetch = [t for t in PRIORITY_TICKERS if t not in cache]
    print(f"Priority tickers to fetch: {len(to_fetch)}")
    print()

    if not to_fetch:
        print("All priority tickers already in cache!")
        return

    successful = 0
    failed = 0
    consecutive_failures = 0

    for i, ticker in enumerate(to_fetch):
        print(f"[{i+1}/{len(to_fetch)}] Fetching {ticker}...", end=" ", flush=True)

        mktcap, name, extra = fetch_market_cap(ticker)

        if mktcap:
            cache[ticker] = {
                "market_cap": mktcap,
                "name": name,
                "currency": extra,
                "fetched_at": datetime.now().isoformat(),
                "source": "yfinance"
            }
            successful += 1
            consecutive_failures = 0
            print(f"OK - {name} - ${mktcap:,.0f}")
            time.sleep(DELAY_BETWEEN_TICKERS)
        else:
            failed += 1
            consecutive_failures += 1
            print(f"FAILED - {extra}")

            if consecutive_failures >= MAX_FAILURES_BEFORE_BACKOFF:
                print(f"\n{MAX_FAILURES_BEFORE_BACKOFF} consecutive failures - backing off for {BACKOFF_DELAY}s...")
                save_cache(cache)
                time.sleep(BACKOFF_DELAY)
                consecutive_failures = 0
            else:
                time.sleep(DELAY_ON_FAILURE)

        # Save every 10 tickers
        if (i + 1) % 10 == 0:
            save_cache(cache)

    save_cache(cache)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total in cache: {len(cache)}")

    # Count by type
    eu_extensions = ['.L', '.DE', '.PA', '.MI', '.AS', '.SW', '.ST', '.OL', '.VI', '.HE', '.BR', '.IR', '.MC', '.LS', '.CO']
    eu_count = sum(1 for t in cache if any(t.endswith(ext) for ext in eu_extensions))
    print(f"EU tickers: {eu_count}")


if __name__ == "__main__":
    main()
