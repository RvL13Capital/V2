"""
Fetch Market Caps for Major EU Tickers
======================================

Run this script after the yfinance rate limit resets (usually 1-2 hours).

Usage:
    python scripts/fetch_major_eu_market_caps.py
"""

import yfinance as yf
import json
import time
from pathlib import Path
from datetime import datetime

MAJOR_EU_TICKERS = [
    # Germany (XETRA)
    'SAP.DE', 'BMW.DE', 'SIE.DE', 'ALV.DE', 'BAS.DE', 'VOW3.DE', 'DTE.DE',
    'MRK.DE', 'ADS.DE', 'MUV2.DE', 'DPW.DE', 'HEN3.DE', 'IFX.DE',
    # UK (LSE)
    'VOD.L', 'HSBA.L', 'SHEL.L', 'BP.L', 'GSK.L', 'RIO.L', 'AZN.L',
    'ULVR.L', 'LLOY.L', 'GLEN.L', 'BARC.L', 'REL.L',
    # France (Euronext Paris)
    'AIR.PA', 'OR.PA', 'SAN.PA', 'BNP.PA', 'MC.PA', 'TTE.PA', 'SU.PA',
    'CAP.PA', 'AI.PA', 'EL.PA',
    # Netherlands (Euronext Amsterdam)
    'ASML.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'UNA.AS',
    # Switzerland (SIX)
    'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ABBN.SW',
    # Spain (BME)
    'SAN.MC', 'IBE.MC', 'ITX.MC', 'TEF.MC',
    # Italy (Borsa Italiana)
    'ENI.MI', 'ISP.MI', 'UCG.MI', 'ENEL.MI',
]

def main():
    cache_file = Path('data/cache/market_cap/market_cap_cache.json')

    # Load existing cache
    with open(cache_file, 'r') as f:
        cache = json.load(f)

    print('='*60)
    print('FETCHING MAJOR EU TICKERS')
    print('='*60)
    print(f'Tickers to fetch: {len(MAJOR_EU_TICKERS)}')
    print(f'Current cache size: {len(cache)}')
    print()

    success = 0
    failed = 0

    for i, ticker in enumerate(MAJOR_EU_TICKERS):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            mc = info.get('marketCap')
            shares = info.get('sharesOutstanding')
            name = info.get('shortName', 'N/A')

            if mc and mc > 0:
                cache[ticker] = {
                    'ticker': ticker,
                    'market_cap': float(mc),
                    'shares_outstanding': float(shares) if shares else None,
                    'source': 'yfinance_major',
                    'fetched_at': datetime.now().isoformat(),
                    'currency': info.get('currency', 'EUR')
                }
                print(f'[{i+1}/{len(MAJOR_EU_TICKERS)}] {ticker}: {name} - ${mc/1e9:.1f}B')
                success += 1
            else:
                print(f'[{i+1}/{len(MAJOR_EU_TICKERS)}] {ticker}: No market cap data')
                failed += 1

            time.sleep(0.5)  # Small delay to avoid rate limit

        except Exception as e:
            error_msg = str(e)
            if 'Rate limited' in error_msg or '429' in error_msg:
                print(f'\nRate limited! Waiting 60 seconds...')
                time.sleep(60)
                # Retry
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    mc = info.get('marketCap')
                    if mc and mc > 0:
                        cache[ticker] = {
                            'ticker': ticker,
                            'market_cap': float(mc),
                            'shares_outstanding': float(info.get('sharesOutstanding')) if info.get('sharesOutstanding') else None,
                            'source': 'yfinance_major',
                            'fetched_at': datetime.now().isoformat(),
                            'currency': info.get('currency', 'EUR')
                        }
                        print(f'[{i+1}/{len(MAJOR_EU_TICKERS)}] {ticker}: ${mc/1e9:.1f}B (retry success)')
                        success += 1
                    else:
                        failed += 1
                except:
                    failed += 1
            else:
                print(f'[{i+1}/{len(MAJOR_EU_TICKERS)}] {ticker}: Error - {error_msg[:50]}')
                failed += 1

        # Save periodically
        if (i + 1) % 10 == 0:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

    # Final save
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print()
    print('='*60)
    print('COMPLETED')
    print('='*60)
    print(f'Success: {success}')
    print(f'Failed: {failed}')
    print(f'Total in cache: {len(cache)}')


if __name__ == '__main__':
    main()
