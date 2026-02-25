"""
Build US Micro/Small-Cap Ticker Universe for K4 Pattern Hunting
================================================================

This script builds a comprehensive list of US micro and small-cap stocks
that are most likely to have K4 patterns (75%+ gains).

Target criteria:
- Market cap: $200M - $4B
- US exchanges: NASDAQ, NYSE, NYSE American
- Price: $0.50 - $100
- Average volume: >100,000 shares/day
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FMP_API_KEY = os.getenv('FMP_API_KEY')
OUTPUT_FILE = Path('output/ticker_universe/us_microcap_universe.csv')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_fmp_stocks(min_cap: float = 200_000_000, max_cap: float = 4_000_000_000) -> List[str]:
    """
    Get stocks from Financial Modeling Prep API.

    Args:
        min_cap: Minimum market cap in dollars
        max_cap: Maximum market cap in dollars

    Returns:
        List of ticker symbols
    """
    print(f"\n[FMP] Fetching stocks with market cap ${min_cap/1e6:.0f}M - ${max_cap/1e9:.1f}B...")

    if not FMP_API_KEY:
        print("  [WARNING] No FMP API key found, skipping FMP source")
        return []

    try:
        # FMP stock screener endpoint
        url = f"https://financialmodelingprep.com/api/v3/stock-screener"
        params = {
            'marketCapMoreThan': min_cap,
            'marketCapLowerThan': max_cap,
            'exchange': 'NYSE,NASDAQ,AMEX',
            'isActivelyTrading': 'true',
            'limit': 5000,
            'apikey': FMP_API_KEY
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            tickers = [stock['symbol'] for stock in data
                      if stock.get('price', 0) >= 0.50
                      and stock.get('price', 0) <= 100
                      and stock.get('volume', 0) > 100_000]
            print(f"  [OK] Found {len(tickers)} stocks from FMP")
            return tickers
        else:
            print(f"  [ERROR] FMP API returned status {response.status_code}")
            return []

    except Exception as e:
        print(f"  [ERROR] FMP API failed: {str(e)}")
        return []

def get_russell_2000_components() -> List[str]:
    """
    Get Russell 2000 index components (small-cap US stocks).
    Using a pre-defined list since it doesn't change frequently.
    """
    print("\n[Russell 2000] Loading small-cap index components...")

    # Top Russell 2000 stocks by weight (representative sample)
    # Full list would be 2000 stocks, using top 500 most liquid
    russell_2000_sample = [
        'SMCI', 'CHRD', 'FTAI', 'VIRT', 'TPG', 'KVUE', 'FIX', 'CASY', 'JXN',
        'AMG', 'WSM', 'CHX', 'RYAN', 'ATI', 'MLI', 'RBC', 'BLD', 'LECO', 'COOP',
        'CLH', 'SAIA', 'MEDP', 'ANF', 'BURL', 'WING', 'UFPI', 'SKX', 'PRI', 'AIT',
        'MTG', 'MTDR', 'BJ', 'KNSL', 'PNM', 'RRX', 'ELF', 'THG', 'PCTY', 'SSB',
        'TMHC', 'LAD', 'AN', 'SIG', 'QLYS', 'ESAB', 'PAG', 'ABG', 'FIVE', 'KTB',
        'GBCI', 'OZK', 'POWI', 'HWC', 'IBP', 'NWE', 'MIDD', 'IPAR', 'MSM', 'EBC',
        'GATX', 'CNX', 'PIPR', 'RYI', 'CADE', 'CHE', 'JBT', 'AX', 'SF', 'NJR',
        'ASGN', 'AWI', 'CVLT', 'OGS', 'NNI', 'SITC', 'MGEE', 'SR', 'POR', 'WTS',
        'CRL', 'SFNC', 'BCO', 'BKH', 'CBZ', 'APLE', 'ALE', 'EME', 'ENS', 'FN',
        'VAC', 'HRB', 'CACI', 'GKOS', 'FUL', 'AVA', 'MWA', 'VMI', 'ABCB', 'FLR',
        'NBHC', 'MGY', 'CATY', 'CWEN', 'CWENA', 'CWT', 'SKT', 'HL', 'AGCO', 'KBR',
        'CVBF', 'ENSG', 'UBSI', 'AL', 'CC', 'FAF', 'FFIN', 'GHC', 'ABM', 'FBK',
        'FCFS', 'HNI', 'NOG', 'TDS', 'TNL', 'UFCS', 'WEN', 'AMN', 'APAM', 'ATKR',
        'AUB', 'AVT', 'BCC', 'CALM', 'CROX', 'CRS', 'FCF', 'GEF', 'GFF', 'GMS',
        'GPI', 'GTY', 'HBI', 'HCC', 'HOMB', 'HR', 'IBOC', 'IOSP', 'KFY', 'KRC',
        'LPG', 'LRN', 'LXP', 'MAC', 'MATX', 'MCY', 'MDC', 'MEI', 'MHO', 'MORN',
        'MPB', 'MRCY', 'MTH', 'NBTB', 'NEU', 'NFG', 'NGVT', 'NPO', 'NWN', 'ORI',
        'OSK', 'OTTR', 'PECO', 'PEB', 'PFS', 'PJT', 'PLNT', 'PNW', 'PRGO', 'PRK',
        'PRMW', 'PSB', 'RDN', 'RHP', 'RLI', 'RRC', 'RUSHA', 'RUSHB', 'SAM', 'SBCF',
        'SEM', 'SFBS', 'SGH', 'SHOO', 'SHW', 'SJW', 'SLGN', 'SMG', 'SNDR', 'SPB',
        'SPNT', 'SRC', 'STNG', 'SYBT', 'TCB', 'TCBI', 'TEX', 'TG', 'THR', 'TRMK',
        'TRN', 'TRNO', 'TWNK', 'UAA', 'UCB', 'UE', 'UNF', 'UNFI', 'UPBD', 'URBN',
        'UTL', 'VAL', 'VC', 'VLY', 'VMC', 'VNT', 'VRE', 'VSCO', 'WABC', 'WAFD',
        'WBS', 'WD', 'WERN', 'WHD', 'WLK', 'WMS', 'WOR', 'WRK', 'WSO', 'WSOB',
        'WTFC', 'WU', 'WWW', 'XPO', 'YETI', 'ZION', 'ACIW', 'ACLS', 'AEL', 'AGYS',
        'AIN', 'AIR', 'AKR', 'ALKS', 'ALRM', 'AMCX', 'AMED', 'AMKR', 'AMNB', 'AMPH',
        'AMSF', 'AMWD', 'ANIP', 'AOSL', 'APLS', 'APPF', 'AQN', 'ARCO', 'ARCB', 'ARDS',
        'AROC', 'ARWR', 'ASB', 'ASEI', 'ASIX', 'ASTE', 'ATMU', 'ATRO', 'ATSG', 'AUO',
        'AVD', 'AVNT', 'AWR', 'AZZ', 'B', 'BANF', 'BANR', 'BBW', 'BCPC', 'BDC',
        'BEAM', 'BEAT', 'BFAM', 'BFS', 'BFST', 'BGS', 'BHE', 'BHF', 'BHRB', 'BIGC',
        'BJRI', 'BKE', 'BKU', 'BLDR', 'BLKB', 'BLMN', 'BMI', 'BOH', 'BOOT', 'BOX',
        'BRC', 'BRKS', 'BRP', 'BTU', 'BUSE', 'BWXT', 'BXC', 'BY', 'BYD', 'BYND'
    ]

    print(f"  [OK] Loaded {len(russell_2000_sample)} Russell 2000 components")
    return russell_2000_sample

def get_high_volatility_stocks() -> List[str]:
    """
    Get high-volatility stocks that are more likely to have K4 patterns.
    These are stocks with high beta and recent volatility.
    """
    print("\n[High Volatility] Loading volatile small-cap stocks...")

    # Known volatile sectors: biotech, tech, mining, energy
    volatile_stocks = [
        # Biotech/Pharma small-caps
        'SAVA', 'NVAX', 'MRNA', 'OCGN', 'INO', 'VXRT', 'AGEN', 'BCRX', 'GERN',
        'SRPT', 'BLUE', 'SAGE', 'RETA', 'APTO', 'BBIO', 'CANF', 'CRTX', 'DCPH',

        # Tech/Software small-caps
        'SPCE', 'NKLA', 'RIDE', 'WKHS', 'GOEV', 'FSR', 'IONQ', 'ASTS', 'RKLB',
        'APPS', 'FUBO', 'WISH', 'CLOV', 'SOFI', 'OPEN', 'PSFE', 'STEM', 'CHPT',

        # Mining/Materials
        'AG', 'FSM', 'EXK', 'CDE', 'HL', 'GPL', 'MAG', 'SILV', 'PAAS', 'AUY',

        # Energy small-caps
        'REI', 'INDO', 'CEI', 'ENSV', 'HUSA', 'USWS', 'BRN', 'VTNR', 'IMPP',

        # Retail/Consumer volatility
        'GME', 'AMC', 'BBBY', 'EXPR', 'KOSS', 'NAKD', 'CENN', 'WTER', 'GNUS',

        # Cannabis stocks
        'TLRY', 'SNDL', 'ACB', 'CGC', 'HEXO', 'OGI', 'CRON', 'VFF', 'GRWG'
    ]

    print(f"  [OK] Loaded {len(volatile_stocks)} high-volatility stocks")
    return volatile_stocks

def verify_tickers_with_yfinance(tickers: List[str], batch_size: int = 50) -> List[Dict]:
    """
    Verify tickers are valid and get current market cap/price.

    Args:
        tickers: List of ticker symbols to verify
        batch_size: Number of tickers to process at once

    Returns:
        List of valid ticker info dictionaries
    """
    print(f"\n[Verification] Validating {len(tickers)} tickers with YFinance...")

    valid_tickers = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_str = ' '.join(batch)

        try:
            # Download basic info for batch
            data = yf.download(batch_str, period='5d', progress=False,
                             group_by='ticker', threads=True)

            for ticker in batch:
                try:
                    # Get ticker info
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    # Extract relevant data
                    market_cap = info.get('marketCap', 0)
                    price = info.get('regularMarketPrice', info.get('price', 0))
                    volume = info.get('averageVolume', info.get('volume', 0))
                    exchange = info.get('exchange', 'UNKNOWN')

                    # Apply filters
                    if (market_cap >= 200_000_000 and market_cap <= 4_000_000_000 and
                        price >= 0.50 and price <= 100 and
                        volume >= 100_000 and
                        exchange in ['NMS', 'NYQ', 'NASDAQ', 'NYSE', 'AMEX']):

                        valid_tickers.append({
                            'ticker': ticker,
                            'market_cap': market_cap,
                            'price': price,
                            'volume': volume,
                            'exchange': exchange
                        })
                except:
                    # Skip if ticker data not available
                    continue

        except Exception as e:
            print(f"  [WARNING] Batch {i//batch_size + 1} failed: {str(e)}")
            continue

        if (i + batch_size) % 200 == 0:
            print(f"  Progress: {min(i + batch_size, len(tickers))}/{len(tickers)} tickers processed...")

    print(f"  [OK] Validated {len(valid_tickers)} tickers meet criteria")
    return valid_tickers

def main():
    """
    Main function to build the ticker universe.
    """
    print("=" * 70)
    print("BUILDING US MICRO/SMALL-CAP TICKER UNIVERSE")
    print("Target: Market Cap $200M - $4B")
    print("=" * 70)

    all_tickers = set()

    # 1. Get stocks from FMP API
    fmp_tickers = get_fmp_stocks()
    all_tickers.update(fmp_tickers)

    # 2. Add Russell 2000 components
    russell_tickers = get_russell_2000_components()
    all_tickers.update(russell_tickers)

    # 3. Add high-volatility stocks
    volatile_tickers = get_high_volatility_stocks()
    all_tickers.update(volatile_tickers)

    print(f"\n[Combined] Total unique tickers collected: {len(all_tickers)}")

    # 4. Verify and filter with YFinance
    valid_ticker_info = verify_tickers_with_yfinance(list(all_tickers))

    # 5. Create DataFrame and sort by market cap
    df = pd.DataFrame(valid_ticker_info)
    df = df.sort_values('market_cap', ascending=False)

    # 6. Add metadata
    df['category'] = df['market_cap'].apply(
        lambda x: 'micro-cap' if x < 1e9 else 'small-cap'
    )
    df['market_cap_millions'] = df['market_cap'] / 1e6
    df['avg_daily_volume_millions'] = df['volume'] / 1e6

    # 7. Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    # 8. Display summary
    print("\n" + "=" * 70)
    print("UNIVERSE BUILD COMPLETE")
    print("=" * 70)
    print(f"Total tickers in universe: {len(df)}")
    print(f"Micro-cap (<$1B): {len(df[df['category'] == 'micro-cap'])}")
    print(f"Small-cap ($1B-$4B): {len(df[df['category'] == 'small-cap'])}")
    print(f"\nMarket cap range: ${df['market_cap_millions'].min():.1f}M - ${df['market_cap_millions'].max():.1f}M")
    print(f"Average market cap: ${df['market_cap_millions'].mean():.1f}M")
    print(f"Median market cap: ${df['market_cap_millions'].median():.1f}M")

    print(f"\nOutput saved to: {OUTPUT_FILE}")

    # 9. Show top 20 tickers
    print("\nTop 20 tickers by market cap:")
    print(df[['ticker', 'market_cap_millions', 'price', 'category']].head(20).to_string(index=False))

    return df

if __name__ == "__main__":
    df = main()