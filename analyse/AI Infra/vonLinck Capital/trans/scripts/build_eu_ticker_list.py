"""
Build Expanded EU Ticker List
=============================

Fetches micro/small cap tickers from European markets:
- Germany (.DE) - Frankfurt/Xetra
- UK (.L) - London
- France (.PA) - Euronext Paris
- Italy (.MI) - Borsa Italiana
- Spain (.MC) - Madrid
- Netherlands (.AS) - Euronext Amsterdam
- Portugal (.LS) - Euronext Lisbon
- Belgium (.BR) - Euronext Brussels
- Switzerland (.SW) - SIX Swiss
- Sweden (.ST) - Stockholm
- Norway (.OL) - Oslo
- Denmark (.CO) - Copenhagen
- Finland (.HE) - Helsinki

Uses multiple data sources:
1. Existing GCS bucket tickers
2. Yahoo Finance screener API
3. FMP stock list API (if available)

Usage:
    python scripts/build_eu_ticker_list.py --output data/eu_tickers.txt
    python scripts/build_eu_ticker_list.py --download-data --workers 4
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env', override=True)


# European market suffixes and names
EU_MARKETS = {
    '.DE': {'name': 'Germany (Frankfurt/Xetra)', 'country': 'DE'},
    '.L': {'name': 'UK (London)', 'country': 'GB'},
    '.PA': {'name': 'France (Euronext Paris)', 'country': 'FR'},
    '.MI': {'name': 'Italy (Borsa Italiana)', 'country': 'IT'},
    '.MC': {'name': 'Spain (Madrid)', 'country': 'ES'},
    '.AS': {'name': 'Netherlands (Euronext Amsterdam)', 'country': 'NL'},
    '.LS': {'name': 'Portugal (Euronext Lisbon)', 'country': 'PT'},
    '.BR': {'name': 'Belgium (Euronext Brussels)', 'country': 'BE'},
    '.SW': {'name': 'Switzerland (SIX)', 'country': 'CH'},
    '.ST': {'name': 'Sweden (Stockholm)', 'country': 'SE'},
    '.OL': {'name': 'Norway (Oslo)', 'country': 'NO'},
    '.CO': {'name': 'Denmark (Copenhagen)', 'country': 'DK'},
    '.HE': {'name': 'Finland (Helsinki)', 'country': 'FI'},
    '.IR': {'name': 'Ireland (Dublin)', 'country': 'IE'},
    '.VI': {'name': 'Austria (Vienna)', 'country': 'AT'},
}

# Market cap thresholds (in USD)
MICRO_CAP_MAX = 300_000_000    # $300M
SMALL_CAP_MAX = 2_000_000_000  # $2B
MID_CAP_MAX = 10_000_000_000   # $10B


def get_existing_gcs_tickers() -> Set[str]:
    """Get tickers already in GCS bucket."""
    try:
        from google.cloud import storage

        project_id = os.getenv('PROJECT_ID')
        bucket_name = os.getenv('GCS_BUCKET_NAME')

        if not project_id or not bucket_name:
            logger.warning("GCS credentials not configured")
            return set()

        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix='tickers/')
        tickers = set()

        for blob in blobs:
            name = blob.name.replace('tickers/', '').replace('.parquet', '').replace('.csv', '')
            if name and any(name.endswith(suffix) for suffix in EU_MARKETS.keys()):
                tickers.add(name)

        logger.info(f"Found {len(tickers)} EU tickers in GCS")
        return tickers

    except Exception as e:
        logger.warning(f"Could not access GCS: {e}")
        return set()


def fetch_tickers_yfinance(market_suffix: str, max_tickers: int = 500) -> List[str]:
    """
    Fetch tickers from Yahoo Finance for a given market.
    Uses yfinance screener functionality.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return []

    market_info = EU_MARKETS.get(market_suffix, {})
    country = market_info.get('country', '')

    logger.info(f"Fetching tickers for {market_suffix} ({market_info.get('name', 'Unknown')})")

    tickers = []

    # Method 1: Try to get from known indices
    index_tickers = get_index_components(market_suffix)
    if index_tickers:
        tickers.extend(index_tickers)
        logger.info(f"  Found {len(index_tickers)} from indices")

    # Method 2: Use screener queries (limited functionality)
    # Yahoo Finance screener is limited, so we supplement with FMP

    return list(set(tickers))[:max_tickers]


def get_index_components(market_suffix: str) -> List[str]:
    """Get components of major indices for each market."""
    try:
        import yfinance as yf
    except ImportError:
        return []

    # Map market suffix to major indices
    indices = {
        '.DE': ['^GDAXI', '^MDAXI', '^SDAXI', '^TECDAX'],  # DAX, MDAX, SDAX, TecDAX
        '.L': ['^FTSE', '^FTMC', '^FTAI'],  # FTSE 100, 250, AIM
        '.PA': ['^FCHI', '^SBF120'],  # CAC 40, SBF 120
        '.MI': ['^FTMIB'],  # FTSE MIB
        '.MC': ['^IBEX'],  # IBEX 35
        '.AS': ['^AEX', '^AMX'],  # AEX, AMX
        '.SW': ['^SSMI'],  # SMI
        '.ST': ['^OMX', '^OMXS30'],  # OMX Stockholm
        '.OL': ['^OSEAX'],  # Oslo All-Share
        '.CO': ['^OMXC25'],  # OMX Copenhagen
        '.HE': ['^OMXH25'],  # OMX Helsinki
    }

    market_indices = indices.get(market_suffix, [])
    all_components = []

    for index_symbol in market_indices:
        try:
            # This is a simplified approach - Yahoo doesn't always return components
            # We'll supplement with FMP API
            pass
        except Exception as e:
            logger.debug(f"Could not get components for {index_symbol}: {e}")

    return all_components


def fetch_tickers_fmp(market_suffix: str, max_tickers: int = 500) -> List[str]:
    """
    Fetch tickers from Financial Modeling Prep API.
    Requires FMP_API_KEY in environment.
    """
    import requests

    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.warning("FMP_API_KEY not set, skipping FMP source")
        return []

    market_info = EU_MARKETS.get(market_suffix, {})
    exchange_map = {
        '.DE': 'XETRA',
        '.L': 'LSE',
        '.PA': 'EURONEXT',
        '.MI': 'MIL',
        '.MC': 'BME',
        '.AS': 'EURONEXT',
        '.LS': 'EURONEXT',
        '.BR': 'EURONEXT',
        '.SW': 'SIX',
        '.ST': 'STO',
        '.OL': 'OSL',
        '.CO': 'CPH',
        '.HE': 'HEL',
    }

    exchange = exchange_map.get(market_suffix)
    if not exchange:
        return []

    try:
        # Get stock list from FMP
        url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            logger.warning(f"FMP API error: {response.status_code}")
            return []

        stocks = response.json()

        # Filter by exchange and add suffix
        tickers = []
        for stock in stocks:
            if stock.get('exchangeShortName') == exchange:
                symbol = stock.get('symbol', '')
                # Add suffix if not present
                if not any(symbol.endswith(s) for s in EU_MARKETS.keys()):
                    symbol = symbol + market_suffix
                tickers.append(symbol)

        logger.info(f"  Found {len(tickers)} from FMP for {exchange}")
        return tickers[:max_tickers]

    except Exception as e:
        logger.warning(f"FMP fetch error: {e}")
        return []


def fetch_tickers_twelvedata(market_suffix: str, max_tickers: int = 500) -> List[str]:
    """
    Fetch tickers from TwelveData API.
    """
    import requests

    api_key = os.getenv('TWELVEDATA_API_KEY')
    if not api_key:
        logger.warning("TWELVEDATA_API_KEY not set, skipping TwelveData source")
        return []

    market_info = EU_MARKETS.get(market_suffix, {})
    country = market_info.get('country', '')

    try:
        url = f"https://api.twelvedata.com/stocks?country={country}&apikey={api_key}"
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            logger.warning(f"TwelveData API error: {response.status_code}")
            return []

        data = response.json()
        stocks = data.get('data', [])

        # Extract symbols and add suffix
        tickers = []
        for stock in stocks:
            symbol = stock.get('symbol', '')
            # TwelveData may already have suffix or not
            if not any(symbol.endswith(s) for s in EU_MARKETS.keys()):
                symbol = symbol + market_suffix
            tickers.append(symbol)

        logger.info(f"  Found {len(tickers)} from TwelveData for {country}")
        return tickers[:max_tickers]

    except Exception as e:
        logger.warning(f"TwelveData fetch error: {e}")
        return []


def get_hardcoded_eu_tickers() -> Dict[str, List[str]]:
    """
    Hardcoded list of major EU small/micro cap tickers by market.
    This serves as a baseline when API access is limited.
    """
    return {
        '.PA': [  # France - Euronext Paris
            'AIR.PA', 'BNP.PA', 'SAN.PA', 'OR.PA', 'MC.PA', 'SU.PA', 'AI.PA',
            'KER.PA', 'RI.PA', 'CAP.PA', 'VIV.PA', 'ORA.PA', 'BN.PA', 'EN.PA',
            'DSY.PA', 'ACA.PA', 'SGO.PA', 'CS.PA', 'VIE.PA', 'HO.PA',
            # Small/Mid caps
            'SOI.PA', 'ERF.PA', 'GTT.PA', 'ATE.PA', 'IPN.PA', 'SOP.PA',
            'FNAC.PA', 'LNA.PA', 'JXR.PA', 'NXI.PA', 'LPE.PA', 'ALBPS.PA',
            'ALBI.PA', 'ALNOV.PA', 'ALSEN.PA', 'ALGAU.PA', 'MLONE.PA',
        ],
        '.MI': [  # Italy - Borsa Italiana
            'ISP.MI', 'UCG.MI', 'ENI.MI', 'ENEL.MI', 'TIT.MI', 'PRY.MI',
            'G.MI', 'FCA.MI', 'TEN.MI', 'LDO.MI', 'SRG.MI', 'MB.MI',
            # Small/Mid caps
            'IOT.MI', 'SRS.MI', 'AEF.MI', 'IT.MI', 'MN.MI', 'RCS.MI',
            'CEM.MI', 'SAL.MI', 'IMA.MI', 'MARR.MI', 'SOL.MI', 'BC.MI',
            'ERG.MI', 'BAMI.MI', 'AMP.MI', 'EVS.MI', 'RWAY.MI',
        ],
        '.MC': [  # Spain - BME Madrid
            'SAN.MC', 'BBVA.MC', 'ITX.MC', 'IBE.MC', 'TEF.MC', 'REP.MC',
            'AMS.MC', 'FER.MC', 'ENG.MC', 'GRF.MC', 'IAG.MC', 'COL.MC',
            # Small/Mid caps
            'CAF.MC', 'PHM.MC', 'LOG.MC', 'LRE.MC', 'GEST.MC', 'FAE.MC',
            'VIS.MC', 'ENC.MC', 'PSG.MC', 'MEL.MC', 'AENA.MC', 'MTS.MC',
        ],
        '.AS': [  # Netherlands - Euronext Amsterdam
            'ASML.AS', 'REN.AS', 'INGA.AS', 'ABN.AS', 'PHIA.AS', 'AD.AS',
            'UNA.AS', 'DSM.AS', 'KPN.AS', 'WKL.AS', 'AGN.AS', 'AKZA.AS',
            # Small/Mid caps
            'CRBN.AS', 'BESI.AS', 'FLOW.AS', 'ALFEN.AS', 'BFIT.AS', 'LIGHT.AS',
            'ACOMO.AS', 'KENDR.AS', 'SBMO.AS', 'VPK.AS', 'TOM2.AS',
        ],
        '.LS': [  # Portugal - Euronext Lisbon
            'EDP.LS', 'GALP.LS', 'JMT.LS', 'BCP.LS', 'SON.LS', 'NOS.LS',
            'SEM.LS', 'CTT.LS', 'PHR.LS', 'ALTR.LS', 'RENE.LS', 'COR.LS',
            'TDSA.LS', 'NBA.LS', 'MRL.LS',
        ],
        '.BR': [  # Belgium - Euronext Brussels
            'ABI.BR', 'KBC.BR', 'UCB.BR', 'SOLB.BR', 'GLPG.BR', 'ACKB.BR',
            'AGS.BR', 'PROX.BR', 'COFB.BR', 'DIE.BR', 'BAR.BR', 'ELI.BR',
            'MELE.BR', 'BPOST.BR', 'VGP.BR', 'ONTEX.BR', 'TINC.BR',
        ],
        '.SW': [  # Switzerland - SIX
            'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'CSGN.SW', 'ABBN.SW',
            'ZURN.SW', 'SREN.SW', 'GIVN.SW', 'LONN.SW', 'GEBN.SW', 'SGSN.SW',
            # Small/Mid caps
            'AMS.SW', 'TEMN.SW', 'VACN.SW', 'SOON.SW', 'BUCN.SW', 'BANB.SW',
            'HELN.SW', 'SENS.SW', 'DKSH.SW', 'BEAN.SW',
        ],
        '.ST': [  # Sweden - Stockholm
            'VOLV-B.ST', 'ERIC-B.ST', 'ATCO-A.ST', 'SEB-A.ST', 'SWED-A.ST',
            'ESSITY-B.ST', 'SAND.ST', 'SKF-B.ST', 'ASSA-B.ST', 'ALFA.ST',
            # Small/Mid caps
            'DUST.ST', 'SINCH.ST', 'CIBUS.ST', 'K2A-PREF.ST', 'TRAD.ST',
            'PACT.ST', 'SYSR.ST', 'COLL.ST', 'TOBII.ST', 'HANZA.ST',
        ],
        '.OL': [  # Norway - Oslo
            'EQNR.OL', 'DNB.OL', 'TEL.OL', 'MOWI.OL', 'YAR.OL', 'ORK.OL',
            'SALM.OL', 'TOM.OL', 'AKRBP.OL', 'STB.OL', 'SUBC.OL', 'AKSO.OL',
            # Small/Mid caps
            'ARR.OL', 'ZAP.OL', 'PEN.OL', 'AKER.OL', 'KOG.OL', 'BWO.OL',
            'NEXT.OL', 'NOD.OL', 'KIT.OL', 'HAUTO.OL', 'AGAS.OL',
        ],
        '.CO': [  # Denmark - Copenhagen
            'NOVO-B.CO', 'DSV.CO', 'VWS.CO', 'MAERSK-B.CO', 'ORSTED.CO',
            'CARL-B.CO', 'COLO-B.CO', 'PNDORA.CO', 'GMAB.CO', 'CHR.CO',
            # Small/Mid caps
            'ROCK-B.CO', 'RBREW.CO', 'TRYG.CO', 'GN.CO', 'JYSK.CO',
            'NKT.CO', 'FLS.CO', 'NZYM-B.CO', 'AMBU-B.CO', 'TOP.CO',
        ],
        '.HE': [  # Finland - Helsinki
            'NOKIA.HE', 'NDA-FI.HE', 'FORTUM.HE', 'SAMPO.HE', 'UPM.HE',
            'KNEBV.HE', 'STERV.HE', 'ELISA.HE', 'OUT1V.HE', 'WRT1V.HE',
            # Small/Mid caps
            'METSO.HE', 'VALMT.HE', 'ORNBV.HE', 'METSB.HE', 'HUH1V.HE',
            'KEMIRA.HE', 'TELIA1.HE', 'CGCBV.HE', 'SSH1V.HE',
        ],
    }


def download_ohlcv_data(ticker: str, start_date: str = '2020-01-01') -> Optional[Dict]:
    """
    Download OHLCV data for a single ticker using yfinance.
    Returns dict with data or None if failed.
    """
    try:
        import yfinance as yf
        import pandas as pd

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, auto_adjust=False)

        if df is None or len(df) < 100:  # Need at least 100 days
            return None

        # Standardize columns
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Rename columns to match expected format
        column_map = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adj_close': 'adj_close',
            'volume': 'volume'
        }

        df = df.rename(columns=column_map)

        # Get market cap info
        info = stock.info
        market_cap = info.get('marketCap', 0)

        return {
            'ticker': ticker,
            'data': df,
            'market_cap': market_cap,
            'rows': len(df)
        }

    except Exception as e:
        logger.debug(f"Failed to download {ticker}: {e}")
        return None


def save_ticker_data(ticker: str, df, output_dir: Path):
    """Save ticker data to parquet file."""
    output_path = output_dir / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def build_eu_ticker_list(args):
    """Main function to build EU ticker list."""

    logger.info("="*60)
    logger.info("Building Expanded EU Ticker List")
    logger.info("="*60)

    all_tickers = set()

    # 1. Get existing GCS tickers
    existing = get_existing_gcs_tickers()
    all_tickers.update(existing)
    logger.info(f"Existing GCS tickers: {len(existing)}")

    # 2. Add hardcoded baseline tickers
    hardcoded = get_hardcoded_eu_tickers()
    for market, tickers in hardcoded.items():
        all_tickers.update(tickers)
        logger.info(f"Added {len(tickers)} hardcoded tickers for {market}")

    # 3. Fetch from APIs if available
    for market_suffix in EU_MARKETS.keys():
        if args.skip_api:
            continue

        # Try TwelveData first (we have API key)
        td_tickers = fetch_tickers_twelvedata(market_suffix, max_tickers=200)
        all_tickers.update(td_tickers)

        # Rate limit
        time.sleep(1)

        # Try FMP
        fmp_tickers = fetch_tickers_fmp(market_suffix, max_tickers=200)
        all_tickers.update(fmp_tickers)

        time.sleep(1)

    # Filter to only EU markets
    eu_tickers = [t for t in all_tickers if any(t.endswith(s) for s in EU_MARKETS.keys())]

    # Report by market
    logger.info("\n" + "="*60)
    logger.info("Ticker Count by Market:")
    logger.info("="*60)

    market_counts = {}
    for suffix, info in EU_MARKETS.items():
        count = len([t for t in eu_tickers if t.endswith(suffix)])
        market_counts[suffix] = count
        logger.info(f"  {suffix} ({info['name']}): {count}")

    logger.info(f"\nTotal EU tickers: {len(eu_tickers)}")

    # Save ticker list
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ticker in sorted(eu_tickers):
            f.write(ticker + '\n')

    logger.info(f"Saved ticker list to {output_path}")

    # Optionally download data
    if args.download_data:
        logger.info("\n" + "="*60)
        logger.info("Downloading OHLCV Data")
        logger.info("="*60)

        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Filter tickers that don't have data yet
        existing_files = set(f.stem for f in data_dir.glob('*.parquet'))
        tickers_to_download = [t for t in eu_tickers if t not in existing_files]

        logger.info(f"Tickers to download: {len(tickers_to_download)}")
        logger.info(f"Already have: {len(existing_files)}")

        if tickers_to_download:
            downloaded = 0
            failed = 0
            micro_small_count = 0

            # Apply limit if specified
            download_list = tickers_to_download[:args.limit] if args.limit else tickers_to_download

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(download_ohlcv_data, ticker, args.start_date): ticker
                    for ticker in download_list
                }

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        if result:
                            # Check if micro/small/mid cap (up to $10B)
                            market_cap = result.get('market_cap', 0)
                            if market_cap > 0 and market_cap <= MID_CAP_MAX:
                                micro_small_count += 1
                                save_ticker_data(ticker, result['data'], data_dir)
                                downloaded += 1
                                cap_label = 'micro' if market_cap <= MICRO_CAP_MAX else 'small' if market_cap <= SMALL_CAP_MAX else 'mid'
                                logger.info(f"  {ticker}: {result['rows']} rows, ${market_cap/1e6:.0f}M ({cap_label})")
                            elif market_cap == 0:
                                # Unknown cap, save anyway
                                save_ticker_data(ticker, result['data'], data_dir)
                                downloaded += 1
                                logger.info(f"  {ticker}: {result['rows']} rows, unknown cap")
                            else:
                                logger.debug(f"  {ticker}: Skipped (${market_cap/1e9:.1f}B > $10B cap)")
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        logger.debug(f"  {ticker}: Error - {e}")

            logger.info(f"\nDownloaded: {downloaded}, Failed: {failed}")
            logger.info(f"Micro/Small/Mid cap: {micro_small_count}")

    return eu_tickers


def main():
    parser = argparse.ArgumentParser(description='Build expanded EU ticker list')
    parser.add_argument('--output', type=str, default='data/eu_tickers.txt',
                       help='Output file for ticker list')
    parser.add_argument('--download-data', action='store_true',
                       help='Download OHLCV data for tickers')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory to save OHLCV data')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for OHLCV data')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of download workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to download (for testing)')
    parser.add_argument('--skip-api', action='store_true',
                       help='Skip API fetching, use only hardcoded tickers')
    args = parser.parse_args()

    tickers = build_eu_ticker_list(args)

    logger.info("\nDone!")
    logger.info(f"Total EU tickers: {len(tickers)}")


if __name__ == '__main__':
    main()
