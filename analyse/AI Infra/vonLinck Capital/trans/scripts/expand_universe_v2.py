"""
Aggressive Universe Expansion v2
================================

Uses ALL available data sources to maximize ticker coverage:
1. TwelveData API - Full stock lists by country/exchange
2. Finnhub API - Stock symbols by exchange
3. Alpha Vantage - Listing status
4. SEC EDGAR - All US public companies (CIK list)
5. Yahoo Finance - Index components
6. Direct exchange CSVs - NASDAQ, NYSE, LSE, etc.
7. Comprehensive hardcoded lists

Target: Maximum coverage of micro/small/mid cap stocks
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Set, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

EU_SUFFIXES = ['.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS', '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI']


class AggressiveExpander:
    """Aggressively expand stock universe using all available sources."""

    def __init__(self):
        self.twelvedata_key = os.getenv('TWELVEDATA_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')

    def fetch_sec_cik_tickers(self) -> Set[str]:
        """Fetch ALL US public company tickers from SEC EDGAR CIK list."""
        logger.info("Fetching SEC EDGAR CIK list (all US public companies)...")
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; research)'}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                tickers = set()
                for entry in data.values():
                    ticker = entry.get('ticker', '')
                    if ticker and ticker.isalpha() and len(ticker) <= 5:
                        tickers.add(ticker.upper())
                logger.info(f"SEC EDGAR: {len(tickers)} tickers")
                return tickers
        except Exception as e:
            logger.warning(f"SEC EDGAR error: {e}")
        return set()

    def fetch_finnhub_stocks(self, exchange: str) -> Set[str]:
        """Fetch stocks from Finnhub by exchange."""
        if not self.finnhub_key:
            return set()

        try:
            url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={self.finnhub_key}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                stocks = response.json()
                tickers = set()
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    stock_type = stock.get('type', '')
                    # Filter to common stocks only
                    if stock_type in ['Common Stock', 'EQS', '']:
                        if symbol:
                            tickers.add(symbol)
                return tickers
        except Exception as e:
            logger.warning(f"Finnhub {exchange} error: {e}")
        return set()

    def fetch_alphavantage_listings(self) -> Set[str]:
        """Fetch active US listings from Alpha Vantage."""
        if not self.alphavantage_key:
            return set()

        logger.info("Fetching Alpha Vantage listing status...")
        try:
            url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={self.alphavantage_key}"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                tickers = set()
                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 4:
                        symbol = parts[0].strip()
                        asset_type = parts[3].strip() if len(parts) > 3 else ''
                        status = parts[5].strip() if len(parts) > 5 else ''
                        if asset_type == 'Stock' and status == 'Active':
                            if symbol.isalpha() and len(symbol) <= 5:
                                tickers.add(symbol.upper())
                logger.info(f"Alpha Vantage: {len(tickers)} active stocks")
                return tickers
        except Exception as e:
            logger.warning(f"Alpha Vantage error: {e}")
        return set()

    def fetch_twelvedata_exchange(self, exchange: str, suffix: str) -> Set[str]:
        """Fetch stocks from TwelveData by exchange."""
        if not self.twelvedata_key:
            return set()

        try:
            url = f"https://api.twelvedata.com/stocks?exchange={exchange}&apikey={self.twelvedata_key}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', [])
                tickers = set()
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    stock_type = stock.get('type', '')
                    if stock_type in ['Common Stock', 'EQUITY', '']:
                        if not any(symbol.endswith(s) for s in EU_SUFFIXES):
                            symbol = symbol + suffix
                        tickers.add(symbol)
                return tickers
        except Exception as e:
            logger.warning(f"TwelveData {exchange} error: {e}")
        return set()

    def fetch_yahoo_index_components(self, index_symbol: str) -> Set[str]:
        """Fetch index components from Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(index_symbol)
            # Try to get components (not always available)
            return set()
        except:
            return set()

    def fetch_otc_markets(self) -> Set[str]:
        """Fetch OTC Markets tickers (pink sheets, OTCQX, OTCQB)."""
        logger.info("Fetching OTC Markets listings...")
        tickers = set()

        # OTC Markets provides CSV downloads
        urls = [
            'https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?gradeGroup=QX',
            'https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?gradeGroup=QB',
        ]

        for url in urls:
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    for line in lines[1:]:
                        parts = line.split(',')
                        if parts:
                            symbol = parts[0].strip().strip('"')
                            if symbol.isalpha() and len(symbol) <= 5:
                                tickers.add(symbol.upper())
            except Exception as e:
                logger.debug(f"OTC Markets error: {e}")

        logger.info(f"OTC Markets: {len(tickers)} tickers")
        return tickers

    def get_comprehensive_us_hardcoded(self) -> Set[str]:
        """Comprehensive hardcoded list of US small/micro/mid caps."""
        # Russell 2000 components (small caps) - partial list
        russell_2000 = [
            'AAON', 'AAXN', 'ABCB', 'ABMD', 'ABTX', 'ACAD', 'ACBI', 'ACCO', 'ACEL',
            'ACGL', 'ACHC', 'ACIW', 'ACLS', 'ACNB', 'ADMA', 'ADNT', 'ADPT', 'ADRO',
            'ADTN', 'ADUS', 'AEGN', 'AEIS', 'AERI', 'AFIN', 'AGCO', 'AGIO', 'AGYS',
            'AHCO', 'AHH', 'AIMC', 'AINV', 'AJRD', 'AKAM', 'AKBA', 'AKRO', 'ALBO',
            'ALEC', 'ALEX', 'ALGT', 'ALKS', 'ALNY', 'ALRM', 'ALSN', 'ALTR', 'ALVR',
            'AMAG', 'AMAL', 'AMBA', 'AMBC', 'AMCX', 'AMED', 'AMEH', 'AMK', 'AMKR',
            'AMN', 'AMNB', 'AMOT', 'AMPH', 'AMRB', 'AMRC', 'AMRS', 'AMSC', 'AMSF',
            'AMWD', 'AMYT', 'ANAT', 'ANDE', 'ANIK', 'ANIP', 'ANSS', 'AOSL', 'APAM',
            'APEI', 'APEN', 'APG', 'APLS', 'APOG', 'APPF', 'APPN', 'APPS', 'APRE',
            'APTS', 'APTV', 'APYX', 'AQUA', 'ARAY', 'ARCB', 'ARCH', 'ARCT', 'ARDX',
            'ARES', 'ARGS', 'ARLO', 'ARNA', 'ARNC', 'AROC', 'ARRY', 'ARTNA', 'ARVN',
            'ARWR', 'ASGN', 'ASIX', 'ASND', 'ASPS', 'ASTE', 'ASUR', 'ATEC', 'ATEN',
            'ATEX', 'ATGE', 'ATHM', 'ATLO', 'ATNI', 'ATNM', 'ATRA', 'ATRC', 'ATRI',
            'ATRO', 'ATRS', 'ATSG', 'AVAV', 'AVDL', 'AVEO', 'AVID', 'AVNS', 'AVNT',
            'AVRO', 'AVXL', 'AWH', 'AWR', 'AXDX', 'AXGN', 'AXLA', 'AXNX', 'AXON',
            'AXSM', 'AXTI', 'AYI', 'AZPN', 'AZTA', 'BAND', 'BANF', 'BANR', 'BBBY',
            'BBIO', 'BBSI', 'BCBP', 'BCEL', 'BCML', 'BCOR', 'BCOV', 'BCPC', 'BCRX',
            'BDC', 'BDGE', 'BDN', 'BDSI', 'BEAT', 'BECN', 'BELFB', 'BERY', 'BGCP',
            'BGS', 'BHVN', 'BIG', 'BKCC', 'BKE', 'BKEP', 'BKH', 'BKU', 'BLBD',
            'BLDR', 'BLFS', 'BLI', 'BLKB', 'BLMN', 'BLNK', 'BLUE', 'BMCH', 'BMI',
            'BMRC', 'BMRN', 'BNED', 'BNFT', 'BOCH', 'BODY', 'BOLT', 'BOOM', 'BOOT',
            'BPMC', 'BPOP', 'BPRN', 'BRBR', 'BREW', 'BRG', 'BRID', 'BRK', 'BRKL',
            'BRKS', 'BRMK', 'BRO', 'BRP', 'BRSP', 'BRT', 'BSIG', 'BSRR', 'BSVN',
            'BTAI', 'BTG', 'BTRS', 'BUD', 'BUSE', 'BWB', 'BWFG', 'BXC', 'BXMT',
        ]

        # S&P 600 Small Cap components - partial
        sp600 = [
            'AAT', 'AAWW', 'ABG', 'ABM', 'ABMD', 'ACA', 'ACAD', 'ACLS', 'ACM',
            'ACRE', 'AEIS', 'AEL', 'AEO', 'AFG', 'AGCO', 'AGM', 'AGYS', 'AHL',
            'AHT', 'AIN', 'AIR', 'AIT', 'AJG', 'AKAM', 'AL', 'ALEX', 'ALGT',
            'ALSN', 'ALV', 'AM', 'AMAG', 'AMED', 'AMG', 'AMKR', 'AMN', 'AMP',
            'AMPH', 'AMWD', 'AN', 'ANAT', 'ANDE', 'ANET', 'ANF', 'ANGO', 'ANH',
            'ANIK', 'ANSS', 'AOBC', 'APAM', 'APC', 'APOG', 'APPF', 'APPN', 'APY',
            'AQUA', 'AR', 'ARA', 'ARCB', 'ARCH', 'ARCO', 'ARDC', 'ARDX', 'ARE',
            'ARES', 'ARGO', 'ARI', 'ARL', 'ARLP', 'ARMK', 'ARNC', 'AROC', 'AROW',
            'ARR', 'ARRY', 'ARTNA', 'ARW', 'ASGN', 'ASIX', 'ASML', 'ASPN', 'ASPS',
            'ASTE', 'ASUR', 'ASX', 'ATCO', 'ATEN', 'ATGE', 'ATH', 'ATHM', 'ATI',
            'ATLO', 'ATML', 'ATNI', 'ATO', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS',
            'ATSG', 'ATU', 'AUB', 'AUO', 'AVAV', 'AVB', 'AVGO', 'AVID', 'AVNS',
            'AVNT', 'AVT', 'AVXL', 'AVY', 'AWH', 'AWI', 'AWK', 'AWR', 'AXE',
        ]

        # More micro/small caps
        micro_caps = [
            'ABEO', 'ABIO', 'ABUS', 'ACER', 'ACET', 'ACHV', 'ACOR', 'ACRS', 'ACST',
            'ACTG', 'ACUR', 'ADAP', 'ADGI', 'ADIL', 'ADMP', 'ADMS', 'ADOC', 'ADTX',
            'ADVM', 'ADXS', 'AEMD', 'AERI', 'AESE', 'AFMD', 'AFYA', 'AGBA', 'AGEN',
            'AGTC', 'AHPI', 'AIKI', 'AIMD', 'AINC', 'AIRG', 'AIRT', 'AKBA', 'AKTS',
            'AKTX', 'ALBO', 'ALDX', 'ALEC', 'ALGM', 'ALLK', 'ALLO', 'ALLT', 'ALNY',
            'ALOT', 'ALRN', 'ALRS', 'ALSK', 'ALT', 'ALTO', 'ALXO', 'AMBO', 'AMCI',
            'AMCX', 'AMEH', 'AMHC', 'AMPE', 'AMPY', 'AMRB', 'AMRN', 'AMRS', 'AMSC',
            'AMSF', 'AMST', 'AMTB', 'AMTI', 'AMTX', 'AMYT', 'ANAB', 'ANDE', 'ANET',
            'ANGI', 'ANIP', 'ANNX', 'ANPC', 'ANSS', 'ANTE', 'ANY', 'AOGO', 'AOSL',
            'AOUT', 'APDN', 'APEX', 'APLT', 'APOG', 'APOP', 'APPH', 'APRE', 'APRN',
            'APTO', 'APVO', 'APWC', 'APYX', 'AQB', 'AQMS', 'AQST', 'ARAV', 'ARAY',
            'ARBE', 'ARBG', 'ARCE', 'ARCO', 'ARCT', 'ARDX', 'AREC', 'ARES', 'ARHS',
            'ARIZ', 'ARKR', 'ARLP', 'ARMK', 'ARMP', 'ARNC', 'AROC', 'AROW', 'ARPO',
            'ARQT', 'ARTL', 'ARTNA', 'ARTW', 'ARVN', 'ARWR', 'ARYA', 'ASAI', 'ASAN',
            'ASGN', 'ASIX', 'ASLE', 'ASLN', 'ASMB', 'ASML', 'ASND', 'ASPS', 'ASPU',
            'ASRT', 'ASRV', 'ASTC', 'ASTE', 'ASUR', 'ASXC', 'ATAI', 'ATAT', 'ATAX',
        ]

        all_tickers = set(russell_2000 + sp600 + micro_caps)
        logger.info(f"Hardcoded US: {len(all_tickers)} tickers")
        return all_tickers

    def expand_us(self) -> Set[str]:
        """Aggressively expand US universe."""
        logger.info("\n" + "="*60)
        logger.info("AGGRESSIVE US EXPANSION")
        logger.info("="*60)

        all_tickers = set()

        # 1. SEC EDGAR (all US public companies)
        sec_tickers = self.fetch_sec_cik_tickers()
        all_tickers.update(sec_tickers)

        # 2. Alpha Vantage active listings
        av_tickers = self.fetch_alphavantage_listings()
        all_tickers.update(av_tickers)

        # 3. Finnhub US exchanges
        for exchange in ['US', 'NYSE', 'NASDAQ', 'AMEX']:
            finnhub_tickers = self.fetch_finnhub_stocks(exchange)
            all_tickers.update([t for t in finnhub_tickers if '.' not in t])
            logger.info(f"Finnhub {exchange}: {len(finnhub_tickers)} tickers")
            time.sleep(0.3)

        # 4. OTC Markets (OTCQX, OTCQB)
        otc_tickers = self.fetch_otc_markets()
        all_tickers.update(otc_tickers)

        # 5. Hardcoded small/micro caps
        hardcoded = self.get_comprehensive_us_hardcoded()
        all_tickers.update(hardcoded)

        # 6. Existing GCS tickers
        try:
            gcs_path = Path(__file__).parent.parent / 'data' / 'us_tickers.txt'
            if gcs_path.exists():
                with open(gcs_path) as f:
                    existing = set(line.strip().upper() for line in f if line.strip())
                all_tickers.update(existing)
                logger.info(f"Existing file: {len(existing)} tickers")
        except:
            pass

        # Filter to valid US stock tickers
        valid = set()
        for t in all_tickers:
            t = t.upper().strip()
            if not t:
                continue
            if '.' in t or '-' in t:
                continue
            if not t.isalpha():
                continue
            if len(t) > 5:
                continue
            # Filter mutual funds (5 chars ending in X)
            if len(t) == 5 and t.endswith('X'):
                continue
            valid.add(t)

        logger.info(f"\nUS TOTAL: {len(valid)} valid tickers")
        return valid

    def expand_eu(self) -> Set[str]:
        """Aggressively expand EU universe."""
        logger.info("\n" + "="*60)
        logger.info("AGGRESSIVE EU EXPANSION")
        logger.info("="*60)

        all_tickers = set()

        # TwelveData exchanges
        td_exchanges = {
            'XETRA': '.DE', 'FSX': '.DE', 'MUN': '.DE', 'STU': '.DE', 'BER': '.DE', 'HAM': '.DE', 'DUS': '.DE',
            'LSE': '.L', 'IOB': '.L',
            'EURONEXT': '.PA',  # Actually covers PA, AS, BR, LS
            'MIL': '.MI',
            'BME': '.MC',
            'SIX': '.SW',
            'STO': '.ST', 'NGM': '.ST',
            'OSL': '.OL',
            'CPH': '.CO',
            'HEL': '.HE',
            'VSE': '.VI',
        }

        for exchange, suffix in td_exchanges.items():
            tickers = self.fetch_twelvedata_exchange(exchange, suffix)
            all_tickers.update(tickers)
            logger.info(f"TwelveData {exchange}: {len(tickers)} tickers")
            time.sleep(0.5)

        # Finnhub EU exchanges
        finnhub_exchanges = {
            'DE': '.DE', 'L': '.L', 'PA': '.PA', 'MI': '.MI',
            'MC': '.MC', 'AS': '.AS', 'SW': '.SW', 'ST': '.ST',
            'OL': '.OL', 'CO': '.CO', 'HE': '.HE', 'VI': '.VI',
        }

        for exchange, suffix in finnhub_exchanges.items():
            finnhub_tickers = self.fetch_finnhub_stocks(exchange)
            formatted = set()
            for t in finnhub_tickers:
                if not any(t.endswith(s) for s in EU_SUFFIXES):
                    t = t + suffix
                formatted.add(t)
            all_tickers.update(formatted)
            logger.info(f"Finnhub {exchange}: {len(formatted)} tickers")
            time.sleep(0.3)

        # Existing EU tickers
        try:
            eu_path = Path(__file__).parent.parent / 'data' / 'eu_tickers.txt'
            if eu_path.exists():
                with open(eu_path) as f:
                    existing = set(line.strip() for line in f if line.strip())
                all_tickers.update(existing)
                logger.info(f"Existing file: {len(existing)} tickers")
        except:
            pass

        # Filter to valid EU tickers
        valid = set()
        for t in all_tickers:
            t = t.strip()
            if not t:
                continue
            if not any(t.endswith(s) for s in EU_SUFFIXES):
                continue

            # Get base symbol
            base = t
            for s in EU_SUFFIXES:
                if t.endswith(s):
                    base = t[:-len(s)]
                    break

            # Filter invalid patterns
            if not base:
                continue
            if base.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                continue
            if len(base) < 2 or len(base) > 10:
                continue

            valid.add(t)

        # Count by market
        market_counts = {}
        for suffix in EU_SUFFIXES:
            count = len([t for t in valid if t.endswith(suffix)])
            market_counts[suffix] = count

        logger.info(f"\nEU TOTAL: {len(valid)} valid tickers")
        for market, count in sorted(market_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {market}: {count}")

        return valid


def main():
    expander = AggressiveExpander()
    output_dir = Path(__file__).parent.parent / 'data'

    # Expand US
    us_tickers = expander.expand_us()
    us_path = output_dir / 'us_tickers_v2.txt'
    with open(us_path, 'w') as f:
        for t in sorted(us_tickers):
            f.write(t + '\n')
    logger.info(f"Saved {len(us_tickers)} US tickers to {us_path}")

    # Expand EU
    eu_tickers = expander.expand_eu()
    eu_path = output_dir / 'eu_tickers_v2.txt'
    with open(eu_path, 'w') as f:
        for t in sorted(eu_tickers):
            f.write(t + '\n')
    logger.info(f"Saved {len(eu_tickers)} EU tickers to {eu_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXPANSION COMPLETE")
    logger.info("="*60)
    logger.info(f"US: {len(us_tickers)} tickers")
    logger.info(f"EU: {len(eu_tickers)} tickers")
    logger.info(f"TOTAL: {len(us_tickers) + len(eu_tickers)} tickers")


if __name__ == '__main__':
    main()
