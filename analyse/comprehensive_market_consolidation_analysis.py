"""
Comprehensive Market-Wide Consolidation Analysis System
Analyzes real stock market data from GCS using 4 different consolidation detection methods
Processes multiple tickers with full historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass, field
import warnings
from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DailyConsolidationMetrics:
    """Metrics for each day of analysis"""
    ticker: str
    date: str
    price: float
    
    # Consolidation detection results (True/False for each method)
    method1_bollinger: bool = False
    method2_range_based: bool = False
    method3_volume_weighted: bool = False
    method4_atr_based: bool = False
    
    # Consolidation boundaries for each method (if in consolidation)
    method1_upper: Optional[float] = None
    method1_lower: Optional[float] = None
    method1_power: Optional[float] = None
    
    method2_upper: Optional[float] = None
    method2_lower: Optional[float] = None
    method2_power: Optional[float] = None
    
    method3_upper: Optional[float] = None
    method3_lower: Optional[float] = None
    method3_power: Optional[float] = None
    
    method4_upper: Optional[float] = None
    method4_lower: Optional[float] = None
    method4_power: Optional[float] = None
    
    # Future price levels reached (actual values or NaN if not enough data)
    price_20d: Optional[float] = None
    price_40d: Optional[float] = None
    price_50d: Optional[float] = None
    price_70d: Optional[float] = None
    price_100d: Optional[float] = None
    
    # Maximum gains in future periods
    max_gain_20d: Optional[float] = None
    max_gain_40d: Optional[float] = None
    max_gain_50d: Optional[float] = None
    max_gain_70d: Optional[float] = None
    max_gain_100d: Optional[float] = None
    
    # Maximum downside risk in future periods
    max_loss_20d: Optional[float] = None
    max_loss_40d: Optional[float] = None
    max_loss_100d: Optional[float] = None
    
    # Additional metrics
    volume: Optional[float] = None
    volatility_20d: Optional[float] = None
    rsi_14: Optional[float] = None
    

class GCSDataLoader:
    """Loads stock market data from Google Cloud Storage"""
    
    def __init__(self, credentials_path: str):
        """Initialize GCS client with credentials"""
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = storage.Client(project="ignition-ki-csv-storage")
        self.bucket_name = "ignition-ki-csv-data-2025-user123"
        self.bucket = self.client.bucket(self.bucket_name)
        
    def list_available_tickers(self) -> List[str]:
        """List all available tickers in GCS"""
        tickers = set()
        
        # Check tickers directory
        blobs = self.bucket.list_blobs(prefix="tickers/")
        for blob in blobs:
            if blob.name.endswith('.csv'):
                ticker = blob.name.split('/')[-1].replace('.csv', '')
                tickers.add(ticker)
        
        # Check market_data directory
        blobs = self.bucket.list_blobs(prefix="market_data/")
        for blob in blobs:
            if blob.name.endswith('.csv'):
                ticker = blob.name.split('/')[-1].replace('.csv', '')
                tickers.add(ticker)
        
        logger.info(f"Found {len(tickers)} tickers in GCS")
        return sorted(list(tickers))
    
    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Load price data for a specific ticker"""
        try:
            # Try different paths
            paths = [
                f"tickers/{ticker}.csv",
                f"market_data/{ticker}.csv",
                f"ml_datasets/{ticker}_features.csv"
            ]
            
            for path in paths:
                try:
                    blob = self.bucket.blob(path)
                    if blob.exists():
                        content = blob.download_as_text()
                        df = pd.read_csv(io.StringIO(content))
                        
                        # Standardize column names
                        df.columns = [col.capitalize() for col in df.columns]
                        
                        # Ensure we have required columns
                        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in df.columns for col in required_cols):
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date')
                            return df
                except Exception as e:
                    continue
            
            logger.warning(f"Could not load data for {ticker}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            return pd.DataFrame()


class MarketWideConsolidationAnalyzer:
    """
    Analyzes consolidation patterns across entire market using 4 methods:
    1. Bollinger Band Width Method
    2. Range-Based Consolidation Method
    3. Volume-Weighted Consolidation Method
    4. ATR-Based Consolidation Method
    """
    
    # Minimum price filter
    MIN_PRICE_FILTER = 0.01
    
    # Minimum data requirements
    MIN_TRADING_DAYS = 252  # At least 1 year of data
    MIN_DATA_COMPLETENESS = 0.65  # At least 65% of days have data
    
    def __init__(self, gcs_loader: GCSDataLoader):
        """Initialize analyzer with GCS data loader"""
        self.gcs_loader = gcs_loader
        self.all_results = []
        self.ticker_summaries = {}
        
    def validate_ticker_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate if ticker data meets quality requirements"""
        if df.empty:
            return False
        
        # Check minimum price filter
        if df['Close'].min() < self.MIN_PRICE_FILTER:
            logger.info(f"Skipping {ticker}: price below ${self.MIN_PRICE_FILTER}")
            return False
        
        # Check data completeness
        date_range = (df['Date'].max() - df['Date'].min()).days
        if date_range > 0:
            data_completeness = len(df) / date_range
            if data_completeness < self.MIN_DATA_COMPLETENESS:
                logger.info(f"Skipping {ticker}: data completeness {data_completeness:.2%} below threshold")
                return False
        
        # Check minimum trading days
        if len(df) < self.MIN_TRADING_DAYS:
            logger.info(f"Skipping {ticker}: only {len(df)} trading days")
            return False
        
        return True
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for analysis"""
        
        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=15).mean()
        df['STD_20'] = df['Close'].rolling(window=20, min_periods=15).std()
        df['BB_Upper'] = df['SMA_20'] + (2 * df['STD_20'])
        df['BB_Lower'] = df['SMA_20'] - (2 * df['STD_20'])
        
        # Handle division by zero
        df['BB_Width'] = np.where(
            df['SMA_20'] != 0,
            (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20'],
            np.nan
        )
        
        # Calculate percentiles with proper window
        df['BB_Width_Percentile'] = df['BB_Width'].rolling(window=100, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        # ATR (Average True Range)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['TR'].rolling(window=14, min_periods=10).mean()
        df['ATR_Percentile'] = df['ATR_14'].rolling(window=100, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        # Volume metrics
        df['Volume_MA_20'] = df['Volume'].rolling(window=20, min_periods=15).mean()
        df['Volume_Ratio'] = np.where(
            df['Volume_MA_20'] > 0,
            df['Volume'] / df['Volume_MA_20'],
            np.nan
        )
        
        # Price range metrics
        df['Daily_Range'] = np.where(
            df['Close'] > 0,
            (df['High'] - df['Low']) / df['Close'],
            np.nan
        )
        df['Range_MA_20'] = df['Daily_Range'].rolling(window=20, min_periods=15).mean()
        df['Range_Ratio'] = np.where(
            df['Range_MA_20'] > 0,
            df['Daily_Range'] / df['Range_MA_20'],
            np.nan
        )
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=10).mean()
        rs = np.where(loss != 0, gain / loss, np.nan)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_20'] = df['Returns'].rolling(window=20, min_periods=15).std() * np.sqrt(252)
        
        # ADX (for trend strength)
        df['DM_Plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        df['DM_Minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        df['DI_Plus'] = np.where(
            df['ATR_14'] > 0,
            100 * (df['DM_Plus'].rolling(14, min_periods=10).mean() / df['ATR_14']),
            np.nan
        )
        df['DI_Minus'] = np.where(
            df['ATR_14'] > 0,
            100 * (df['DM_Minus'].rolling(14, min_periods=10).mean() / df['ATR_14']),
            np.nan
        )
        
        df['DX'] = np.where(
            (df['DI_Plus'] + df['DI_Minus']) > 0,
            100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus']),
            np.nan
        )
        df['ADX'] = df['DX'].rolling(14, min_periods=10).mean()
        
        return df
    
    def detect_consolidation_method1_bollinger(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Optional[Dict]]:
        """
        Method 1: Bollinger Band Width Consolidation
        Consolidation when BB Width is in bottom 30th percentile
        """
        if idx < 100:  # Need enough history
            return False, None
            
        row = df.iloc[idx]
        
        # Check if BB Width is in bottom 30th percentile
        if pd.notna(row['BB_Width_Percentile']) and row['BB_Width_Percentile'] < 0.30:
            # Calculate consolidation boundaries
            window_data = df.iloc[max(0, idx-20):idx+1]
            
            boundaries = {
                'upper': window_data['High'].max(),
                'lower': window_data['Low'].min(),
                'power': window_data['High'].max() * 1.005  # 0.5% above upper
            }
            
            return True, boundaries
        
        return False, None
    
    def detect_consolidation_method2_range(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Optional[Dict]]:
        """
        Method 2: Range-Based Consolidation
        Consolidation when daily range < 65% of 20-day average and ADX < 32
        """
        if idx < 20:
            return False, None
            
        row = df.iloc[idx]
        
        # Check range and ADX conditions
        if (pd.notna(row['Range_Ratio']) and row['Range_Ratio'] < 0.65 and
            pd.notna(row['ADX']) and row['ADX'] < 32):
            
            # Calculate consolidation boundaries over last 10 days
            window_data = df.iloc[max(0, idx-10):idx+1]
            
            boundaries = {
                'upper': window_data['High'].max(),
                'lower': window_data['Low'].min(),
                'power': window_data['High'].max() * 1.005
            }
            
            # Additional check: range should be < 15% of price
            if row['Close'] > 0 and (boundaries['upper'] - boundaries['lower']) / row['Close'] < 0.15:
                return True, boundaries
        
        return False, None
    
    def detect_consolidation_method3_volume(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Optional[Dict]]:
        """
        Method 3: Volume-Weighted Consolidation
        Consolidation when volume < 35% of 20-day average and price range is tight
        """
        if idx < 20:
            return False, None
            
        row = df.iloc[idx]
        
        # Check volume condition
        if pd.notna(row['Volume_Ratio']) and row['Volume_Ratio'] < 0.35:
            # Check if price is in tight range (last 15 days)
            window_data = df.iloc[max(0, idx-15):idx+1]
            
            if row['Close'] > 0:
                price_range = (window_data['High'].max() - window_data['Low'].min()) / row['Close']
                
                if price_range < 0.12:  # Less than 12% range
                    boundaries = {
                        'upper': window_data['High'].max(),
                        'lower': window_data['Low'].min(),
                        'power': window_data['High'].max() * 1.005
                    }
                    return True, boundaries
        
        return False, None
    
    def detect_consolidation_method4_atr(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Optional[Dict]]:
        """
        Method 4: ATR-Based Consolidation
        Consolidation when ATR is in bottom 30th percentile and volatility is low
        """
        if idx < 100:
            return False, None
            
        row = df.iloc[idx]
        
        # Check ATR and volatility conditions
        if (pd.notna(row['ATR_Percentile']) and row['ATR_Percentile'] < 0.30 and
            pd.notna(row['Volatility_20']) and row['Volatility_20'] < 0.40):  # Less than 40% annualized vol
            
            # Calculate boundaries using ATR
            window_data = df.iloc[max(0, idx-20):idx+1]
            current_atr = row['ATR_14']
            
            if pd.notna(current_atr) and current_atr > 0:
                boundaries = {
                    'upper': row['Close'] + (1.5 * current_atr),
                    'lower': row['Close'] - (1.5 * current_atr),
                    'power': (row['Close'] + (1.5 * current_atr)) * 1.005
                }
                
                # Validate with actual price range
                actual_upper = window_data['High'].max()
                actual_lower = window_data['Low'].min()
                
                # Use the tighter of the two
                boundaries['upper'] = min(boundaries['upper'], actual_upper)
                boundaries['lower'] = max(boundaries['lower'], actual_lower)
                boundaries['power'] = boundaries['upper'] * 1.005
                
                return True, boundaries
        
        return False, None
    
    def calculate_future_metrics(self, df: pd.DataFrame, idx: int) -> Dict:
        """Calculate future price levels and risk metrics"""
        metrics = {}
        current_price = df.iloc[idx]['Close']
        
        if current_price <= 0:
            # Return NaN for all metrics if price is invalid
            for days in [20, 40, 50, 70, 100]:
                metrics[f'price_{days}d'] = np.nan
                metrics[f'max_gain_{days}d'] = np.nan
            for days in [20, 40, 100]:
                metrics[f'max_loss_{days}d'] = np.nan
            return metrics
        
        # Future price levels
        for days in [20, 40, 50, 70, 100]:
            future_idx = idx + days
            if future_idx < len(df):
                metrics[f'price_{days}d'] = df.iloc[future_idx]['Close']
                
                # Calculate max gain in period
                future_window = df.iloc[idx+1:min(future_idx+1, len(df))]
                if not future_window.empty:
                    max_price = future_window['High'].max()
                    metrics[f'max_gain_{days}d'] = ((max_price - current_price) / current_price) * 100
            else:
                metrics[f'price_{days}d'] = np.nan
                metrics[f'max_gain_{days}d'] = np.nan
        
        # Downside risk (max loss in periods)
        for days in [20, 40, 100]:
            future_idx = idx + days
            if future_idx < len(df):
                future_window = df.iloc[idx+1:min(future_idx+1, len(df))]
                if not future_window.empty:
                    min_price = future_window['Low'].min()
                    metrics[f'max_loss_{days}d'] = ((min_price - current_price) / current_price) * 100
            else:
                metrics[f'max_loss_{days}d'] = np.nan
        
        return metrics
    
    def analyze_ticker(self, ticker: str) -> pd.DataFrame:
        """Perform comprehensive analysis for a ticker"""
        logger.info(f"Analyzing {ticker}")
        
        # Load data from GCS
        df = self.gcs_loader.load_ticker_data(ticker)
        
        # Validate data
        if not self.validate_ticker_data(df, ticker):
            return pd.DataFrame()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Analyze each day
        results = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Skip if price below minimum filter
            if row['Close'] < self.MIN_PRICE_FILTER:
                continue
            
            # Create daily metrics
            metrics = DailyConsolidationMetrics(
                ticker=ticker,
                date=row['Date'].strftime('%Y-%m-%d'),
                price=row['Close'],
                volume=row['Volume'],
                volatility_20d=row.get('Volatility_20'),
                rsi_14=row.get('RSI_14')
            )
            
            # Method 1: Bollinger Band Width
            is_consolidation, boundaries = self.detect_consolidation_method1_bollinger(df, idx)
            metrics.method1_bollinger = is_consolidation
            if boundaries:
                metrics.method1_upper = boundaries['upper']
                metrics.method1_lower = boundaries['lower']
                metrics.method1_power = boundaries['power']
            
            # Method 2: Range-Based
            is_consolidation, boundaries = self.detect_consolidation_method2_range(df, idx)
            metrics.method2_range_based = is_consolidation
            if boundaries:
                metrics.method2_upper = boundaries['upper']
                metrics.method2_lower = boundaries['lower']
                metrics.method2_power = boundaries['power']
            
            # Method 3: Volume-Weighted
            is_consolidation, boundaries = self.detect_consolidation_method3_volume(df, idx)
            metrics.method3_volume_weighted = is_consolidation
            if boundaries:
                metrics.method3_upper = boundaries['upper']
                metrics.method3_lower = boundaries['lower']
                metrics.method3_power = boundaries['power']
            
            # Method 4: ATR-Based
            is_consolidation, boundaries = self.detect_consolidation_method4_atr(df, idx)
            metrics.method4_atr_based = is_consolidation
            if boundaries:
                metrics.method4_upper = boundaries['upper']
                metrics.method4_lower = boundaries['lower']
                metrics.method4_power = boundaries['power']
            
            # Calculate future metrics
            future_metrics = self.calculate_future_metrics(df, idx)
            for key, value in future_metrics.items():
                setattr(metrics, key, value)
            
            results.append(metrics)
        
        # Convert to DataFrame
        if results:
            results_df = pd.DataFrame([vars(m) for m in results])
            
            # Store summary for this ticker
            self.ticker_summaries[ticker] = {
                'total_days': len(results_df),
                'date_range': f"{results_df['date'].min()} to {results_df['date'].max()}",
                'consolidation_days': {
                    'method1': results_df['method1_bollinger'].sum(),
                    'method2': results_df['method2_range_based'].sum(),
                    'method3': results_df['method3_volume_weighted'].sum(),
                    'method4': results_df['method4_atr_based'].sum()
                }
            }
            
            return results_df
        
        return pd.DataFrame()
    
    def analyze_market(self, max_tickers: int = None, parallel_workers: int = 5) -> pd.DataFrame:
        """Analyze all available tickers in the market"""
        # Get list of available tickers
        tickers = self.gcs_loader.list_available_tickers()
        
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        logger.info(f"Analyzing {len(tickers)} tickers")
        
        all_results = []
        
        # Process tickers in parallel
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_ticker, ticker): ticker 
                              for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results_df = future.result()
                    if not results_df.empty:
                        all_results.append(results_df)
                        logger.info(f"Completed {ticker}: {len(results_df)} days analyzed")
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Total records analyzed: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def export_comprehensive_results(self, results_df: pd.DataFrame, output_prefix: str = "market_consolidation"):
        """Export comprehensive analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export full dataset to CSV (can handle large data)
        csv_filename = f"{output_prefix}_full_data_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        logger.info(f"Full dataset saved to {csv_filename}")
        
        # Export summary to Excel with multiple sheets
        excel_filename = f"{output_prefix}_summary_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Overall summary
            summary_data = {
                'Metric': [
                    'Total Tickers Analyzed',
                    'Total Days Analyzed',
                    'Date Range',
                    'Method 1 (Bollinger) Total Days',
                    'Method 2 (Range) Total Days',
                    'Method 3 (Volume) Total Days',
                    'Method 4 (ATR) Total Days',
                    'Average Future Gain (20d)',
                    'Average Future Gain (40d)',
                    'Average Future Loss (20d)',
                    'Average Future Loss (40d)'
                ],
                'Value': [
                    results_df['ticker'].nunique(),
                    len(results_df),
                    f"{results_df['date'].min()} to {results_df['date'].max()}",
                    results_df['method1_bollinger'].sum(),
                    results_df['method2_range_based'].sum(),
                    results_df['method3_volume_weighted'].sum(),
                    results_df['method4_atr_based'].sum(),
                    results_df['max_gain_20d'].mean(),
                    results_df['max_gain_40d'].mean(),
                    results_df['max_loss_20d'].mean(),
                    results_df['max_loss_40d'].mean()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
            
            # Per ticker summary
            ticker_summary = []
            for ticker in results_df['ticker'].unique():
                ticker_data = results_df[results_df['ticker'] == ticker]
                ticker_summary.append({
                    'Ticker': ticker,
                    'Days': len(ticker_data),
                    'Method1_Days': ticker_data['method1_bollinger'].sum(),
                    'Method2_Days': ticker_data['method2_range_based'].sum(),
                    'Method3_Days': ticker_data['method3_volume_weighted'].sum(),
                    'Method4_Days': ticker_data['method4_atr_based'].sum(),
                    'Avg_Gain_20d': ticker_data['max_gain_20d'].mean(),
                    'Avg_Loss_20d': ticker_data['max_loss_20d'].mean()
                })
            
            ticker_summary_df = pd.DataFrame(ticker_summary)
            ticker_summary_df.to_excel(writer, sheet_name='Ticker_Summary', index=False)
            
            # Method comparison
            method_comparison = pd.DataFrame({
                'Method': ['Bollinger Band Width', 'Range-Based', 'Volume-Weighted', 'ATR-Based'],
                'Total_Consolidation_Days': [
                    results_df['method1_bollinger'].sum(),
                    results_df['method2_range_based'].sum(),
                    results_df['method3_volume_weighted'].sum(),
                    results_df['method4_atr_based'].sum()
                ],
                'Percentage_of_Days': [
                    results_df['method1_bollinger'].mean() * 100,
                    results_df['method2_range_based'].mean() * 100,
                    results_df['method3_volume_weighted'].mean() * 100,
                    results_df['method4_atr_based'].mean() * 100
                ],
                'Avg_Future_Gain_When_Consolidating': [
                    results_df[results_df['method1_bollinger'] == True]['max_gain_20d'].mean(),
                    results_df[results_df['method2_range_based'] == True]['max_gain_20d'].mean(),
                    results_df[results_df['method3_volume_weighted'] == True]['max_gain_20d'].mean(),
                    results_df[results_df['method4_atr_based'] == True]['max_gain_20d'].mean()
                ]
            })
            method_comparison.to_excel(writer, sheet_name='Method_Comparison', index=False)
            
            # Sample of daily data (first 10000 rows to avoid Excel limitations)
            sample_df = results_df.head(10000)
            sample_df.to_excel(writer, sheet_name='Sample_Daily_Data', index=False)
        
        logger.info(f"Summary saved to {excel_filename}")
        
        # Export JSON summary
        json_summary = {
            'analysis_timestamp': timestamp,
            'total_tickers': int(results_df['ticker'].nunique()),
            'total_records': len(results_df),
            'date_range': {
                'start': results_df['date'].min(),
                'end': results_df['date'].max()
            },
            'consolidation_statistics': {
                'method1_bollinger': {
                    'total_days': int(results_df['method1_bollinger'].sum()),
                    'percentage': float(results_df['method1_bollinger'].mean() * 100)
                },
                'method2_range_based': {
                    'total_days': int(results_df['method2_range_based'].sum()),
                    'percentage': float(results_df['method2_range_based'].mean() * 100)
                },
                'method3_volume_weighted': {
                    'total_days': int(results_df['method3_volume_weighted'].sum()),
                    'percentage': float(results_df['method3_volume_weighted'].mean() * 100)
                },
                'method4_atr_based': {
                    'total_days': int(results_df['method4_atr_based'].sum()),
                    'percentage': float(results_df['method4_atr_based'].mean() * 100)
                }
            },
            'ticker_summaries': self.ticker_summaries
        }
        
        json_filename = f"{output_prefix}_summary_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        logger.info(f"JSON summary saved to {json_filename}")
        
        return csv_filename, excel_filename, json_filename


def main():
    """Main execution function"""
    logger.info("Starting Comprehensive Market-Wide Consolidation Analysis")
    
    # Set up GCS credentials
    credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
    
    try:
        # Initialize GCS loader
        logger.info("Initializing GCS connection...")
        gcs_loader = GCSDataLoader(credentials_path)
        
        # Initialize analyzer
        analyzer = MarketWideConsolidationAnalyzer(gcs_loader)
        
        # Analyze market (limit to first 20 tickers for initial test)
        logger.info("Starting market analysis...")
        results_df = analyzer.analyze_market(max_tickers=20, parallel_workers=5)
        
        if not results_df.empty:
            # Export results
            csv_file, excel_file, json_file = analyzer.export_comprehensive_results(results_df)
            
            # Print summary statistics
            print("\n" + "="*60)
            print("MARKET-WIDE CONSOLIDATION ANALYSIS COMPLETE")
            print("="*60)
            print(f"Total Tickers Analyzed: {results_df['ticker'].nunique()}")
            print(f"Total Trading Days Analyzed: {len(results_df):,}")
            print(f"Date Range: {results_df['date'].min()} to {results_df['date'].max()}")
            
            print("\n--- Consolidation Detection Results ---")
            print(f"Method 1 (Bollinger): {results_df['method1_bollinger'].sum():,} days ({results_df['method1_bollinger'].mean()*100:.2f}%)")
            print(f"Method 2 (Range-Based): {results_df['method2_range_based'].sum():,} days ({results_df['method2_range_based'].mean()*100:.2f}%)")
            print(f"Method 3 (Volume-Weighted): {results_df['method3_volume_weighted'].sum():,} days ({results_df['method3_volume_weighted'].mean()*100:.2f}%)")
            print(f"Method 4 (ATR-Based): {results_df['method4_atr_based'].sum():,} days ({results_df['method4_atr_based'].mean()*100:.2f}%)")
            
            print("\n--- Average Future Performance ---")
            print(f"Average 20-day gain: {results_df['max_gain_20d'].mean():.2f}%")
            print(f"Average 40-day gain: {results_df['max_gain_40d'].mean():.2f}%")
            print(f"Average 20-day loss: {results_df['max_loss_20d'].mean():.2f}%")
            print(f"Average 40-day loss: {results_df['max_loss_40d'].mean():.2f}%")
            
            print("\n--- Output Files ---")
            print(f"Full Data CSV: {csv_file}")
            print(f"Summary Excel: {excel_file}")
            print(f"JSON Summary: {json_file}")
            
        else:
            logger.warning("No data available for analysis")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()