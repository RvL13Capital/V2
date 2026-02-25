
import unittest
import pandas as pd
import numpy as np
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from core.pattern_scanner import ConsolidationPatternScanner

class TestConsolidationPatternScanner(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for tests
        self.test_dir = Path(tempfile.mkdtemp())
        self.scanner = ConsolidationPatternScanner(
            min_liquidity_dollar=0,  # disable liquidity check for synthetic data
            enable_market_cap=False # disable external calls
        )
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_synthetic_data(self, pattern_type="flat"):
        """Create synthetic OHLCV data with specific patterns"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Base price
        price = 100.0
        data = []
        
        for i, date in enumerate(dates):
            # 0-50: Random movement
            if i < 50:
                price += np.random.normal(0, 1.0)
            
            # 50-70: Qualification (Tight range)
            elif 50 <= i < 70:
                if pattern_type == "valid":
                    # Tight consolidation
                    price += np.random.normal(0, 0.2)
                elif pattern_type == "volatile":
                    # Too volatile
                    price += np.random.normal(0, 2.0)
                    
            # 70-150: Active phase
            elif 70 <= i < 150:
                if pattern_type == "valid":
                    # Stay within range
                    price += np.random.normal(0, 0.3)
                elif pattern_type == "breakout":
                    if i == 100:
                        price = 120.0 # Huge jump
            
            # 150+: Outcome
            else:
                price += np.random.normal(0, 1.0)
                
            # Create candles
            high = price + abs(np.random.normal(0, 0.5))
            low = price - abs(np.random.normal(0, 0.5))
            close = price
            volume = 1000000
            
            data.append({
                'date': date,
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df

    def test_scanner_initialization(self):
        """Test scanner initializes with correct defaults"""
        self.assertEqual(self.scanner.qualifying_days, 10)
        self.assertFalse(self.scanner.enable_market_cap)

    def test_scan_synthetic_valid(self):
        """Test scanning on synthetic valid data"""
        df = self.create_synthetic_data(pattern_type="valid")
        # Ensure we have enough data (min_data_days=100)
        
        # We need to mock _load_and_optimize or pass df directly
        # scan_ticker accepts df
        result = self.scanner.scan_ticker("SYNTH", df=df)
        
        # We expect some patterns if our synthetic data is good enough
        # But sleeper_scanner logic is complex (BBW, ADX etc)
        # So mainly we check it doesn't crash and returns a result object
        self.assertIsNotNone(result)
        self.assertEqual(result.ticker, "SYNTH")
        self.assertTrue(result.success)

    def test_scan_insufficient_data(self):
        """Test scanning with too little data"""
        df = self.create_synthetic_data().iloc[:50]
        
        result = self.scanner.scan_ticker("SHORT", df=df)
        self.assertFalse(result.success)
        self.assertIn("Insufficient data", result.error)

if __name__ == '__main__':
    unittest.main()
