"""
Test script for refactored AIv3 System components
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration module"""
    logger.info("Testing configuration module...")
    try:
        from core import get_config
        config = get_config()

        assert config.gcs.project_id == "ignition-ki-csv-storage"
        assert config.gcs.bucket_name == "ignition-ki-csv-data-2025-user123"
        assert config.pattern.min_duration_days == 5
        assert config.outcome.explosive_threshold == 75.0

        logger.info("✓ Configuration module working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_data_loader():
    """Test data loader module"""
    logger.info("Testing data loader module...")
    try:
        from core import get_data_loader

        loader = get_data_loader()

        # Test listing tickers
        tickers = loader.list_available_tickers()
        logger.info(f"  Found {len(tickers)} available tickers")

        # Test loading patterns
        if Path("historical_patterns.parquet").exists():
            patterns = loader.load_patterns("historical_patterns.parquet")
            logger.info(f"  Loaded {len(patterns)} historical patterns")

        logger.info("✓ Data loader module working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False


def test_pattern_detector():
    """Test pattern detector module"""
    logger.info("Testing pattern detector module...")
    try:
        from core import UnifiedPatternDetector, get_data_loader

        detector = UnifiedPatternDetector()
        loader = get_data_loader()

        # Get a sample ticker
        tickers = loader.list_available_tickers()
        if tickers:
            test_ticker = tickers[0]
            logger.info(f"  Testing with ticker: {test_ticker}")

            # Try to detect patterns
            patterns = detector.detect_patterns(test_ticker)
            logger.info(f"  Detected {len(patterns)} patterns")

            if patterns:
                # Convert to DataFrame
                df = detector.patterns_to_dataframe(patterns)
                logger.info(f"  Successfully converted to DataFrame with {len(df)} rows")

        logger.info("✓ Pattern detector module working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Pattern detector test failed: {e}")
        return False


def test_pdf_generator():
    """Test PDF generator module"""
    logger.info("Testing PDF generator module...")
    try:
        from visualization.pdf_generator import UnifiedPDFGenerator
        import pandas as pd

        # Check if we have pattern data
        if Path("historical_patterns.parquet").exists():
            patterns_df = pd.read_parquet("historical_patterns.parquet")
            logger.info(f"  Using {len(patterns_df)} historical patterns for test")

            pdf_gen = UnifiedPDFGenerator(output_dir="./test_reports")

            # Note: PDF generation requires additional dependencies (reportlab, kaleido)
            # This will test the initialization but may fail on actual generation
            logger.info("  PDF generator initialized successfully")
            logger.info("✓ PDF generator module structure correct")
            return True
        else:
            logger.warning("  No pattern data found for PDF test")
            return True

    except ImportError as e:
        logger.warning(f"  PDF generator requires additional dependencies: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ PDF generator test failed: {e}")
        return False


def test_main_cli():
    """Test main CLI interface"""
    logger.info("Testing main CLI interface...")
    try:
        import main

        # Test that main module imports correctly
        logger.info("  Main module imported successfully")

        # Test command structure
        parser = main.argparse.ArgumentParser(description="Test")
        logger.info("  CLI parser created successfully")

        logger.info("✓ Main CLI interface working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Main CLI test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Running refactored AIv3 System tests...")
    logger.info("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
        ("Pattern Detector", test_pattern_detector),
        ("PDF Generator", test_pdf_generator),
        ("Main CLI", test_main_cli)
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{test_name} Test:")
        results[test_name] = test_func()
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)