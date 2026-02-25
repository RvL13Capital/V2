"""
System Health Check Script
==========================

Quick health check for TRANS production system components.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    errors = []

    required_modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('fastapi', 'FastAPI'),
        ('pydantic', 'Pydantic')
    ]

    for module, name in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            errors.append(f"  ✗ {name}: {e}")
            print(f"  ✗ {name}: Missing")

    return len(errors) == 0, errors


def check_configuration():
    """Check if configuration is valid."""
    print("\nChecking configuration...")
    errors = []

    try:
        from config.production import get_config
        config = get_config()

        print(f"  ✓ Configuration loaded")
        print(f"    Environment: {config.environment.value}")
        print(f"    Database: {config.database.url[:30]}...")

        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            for error in validation_errors:
                errors.append(f"    Config error: {error}")
                print(f"  ⚠ {error}")
        else:
            print(f"  ✓ Configuration valid")

    except Exception as e:
        errors.append(f"Configuration error: {e}")
        print(f"  ✗ Configuration error: {e}")

    return len(errors) == 0, errors


def check_database():
    """Check database connectivity."""
    print("\nChecking database...")
    errors = []

    try:
        from database.connection import db_manager

        # Initialize database
        db_manager.initialize()
        print(f"  ✓ Database initialized")

        # Run health check
        health = db_manager.health_check()
        if health['status'] == 'healthy':
            print(f"  ✓ Database healthy")
            print(f"    Pattern count: {health.get('pattern_count', 0)}")
            print(f"    Prediction count: {health.get('prediction_count', 0)}")
            print(f"    Model count: {health.get('model_count', 0)}")
        else:
            errors.append(f"Database unhealthy: {health.get('error')}")
            print(f"  ✗ Database unhealthy: {health.get('error')}")

        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(db_manager.engine)
        tables = inspector.get_table_names()
        required_tables = ['patterns', 'predictions', 'model_versions']

        for table in required_tables:
            if table in tables:
                print(f"  ✓ Table exists: {table}")
            else:
                errors.append(f"Missing table: {table}")
                print(f"  ✗ Missing table: {table}")

    except Exception as e:
        errors.append(f"Database error: {e}")
        print(f"  ✗ Database error: {e}")

    return len(errors) == 0, errors


def check_models():
    """Check model loading capability."""
    print("\nChecking models...")
    errors = []

    try:
        from models.temporal_hybrid_v18 import HybridFeatureNetwork

        # Create model instance
        model = HybridFeatureNetwork()
        print(f"  ✓ Model instantiated")
        print(f"    Architecture: HybridFeatureNetwork")
        print(f"    Labeling version: v17")
        print(f"    Output classes: {model.num_classes}")

        # Check model manager
        from models.model_manager import get_model_manager
        manager = get_model_manager()
        print(f"  ✓ Model manager initialized")

    except Exception as e:
        errors.append(f"Model error: {e}")
        print(f"  ✗ Model error: {e}")

    return len(errors) == 0, errors


def check_scanner():
    """Check pattern scanner."""
    print("\nChecking pattern scanner...")
    errors = []

    try:
        from core.pattern_scanner import ConsolidationPatternScanner
        from config import MIN_LIQUIDITY_DOLLAR

        scanner = ConsolidationPatternScanner()
        print(f"  ✓ Scanner initialized")
        print(f"    Min liquidity: ${MIN_LIQUIDITY_DOLLAR:,.0f}")
        print(f"    Labeling: {scanner.labeling_version}")

    except Exception as e:
        errors.append(f"Scanner error: {e}")
        print(f"  ✗ Scanner error: {e}")

    return len(errors) == 0, errors


def check_api():
    """Check if API can be imported."""
    print("\nChecking API...")
    errors = []

    try:
        from api.main import app
        print(f"  ✓ API application loaded")
        print(f"    Title: {app.title}")
        print(f"    Version: {app.version}")

        # List endpoints
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)

        print(f"    Endpoints: {len(routes)}")
        for route in routes[:5]:  # Show first 5
            print(f"      {route}")

    except Exception as e:
        errors.append(f"API error: {e}")
        print(f"  ✗ API error: {e}")

    return len(errors) == 0, errors


def check_logging():
    """Check logging system."""
    print("\nChecking logging...")
    errors = []

    try:
        from utils.logging_config import get_production_logger_manager

        logger_mgr = get_production_logger_manager()
        logger = logger_mgr.get_logger("health_check", "test")

        print(f"  ✓ Logging initialized")
        print(f"    Log level: {logger_mgr.log_level}")
        print(f"    Log directory: {logger_mgr.log_dir}")
        print(f"    JSON format: {logger_mgr.use_json_format}")

        # Test logging
        with logger_mgr.request_context() as request_id:
            logger.info("Health check test", request_id=request_id)
            print(f"  ✓ Test log written (request_id: {request_id[:8]}...)")

    except Exception as e:
        errors.append(f"Logging error: {e}")
        print(f"  ✗ Logging error: {e}")

    return len(errors) == 0, errors


def run_health_check():
    """Run complete health check."""
    print("=" * 60)
    print("TRANS SYSTEM HEALTH CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    all_errors = []
    checks = [
        ("Imports", check_imports),
        ("Configuration", check_configuration),
        ("Database", check_database),
        ("Models", check_models),
        ("Scanner", check_scanner),
        ("API", check_api),
        ("Logging", check_logging)
    ]

    results = {}
    for name, check_func in checks:
        try:
            success, errors = check_func()
            results[name] = success
            if errors:
                all_errors.extend(errors)
        except Exception as e:
            results[name] = False
            all_errors.append(f"{name} check failed: {e}")
            print(f"\n{name} check failed with exception: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("HEALTH CHECK SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:20} {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if all_errors:
        print("\nErrors encountered:")
        for error in all_errors:
            print(f"  - {error}")

    # Write results to file
    results_file = Path("health_check_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "errors": all_errors,
            "passed": passed,
            "total": total
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return passed == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="System health check")
    parser.add_argument("--quick", action="store_true", help="Quick check (imports only)")
    args = parser.parse_args()

    if args.quick:
        success, _ = check_imports()
        sys.exit(0 if success else 1)
    else:
        success = run_health_check()
        sys.exit(0 if success else 1)