"""
Local Development Startup Script
=================================

Start TRANS system locally without Docker for development/testing.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import signal
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_requirements():
    """Check if required packages are installed."""
    print("Checking requirements...")

    try:
        import numpy
        import pandas
        import torch
        import sqlalchemy
        import fastapi
        print("  ✓ Core packages installed")
        return True
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        print("  pip install -r api/requirements.txt")
        return False


def setup_environment():
    """Setup environment variables from .env file."""
    print("Setting up environment...")

    env_file = Path(".env")
    if not env_file.exists():
        # Copy from example if it doesn't exist
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            print("  ✓ Created .env from .env.example")
            print("  ⚠ Please edit .env with your configuration")
        else:
            print("  ✗ No .env file found")
            return False

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    print("  ✓ Environment variables loaded")

    # Set default for local development
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("DATABASE_URL", "sqlite:///trans_development.db")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")

    return True


def initialize_database():
    """Initialize database if needed."""
    print("Initializing database...")

    try:
        from database.migrate import run_migration
        success = run_migration()
        if success:
            print("  ✓ Database initialized")
        else:
            print("  ✗ Database initialization failed")
        return success
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def start_api_server():
    """Start FastAPI server in a subprocess."""
    print("Starting API server...")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        print("  ✓ API server started (PID: {})".format(process.pid))
        print("  → Access at: http://localhost:8000")
        print("  → API docs: http://localhost:8000/docs")
        return process
    except Exception as e:
        print(f"  ✗ Failed to start API: {e}")
        return None


def monitor_process(process, name):
    """Monitor a subprocess and print its output."""
    def print_output():
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.strip()}")

    thread = threading.Thread(target=print_output, daemon=True)
    thread.start()
    return thread


def wait_for_api():
    """Wait for API to be ready."""
    print("Waiting for API to be ready...")

    import requests
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("  ✓ API is ready")
                return True
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"  ... waiting ({i}/{max_attempts})")

    print("  ✗ API failed to start")
    return False


def run_quick_test():
    """Run a quick test to verify system is working."""
    print("\nRunning quick test...")

    try:
        import requests
        import json

        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ Health check passed")
            print(f"    Status: {health.get('status')}")
            print(f"    Database: {health.get('database_status')}")
        else:
            print(f"  ✗ Health check failed: {response.status_code}")

        # Test root endpoint
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            info = response.json()
            print(f"  ✓ API info retrieved")
            print(f"    Service: {info.get('service')}")
            print(f"    Version: {info.get('version')}")
            print(f"    Labeling: {info.get('labeling_version')}")
        else:
            print(f"  ✗ API info failed: {response.status_code}")

        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Main startup sequence."""
    print("=" * 60)
    print("TRANS LOCAL DEVELOPMENT STARTUP")
    print("=" * 60)

    # Step 1: Check requirements
    if not check_requirements():
        return 1

    # Step 2: Setup environment
    if not setup_environment():
        return 1

    # Step 3: Initialize database
    if not initialize_database():
        print("⚠ Database initialization failed - continuing anyway")

    # Step 4: Start API server
    api_process = start_api_server()
    if not api_process:
        return 1

    # Monitor API output
    monitor_thread = monitor_process(api_process, "API")

    # Step 5: Wait for API to be ready
    if not wait_for_api():
        api_process.terminate()
        return 1

    # Step 6: Run quick test
    run_quick_test()

    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("API Server:  http://localhost:8000")
    print("API Docs:    http://localhost:8000/docs")
    print("Health:      http://localhost:8000/health")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    # Keep running until interrupted
    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        api_process.terminate()
        time.sleep(1)
        if api_process.poll() is None:
            api_process.kill()
        print("Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())