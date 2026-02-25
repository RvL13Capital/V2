"""
Pytest configuration for TRANS tests.

Loads environment variables from .env file before tests run.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / '.env', override=True)

# Verify GCS credentials are set
gcs_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if gcs_creds:
    print(f"GCS credentials loaded: {gcs_creds}")
