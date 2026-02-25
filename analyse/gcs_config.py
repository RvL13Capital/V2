"""
GCS Configuration and Data Loading Module
Handles Google Cloud Storage authentication and data retrieval
"""

import os
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GCSDataLoader:
    """Handles GCS authentication and data loading for AIv3 system"""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize GCS data loader
        
        Args:
            credentials_path: Path to GCS service account JSON file
        """
        self.credentials_path = credentials_path or self._find_credentials()
        self.project_id = "ignition-ki-csv-storage"
        self.bucket_name = "ignition-ki-csv-data-2025-user123"
        self.client = None
        self.bucket = None
        
        # Initialize GCS client
        self._initialize_client()
        
    def _find_credentials(self) -> str:
        """Find GCS credentials file"""
        # Check common locations
        possible_paths = [
            r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json",
            r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json",
            "./gcs_credentials.json",
            os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"Found credentials at: {path}")
                return path
        
        raise FileNotFoundError("GCS credentials file not found. Please provide path.")
    
    def _initialize_client(self):
        """Initialize GCS client with credentials"""
        try:
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Create client
            self.client = storage.Client(
                credentials=credentials,
                project=self.project_id
            )
            
            # Get bucket
            self.bucket = self.client.bucket(self.bucket_name)
            
            logger.info(f"Successfully connected to GCS bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def list_pattern_files(self, prefix: str = "patterns/") -> List[str]:
        """List all pattern files in GCS bucket"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            files = [blob.name for blob in blobs if blob.name.endswith('.json')]
            logger.info(f"Found {len(files)} pattern files in GCS")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def load_patterns_from_gcs(self, pattern_file: str = None) -> List[Dict]:
        """Load consolidation patterns from GCS"""
        patterns = []
        
        try:
            if pattern_file:
                # Load specific file
                blob = self.bucket.blob(pattern_file)
                if blob.exists():
                    content = blob.download_as_text()
                    patterns = json.loads(content)
                    logger.info(f"Loaded {len(patterns)} patterns from {pattern_file}")
            else:
                # Load all pattern files
                pattern_files = self.list_pattern_files()
                for file in pattern_files:
                    blob = self.bucket.blob(file)
                    content = blob.download_as_text()
                    file_patterns = json.loads(content)
                    patterns.extend(file_patterns)
                    logger.info(f"Loaded {len(file_patterns)} patterns from {file}")
            
        except Exception as e:
            logger.error(f"Error loading patterns from GCS: {e}")
        
        return patterns
    
    def load_price_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load price data for a specific ticker from GCS"""
        try:
            # Construct file path (adjust based on your GCS structure)
            file_path = f"price_data/{ticker}.csv"
            blob = self.bucket.blob(file_path)
            
            if not blob.exists():
                logger.warning(f"Price data not found for {ticker}")
                return pd.DataFrame()
            
            # Download and read CSV
            content = blob.download_as_text()
            df = pd.read_csv(pd.io.common.StringIO(content))
            
            # Parse dates
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            logger.info(f"Loaded {len(df)} days of price data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def save_analysis_results(self, results: Dict, filename: str = None) -> bool:
        """Save analysis results back to GCS"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"analysis_results/consolidation_analysis_{timestamp}.json"
            
            blob = self.bucket.blob(filename)
            
            # Convert results to JSON
            json_content = json.dumps(results, indent=2, default=str)
            
            # Upload to GCS
            blob.upload_from_string(json_content, content_type='application/json')
            
            logger.info(f"Saved analysis results to GCS: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to GCS: {e}")
            return False
    
    def load_historical_patterns(self, 
                               start_date: str = None,
                               end_date: str = None,
                               methods: List[str] = None) -> List[Dict]:
        """Load historical patterns with filtering options"""
        
        # Load all patterns
        patterns = self.load_patterns_from_gcs()
        
        # Filter patterns
        filtered_patterns = []
        for pattern in patterns:
            # Date filtering
            if start_date and pd.to_datetime(pattern.get('start_date')) < pd.to_datetime(start_date):
                continue
            if end_date and pd.to_datetime(pattern.get('end_date')) > pd.to_datetime(end_date):
                continue
            
            # Method filtering
            if methods and pattern.get('detection_method') not in methods:
                continue
            
            filtered_patterns.append(pattern)
        
        logger.info(f"Filtered to {len(filtered_patterns)} patterns")
        return filtered_patterns
    
    def create_pattern_dataset(self, patterns: List[Dict]) -> pd.DataFrame:
        """Convert patterns list to DataFrame for analysis"""
        
        df = pd.DataFrame(patterns)
        
        # Convert date columns
        date_columns = ['start_date', 'end_date', 'breakout_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Add calculated columns
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df['duration'] = (df['end_date'] - df['start_date']).dt.days
        
        if 'upper_boundary' in df.columns and 'lower_boundary' in df.columns:
            df['boundary_width'] = df['upper_boundary'] - df['lower_boundary']
            df['boundary_width_pct'] = (df['boundary_width'] / df['lower_boundary']) * 100
        
        return df


def setup_gcs_environment():
    """Setup GCS environment variables and configuration"""
    
    # Set credentials path
    credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
    
    if os.path.exists(credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logger.info("GCS credentials configured successfully")
        return True
    else:
        logger.error(f"Credentials file not found at: {credentials_path}")
        return False


# Integration with the analysis runner
def load_patterns_for_analysis(use_gcs: bool = True, local_file: str = None) -> List[Dict]:
    """
    Load patterns from GCS or local file for analysis
    
    Args:
        use_gcs: Whether to load from GCS
        local_file: Path to local JSON file (if not using GCS)
    
    Returns:
        List of pattern dictionaries
    """
    
    if use_gcs:
        # Setup GCS
        setup_gcs_environment()
        
        # Initialize loader
        loader = GCSDataLoader()
        
        # Load patterns
        patterns = loader.load_historical_patterns()
        
        if not patterns:
            logger.warning("No patterns found in GCS, generating sample data...")
            return None
        
        return patterns
        
    elif local_file:
        # Load from local file
        with open(local_file, 'r') as f:
            patterns = json.load(f)
        logger.info(f"Loaded {len(patterns)} patterns from {local_file}")
        return patterns
    
    else:
        logger.warning("No data source specified")
        return None