"""
GCS Connection and Data Diagnostic Tool
Helps identify and fix GCS data loading issues
"""

import os
import sys
import json
import traceback
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import pandas as pd
from datetime import datetime

def test_gcs_connection():
    """Test GCS connection and diagnose issues"""
    
    print("="*80)
    print("GCS CONNECTION DIAGNOSTIC TOOL")
    print("="*80)
    
    results = {
        'credentials_found': False,
        'credentials_valid': False,
        'gcs_connected': False,
        'bucket_accessible': False,
        'files_found': [],
        'csv_files': [],
        'json_files': [],
        'errors': []
    }
    
    # Step 1: Check credentials file
    print("\n1. Checking for credentials file...")
    credential_paths = [
        r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json",
        r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json",
        os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    ]
    
    credentials_path = None
    for path in credential_paths:
        if path and os.path.exists(path):
            credentials_path = path
            print(f"   ✓ Found credentials at: {path}")
            results['credentials_found'] = True
            break
    
    if not credentials_path:
        print("   ✗ No credentials file found!")
        print("   Expected locations:")
        for path in credential_paths:
            print(f"     - {path}")
        return results
    
    # Step 2: Validate credentials
    print("\n2. Validating credentials...")
    try:
        with open(credentials_path, 'r') as f:
            cred_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in cred_data]
        
        if missing_fields:
            print(f"   ✗ Missing fields in credentials: {missing_fields}")
            results['errors'].append(f"Missing credential fields: {missing_fields}")
        else:
            print(f"   ✓ Credentials valid")
            print(f"   - Project ID: {cred_data['project_id']}")
            print(f"   - Service Account: {cred_data['client_email']}")
            results['credentials_valid'] = True
            
    except Exception as e:
        print(f"   ✗ Error reading credentials: {e}")
        results['errors'].append(f"Credential read error: {str(e)}")
        return results
    
    # Step 3: Test GCS connection
    print("\n3. Testing GCS connection...")
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        client = storage.Client(
            credentials=credentials,
            project=cred_data['project_id']
        )
        
        # Test connection by listing buckets
        buckets = list(client.list_buckets())
        print(f"   ✓ Connected to GCS")
        print(f"   - Found {len(buckets)} buckets")
        for bucket in buckets[:5]:  # Show first 5
            print(f"     • {bucket.name}")
        
        results['gcs_connected'] = True
        
    except Exception as e:
        print(f"   ✗ Failed to connect to GCS: {e}")
        results['errors'].append(f"GCS connection error: {str(e)}")
        return results
    
    # Step 4: Check specific bucket
    print("\n4. Checking bucket access...")
    bucket_name = "ignition-ki-csv-data-2025-user123"
    
    try:
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        if bucket.exists():
            print(f"   ✓ Bucket '{bucket_name}' exists and is accessible")
            results['bucket_accessible'] = True
        else:
            print(f"   ✗ Bucket '{bucket_name}' not found")
            
            # Try alternative bucket names
            print("\n   Searching for alternative buckets...")
            for b in buckets:
                if 'ignition' in b.name.lower() or 'csv' in b.name.lower():
                    print(f"     Found potential bucket: {b.name}")
                    
                    # Ask user if this is correct
                    response = input(f"     Is '{b.name}' your data bucket? (y/n): ")
                    if response.lower() == 'y':
                        bucket_name = b.name
                        bucket = client.bucket(bucket_name)
                        results['bucket_accessible'] = True
                        break
            
    except Exception as e:
        print(f"   ✗ Error accessing bucket: {e}")
        results['errors'].append(f"Bucket access error: {str(e)}")
        return results
    
    # Step 5: List files in bucket
    print(f"\n5. Listing files in bucket '{bucket_name}'...")
    try:
        blobs = bucket.list_blobs()
        file_list = []
        csv_files = []
        json_files = []
        parquet_files = []
        
        print("   Scanning bucket contents...")
        for blob in blobs:
            file_list.append(blob.name)
            
            # Categorize files
            if blob.name.endswith('.csv'):
                csv_files.append(blob.name)
            elif blob.name.endswith('.json'):
                json_files.append(blob.name)
            elif blob.name.endswith('.parquet'):
                parquet_files.append(blob.name)
        
        results['files_found'] = file_list
        results['csv_files'] = csv_files
        results['json_files'] = json_files
        
        print(f"   ✓ Found {len(file_list)} total files")
        print(f"     - CSV files: {len(csv_files)}")
        print(f"     - JSON files: {len(json_files)}")
        print(f"     - Parquet files: {len(parquet_files)}")
        
        # Show sample files
        if csv_files:
            print("\n   Sample CSV files:")
            for f in csv_files[:5]:
                print(f"     • {f}")
        
        if json_files:
            print("\n   Sample JSON files:")
            for f in json_files[:5]:
                print(f"     • {f}")
        
        # Look for pattern-related files
        print("\n   Searching for pattern files...")
        pattern_files = [f for f in file_list if 'pattern' in f.lower() or 'consolidation' in f.lower()]
        if pattern_files:
            print(f"   Found {len(pattern_files)} pattern-related files:")
            for f in pattern_files[:10]:
                print(f"     • {f}")
        else:
            print("   ✗ No files with 'pattern' or 'consolidation' in name")
            
            # Search for other potential data files
            print("\n   Looking for potential data files...")
            data_dirs = set()
            for f in file_list:
                if '/' in f:
                    dir_name = f.split('/')[0]
                    data_dirs.add(dir_name)
            
            if data_dirs:
                print(f"   Found {len(data_dirs)} directories:")
                for d in sorted(data_dirs)[:10]:
                    dir_files = [f for f in file_list if f.startswith(f"{d}/")]
                    print(f"     • {d}/ ({len(dir_files)} files)")
        
    except Exception as e:
        print(f"   ✗ Error listing files: {e}")
        results['errors'].append(f"File listing error: {str(e)}")
        traceback.print_exc()
    
    # Step 6: Try to read a sample file
    if csv_files or json_files:
        print("\n6. Testing file reading...")
        
        # Try CSV first
        if csv_files:
            test_file = csv_files[0]
            print(f"   Testing CSV file: {test_file}")
            try:
                blob = bucket.blob(test_file)
                content = blob.download_as_text()
                
                # Try to parse as CSV
                import io
                df = pd.read_csv(io.StringIO(content), nrows=5)
                print(f"   ✓ Successfully read CSV file")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
                
            except Exception as e:
                print(f"   ✗ Error reading CSV: {e}")
        
        # Try JSON
        if json_files:
            test_file = json_files[0]
            print(f"\n   Testing JSON file: {test_file}")
            try:
                blob = bucket.blob(test_file)
                content = blob.download_as_text()
                
                # Try to parse as JSON
                data = json.loads(content)
                print(f"   ✓ Successfully read JSON file")
                
                if isinstance(data, list):
                    print(f"   Type: List with {len(data)} items")
                    if data:
                        print(f"   Sample keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                elif isinstance(data, dict):
                    print(f"   Type: Dictionary")
                    print(f"   Keys: {list(data.keys())[:10]}")
                
            except Exception as e:
                print(f"   ✗ Error reading JSON: {e}")
    
    # Step 7: Provide recommendations
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if results['bucket_accessible'] and results['files_found']:
        print("✓ GCS connection successful")
        print(f"✓ Found {len(results['files_found'])} files in bucket")
        
        print("\nRECOMMENDATIONS:")
        print("-"*40)
        
        if not any('pattern' in f.lower() for f in results['files_found']):
            print("⚠ No pattern files found. Possible issues:")
            print("  1. Patterns might be stored with different naming")
            print("  2. Patterns might be in a different bucket")
            print("  3. Patterns need to be generated first")
            
            print("\nTo fix:")
            print("  1. Check if patterns are stored in CSV files:")
            for csv in csv_files[:3]:
                print(f"     - {csv}")
            
            print("\n  2. Update gcs_config.py to look for correct file patterns")
            print("  3. Or generate patterns using your detection system first")
    else:
        print("✗ Issues detected with GCS connection")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Save diagnostic report
    report_file = 'gcs_diagnostic_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostic report saved to: {report_file}")
    
    return results


def find_pattern_data(client, bucket_name):
    """Deep search for pattern data in various formats"""
    
    print("\n" + "="*80)
    print("DEEP PATTERN SEARCH")
    print("="*80)
    
    bucket = client.bucket(bucket_name)
    
    # Search for various pattern indicators
    keywords = ['pattern', 'consolidation', 'breakout', 'detection', 'signal', 
                'stateful', 'boundary', 'qualification', 'outcome']
    
    found_files = {}
    
    print("Searching for pattern-related data...")
    for blob in bucket.list_blobs():
        name_lower = blob.name.lower()
        
        for keyword in keywords:
            if keyword in name_lower:
                if keyword not in found_files:
                    found_files[keyword] = []
                found_files[keyword].append(blob.name)
    
    if found_files:
        print("\nFound potential pattern files:")
        for keyword, files in found_files.items():
            print(f"\n  Keyword '{keyword}': {len(files)} files")
            for f in files[:3]:
                print(f"    • {f}")
    
    # Check for ticker data that might contain patterns
    print("\n\nChecking for ticker-based data...")
    ticker_files = {}
    
    for blob in bucket.list_blobs():
        # Look for ticker patterns (e.g., AAPL.csv, MSFT.json)
        import re
        ticker_match = re.search(r'([A-Z]{2,5})\.(csv|json|parquet)', blob.name)
        if ticker_match:
            ticker = ticker_match.group(1)
            if ticker not in ticker_files:
                ticker_files[ticker] = []
            ticker_files[ticker].append(blob.name)
    
    if ticker_files:
        print(f"\nFound {len(ticker_files)} ticker-based files")
        for ticker in list(ticker_files.keys())[:5]:
            print(f"  • {ticker}: {ticker_files[ticker]}")
    
    return found_files, ticker_files


if __name__ == "__main__":
    # Run diagnostic
    results = test_gcs_connection()
    
    # If connected, do deep search
    if results['gcs_connected'] and results['bucket_accessible']:
        print("\nWould you like to perform a deep search for pattern data? (y/n): ", end='')
        if input().lower() == 'y':
            try:
                # Re-establish connection for deep search
                credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                client = storage.Client(credentials=credentials)
                
                bucket_name = "ignition-ki-csv-data-2025-user123"
                pattern_files, ticker_files = find_pattern_data(client, bucket_name)
                
            except Exception as e:
                print(f"Error during deep search: {e}")
                traceback.print_exc()