"""
Simple GCS Test - Minimal code to test connection and list files
"""

import os
from google.cloud import storage
import json

def simple_gcs_test():
    """Simplest possible GCS test"""
    
    print("SIMPLE GCS CONNECTION TEST")
    print("-" * 40)
    
    # Set credentials
    cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
    
    # Check if file exists
    if not os.path.exists(cred_path):
        print(f"ERROR: Credentials file not found at:\n{cred_path}")
        
        # Check alternative path
        alt_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
        if os.path.exists(alt_path):
            print(f"\nFound at alternative path:\n{alt_path}")
            cred_path = alt_path
        else:
            return
    
    print(f"✓ Credentials found: {cred_path}")
    
    # Set environment variable
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
    
    try:
        # Create client
        print("\nConnecting to GCS...")
        client = storage.Client()
        
        # List all buckets
        print("\nListing all buckets:")
        print("-" * 40)
        buckets = list(client.list_buckets())
        
        if not buckets:
            print("No buckets found!")
            return
        
        for i, bucket in enumerate(buckets, 1):
            print(f"{i}. {bucket.name}")
        
        # Try each bucket
        for bucket in buckets:
            print(f"\n\nChecking bucket: {bucket.name}")
            print("-" * 40)
            
            try:
                # List first 10 files
                blobs = list(bucket.list_blobs(max_results=10))
                
                if not blobs:
                    print("  Empty bucket")
                    continue
                
                print(f"  Found {len(blobs)} files (showing first 10):")
                for blob in blobs:
                    size_kb = blob.size / 1024 if blob.size else 0
                    print(f"    • {blob.name} ({size_kb:.1f} KB)")
                
                # Count total files
                all_blobs = list(bucket.list_blobs())
                print(f"\n  Total files in bucket: {len(all_blobs)}")
                
                # Look for patterns
                pattern_count = sum(1 for b in all_blobs if 'pattern' in b.name.lower())
                csv_count = sum(1 for b in all_blobs if b.name.endswith('.csv'))
                json_count = sum(1 for b in all_blobs if b.name.endswith('.json'))
                
                print(f"  Files with 'pattern': {pattern_count}")
                print(f"  CSV files: {csv_count}")
                print(f"  JSON files: {json_count}")
                
            except Exception as e:
                print(f"  Error accessing bucket: {e}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPossible issues:")
        print("1. Invalid credentials")
        print("2. No internet connection")
        print("3. Credentials don't have proper permissions")
        print("4. Project doesn't exist")
        
        # Try to read credentials to check project
        try:
            with open(cred_path, 'r') as f:
                creds = json.load(f)
                print(f"\nCredentials info:")
                print(f"  Project: {creds.get('project_id', 'NOT FOUND')}")
                print(f"  Email: {creds.get('client_email', 'NOT FOUND')}")
        except:
            pass


if __name__ == "__main__":
    simple_gcs_test()