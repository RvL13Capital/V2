"""
Basic GCS Test - Find and list data
"""

import os
import sys
from google.cloud import storage
import json

def test_gcs():
    """Test GCS connection and find data"""
    
    print("GCS CONNECTION TEST")
    print("-" * 40)
    
    # Credentials path
    cred_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
    
    # Check if file exists
    if not os.path.exists(cred_path):
        print(f"ERROR: Credentials not found at: {cred_path}")
        
        # Try alternative
        alt_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
        if os.path.exists(alt_path):
            print(f"Found at: {alt_path}")
            cred_path = alt_path
        else:
            print("\nPlease check your credentials file location")
            return
    
    print(f"OK: Credentials found")
    
    # Read credentials to get project info
    try:
        with open(cred_path, 'r') as f:
            creds = json.load(f)
            project_id = creds.get('project_id', 'unknown')
            print(f"Project ID: {project_id}")
    except Exception as e:
        print(f"Error reading credentials: {e}")
        return
    
    # Set environment variable
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
    
    try:
        # Create client
        print("\nConnecting to GCS...")
        client = storage.Client(project=project_id)
        
        # List all buckets
        print("\n" + "="*50)
        print("AVAILABLE BUCKETS:")
        print("="*50)
        
        buckets = list(client.list_buckets())
        
        if not buckets:
            print("ERROR: No buckets found in project!")
            print("\nPossible issues:")
            print("1. Wrong project ID")
            print("2. No buckets created yet")
            print("3. No permissions to list buckets")
            return
        
        # Show all buckets
        for i, bucket in enumerate(buckets, 1):
            print(f"\n{i}. Bucket: {bucket.name}")
            
            # Check each bucket for data
            try:
                # Get first few files to check
                blobs = list(bucket.list_blobs(max_results=5))
                
                if blobs:
                    print(f"   Contains data - Sample files:")
                    for blob in blobs[:3]:
                        print(f"     - {blob.name}")
                    
                    # Count total
                    total = sum(1 for _ in bucket.list_blobs())
                    print(f"   Total files: {total}")
                else:
                    print("   [Empty bucket]")
                    
            except Exception as e:
                print(f"   Error accessing: {e}")
        
        # Try the expected bucket
        print("\n" + "="*50)
        print("CHECKING EXPECTED BUCKET:")
        print("="*50)
        
        expected_bucket = "ignition-ki-csv-data-2025-user123"
        print(f"Looking for: {expected_bucket}")
        
        try:
            bucket = client.bucket(expected_bucket)
            if bucket.exists():
                print("OK: Bucket exists!")
                
                # List contents
                print("\nListing contents...")
                blobs = list(bucket.list_blobs())
                print(f"Total files: {len(blobs)}")
                
                if blobs:
                    # Categorize files
                    csv_files = [b.name for b in blobs if b.name.endswith('.csv')]
                    json_files = [b.name for b in blobs if b.name.endswith('.json')]
                    
                    print(f"CSV files: {len(csv_files)}")
                    print(f"JSON files: {len(json_files)}")
                    
                    # Show samples
                    if csv_files:
                        print("\nSample CSV files:")
                        for f in csv_files[:5]:
                            print(f"  - {f}")
                    
                    if json_files:
                        print("\nSample JSON files:")
                        for f in json_files[:5]:
                            print(f"  - {f}")
                    
                    # Look for patterns
                    pattern_files = [b.name for b in blobs if 'pattern' in b.name.lower()]
                    if pattern_files:
                        print(f"\nFiles with 'pattern': {len(pattern_files)}")
                        for f in pattern_files[:5]:
                            print(f"  - {f}")
                    else:
                        print("\nNO FILES WITH 'pattern' IN NAME")
                        print("Data might be stored differently")
                        
                        # Show directory structure
                        dirs = set()
                        for b in blobs:
                            if '/' in b.name:
                                dir_name = b.name.split('/')[0]
                                dirs.add(dir_name)
                        
                        if dirs:
                            print(f"\nFound {len(dirs)} directories:")
                            for d in sorted(dirs)[:10]:
                                count = sum(1 for b in blobs if b.name.startswith(f"{d}/"))
                                print(f"  - {d}/ ({count} files)")
                else:
                    print("Bucket is empty!")
            else:
                print(f"ERROR: Bucket '{expected_bucket}' does not exist!")
                print("\nYour data might be in one of the buckets listed above")
                
        except Exception as e:
            print(f"Error checking bucket: {e}")
        
    except Exception as e:
        print(f"\nCONNECTION ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify credentials are valid")
        print("3. Ensure project ID is correct")
        print("4. Check service account permissions")
        
        import traceback
        print("\nFull error:")
        traceback.print_exc()


if __name__ == "__main__":
    test_gcs()