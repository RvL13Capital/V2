"""
Check and configure GCP permissions for BigQuery Storage API
"""

import os
import json
from google.cloud import bigquery
from google.cloud import bigquery_storage
import sys

def check_permissions():
    """Check current GCP permissions and configuration"""

    # Get credentials path
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("[ERROR] GOOGLE_APPLICATION_CREDENTIALS not set!")
        print("   Set it with: set GOOGLE_APPLICATION_CREDENTIALS=path\\to\\your\\credentials.json")
        return False

    print(f"[OK] Credentials path: {creds_path}")

    # Load service account info
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            service_account = creds_data.get('client_email', 'Unknown')
            project_id = creds_data.get('project_id', 'Unknown')

        print(f"[OK] Service Account: {service_account}")
        print(f"[OK] Project ID: {project_id}")
    except Exception as e:
        print(f"[ERROR] Error reading credentials: {e}")
        return False

    print("\n" + "="*60)
    print("Testing BigQuery Standard API...")
    print("="*60)

    # Test BigQuery standard API
    try:
        bq_client = bigquery.Client(project=project_id)
        # Try a simple query
        query = "SELECT 1 as test"
        result = bq_client.query(query).result()
        for row in result:
            pass
        print("[OK] BigQuery Standard API works!")
    except Exception as e:
        print(f"[ERROR] BigQuery Standard API error: {e}")
        return False

    print("\n" + "="*60)
    print("Testing BigQuery Storage API...")
    print("="*60)

    # Test BigQuery Storage API
    try:
        storage_client = bigquery_storage.BigQueryReadClient()
        print("[OK] BigQuery Storage API client created!")

        # Try to read from a table (will fail if no permissions)
        try:
            # Test with a small query
            query = """
            SELECT 'test' as col
            """
            query_job = bq_client.query(query)
            results = query_job.result()

            # Try to convert with Storage API
            df = results.to_dataframe(create_bqstorage_client=True)
            print("[OK] BigQuery Storage API fully functional!")
            return True

        except Exception as e:
            if 'permission' in str(e).lower() or 'readsession' in str(e).lower():
                print(f"[ERROR] Missing permission: bigquery.readsessions.create")
                print("\nTO FIX THIS:")
                print("="*60)
                print("Option 1: Via Google Cloud Console")
                print("-"*40)
                print(f"1. Go to: https://console.cloud.google.com/iam-admin/iam?project={project_id}")
                print(f"2. Find service account: {service_account}")
                print("3. Click 'Edit' (pencil icon)")
                print("4. Add role: 'BigQuery Read Session User'")
                print("5. Click 'Save'\n")

                print("Option 2: Via gcloud CLI")
                print("-"*40)
                print("Run this command:")
                print(f"\ngcloud projects add-iam-policy-binding {project_id} \\")
                print(f'    --member="serviceAccount:{service_account}" \\')
                print('    --role="roles/bigquery.readSessionUser"\n')

                print("Option 3: Ask your GCP admin")
                print("-"*40)
                print(f"Request 'BigQuery Read Session User' role for: {service_account}")

                return False
            else:
                print(f"[ERROR] Storage API error: {e}")
                return False

    except Exception as e:
        if 'Could not load the default credentials' in str(e):
            print("[ERROR] Authentication error. Check your credentials file.")
        else:
            print(f"[ERROR] BigQuery Storage API not available: {e}")
            print("\nTO FIX THIS:")
            print("="*60)
            print(f"1. Go to: https://console.cloud.google.com/apis/library?project={project_id}")
            print("2. Search for 'BigQuery Storage API'")
            print("3. Click on it and press 'ENABLE'")
            print("\nOr run:")
            print(f"gcloud services enable bigquerystorage.googleapis.com --project={project_id}")
        return False

def estimate_costs(num_tickers=3000):
    """Estimate costs for processing tickers"""
    print("\n" + "="*60)
    print(f"COST ESTIMATION FOR {num_tickers} TICKERS")
    print("="*60)

    # Estimates
    avg_days_per_ticker = 365 * 2  # 2 years of data
    bytes_per_row = 100  # Approximate
    total_rows = num_tickers * avg_days_per_ticker
    total_gb = (total_rows * bytes_per_row) / 1e9

    print(f"Estimated data size: {total_gb:.2f} GB")
    print(f"Free tier limit: 1000 GB/month, 33 GB/day")

    if total_gb <= 33:
        print(f"[OK] Will fit in daily free tier!")
    elif total_gb <= 1000:
        print(f"[OK] Will fit in monthly free tier (split over {int(total_gb/33)+1} days)")
    else:
        cost = (total_gb - 1000) * 0.005  # $5 per TB after free tier
        print(f"[WARNING] Exceeds free tier. Estimated cost: ${cost:.2f}")

    print("\nRECOMMENDATIONS:")
    print("-"*40)
    if total_gb > 33:
        chunks = int(total_gb / 30) + 1
        tickers_per_chunk = num_tickers // chunks
        print(f"* Process in {chunks} chunks of {tickers_per_chunk} tickers")
        print(f"* Run one chunk per day to stay in free tier")
    else:
        print(f"* Can process all {num_tickers} tickers in one run")

    print("\nOPTIMIZATIONS:")
    print("-"*40)
    print("* Use BigQuery Storage API (10x faster downloads)")
    print("* Filter date ranges to reduce data")
    print("* Use columnar selection in queries")
    print("* Cache results locally after processing")

if __name__ == "__main__":
    print("GCP BigQuery Configuration Checker")
    print("="*60 + "\n")

    # Check permissions
    success = check_permissions()

    # Show cost estimates
    estimate_costs(3000)

    if success:
        print("\n[SUCCESS] ALL CHECKS PASSED! Ready to process 3000+ tickers!")
    else:
        print("\n[WARNING] Fix the issues above before processing large datasets")
        sys.exit(1)