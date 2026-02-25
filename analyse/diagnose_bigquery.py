"""
Diagnose BigQuery issues
"""
import os
import sys
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

def check_bigquery():
    print("="*60)
    print("BIGQUERY DIAGNOSIS")
    print("="*60)

    # 1. Check if BigQuery client can be created
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project='ignition-ki-csv-storage')
        print("[OK] BigQuery client created")
    except Exception as e:
        print(f"[ERROR] Cannot create BigQuery client: {e}")
        return False

    # 2. Check if dataset exists
    try:
        dataset_id = 'ignition-ki-csv-storage.market_analysis'
        dataset = client.get_dataset(dataset_id)
        print(f"[OK] Dataset exists: {dataset_id}")
    except Exception as e:
        print(f"[INFO] Dataset doesn't exist: {e}")
        try:
            # Try to create it
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset = client.create_dataset(dataset)
            print(f"[OK] Created dataset: {dataset_id}")
        except Exception as e2:
            print(f"[ERROR] Cannot create dataset: {e2}")

    # 3. Check if table exists and has data
    try:
        table_id = 'ignition-ki-csv-storage.market_analysis.market_data'
        table = client.get_table(table_id)
        print(f"[OK] Table exists: {table_id}")
        print(f"    Rows: {table.num_rows}")
        print(f"    Size: {table.num_bytes / 1e6:.2f} MB")

        # Get sample data
        query = f"SELECT COUNT(DISTINCT ticker) as tickers FROM `{table_id}` LIMIT 1"
        result = client.query(query).result()
        for row in result:
            print(f"    Unique tickers in table: {row.tickers}")

    except Exception as e:
        print(f"[WARNING] Table doesn't exist or no data: {e}")
        print("[INFO] This means data needs to be loaded from GCS first")

    # 4. Test a simple query
    try:
        query = "SELECT 1 as test"
        result = client.query(query).result()
        print("[OK] Can execute queries")
    except Exception as e:
        print(f"[ERROR] Cannot execute queries: {e}")
        return False

    # 5. Check Storage API
    try:
        from google.cloud import bigquery_storage
        storage_client = bigquery_storage.BigQueryReadClient()
        print("[OK] BigQuery Storage API available")
    except Exception as e:
        print(f"[WARNING] Storage API issue: {e}")

    return True

if __name__ == "__main__":
    success = check_bigquery()

    if not success:
        print("\n[RECOMMENDATION] Use local processing instead:")
        print("python cloud_market_analysis.py --local-only --num-stocks 100")