"""
Test GCS Connection and Diagnose SSL Issues
"""
import os
import sys

print("="*70)
print("GCS CONNECTION DIAGNOSTIC TEST")
print("="*70)
print()

# Test 1: Check credentials file
print("[TEST 1] Checking GCS credentials file...")
cred_file = "gcs-key.json"
if os.path.exists(cred_file):
    print(f"  [OK] Found: {cred_file}")
    import json
    with open(cred_file, 'r') as f:
        creds = json.load(f)
        print(f"  [OK] Project: {creds.get('project_id', 'N/A')}")
else:
    print(f"  [ERROR] Not found: {cred_file}")
    sys.exit(1)

print()

# Test 2: Check proxy settings
print("[TEST 2] Checking proxy settings...")
http_proxy = os.environ.get('HTTP_PROXY', 'Not set')
https_proxy = os.environ.get('HTTPS_PROXY', 'Not set')
no_proxy = os.environ.get('NO_PROXY', 'Not set')
print(f"  HTTP_PROXY: {http_proxy}")
print(f"  HTTPS_PROXY: {https_proxy}")
print(f"  NO_PROXY: {no_proxy}")

if http_proxy != 'Not set' or https_proxy != 'Not set':
    print("  [WARNING] Proxy detected - may cause SSL issues")
    print("  [FIX] Set NO_PROXY=*.googleapis.com")
else:
    print("  [OK] No proxy configured")

print()

# Test 3: Test basic HTTPS connectivity to Google
print("[TEST 3] Testing HTTPS connectivity to Google APIs...")
try:
    import requests
    response = requests.get('https://www.googleapis.com/', timeout=10)
    print(f"  [OK] Can reach googleapis.com (Status: {response.status_code})")
except requests.exceptions.SSLError as e:
    print(f"  [ERROR] SSL Error: {e}")
    print("  [FIX] This is likely a proxy/firewall issue")
    print("        - Disable VPN/Proxy")
    print("        - Check corporate firewall SSL inspection")
except Exception as e:
    print(f"  [ERROR] Connection failed: {e}")

print()

# Test 4: Try to authenticate
print("[TEST 4] Testing GCS authentication...")
try:
    from google.cloud import storage
    from google.oauth2 import service_account

    # Set environment variable to disable mTLS (may help with SSL issues)
    os.environ['GOOGLE_AUTH_DISABLE_MTLS'] = '1'

    credentials = service_account.Credentials.from_service_account_file(
        cred_file,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    print("  [OK] Credentials loaded")

    # Try to create client
    client = storage.Client(
        credentials=credentials,
        project=creds.get('project_id')
    )

    print("  [OK] GCS Client created")

    # Try to access bucket
    bucket_name = "ignition-ki-csv-data-2025-user123"
    bucket = client.bucket(bucket_name)

    print(f"  [OK] Connected to bucket: {bucket_name}")

    # Try to list a few blobs
    blobs = list(bucket.list_blobs(max_results=5))
    print(f"  [OK] Can list blobs (found {len(blobs)} files)")

    print()
    print("="*70)
    print("[SUCCESS] GCS connection is working!")
    print("="*70)

except Exception as e:
    print(f"  [ERROR] Authentication failed: {e}")
    print()
    print("="*70)
    print("[FAILED] GCS connection not working")
    print("="*70)
    print()
    print("TROUBLESHOOTING:")
    print("1. Check if you're behind a proxy/VPN")
    print("2. Try: set NO_PROXY=*.googleapis.com")
    print("3. Try: set GOOGLE_AUTH_DISABLE_MTLS=1")
    print("4. Disable antivirus SSL scanning temporarily")
    print("5. Check if corporate firewall is blocking access")
    print()
    sys.exit(1)
