# Set GCS credentials for TRANS system
# Run this before using the system: . .\set_env.ps1

$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\vonLinck Capital\trans\gcs_credentials.json"
$env:PROJECT_ID="ignition-ki-csv-storage"
$env:GCS_BUCKET_NAME="ignition-ki-csv-data-2025-user123"

Write-Host ""
Write-Host "========================================"
Write-Host "TRANS Environment Configured"
Write-Host "========================================"
Write-Host "GCS Credentials: $env:GOOGLE_APPLICATION_CREDENTIALS"
Write-Host "Project ID: $env:PROJECT_ID"
Write-Host "Bucket: $env:GCS_BUCKET_NAME"
Write-Host ""
Write-Host "Environment variables set for this session."
Write-Host "You can now run Python scripts."
Write-Host ""
