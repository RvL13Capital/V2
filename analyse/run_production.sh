#!/bin/bash

# Production Deployment Script for Linux/Mac
# Runs full consolidation analysis on all available GCS data

echo "========================================"
echo "   PRODUCTION ANALYSIS DEPLOYMENT"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo ""
echo "Installing required packages..."
pip install -r requirements.txt --quiet

# Set environment variables
echo ""
echo "Setting environment variables..."
export GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"
export PROJECT_ID="ignition-ki-csv-storage"
export GCS_BUCKET_NAME="ignition-ki-csv-data-2025-user123"

# Create necessary directories
mkdir -p logs
mkdir -p production_analysis

# Run production deployment
echo ""
echo "========================================"
echo "   Starting Production Analysis..."
echo "========================================"
echo ""

python3 deploy_full_analysis.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "   DEPLOYMENT SUCCESSFUL"
    echo "========================================"
    echo "Results saved in production_analysis folder"
else
    echo ""
    echo "========================================"
    echo "   DEPLOYMENT FAILED"
    echo "========================================"
    echo "Check logs for details."
    exit 1
fi

echo ""
read -p "Press Enter to continue..."